# 运行与接入

## 当前运行方式

更新：server 现在默认 `H3_MODEL_LIFETIME=resident`。父进程仅负责接单，专用推理子进程常驻 conditioner 和 Eros generator 的 CPU 权重；启动完成权重校验、加载、按需 Eros 转换和两侧 GPU 放置探测后 ready。每单只做 CPU/GPU 搬运、编码和推理，不再从磁盘重新加载模型或重新转换 Eros。两套未量化模型不会同时放上 GPU。下面关于逐阶段子进程/每单冷加载的说明仅适用于显式 `H3_MODEL_LIFETIME=cold` 和未加 `--resident` 的 benchmark。

常驻启动要求有效可用 RAM 至少 160 GiB，可用 `H3_RESIDENT_MIN_RAM_GIB` 调整；加载后/每单开始前保留至少 16 GiB 有效可用量。有效可用量取主机 MemAvailable 与 cgroup 剩余配额加 inactive_file 可回收缓存的较小值，这是容量估计，不保证未来无 OOM。内存不足明确失败，不自动退到冷加载。已提供的 Vast 容器上限约 342 GiB、当前约 180 GiB，不能按宿主机 1 TiB 预算。

常驻推理发生错误、超时或失去续租时，整个模型进程回收，server 退出 ready 并停止消费；当前任务依既有终态/重投规则处理。恢复需要重启 worker，届时重新加载一次。不会继续复用 CUDA 错误后的进程。

连续两单复用验收入口（新输出目录，每次成片保存在 run-01/run-02）：

```bash
python benchmark.py --resident --runs 2 --image /workspace/test1.jpg \
  --prompt-file /workspace/prompt.txt --output /workspace/h3-resident-check
```

该入口和 server 复用同一个 ResidentEngine，第二单应无模型读盘/权重转换阶段；仍会记录 GPU 搬运与回 CPU 耗时。首次算子初始化仍可能使第一单更慢。该模式会在 benchmark 结束时关闭常驻进程；server 则保留至停止或异常。

worker 父进程不导入 torch。启动时核对固定依赖和文件哈希，分别启动 conditioner/generator probe，把模型实际放上 GPU 后退出；通过后开始 SQS 消费。每单也是先 conditioner 子进程、后 generator 子进程，阶段退出后 CUDA context 随进程释放。

与设计中的后续常驻优化相比，首版降低了两份模型同时占主机内存的需求，但增加了每单磁盘读取和转换时间。实际 RAM、VRAM、速度尚待 GPU 测量。默认要求空闲显存至少 88 GiB，可由 `H3_MIN_FREE_GIB` 调整；调小不代表较小显卡已经受支持。

## 安装及启动

脚本以本目录为工作目录。GPU 主机需要 Python 3.12、git、ffmpeg/ffprobe、能够支持 cu130 wheel 的驱动。`deploy.sh` 不安装驱动，不启动消费任务，不修改 Wan 环境。

```bash
./deploy.sh
./start.sh
```

`deploy.sh` 创建独立 `.venv`，安装候选固定 GPU 依赖、下载指定 revisions、校验 Eros SHA-256 并生成基础文件哈希清单。`start.sh` 加载可选 `.env`，默认只监听 `127.0.0.1:8001`，单个 uvicorn worker。同一数据目录通过 flock 防止重复启动；跨机器仍必须确保只有一个 worker 消费该队列。

必填运行环境：`H3_S3_BUCKET`、`H3_QUEUE_URL`、`H3_PUBLIC_BASE_URL`。CDN base 是桶映射根路径，不要重复加 `video/h3`。凭证由 IAM role 或标准 AWS 环境注入；回调密钥通过 `H3_WEBHOOK_SECRET` 注入。配置变量说明在 `.env.example`，不含真实凭证。

`deploy.sh` 默认调用 `setup-aws.sh`：缺少凭证时隐藏输入，先验证 STS，再原子保存到本机 `.env`（0600，已忽略 Git）。已有 AWS_PROFILE/角色继续使用其 provider，不复制临时角色凭证。设置 `H3_AWS_SETUP=0` 可跳过部署时配置。非交互部署可注入标准 AWS 环境变量；显式配置流程会保存静态 key/token，临时 token 到期后必须更新。

`./setup-aws.sh` 现在只配置并保存，不启动或重启 GPU 服务；`./setup-aws.sh --force` 用于轮换。之后 `./start.sh` 自动读取保存值。start 和直接 uvicorn 的 worker 都会在模型预加载前验证 STS 与队列 GetQueueAttributes；该检查不消费消息。缺权限、队列 region 错误等可提前失败。STS/队列检查不代表 S3 读写权限全部验证。

Vast 没有 EC2 instance role，交互配置入口默认禁用 EC2 metadata 凭证探测；显式设置 `AWS_EC2_METADATA_DISABLED=false` 可覆盖。修改准则：云端 worker 在昂贵的模型预加载之前验证 AWS 身份，避免加载十几分钟后才发现缺少凭证。

模型下载约涉及 140+ GB 的基础组件及约 40 GB Eros，磁盘还需为缓存、输出和安装预留余量。部署脚本不自动清理既有模型。重新下载会重建本地组件路径与 manifest；移动模型目录后须重建该清单。

## 单机生成验收入口

`benchmark.py` 接受本地图片和提示词文件，生成真实视频；不会访问队列、S3 或业务回调。输出目录必须不存在。

```bash
.venv/bin/python benchmark.py --image /path/to/input.png --prompt-file /path/to/prompt.txt \
  --output data/bench-540p-5s --resolution 540p --duration 5 --steps 6 --seed 42 --audio
```

输出包含 `output.mp4`、转换覆盖报告、各阶段报告、音频指标及总报告。临时提示词 job 和 embeddings 完成后删除。所有生成都含冷加载，不能把报告称为常驻模型吞吐测试。失败退出非零，不生成伪成功结果。

benchmark 默认 preflight 只运行依赖/CUDA kernel 检查和全量权重哈希校验；不额外加载两套模型作空探测，避免正式生成前重复读取、转换和 GPU 放置。`--load-probes` 可恢复额外探测；worker 启动仍保留完整探测。

哈希阶段每 5 秒报告当前文件、读取 GiB 和速率；模型阶段报告当前组件、Eros 转换进度及 GPU 放置，并至少每 10 秒报告子进程 PID/存活等待时间。子进程第三方 stdout/stderr 仍不直接透传，防止提示词出现在日志；受控阶段状态通过独立 JSON 文件传回。benchmark 的 preflight 报告保留在输出目录的 `preflight/`。

`--output` 必须是新目录，不是 MP4 文件名。例如传入 `/workspace/test.mp4`，实际视频路径会是 `/workspace/test.mp4/output.mp4`。已有目录会拒绝使用，且不修改其报告或删除其临时文件。

Vast 的系统 `nvcc=13.2` 与旧 venv 的 `torch.version.cuda=12.8` 可以并存。H3 固定安装 `torch==2.11.0+cu130` / `torchvision==0.26.0+cu130`，不复用 cu128 环境；当前没有编译 Sage/FlashAttention 扩展，不要求把系统 Toolkit 改为 13.0。部署先检查驱动 R580+，安装后实际运行 BF16/cuDNN kernel，再下载模型。驱动下限仅为初步条件，不代替实际运算检查。参考 [PyTorch wheel 安装组合](https://pytorch.org/get-started/previous-versions/) 和 [NVIDIA CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 现有 Wan Lambda 接入（唯一入口）

继续使用已部署的 Wan API Gateway/Lambda，不部署本目录的 lambda_api；该目录仅保留历史实现，当前不作为上线步骤。H3 是现有接单链路的 GPU 消费端。

worker 接受 Wan 平铺 SQS JSON：task_id、image_url、last_image_url、prompt、negative_prompt、duration、resolution、webhook_url、seed、steps、quality、audio。null 的 steps/resolution/duration 按 H3 默认处理。已有嵌套 request 格式保留兼容，但不要求生产者改变格式。

S3 中 request 保留 Wan 原始消息；内部仅将其归一化后比对，阶段/终态写回原始结构。回调保持 Wan 的 id/task_id/status/video_url/error/seed/duration/resolution 字段，以及 X-Wan-Signature 签名头。实际画幅、帧数和音轨信息保留在状态 result 内，不扩展旧回调字段。

配置直接兼容 SQS_QUEUE_URL、WAN22_S3_BUCKET（或 S3_BUCKET）、WAN22_S3_PREFIX、WAN22_STATUS_PREFIX（或 STATUS_PREFIX）、WAN22_S3_PUBLIC_BASE_URL、WAN22_WEBHOOK_SECRET/WEBHOOK_HOSTS/IMAGE_HOSTS；H3_* 显式值可覆盖。视频和状态前缀分开：H3_S3_PREFIX 可以放 video/h3/，状态默认仍为旧 Lambda 的 video/wan22/status/；若现网 Lambda 修改过 STATUS_PREFIX，则在 GPU 机配置同值的 STATUS_PREFIX 或 H3_STATUS_PREFIX。旧 .env 已显式填写的值不会被自动覆盖。

本地已通过原始 server/lambda_api/status.py 的 describe 实现测试，确认其能从同一 key 读取 H3 终态和视频 URL。检查真实任务参数可运行 python -m scripts.check_task MESSAGE_FILE；不涉及 GPU 或队列操作。

同一无路由队列中的消息没有模型选择字段，同时启动 Wan 与 H3 消费者时会由抢到消息的一方生成，不能保证模型选择。当前可以用原接单入口/原队列，仅保持所需模型的 worker 活跃；统一项目与 adapter 路由按用户要求后续重构。

```json
{
  "image": "https://images.example.com/frame.png",
  "prompt": "A quiet forest in soft morning light, leaves moving gently in the breeze.",
  "duration": 5,
  "resolution": "540p",
  "steps": 6,
  "seed": 42,
  "audio": true
}
```

支持范围：2–14 秒请求、540p/720p 档、横竖 16:9/9:16、4–8 步。5 秒实际为 124 帧 / 24 fps；查询与回调给出实际尺寸和时长。GPU 未验收前，这些是代码接受范围，不是已认证能力清单。

`negativePrompt` 非空、`quality` 非空、480p/1080p 和未知字段返回 422。`audio=false` 输出无音轨 MP4，但仍进行模型联合生成。图片仅 HTTPS，最大 25 MiB / 4000 万像素，最多 3 次重定向；每次连接解析并固定目标 IP，拒绝图片私网地址。回调只有明确配置 host 白名单才允许私网，始终拒绝链路本地和 loopback。

提交成功为 202/queued。503/enqueue_unconfirmed 会附带 task_id，表示提交结果不确定，应先查询或核对队列，不能盲目重复提交。查询状态不存在为 unknown，S3 故障为 503。

## AWS 权限及队列

使用现有接单 Standard queue；保留 DLQ redrive policy，建议 maxReceiveCount=5。完全非法的 envelope 保留给 DLQ，避免未知任务被静默丢弃。当前由所需模型的 worker 消费现有队列。

| 角色 | 必需权限 |
| --- | --- |
| Lambda | 对 H3 队列 SendMessage；启用积压限制时 GetQueueAttributes；对 H3 status 前缀 GetObject/PutObject；Lambda 常规日志权限 |
| Worker | 对 H3 队列 GetQueueAttributes/ReceiveMessage/DeleteMessage/ChangeMessageVisibility；对 H3 前缀 GetObject/PutObject 和 multipart 上传所需权限 |

S3 使用 KMS 时另需相应密钥权限。视频前缀可经 CDN 服务，状态对象保持私有，经 Lambda 查询；不要把包含业务请求的状态对象暴露到公共 CDN。

默认 visibility=600 秒，每 60 秒续租。续租或状态 heartbeat 失败时，worker 取消当前 GPU 子进程、停止接单并退出 ready；由运维重启，不能在未知执行权下继续生成。尚未提供多 worker fencing，禁止水平扩容。

## 恢复与排错

任务生命周期日志现覆盖：SQS 收到、校验/接单、读取及写入状态、首尾图下载及重试、画布处理、生成阶段/耗时、上传、回调每次 HTTP 响应、重复终态跳过、receive 上限失败、DeleteMessage 前后和任务总耗时。以 task_id 关联，不打印提示词、图片/回调 URL 或 receipt handle。空队列每约 60 秒输出轮询仍存活，长任务按续租间隔输出阶段。/health 增加 task_stage、received_messages、deleted_messages、last_task（终态/错误/耗时）。

图片失败日志记录自有错误码、HTTP status（若有）、异常类型及不含输入的文件/函数/行号。无需重启模型即可诊断已失败任务：`bash check-download.sh --task-id REAL_TASK_ID`，它只读任务状态，临时下载并检查首尾图，不调用 GPU、不领取消息、不写 S3；输出不包含图片 URL 或提示词。

AWS AccessDenied 排查运行 `./check-aws.sh --task-id REAL_TASK_ID`，必须提供现有 Lambda 返回的真实任务 ID；默认仅读取该状态对象，已完成任务才可加 `--video` 测试视频。无需重启 GPU 服务或加载模型，不领取/删除消息、不写对象。输出实际 bucket/prefix、操作名、HTTP 错误码和 request ID；使用当前 shell/.env 的凭证，若它与运行中 server 的环境已不同，结果不能代表旧进程。旧脚本默认生成随机 ID，其 403 不能证明真实已存在对象也不可读。

S3 的 GetObject 请求在目标不存在且身份缺少 s3:ListBucket 时可能返回 403，而不是 404。H3 的终态/上传恢复检查不能把所有 403 当作不存在，否则真实权限故障会被掩盖。运行身份需要 S3 对象 GetObject/PutObject，以及使缺失对象可正确判定的桶 ListBucket 权限；若视频 prefix 改成 video/h3/，IAM/桶策略也需覆盖该前缀。STS/队列属性通过不代表这些 S3 权限已通过。参考 [S3 GetObject 权限与缺失对象响应](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html)。

新任务现在只读取 Lambda 已写入的状态，不 GET 尚未生成的视频。进入 upload 阶段前持久化 upload_started；只有存在这个标记（或旧版本 stage=upload）的重投任务才读取视频恢复结果，避免为普通新任务增加缺失对象读取要求。真实状态读取 AccessDenied 仍明确报错，不静默绕过幂等检查。Wan 的旧实现对状态读取异常只记日志并继续，因此“Wan 能生成”不等同于它的状态读取未出过错。

- 视频上传与其 metadata 一起提交，metadata 包含 task_id、模型/请求指纹和结果参数。终态写失败时不删消息，重投校验该 metadata 后补终态。
- 已有终态的重复投递不会重新生成；即使 receive_count 已超上限，也先保护已有终态。
- 回调最多尝试 3 次；失败终态保存在 callback 字段。回调确认和 SQS 删除之间仍可能重复通知，接收端须按 task_id 幂等。
- 人工补回调入口：停止此队列的 worker 后运行 `.venv/bin/python -m h3_service.retry_callback TASK_ID`。它会实际发送回调；不能和 worker 并发改同一状态。
- `/health` 提供进程、ready、当前任务和初始化错误类型；`/ready` 在模型探测未完成或失去续租时返回 503。
- 普通日志不记录提示词、图片签名 URL 或凭证。GPU 子进程错误记录异常类型及文件/函数/行号，不输出可能含输入的第三方异常消息。
- SIGTERM 停止领取并取消当前 phase，未完成消息等待重新投递；这不是保证完成当前单的优雅排空模式。重启前先等 current task 为空可以避免重算。

## 本地测试

在具有 service 依赖以及 torch、safetensors、numpy、diffusers、av 的环境中执行：

```bash
python -m unittest discover -s tests -v
```

测试中的 fake engine 只存在于 `tests/`，不会上传或消费真实 AWS 资源。真实媒体封装测试需要 ffprobe；缺少时会明确 skip。完整 GPU 验收必须另跑 benchmark。

## 重复部署和重启成本

CPU 线程池现在默认 8 线程，并按可见 cgroup `cpu.max` 与进程 affinity 取更小上限；`H3_CPU_THREADS` 可调整。用户提供的配额 1152000/100000 等于 11.52 核，不能按 96 个可见逻辑 CPU 设置 PyTorch/OpenMP/MKL 线程数。该设置在模型子进程导入 torch 前应用，并显式设置 torch intra-op 线程；现有进程需下次重启才生效。

`python -m scripts.transfer_probe` 可在 GPU 空闲时测 256 MiB 预分配缓冲的 pageable/pinned 双向复制（各 3 次取中位数）。无需加载模型或重启 server，但应避免与生成任务同时运行。该工具测复制路径，不包含模型权重分配、磁盘缺页或数百个 tensor 的调度开销，结果不能直接当作完整模型搬运速度。

- `git pull` 不会主动重启运行中的 server；Python 已加载的业务代码需要重启后才更新。不要对 GPU 服务使用 `--reload`。
- requirements 内容未变化且存在部署标记时，deploy 复用 venv，跳过 pip install；仍执行 pip check。需要修复依赖时 `H3_REINSTALL=1 ./deploy.sh`。
- 已有固定模型和 manifest 验证通过时不再走 HF 下载/重建清单。
- `H3_VERIFY_MODE=cached` 默认首次完整 SHA-256，随后只在预期哈希、路径、设备/inode、尺寸、mtime/ctime 全部一致时复用校验结果。变化或缓存丢失会重新校验；这是本机可信磁盘的启动优化，不是对恶意篡改者的真实性证明。`H3_VERIFY_MODE=full` 强制逐次全量 SHA-256。
- Resident 默认 `H3_CONVERTED_CACHE=1`。首次把转换好的完整 Transformer 保存到 models/converted，另需约 62 GiB 磁盘和一次写入/校验耗时；其后直接加载该 Diffusers checkpoint，省去原始 Transformer 加载后的 Eros 逐张量转换。基线/模型/转换格式版本不匹配或文件校验失败会重建。修改转换算法必须提升 converter 版本。
- 模型和缓存目录需要持久存储。新机器首次部署仍有下载、验证、转换成本；复制缓存后文件元信息变化会触发哈希验证。
- 重启仍要加载两套模型并探测 GPU 放置，不能保证零冷启动或承诺具体秒数。业务代码与 GPU 常驻进程跨重启复用尚未实现。
