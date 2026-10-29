# 运行与接入

## 当前运行方式

更新：server 现在默认 `H3_MODEL_LIFETIME=resident`。父进程仅负责接单，专用推理子进程常驻 conditioner 和 Eros generator 的 CPU 权重；启动完成全量哈希、加载、一次 Eros 转换和两侧 GPU 放置探测后 ready。每单只做 CPU/GPU 搬运、编码和推理，不再从磁盘重新加载模型或重新转换 Eros。两套未量化模型不会同时放上 GPU。下面关于逐阶段子进程/每单冷加载的说明仅适用于显式 `H3_MODEL_LIFETIME=cold` 和未加 `--resident` 的 benchmark。

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

## Lambda 接入

打包入口 `./lambda_api/package.sh`，产物 `dist/h3-lambda.zip`，Python 3.12 / x86_64，handler 为 `lambda_api.handler.handler`。包只包含服务依赖和轻量模块，不包含 torch、模型或 worker 推理代码。

API Gateway 将 `/v1/generate` 和 `/v1/tasks/{id}` 代理给 Lambda。身份认证由 API Gateway/IAM 或已有业务网关承担；应用不自行实现用户鉴权。Lambda 配置使用与 worker 一致的 H3 queue/bucket/prefix/public base。

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

使用独立 H3 Standard queue；设置 DLQ redrive policy，建议 maxReceiveCount=5。完全非法的 envelope 保留给 DLQ，避免未知任务被静默丢弃。Lambda 和 worker 均不能共享 Wan 队列。

| 角色 | 必需权限 |
| --- | --- |
| Lambda | 对 H3 队列 SendMessage；启用积压限制时 GetQueueAttributes；对 H3 status 前缀 GetObject/PutObject；Lambda 常规日志权限 |
| Worker | 对 H3 队列 ReceiveMessage/DeleteMessage/ChangeMessageVisibility；对 H3 前缀 GetObject/PutObject 和 multipart 上传所需权限 |

S3 使用 KMS 时另需相应密钥权限。视频前缀可经 CDN 服务，状态对象保持私有，经 Lambda 查询；不要把包含业务请求的状态对象暴露到公共 CDN。

默认 visibility=600 秒，每 60 秒续租。续租或状态 heartbeat 失败时，worker 取消当前 GPU 子进程、停止接单并退出 ready；由运维重启，不能在未知执行权下继续生成。尚未提供多 worker fencing，禁止水平扩容。

## 恢复与排错

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
