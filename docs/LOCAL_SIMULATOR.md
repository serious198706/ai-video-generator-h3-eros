# 本地 SQS 延迟模拟服务

分支 `experiment/local-queue-simulator` 基于 `feat/supervisor-operations`。

SQS/S3 使用真实 AWS；只有生成 worker 在本机运行。复用正式 Worker 的请求校验、双在途调度、消息续租、状态读写、S3 上传、上传后恢复、回调和删消息。单独入口 sim.sh 才能启用模拟；正式 start.sh 不会选择假生成器。

模拟环境不安装 torch、diffusers、CUDA、超分模型或 ffmpeg，不下载任何模型。图片 URL 也不下载：用本地纯色图片走现有尺寸准备逻辑，默认模拟 1080×1920 竖图。真实回调 URL 若随消息提供，仍按现有逻辑校验和回调。

## 启动

```bash
./deploy-simulator.sh
./sim.sh start
./sim.sh status
./sim.sh logs
```

部署脚本只装 requirements-service.txt 到 `.venv-simulator`，保留正式 `.venv`。默认继承本机 `.env` 的 AWS、SQS、bucket、status/video prefix 和 public URL，`.env.simulator` 可覆盖这些值。没有凭证时使用既有隐藏输入并保存到权限 0600 的 `.env.simulator`；已有 AWS CLI profile 也可用，不把 profile 凭证复制到文件。check 只做 STS/队列检查及读取可见/在途数量，不领取消息、不写 S3、不要求 ListBucket。

本地已安装依赖时，后续无需重跑 deploy：`./sim.sh install` 安装托管配置，`./sim.sh start` 启动即可。simulator 的 Supervisor/socket/report/log 与正式服务分开，默认 HTTP 为 127.0.0.1:8021。

macOS 上服务会启动绑定自身 PID 的 caffeinate 防止空闲自动睡眠，退出时释放，不修改系统电源设置。手动睡眠、合盖和网络断开仍可能影响长时间测试。

**只有模拟 worker 应消费本次测试队列。** 若继续使用现有 Lambda/API 发送 50 条，保持相同 queue 和 status prefix，同时停用该队列上的真实 GPU worker；否则两种 worker 会竞争任务。也可以在 .env.simulator 指向独立测试队列和前缀。模拟服务不会创建或清空 AWS 队列/bucket，不更改已部署 Lambda。

请求沿用已跑通的格式：

```json
{
  "task_id": "32位十六进制任务ID",
  "prompt": "Animate the scene.",
  "image": "https://d2ud1xskuh00y0.cloudfront.net/test1.jpg",
  "duration": 15,
  "resolution": "1080p",
  "steps": 4,
  "audio": false
}
```

直接 SendMessage 需要 task_id；调用既有 Lambda 则仍由 Lambda 生成 ID。初始 status 可由原 Lambda 创建；直接向 SQS 发送时 Worker 也能按现有约定创建状态，但需要现有缺失对象读取权限行为与真实 worker 一致。标准 SQS 不能保证严格 FIFO，本测试测实际每条消息等待，不假定第50条一定最后完成。

若当前 IAM 没有 ListBucket，直接发一条没有初始 status 的消息可能在读取不存在对象时返回403。推荐继续用现有 Lambda，或用提供的批量提交工具：

```bash
./sim.sh submit --count 50
# 也可仅测一种组合：
./sim.sh submit --count 50 --durations 15 --resolutions 1080p
```

工具先写50份 queued status，再用每批最多10条的 SendMessageBatch 提交到真实SQS，不下载图片。默认六种组合轮流分配；输出 batch.json 保存全部ID及发送结果。网络超时等发送结果不确定时不自动重试，防止意外重复提交；命令非零退出并保留 manifest 供核查。

## 时间与调度

`.env.simulator` 中设置 `H3_SIM_PROFILE=6000-maxq` 或 `6000-s`，默认 Max-Q。支持 5/10/15s × 720p/1080p、4步、无音频，其余组合会标记业务失败，不花 GPU 模拟时间。

| 组合 | 6000-S 总处理基准 | 6000-Max-Q 总处理基准 |
|---|---:|---:|
| 5s / 720p | 52.285秒 | 70.321秒 |
| 5s / 1080p | 55.787秒 | 79.332秒 |
| 10s / 720p | 127.338秒 | 175.167秒 |
| 10s / 1080p | 139.598秒 | 196.698秒 |
| 15s / 720p | 213.895秒 | 301.062秒 |
| 15s / 1080p | 232.391秒 | 331.585秒 |

每个组合按之前实测的 condition / generate / upscale 三段分别 sleep，两个独立锁模拟 GPU：GPU0 的 condition 和 upscale 共用，GPU1 的 generate 独占。每台最多两单在途，不把整个处理时间当作可同时跑两份的去噪，也不启动50个线程同时 sleep。短边 768 的生成规格、24fps 和 15s→14.375s 规则继续走原有代码。

`H3_SIM_TIME_SCALE=1` 为真实等待秒数。0.01 等值用于快速检查流程；报表记录实际墙钟，不把时间放大后伪装成真实测试。小倍率时 AWS 网络、图片处理和状态读写开销不缩放，因此不能直接用快速测试乘100预测真实排队。

这些是固定样本延迟模拟，不包含真实GPU抖动、并发争用导致的降速、冷启动、模型OOM或Vast断线。profile 的微小720p/1080p生成差异保留自两次实测，不代表原生生成分辨率不同。

## 50 单排队报告

先启动 simulator，再自行向 SQS 或现有 Lambda 提交 50 条。另一个终端执行：

```bash
./sim.sh report --expected 50 --watch
```

使用上述 submit 工具时，优先执行它打印的批次监控命令，确保只统计该50个ID，即使服务中途重启也不漏掉前一 run：

```bash
./sim.sh report --batch data/simulator/batches/<批次目录>/batch.json --watch
```

默认从本机 health 选择正在运行的 run_id；服务停止时选择最新报告所属 run，避免混入之前启动的任务。一次启动做多批测试时，应按本次提交前的 ISO 时间筛选，例如：

```bash
./sim.sh report --expected 50 --watch --since '2026-11-03T12:00:00+08:00' --output data/simulator/batch-2
```

也可用 `--run <run_id>`。watch 最多等8小时，超时不删任务，继续用相同筛选运行即可。

默认输出 `data/simulator/queue-report/report.json` 和 report.csv，包含每个 task_id、规格、机器标识、时间点，以及：

- queue_s：SQS SentTimestamp 到 worker 领取执行的等待时间，**即纯 SQS 排队**。
- wait_until_generation_s：同一发送时间到实际进入去噪阶段，包含 SQS 等待、输入准备、编码和内部 GPU 等待。
- processing_s：实际模拟编码、生成、超分耗时，不含 GPU 等待和网络。
- submitted_to_result_s：同一发送时间到首次写入最终成功/失败状态前的时长，含等待、模拟处理和上传；不等于客户端发起 HTTP 到收到响应的计时。

报告分别统计 min/p50/p95/max，最大 queue_s 和最大 wait_until_generation_s 回答不同口径的“排队多久”。来自重投的 queue_s 也包含前次执行及不可见期，应同时看执行报告的 receive_count。report 读取本地诊断，不反复查询 S3；只有已结束的任务进入统计，等50条都完成才是完整结论。

即使不运行 report 命令，worker 每完成一条也自动更新 `data/simulator/auto-report/report.json` 和 report.csv，按当前 run 汇总；默认目标50条，可用 H3_SIM_EXPECTED_TASKS 调整。交互报表与自动报表分开存储，互不覆盖。

## status 与占位文件

状态尽量复用真实格式：queued/pending/generating/succeeded/failed、stage、result、producer、timing、callback 和 video_url 都由正常 runner 维护，额外携带 simulation.enabled/profile/time_scale、SQS发送/领取/开始生成/完成时间。

上传的是24字节 MP4容器头占位文件，没有帧、不可播放；result 的宽高、帧数和时长是模拟规格。result.simulation=true、model=simulation-...，S3 metadata 也保留模拟标记。使用独立的 simulation fingerprint，真实生成器不能把模拟上传当成其合法恢复成片。24字节而非0字节，是为兼容已有 Range GET 上传恢复逻辑。

模拟结果会真实写到配置好的 bucket/status/video keys。请使用测试任务验收，勿把其成功状态作为可播放的真实成片交付。

```bash
./sim.sh trace --task <task_id>
./sim.sh trace --errors --since '1 hour ago'
./sim.sh stop
./sim.sh shutdown
```

停止、失败预算和诊断导出沿用 Supervisor 机制。默认 stop 等在途任务完成，不会把 SQS 中剩余任务清空。

## 验证

独立解释器测试会禁止导入 torch/diffusers/transformers/torchao，确认服务初始化不依赖模型。50单集成测试用 Moto 本地模拟 AWS 协议，走真实 boto3/SQS/S3/Worker，在缩短时间倍率下验证50条均成功上传并删除、状态和时间统计可用；该测试不会向真实 AWS 提交50条任务。
