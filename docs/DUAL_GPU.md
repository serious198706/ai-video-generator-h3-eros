# 双卡 INT8 本地超分流水线

原始双卡实现分支：`experiment/dual-gpu-local-upscale`，已完成两台 2×96GB 机器的六种组合生成测试。当前运维改造分支：`feat/supervisor-operations`，托管和日志见 [Supervisor 运维](SUPERVISOR.md)。

## 设备分工与并发

GPU 0 常驻 INT8 text encoder 和 RealESRGAN compact；GPU 1 常驻 INT8 H3 transformer、视频 VAE 和音频 VAE。两个独立 Python 模型进程分别设置 CUDA_VISIBLE_DEVICES，每个进程内部均使用 cuda:0。启动时验证两个设备可用，并在每个设备执行 runtime/INT8 检查。

生成队列最多同时领取两条任务。每条任务独立维护状态、visibility 心跳、回调和计时。GPU 0 的编码与超分共享一个锁；GPU 1 的生成共享另一个锁。例如 A 在 GPU 1 生成时 B 可在 GPU 0 编码；A 在 GPU 0 超分时 B 可在 GPU 1 生成。不是两条 H3 去噪同时运行，不能假定吞吐翻倍。

本机已有 worker.lock 阻止两个服务实例同时领取任务；同一任务的重复消息在本机有在途记录时延后 30 秒。跨机器同时运行仍应按既有队列部署策略隔离，SQS 标准队列不是全局互斥锁。

## 部署

在 H3 的 .env 中设置：

```bash
H3_MODEL_LIFETIME=dual_gpu
H3_CONDITION_GPU=0
H3_GENERATE_GPU=1
```

设备变量可使用物理 GPU 序号或 CUDA 支持的 GPU UUID。两个值必须指向不同设备。启动前由小进程验证两张可见卡，避免仅靠两个不同字符串判断。主机需要同时容纳启动加载、量化的 CPU 权重，仍沿用现有 RAM 预检；不能只按 GPU 总显存估算主机 RAM。

```bash
./deploy.sh
./start.sh
./service.sh status
```

部署新增 opencv-python-headless、spandrel、spandrel_extra_arches，并下载带固定 SHA256 校验的 realesr-general-x4v3.pth。可用 H3_SR_MODEL 指定路径。模型进程的首次量化仍会产生冷启动成本。

双卡模式只使用生成队列，1080p 不再需要 H3_UPSCALE_QUEUE_URL。完成原生 MP4 后，通过本地路径交给 GPU 0，超分验证通过后上传最终 S3 视频、写 succeeded、回调、删除生成消息。staging 视频不上传 S3，不发送超分 SQS 消息。独立超分服务不应再收到双卡模式的新任务。

## 日志与最终输出

`logs/h3.log` 包含线程名、task、GPU、role、phase、进度和计时；编码/生成/超分等待 GPU 时也更新 stage。SQS 任务的阶段日志使用原始 32 位 task_id，可直接检索整条链路。

`data/reports/{task_id}/{attempt_id}.json` 保存每次执行的完整阶段诊断：请求的安全摘要、执行耗时、GPU 等待时间、显存峰值、失败阶段、SQS 删除结果和回调结果；summary.json 指向最新执行的副本。文件权限 0600；不保存提示词正文和原图地址。

status 的 result 仅含 seed、最终 width/height、num_frames、fps、actual_duration、audio_present、model 以及可选 upscaled。producer 保存产出机器和代码版本。timing 仅保存关键耗时，显存及内部推理明细保留在本地诊断报告。

wall_s 是拿到 GPU 执行权后的阶段耗时，gpu_wait_s 单列等待时间。完整端到端时间见日志和任务 summary，不能把有嵌套关系的 stages_s 全部相加。

S3 上传前记录 upload_started。上传成功但终态写入失败时，下一次投递通过成片 metadata 恢复，不重复生成或超分。上传前发生进程退出或本地超分失败时，本地临时文件会清理，重投可能重新生成；本地文件不是跨机器恢复介质。

任一模型进程失败，停止接受新任务并使其余在途任务取消，保留未确认消息。OOM 会记录 GPU 内存和错误码 cuda_oom。重启后恢复接单。

## 验收顺序

先使用 benchmark.py --resident 跑 5s/1080p，确认 text encoder、SR 进程都在 GPU 0，generator 在 GPU 1，输出音轨/尺寸/帧数正确。再通过现有 Lambda 快速提交两条独立 1080p 任务，在 /health 的 active_tasks 中确认至多两个在途任务，观察日志中 GPU 1 generate 与 GPU 0 upscale 的重叠时间。

最后测 15s/720p 与 15s/1080p 峰值和总耗时。API 九宫格脚本默认串行，只能测单任务耗时，不能用来证明双任务流水线吞吐。
