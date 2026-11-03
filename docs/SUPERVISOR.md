# H3 服务托管与排障

本分支使用项目自己的 Supervisor，不依赖 systemd、root 或系统 PID 1。配置、Unix socket、PID、重启记录在 `data/service/`，目录 0700、配置/socket 0600。没有管理用 TCP 端口，不修改 Vast 自带的 Supervisor、Caddy 或 Portal。

## 安装和日常管理

首次安装或依赖变化：

```bash
./deploy.sh
./start.sh
./service.sh status
```

deploy 最后生成 Supervisor 配置，不启动或重启服务。已有依赖时可以直接 `./service.sh install`。默认 `./start.sh` 等同于 `./service.sh start`，断开 SSH 不影响运行；重复 start 不重启已有进程。

```bash
./service.sh stop       # 停止领单，等待在途任务；默认最多等 900 秒
./service.sh start
./service.sh restart    # 明确停止后重启，会重新加载模型
./service.sh status     # 托管进程状态 + 模型就绪、初始化进度、机器信息、在途任务
./service.sh logs       # 实时原始控制台，包含 Python/启动失败
./service.sh shutdown   # 停止 H3 和项目自己的 Supervisor
```

status 退出码：0 就绪；2 初始化中或资源不足暂停领单；1 健康检查失败；Supervisor 自身返回的失败码也会保留。RUNNING 只代表进程存在，`model_ready=true` 和 `accepting_tasks=true` 才能接单。

迁移旧版前台进程时，先在原终端 Ctrl+C。单实例锁会阻止另一套服务重复加载模型和接单；管理脚本不会按进程名批量杀进程。调试需要前台时使用 `./start.sh --foreground`，同样有 worker 锁和进程健康检查。

换机后部署路径变化，重新 `./service.sh install`；若项目自己的 Supervisor 正在使用不同配置，先 shutdown 再 install。配置相同的重复 install 不重启。正常 git pull 不自动重启；需要使用新代码时显式 restart。

Supervisor 能维护运行中的服务，但机器启动时仍需要入口。Vast 的 **On-start Script** 配置为实际路径，例如：

```bash
bash /workspace/server/service.sh start
```

该设置随创建实例的模板保存；本脚本不擅自修改云端模板。容器镜像若有自己的启动入口，也可调用同一命令。仅执行 install 不能保证容器重建后自动启动。

## 故障、退出和资源保护

- AWS STS/队列检查在模型加载前执行；失败会导致服务非零退出。S3 仍在实际操作时校验，不要求 ListBucket。
- worker 初始化失败、worker 异常退出、空闲模型子进程死亡、阶段超时会使服务不可用并退出，由 Supervisor 重启。初始化总超时默认 3600 秒；模型阶段沿用 H3_PHASE_TIMEOUT。
- 一小时最多自动重试三次失败运行；达到限制后退出 78，Supervisor 不再无限预热。检查原因后执行 `./service.sh reset`，再 start。手动 start 不自动清空失败预算。
- 正常 stop 先停止领单，等待在途任务完成。超出 H3_SHUTDOWN_GRACE 后取消，尚未确认的消息保留，等 SQS visibility 到期重投。模型回收额外留 60 秒，之后 Supervisor 强制结束进程组。
- Linux 模型子进程启用 parent-death signal，API 即使被 SIGKILL，也不会留下常驻 GPU 模型。该机制在 macOS 不启用。
- 每 30 秒检查工作盘和 cgroup 可用内存；磁盘不足 5 GiB 或主机可用 RAM 不足 4 GiB 时暂停领取新单，保留模型和在途任务；资源恢复后恢复领取。health 显示 resource_warnings，ready 返回 503。

AWS 静态凭证继续由 deploy 的隐藏输入流程写入 `.env`，权限 0600；Supervisor 配置无凭证，守护进程也不保留管理命令继承的静态 AWS 凭证。H3 每次启动从 `.env` 重新读取，也支持 AWS profile/role provider。只有临时 shell 环境中的静态凭证不适合托管运行，应使用现有保存流程或 profile。

## 机器与成片来源

worker_id 优先读 H3_WORKER_ID，其次 `vast-$CONTAINER_ID`，最后主机名。新机器即使拷贝整份项目目录，也会从环境重新识别 Vast ID。可显式指定稳定的业务机器名称。

run_id 每次主服务启动生成，attempt_id 每次 SQS 投递处理生成。文本日志带 worker/run，结构化日志带任务及 attempt，成片 S3 metadata 的 h3-result 和最终 status 的 producer 保存真实产出来源及 Git revision。上传后终态写入失败，由另一台机器恢复时保留原产出者，不改成恢复机器。

这是 SQS 至少一次投递的服务，不提供跨机器全局排他锁。当前实现保留本机在途重复投递保护、任务请求指纹、成片恢复；多台 worker 同队列不等于严格 exactly-once。

## 精简 status

保留 task_id、status、stage、时间戳、必要 request、seed、result、video_url、callback、producer 和 timing。request 是现有 Lambda/回调/重投一致性校验需要的数据，本轮不移除、不改 Lambda。未完成任务仍保留必要恢复标记；成功后清理 resolved、upload_started 和已完成的超分交接信息。

```json
{
  "producer": {
    "worker_id": "vast-58925531",
    "run_id": "example-run",
    "attempt_id": "example-attempt",
    "revision": "example-revision"
  },
  "timing": {
    "conditioning_s": 0.751,
    "generation_s": 300.801,
    "upscale_s": 30.033,
    "processing_s": 331.585,
    "gpu_wait_s": 0.5,
    "upload_s": 1.9,
    "task_total_s": 337.2
  }
}
```

processing_s = 编码 + 生成（含解码/导出）+ 本地超分，不含等待 GPU、S3 上传下载或状态操作。task_total_s 从本次 worker 接单计时到最后一次终态写入前，包含业务处理和 GPU 等待，不是从 Lambda 提交算的排队/端到端时间，也不包含尚未发生的 SQS 删除。每次执行的准确耗时和删除结果见本地报告 total_s/sqs_deleted。异常重投时该字段表示当前投递，不把两次执行的时间混加。

详细峰值显存、量化、GPU 编号、内部阶段及异常诊断保留在 `data/reports/<task_id>/<attempt_id>.json`。`summary.json` 是最新执行的兼容索引，历史执行不会覆盖。现有独立超分 worker 仍可能追加自己的旧格式 timing；本规范面向双卡本地超分链路。

## 日志筛选与导出

```bash
./logs.sh --task <32位task_id>
./logs.sh --task <32位task_id> --follow
./logs.sh --errors --since '1 hour ago'
./logs.sh --worker vast-58925531 --run <run_id>
./logs.sh --summary --since today
./logs.sh --task <32位task_id> --export data/task-diagnostics.tar.gz
```

JSON 日志时间使用 UTC。since 可用带时区的 ISO 时间；today/无时区输入按机器本地时区解释。summary 从每次投递报告读取，能显示重试和失败；兼容旧版只有 summary.json 的记录。export 只包含任务结构化日志及脱敏诊断，不包含 `.env`、原始提示词、图片或视频；文件权限 0600。

`logs/h3.log` 为可读文本，`logs/h3.jsonl` 用于查询。两者默认每文件 20 MiB、10 个备份，合计上限约 440 MiB。项目 Supervisor 的 console.log 最多约 100 MiB，supervisord.log 最多约 20 MiB。文件轮转继续保持 0600。INFO 默认不重复打印整份推理诊断，需要时 H3_LOG_LEVEL=DEBUG。

诊断报告每小时清理一次：默认保留 30 天，超过 500 MiB 时优先清理一天以前的旧记录，今日记录不因容量目标删除。权重、成片及正在执行的任务目录不会被清理。参数见 `.env.example`。

## 验收范围

本地检查覆盖真实 Supervisor 启停、退出78不重启、失败预算、优雅停止、空闲子进程检测、产出者跨机器恢复、脱敏日志筛选、状态时间口径和回归接单测试。Linux parent-death signal 另有 Linux 专用集成测试。完整双卡生成、故障重启后的真实 GPU 恢复，需要在实际机器验收；开发不自动租用或启动已关闭的实例。
