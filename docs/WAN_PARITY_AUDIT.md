# Wan 对照检查与部署成本

2026-10-29。依据本地 server/deploy.sh、setup-aws.sh、start.sh、SQS runner 和 H3 实现检查。

接入决定更新：继续使用既有 Wan Lambda/队列，不部署新的 H3 Lambda；H3 worker 已对齐平铺 SQS 消息、原样 S3 request、默认状态 key 与回调字段。统一项目和 adapter 重构后续再做。同一队列当前仅保持目标模型的 worker 活跃，避免无路由情况下随机选择模型。

| 检查项 | H3 结果/本次处理 |
| --- | --- |
| 部署阶段请求并保存 AWS 凭证 | 已补：隐藏输入、STS 验证后原子写 .env，0600，不打印值，--force 轮换 |
| 标准 AWS provider、Session Token、Region | 支持环境/profile/角色；兼容 AWS_DEFAULT_REGION；静态凭证保存时清理旧 token |
| 云端依赖先检查，再加载模型 | STS + GetQueueAttributes；S3 操作权限仍在实际使用时验证 |
| 重复 deploy 不浪费安装和下载 | venv/requirements 内容标记、pip check、已有固定模型验证后跳过下载 |
| 每次重启全量读取权重做哈希 | 默认复用本机文件身份/元信息一致的既有校验；可强制 full |
| Eros 每次重新转换 | 默认保存转换后 checkpoint；后续直接加载。首次额外磁盘约 62 GiB |
| 每单模型生命周期 | server 两套 CPU 权重常驻，每单轮流上 GPU，不重新读盘/转换 |
| 服务启动阶段是否创建生成任务 | 不创建任务；模型 ready 后才消费队列，已有积压会自动开始 |
| 失败/重复投递/终态与回调恢复 | 已有相关本地测试；不保证 exactly-once |
| 运行目录隔离 | H3 独立 venv、模型、队列、S3 prefix；不复用 Wan 的硬编码路径 |
| 日志和预加载进度 | 有阶段/哈希进度，未复制 Wan 的按日滚动文件日志/飞书 watchdog |
| 停止服务 | 取消模型进程；尚无“停止接新单并等当前单完成”的独立 drain 命令 |
| 多台机器共享一条队列 | 尚未实现分布式租约/fencing，不能宣称可靠水平扩容；现阶段每条队列只运行一个 worker |
| 只更新业务代码，保持 GPU 进程跨主服务重启 | 尚未实现。git pull 本身不退出模型，但重启 server 会回收子进程 |

本次没有修改正在运行的远端机器、重启其 worker 或生成真实视频。首次建立转换缓存与后续复用的 GPU 加载时长仍需远端测量。

按用户提供的 1.4 美元/小时，每次 15 分钟准备约 0.35 美元。缓存优化针对重复校验和转换，不把全部 15 分钟都算作已消除。
