# H3 / 10Eros 视频生成服务

当前阶段：INT8 双卡生成及本地超分已完成六种组合实机测试；本分支新增 Supervisor 托管、机器来源、精简状态和日志检索，待实机验收这些运维改动。更新日期：2026-11-03。

目标：在自有 GPU 上运行 MiniMax-H3 + 10Eros beta5 Turbo，生成带原生音轨的视频，通过现有 Wan Lambda / SQS 接单，上传 S3 并回调业务端。

- [技术设计](docs/ARCHITECTURE.md)：模型组合、部署拓扑、接口与可靠性约定。
- [实现与验收计划](docs/IMPLEMENTATION.md)：阶段产物、测试、GPU 验证和上线门槛。
- [运行与接入](docs/OPERATIONS.md)：实际脚本、配置、接口、故障恢复和部署包。
- [本地验证记录](docs/VALIDATION.md)：已执行检查及明确未验证的部分。
- [Wan 对照检查](docs/WAN_PARITY_AUDIT.md)：部署成本、已补项和剩余多机运行边界。
- [Supervisor 运维](docs/SUPERVISOR.md)：启动、停止、日志筛选、机器标识、状态字段和故障恢复。
- [双卡流水线](docs/DUAL_GPU.md)：GPU 分工和本地超分。
- [本地排队模拟](docs/LOCAL_SIMULATOR.md)：真实 SQS/S3、无模型、按两台实测时长模拟生成与超分，统计50单排队。
- [DynamoDB 接入](docs/DYNAMODB.md)：现有 Lambda、建表、容量判断与 429、任务租约、权限及切换步骤。

已选路线：复用 H3EROS 的 Diffusers 权重转换思路；条件编码也由自己部署，不调用公共 Space。当前双卡路线使用 INT8WO，关闭额外 Turbo LoRA、AoTI 和提示词自动改写。

本目录已经初始化独立 Git 仓库，与 `../server/` 的 Wan 实现分开。

已实现：Wan 接单协议兼容、单任务 SQS worker、续租与取消、S3 终态恢复、HMAC 回调、固定权重下载校验、Eros 转换、本地条件编码、GPU 子进程调度、真实 MP4 导出检查，以及无云端副作用的 benchmark。

双卡部署设置 `H3_MODEL_LIFETIME=dual_gpu`：GPU 0 常驻 INT8 conditioner 和本地超分，GPU 1 常驻 INT8 generator 和 VAE。最多两项任务在途，按 GPU 串行、跨 GPU 重叠。保留 resident、cold 和单卡 INT8 路径，配置需显式选择。

GPU 安装入口为 `deploy.sh`，`start.sh` 默认启动 Supervisor 托管服务，日常使用 `service.sh` 和 `logs.sh`。接单继续使用已部署的 Wan Lambda；本目录 lambda_api 不作为部署入口。无生产占位视频或自动降级模型路径。

统一项目、共用 task/runner、各模型 generator adapter 的结构调整留到后续；当前只修现有链路兼容，不迁移目录或重构 Wan。
