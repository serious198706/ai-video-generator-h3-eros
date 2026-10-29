# H3 / 10Eros 视频生成服务

当前阶段：首版代码已实现，本地 CPU/媒体封装测试通过；尚未部署云端或完成真实 GPU 生成验收。更新日期：2026-10-29。

目标：在自有 GPU 上运行 MiniMax-H3 + 10Eros beta5 Turbo，生成带原生音轨的视频，通过独立的 Lambda / SQS 接单，上传 S3 并回调业务端。

- [技术设计](docs/ARCHITECTURE.md)：模型组合、部署拓扑、接口与可靠性约定。
- [实现与验收计划](docs/IMPLEMENTATION.md)：阶段产物、测试、GPU 验证和上线门槛。
- [运行与接入](docs/OPERATIONS.md)：实际脚本、配置、接口、故障恢复和部署包。
- [本地验证记录](docs/VALIDATION.md)：已执行检查及明确未验证的部分。

已选路线：复用 H3EROS 的 Diffusers 权重转换思路；条件编码也由自己部署，不调用公共 Space。首版关闭额外 Turbo LoRA、AoTI、量化和提示词自动改写，以完成准确、可复现的端到端验证为先。

本目录已经初始化独立 Git 仓库，与 `../server/` 的 Wan 实现分开。

已实现：请求契约、Lambda 入队/查询、单任务 SQS worker、续租与取消、S3 终态恢复、HMAC 回调、固定权重下载校验、Eros 转换、本地条件编码、GPU 子进程调度、真实 MP4 导出检查，以及无云端副作用的 benchmark。

首版采用**每阶段独立子进程**，编码进程退出后再启动生成进程，每单均冷加载。这样先保证显存释放和中断回收可控；尚未实现 CPU 常驻权重或独立编码 GPU 服务。worker 的 ready 表示启动时两侧模型加载/GPU 放置探测通过，不等于完整采样或画质已通过。

GPU 安装入口为 `deploy.sh`，运行入口为 `start.sh`，Lambda 打包入口为 `lambda_api/package.sh`。无生产占位视频或自动降级模型路径。
