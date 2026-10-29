# 历史实现，当前不部署

按用户决定，生产接单唯一入口为已部署的 Wan Lambda/API Gateway，H3 worker 兼容其 SQS 消息、S3 状态和回调协议。这里的独立 Lambda 实现保留供后续整理，当前无需打包或部署。

统一项目中共用接单、task、runner，generator 通过 adapter 切换，属于后续工作。
