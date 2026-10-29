# AccessDenied 只读排查结果

2026-10-29，使用本机 AWS CLI/SDK，只读查询。未修改权限、对象、消息或服务。

## 身份和资源

本机 CLI/SDK 为账号 905641673118 的 root 身份，不能把本机 S3 读取成功当作 Vast 身份成功。

账号中列出的 IAM 用户 auto-scene 没有 permissions boundary 或组，挂载 AmazonSQSFullAccess；内联策略 Wan22S3StatusAccess 允许 s3:GetObject 和 s3:PutObject，资源为该视频桶 video/wan22/*。策略模拟显示这两个动作允许，桶上的 s3:ListBucket 为 implicitDeny。尚未直接读取 Vast 的 STS 身份，用户匹配仍有这一限制。

现网 videoQueueApi Lambda 配置的桶为 generated-video-905641673118-us-east-1-an，STATUS_PREFIX 为 video/wan22，队列为 VideoTaskQueue。与用户诊断打印的 bucket/prefix 一致。

桶策略只有 CloudFront 服务读对象的 Allow，没有查到显式 Deny。

## 对应任务证据

- 08e790291a2a4364ac14a67fed9b3a6d 的状态对象实际存在，status=pending、stage=recovery、updated_at=2026-10-29T09:47:33.025351+00:00。
- 用户报错时间为 09:47:33.367；此版本 worker 在写 pending/recovery 之后无条件 GET 视频。
- 同 task_id 的 MP4 用本机 root 执行 HeadObject 返回 404，确认该视频不存在。
- 另一个近期任务同样停留 pending/recovery。

因此，时间线与“视频尚不存在，执行身份又没有 ListBucket，导致 GetObject 返回 403”高度一致，而不支持“Access Key 无效”的判断。最终与 Vast 身份一一对应仍需其 STS 身份或新增详细错误日志；没有伪称使用了 Vast 的凭证进行实测。

## 处理边界

代码已修正：普通新任务不再读取不存在的视频，只有持久化上传开始标记的重投才尝试视频恢复。原来的随机 ID 诊断也已撤掉，改为真实 task_id。

若希望当前旧进程不重启即继续工作，可以评估给其实际 IAM 身份添加此桶的 s3:ListBucket，让缺失对象返回 404。该权限修改尚未执行，当前授权只涵盖排查。

AWS 官方依据：https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html （目标不存在时，是否具备 ListBucket 决定返回 404 或 403）。
