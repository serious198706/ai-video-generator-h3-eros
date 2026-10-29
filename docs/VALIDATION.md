# 本地验证记录

日期：2026-10-29，执行平台 macOS / CPU。

## 已执行

- 36 项 CPU 单元及集成测试全部通过，无跳过：契约、帧网格、输入限制、图片裁剪、DNS/IP 策略、Eros QKV/SwiGLU/AdaLN 运算、conditioning 格式、S3 typed errors/流关闭、API 入队/查询、终态与回调恢复、续租失败取消和回调签名。
- 用合成小尺寸帧和立体声音频实际编码 MP4，通过 ffprobe 检查有声/无声轨道、帧数、FPS、宽高和音轨时长。此项不是模型生成。
- 使用 HF HTTP Range 只读取指定 Eros checkpoint 文件头；以固定 Diffusers commit 的 H3 配置在 PyTorch meta device 构建目标结构，全部 634 个映射参数的键和形状匹配，4 个 time-embedder 参数保留。结果见 [header-validation.json](header-validation.json)。
- 固定 Diffusers 源码的 ConditionerBlocks/GeneratorBlocks 能构造，并确认组件切分；生成侧没有 text_encoder。检查时使用临时固定源码目录，不覆盖既有环境。
- Lambda Python 3.12 / Linux x86_64 包已实际构建为 `dist/h3-lambda.zip`。构建不等同于在 AWS Lambda 运行通过。
- Python compileall、Shell bash -n、Markdown 本地链接/代码块和 ZIP 完整性检查通过。部署包解压后约 40.5 MB，未包含 torch、推理模块或测试目录。

测试使用本机既有 Python 3.14.6、torch 2.14.1、transformers 5.20.2、FastAPI 0.161.1、Pydantic 2.14.1、boto3 1.43.118、safetensors 0.8.0、av 19.0.2。它们不冒充部署候选版本。模型文件头验证另外使用固定 Diffusers 源码 `665f5782`；其他 CPU 依赖仍为本机版本。

发现本机原有 Diffusers 0.42.0.dev0 不再包含本方案依赖的旧 H3 resize 类，不能直接拿现有 venv 跑此服务。已在启动 preflight 检查 Diffusers git commit 及关键依赖版本，要求独立安装候选环境。

## 尚未验证

- Linux/CUDA 主机上安装完整候选依赖、驱动与 GPU kernel 兼容性。
- 完整模型下载/哈希校验、真实权重数值覆盖及 AdaLN 与 reference 计算的实权重误差。
- GPU 条件编码、Eros 采样、原生音轨、首尾帧效果、14 秒/720p 峰值与画质。
- 真实 AWS SQS/S3/Lambda/业务回调的 staging 闭环。
- 常驻模式的真实 GPU 验收；AoTI、量化、多卡和多 worker 尚未实现或启用。

当前结论：代码具备可部署和可验证的首版结构，本地服务逻辑与文件头映射有证据；不宣称生产已上线或 GPU 成片已验收。

## 2026-10-29 部署检查与 preflight 进度修复

本地回归更新为 46 项全部通过，无跳过。新增 CUDA wheel/驱动门槛、哈希进度与取消、哈希损坏仍然拒绝、benchmark 省略重复加载探测但保留完整哈希、worker 默认仍保留探测、已有输出目录不被修改等验证。

保留用户补充的 benchmark 日志，并增加跨进程阶段状态。实际服务器停在哪个操作仍需新日志确认；没有把本地测试描述为远端问题已复现或 GPU 验收通过。

## 常驻生命周期改造

server 默认使用 ResidentEngine，新增本地测试验证连续两单只加载两个 pipeline、Eros 仅转换一次、相同模型对象被重复使用、容器配额/可回收缓存预算、内存不足提前失败、取消回收与死进程拒绝复用。模型执行使用测试替身，未冒充真实 GPU 连续成片结果。

改造后共 51 项本地测试全部通过，无跳过；compileall 与 git diff --check 通过。

## AWS 凭证启动检查

补充隐藏输入、进程环境传递和 STS 验证，密钥不持久化；验证缺少凭证时 worker 在模型加载前退出。增加已有凭证链、非交互缺失、临时 token 清理及 fail-fast 测试，真实 AWS 身份和 SQS/S3 权限仍由目标机器实际验证。

后续按用户明确授权，setup/deploy 已改为 STS 验证后保存本机 .env（0600）。增加持久化保留业务配置、特殊字符 shell 引用、旧 token 清理、验证失败不覆盖、权重校验缓存失效、转换 checkpoint 复用和损坏重建测试。另增加队列检查失败时模型不加载测试。GPU 上的 checkpoint 保存/重新加载时长和数值一致性尚需实际验证。

benchmark 提供 `--resident --runs 2` 与 server 相同生命周期的验收入口。现有阶段冷加载路径仍可显式使用。GPU 实际显存回落、CPU RSS 稳定性和连续两单成片需在 Vast 主机验证。
