# 上游来源记录

本实现没有复制或启动 Gradio/ZeroGPU 应用，也没有调用公共 conditioner。

- H3EROS：`Daankular/H3EROS`，commit `157975f3eb07b3d17a1601f5f3f8dbf0949758ab`。
  - `h3_eros.py` 的参考键映射、SwiGLU 排列与 AdaLN 公式用于本地 `infer/eros.py`。
  - `h3_split_blocks.py` 的 fl2va 切分方式用于 `infer/blocks.py`。
  - 本地增加完整覆盖检查、重复目标检查、有限值检查、本地文件加载和进程隔离。
  - Space 未发现单独 LICENSE 声明；保留来源，不将其代码擅自标为 Apache/MIT。再分发前需明确适用授权。
- Diffusers commit `665f578278365ea4a3318cb8c9b66ce6c01204b9`：H3 模块源文件标注 Apache-2.0，Copyright 2026 The MiniMax and HuggingFace Teams。
- 基础模型 `MiniMaxAI/MiniMax-H3` revision `42ed227ee7df40d41602854ae760620d6eb651fe`，适用仓库 LICENSE。
- Eros `TenStrip/10Eros-Max` revision `5794cf604068adad01d9232db1c4e55964d3cbfb`，模型卡说明基础 H3 及所涉及源模型条款；本仓库不包含模型权重。

读取日期 2026-10-29。上游运行报告不是本项目的 GPU 验证结果。
