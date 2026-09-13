# HumanMatting (third_party 资产与来源说明)

接入 skill: **MODNet 人像抠图引擎** (agent T01_matting 的 `humanmatting` / `humanmatting_fast`)。

## 来源链与合规

| 层 | 来源 | 许可证 |
|---|---|---|
| 触发入口 | https://github.com/davilsu/HumanMatting (Android APP, ncnn 推理) | AGPL-3.0 |
| 模型真身 | **PaddleSeg/Matting 官方 MODNet** 双 backbone (层名 `instance_norm` 与 MODNet 结构吻合) | Apache-2.0 |
| 本项目实际采用 | PaddleSeg 官方 inference model → paddle2onnx (opset 12) → ONNX | Apache-2.0 |

**合规说明**: 本项目**未复用** davilsu 仓库的 AGPL 代码/权重（其 ncnn 模型仅作格式参考，
`assets/` 目录保留仅为溯源，可随时删除）。运行时模型为 PaddleSeg 官方 Apache-2.0 权重，
无 AGPL 传染问题。

## 目录内容

- `assets/` — davilsu 仓库的 ncnn 模型 (hrnet_w18 / mobilenet_v2, 含 int8)。**运行时不使用**。
- `paddle/` — PaddleSeg 官方 inference model (`modnet-hrnet_w18` / `modnet-mobilenetv2`)。
- `onnx/` — 转换后的 ONNX，**引擎实际加载的文件**。
  - `modnet_hrnet_w18.onnx` (41MB, 精度优先, CPU ~0.2-0.4s)
  - `modnet_mobilenetv2.onnx` (26MB, 速度优先, CPU ~0.1-0.2s)
- `pnnx/`, `pnnx_old.zip` — pnnx 工具残留（转换尝试未用上，可删）。
- `photo_jni.cpp` — davilsu 上游推理参考（预处理对齐依据: 512x512, x/127.5-1, sigmoid alpha）。

## 重建 ONNX（如需）

```bash
# 1. 下载官方 inference model (国内直连)
curl -L -o paddle/modnet-hrnet_w18.zip \
  https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-hrnet_w18.zip
curl -L -o paddle/modnet-mobilenetv2.zip \
  https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-mobilenetv2.zip
# 2. 转 ONNX (需 py<=3.12 + paddlepaddle + paddle2onnx>=2.0)
paddle2onnx --model_dir paddle/modnet-hrnet_w18 --model_filename model.pdmodel \
  --params_filename model.pdiparams --save_file onnx/modnet_hrnet_w18.onnx --opset_version 12
```

## 为什么不是 ncnn (排障记录)

davilsu 原版是 ncnn 格式。`pip install ncnn` (20260526/20240820, py3.12/3.13) 在本机
`extract()` 稳定段错误: 输入 ≥320 时崩（288 以下正常），关闭 winograd/sgemm/packing/vulkan、
单线程均无效，沙箱内外一致——wheel 内部 SIMD dispatch 问题。故改走 onnxruntime
（接口 `matting/humanmatting_engine.py`），预处理对齐上游 photo_jni.cpp。

## 引擎接口

```python
from matting.humanmatting_engine import human_matting
alpha = human_matting(image_bgr, model="hrnet")          # "mobilenet" 更快
alpha, ms = human_matting(img, model="hrnet", return_time=True)
```

Agent 触发: 指令点名 "MODNet / humanmatting / 人像语义抠图" → T01 params.engine。
CLI: `python studio_cli.py matting <img> --engine humanmatting`。
