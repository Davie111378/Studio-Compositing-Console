# bsrc — B 组算法核心引入

来源仓库: <https://github.com/Davie111378/Studio-Compositing-Console>
上游提交: `b3107bc42dd6c191c6f20f54dad8f56aefb25bad`（2026-09-13，"翻入项目源码与文档"）

## 内容

| 子包 | 上游路径 | 说明 |
|---|---|---|
| `matting/` | `ai-service/src/matting` | BiRefNet 后端选择链（真实权重 → carvekit → SimplifiedBiRefNet） |
| `lighting/` | `ai-service/src/lighting` | Lambert 半球光照估计（方向/色温/SH9） |
| `relighting/` | `ai-service/src/relighting` | 方向性重打光 + MKL 色彩迁移 |
| `shadow/` | `ai-service/src/shadow` | 程序化接触阴影（alpha 模糊+偏移+压暗） |
| `harmonization/` | `ai-service/src/harmonization` | v2 和谐化（Lab 色度对齐+低频亮度迁移+肤色保护+FDR 防压黑） |
| `composite/` | `ai-service/src/composite` | alpha-over / 羽化泊松近似合成 |
| `fx/` | `ai-service/src/fx` | 特效链（spotlight/bokeh/fog/vignette/color_temp/depth_blur，支持区域选区） |

## 约定

- 本目录下文件与上游 **逐字一致**，便于 B 组后续 PR 直接 diff 合并；
- 依赖 `opencv-python`（cv2）、`numpy`；`matting` 真实链另需 `torch/torchvision` 与权重
  （环境变量 `BIREFNET_WEIGHT_PATH` 指向 safetensors/ckpt）；
- 缺依赖或加载失败时，A 组 impl 自动回落到纯 PIL 的 mock 引擎（信封契约不变，
  见 `aiservice/bsrc_loader.py` 与 `IMC_ENGINE` 环境变量）。
