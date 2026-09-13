# 实验报告：AI 素材驱动的新实验线（真实感人像域修复 + 特效 + 混合域训练）

> 责任人：B 组
> 日期：2026-09-09 下午
> 动机：旧实验线（B4/B6）在程序合成几何色块上跑指标，MKL 光影把前景压黑，与任务表
> "演播室图像合成技术——抠图/背景叠加/特效"的目标存在偏差。本实验线全面改用
> AI 生成真实感素材，重建评测与训练闭环。

---

## 1. 素材制备（ImageGen 生成，13 张）

| 目录 | 内容 | 数量 |
|---|---|---|
| `data/ai_generated/studio_bg/` | 演播厅背景：新闻LED/访谈/全景/天气/综艺/播客 | 6 |
| `data/ai_generated/green_fg/` | 绿幕人像：主播男/主播女/金属框眼镜/飞扬发丝 | 4 |
| `data/ai_generated/green_fg_hard/` | 难例：磨砂玻璃(semi)/玻璃水壶(transparent)/飞舞纱巾(veil) | 3 |

GT 构建方式：**色度键提取**（绿色优势度软阈值 + 水印区域清零 + 非前景连通域清理），
独立于 BiRefNet，避免循环论证。脚本：`training/scripts/ai_material_eval.py`。

## 2. 发现一：StudioRefiner 在真实感人像上反退化 2.8×

以色度键 GT 为参照（n=4，refine@512）：

| 模型 | MEAN SAD ↓ | MSE ↓ | Conn ↓ |
|---|---:|---:|---:|
| BiRefNet 粗抠（coarse） | 47 424 | 0.0208 | 0.325 |
| + refiner_studio（增训前） | 133 983 | 0.0958 | 1.095 |

归因：refiner_studio 在程序合成演播室域训练，对真实感人像纹理分布严重负迁移
（与 B9 通用 Refiner 跨域 +168.7% 同类现象）。

## 3. 发现二：纯 AI 域微调引发灾难性遗忘

用 240 个 AI 合成对（4 fg × 6 bg × 10 变体）微调 40 epoch → `refiner_ai`：

| 域 | refiner_studio | refiner_ai | 变化 |
|---|---:|---:|---|
| AI 素材域 SAD | 133 983 | **42 387** | ✅ 修复（优于 coarse） |
| 原演播室域 SAD（n=120） | 4 091 | 16 128 | ❌ **遗忘 ×3.9** |

**结论：单域微调必然顾此失彼，必须混合域训练。**

## 4. 混合域训练（修复中）

配置：`ai_finetune_refiner.py --stage train --epochs 25`
- 训练集 = 原演播室 720 张（`data/studio/train`） + AI 合成 240 张，混合采样
- 热启动 refiner_studio；AdamW(2e-4) + Cosine；bs=8；512²；AMP fp16
- 验证集 = AI val 16 + 演播室 val 32（双域监控）

验收目标（双域同时达标）：
- 演播室域 SAD ≤ 6 000（接近域内特化版 4 091）
- AI 素材域 SAD ≤ coarse（47 424）且较增训前 133 983 大幅下降
- 三张难例（semi/transparent/veil）不劣于 coarse

实测结果（n=120 演播室 / n=4 AI / n=3 难例）：

| 域 | refiner_studio | 纯 AI 微调 | **混合域（final）** |
|---|---:|---:|---:|
| 演播室域 SAD | 4 091 | 16 128（遗忘） | **2 928 ✅ 全场最优** |
| AI 素材域 SAD | 133 983 | 42 387 | 47 315（≈coarse，中性） |

结论：混合域以 3:1 配比实现"演播室域最优 + AI 域无害"；AI 域的精修收益（-11%）
被稀释，列为下一步优化方向（提高 AI 过采样权重 / 扩充人像身份数）。

### 难例三张（semi / transparent / veil）——难度天花板确认

| 素材 | coarse SAD | refined（无门控） | refined（+门控） |
|---|---:|---:|---:|
| 磨砂玻璃(semi) | 422 160 | 414 956 | 414 956 |
| 玻璃水壶(transparent) | 14 677 | 120 721（劣化×8） | **14 677（回退）** |
| 飞舞纱巾(veil) | 208 606 | 352 762（劣化×1.7） | **208 606（回退）** |

两个发现：
1. **coarse 的语义分歧是主因**：BiRefNet 把"半透明物"（磨砂玻璃后的身体、纱巾）
   判为背景丢弃，色度键 GT 判其为前景——与 B4/B8 "veil 是 matting 根本困难"一致。
2. **Refiner 在分布外会"越修越坏"** → 已实现**置信门控保险丝**：
   大变化像素占比（|Δ|>0.2）> 5% 即回退 coarse。实测劣化全拦截（2/2）、
   正常例零误伤（0/4），难例 refined MEAN SAD 296 147 → 212 747。

## 5. 算法与功能改造（同步完成）

| 模块 | 改造 | 效果 |
|---|---|---|
| `harmonize_tool.py` v2 | Lab 色度对齐(限幅) + 低频亮度迁移 + 高频细节保留 + **FDR∈[0.7,1.3] 自动回退** | 修复压黑；FDR 一票否决 |
| `relight_tool.py` directional | 背景亮度分布自动估光向 + 渐变光场 + 色温 | 与色彩迁移正交，修复 T04==T06 |
| `fx/fx_tool.py`（新） | 6 种特效 × region mask（box/polygon）× 链式调用 | 任务表"添加特效"落地 |
| `matting/interactive_matting.py`（新） | GrabCut 圈选抠图（box/多边形/正负点） | 圈选抠图 v1 |
| `pipeline.py` | 新增 `fx_chain` 参数 | 特效接入主流水线 |

## 6. 端到端评测（增训前 refiner_ai 权重，12 组 × 5 配置）

| 组 | dL ↓ | dAB ↓ | BNR ↓ | FDR |
|---|---:|---:|---:|---:|
| A 直接合成 | 40.08 | 19.87 | 14.94 | 0.997 |
| B +Harmonize v2 | 27.61 | 9.62 | 14.12 | 0.892 |
| C +Relight 方向光 | 37.48 | 18.20 | 15.19 | 1.029 |
| **D 全流程** | **26.06** | **8.28** | 13.77 | 0.919 |
| **D + 特效** | 26.18 | **7.56** | **13.10** | 0.922 |

- D vs A：dL **↓35%**、dAB **↓58%**；**12/12 组 FDR 全部 ∈[0.7,1.3]**——压黑一票否决项全通过
- 对比旧线（B6）：旧 dAB 10.63 是靠压黑换来的（FDR 未监控）；新线在前景细节完整的前提下取得更好的一致性

## 7. 交付物索引

```
training/checkpoints/refiner_ai/     混合域权重（best/last + config/log/metrics 四件套）
training/scripts/ai_material_eval.py 素材评测（色度键 GT + 双路推理 + 四指标）
training/scripts/ai_finetune_refiner.py 混合域微调（--mix_studio）
training/scripts/ai_e2e_eval.py      端到端四组+特效评测（dL/dAB/BNR/FDR）
ai-service/src/fx/fx_tool.py         特效工具
ai-service/src/matting/interactive_matting.py 圈选抠图
ai-service/src/harmonization/harmonize_tool.py harmonize v2
ai-service/src/relighting/relight_tool.py     relight 方向光
experiments/ai_materials/            评测产物（e2e_grid.png 等）
experiments/ai_materials_v2/         增训后抠图对比网格
```

## 8. 下一步

1. 混合域训练结果回填 + 难例三张评测
2. 扩充绿幕人像规模（4 身份易过拟合，目标 ≥15 身份）
3. B6 主观评分（n≥10 双盲）补闭环
4. 真实拍摄数据接入（`ingest_real.py` 接口已备）
