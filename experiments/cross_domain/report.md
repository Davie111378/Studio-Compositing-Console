# 跨域对比实验报告
评估集: studio/test (120 张)

## 0. 实验设置
- S0: BiRefNet 直出（粗 alpha, 512 输入, 权重来自 hf-mirror）
- S1: BiRefNet + 通用域 AlphaRefiner（跨域, 在 data/matting 上训练）
- S2: BiRefNet + 演播室域 AlphaRefiner（域内, 在 data/studio 上训练）

## 1. 指标对比

| Model | SAD | MSE | Grad | Conn | 相对 S0 SAD 增益 |
|---|---|---|---|---|---|
| S0_BiRefNet512 | 13981.63 | 0.02609 | 0.01 | 0.1264 |  |
| S1_GeneralRefiner | 37574.47 | 0.06560 | 0.01 | 0.1799 | -168.74% |
| S2_StudioRefiner | 4090.62 | 0.00221 | 0.00 | 0.0363 | +70.74% |

## 2. 视觉对比
![cross domain](cross_domain_grid.png)
