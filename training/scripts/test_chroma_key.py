# -*- coding: utf-8 -*-
"""T1 补充：绿幕色键（chroma key + despill）基线 —— 绿幕演播室的工业标准做法
S3: chroma key，与 S0(BiRefNet) / S2(Refiner) 三方对比
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("D:/AIcode/生产实习")
OUT = ROOT / "experiments" / "studio" / "single_image_test"
img = np.array(Image.open(ROOT / "演播室图片生成需求.png").convert("RGB")).astype(np.float32)

r, g, b = img[..., 0], img[..., 1], img[..., 2]
# 绿色优势度：g 超过 max(r,b) 的程度（绿幕区域 g >> max(r,b)）
excess = g - np.maximum(r, b)
# 软阈值：excess > 60 全透明， < 15 全不透明，中间线性过渡
alpha = np.clip((60.0 - excess) / (60.0 - 15.0), 0, 1)
# 只在"原本就偏绿"的区域做抠除，避免误伤白色台面高光里的轻微绿色反射
greenish = (excess > 10).astype(np.float32)
alpha = alpha * greenish + (1 - greenish) * (alpha > 0).astype(np.float32) * alpha

# 时间信息只是演示，不精确计时
alpha8 = (alpha * 255).astype(np.uint8)
Image.fromarray(alpha8).save(OUT / "alpha_chroma.png")

# despill：前景像素里 g 超出 max(r,b) 的部分压回
spill = np.clip(g - np.maximum(r, b), 0, None) * (alpha > 0.1)
g2 = g - spill
fg = np.stack([r, g2, b], axis=-1) * alpha[..., None]
Image.fromarray(fg.astype(np.uint8)).save(OUT / "fg_chroma.png")

stats = {
    "fg_ratio": f"{float((alpha > 0.5).mean()) * 100:.1f}%",
    "soft_edge_pixels": f"{float(((alpha > 0.05) & (alpha < 0.95)).mean()) * 100:.2f}%",
}
print("[S3 chroma]", json.dumps(stats))
