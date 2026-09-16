# -*- coding: utf-8 -*-
"""生成 02 EXTRACT 章的极细主体描边层（评审建议：mask 描边优于矩形框）。

从 subject_clean.png 的 alpha 提取 1px 级轮廓，着色 rgba(126,167,180,.45)，
输出与原照片同帧（1024x1536）的 PNG，前端按主体摆位对齐叠加，随扫描带渐显。
"""
import os
import numpy as np
from PIL import Image, ImageFilter

M = r"G:\myself\作业\生产实习\imagecompose-site\media"

src = Image.open(os.path.join(M, "subject_clean.png")).convert("RGBA")
a = src.split()[3].filter(ImageFilter.GaussianBlur(0.8))
arr = np.asarray(a).astype(np.float32)

gx = np.abs(arr - np.roll(arr, 1, axis=1))
gy = np.abs(arr - np.roll(arr, 1, axis=0))
edge = np.clip(gx + gy, 0, 255)

edge_img = Image.fromarray(edge.astype(np.uint8), "L").filter(ImageFilter.MaxFilter(3))
edge_img = edge_img.filter(ImageFilter.GaussianBlur(0.7))
e = np.asarray(edge_img).astype(np.float32) / 255.0

alpha = (e * 0.45 * 255).astype(np.uint8)
rgb = np.zeros((*edge.shape, 3), dtype=np.uint8)
rgb[:, :] = (126, 167, 180)

out = np.concatenate([rgb, alpha[:, :, None]], axis=2)
Image.fromarray(out, "RGBA").save(os.path.join(M, "subject_edge.png"), optimize=True)
print("subject_edge", out.shape)
