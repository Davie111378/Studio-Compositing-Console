# -*- coding: utf-8 -*-
"""生成 06 GROUND 章节用的「无阴影」脚部条带（与 detail_feet.jpg 像素对齐）。

复刻 tools/compose_stage.py 的 build(with_shadow=False, with_grade=True) 基线，
再按 detail_feet.jpg 同一裁切框 (253, 418, 1283, 914) 输出。
"""
import os
import numpy as np
from PIL import Image

M = r"G:\myself\作业\生产实习\imagecompose-site\media"

scene = Image.open(os.path.join(M, "scene_lakeside.png")).convert("RGB")
W, H = scene.size
sub = Image.open(os.path.join(M, "subject_alpha.png")).convert("RGBA")
sub = sub.crop(sub.getchannel("A").getbbox())
SH = 700
SW = int(sub.width * SH / sub.height)
s = sub.resize((SW, SH), Image.LANCZOS)
FEET_Y, CX = 872, 700
px, py = int(CX - SW / 2), int(FEET_Y - SH)


def grade(img, strength=1.0):
    a = np.asarray(img).astype(np.float32)
    w, h = img.size
    xs = np.linspace(0, 1, w)[None, :, None]
    warm = np.array([1.26, 1.02, 0.74], dtype=np.float32)
    cool = np.array([0.78, 0.86, 0.98], dtype=np.float32)
    k = np.clip(xs * 1.25, 0, 1)
    g = warm * (1 - k) + cool * k
    f = 1 + strength * (g - 1)
    rgb = np.clip(a[:, :, :3] * f, 0, 255)
    return Image.fromarray(
        np.concatenate([rgb, a[:, :, 3:4]], axis=2).astype(np.uint8), "RGBA")


canvas = scene.copy()
s2 = grade(s, 0.85)
canvas.paste(s2, (px, py), s2)
band = canvas.crop((253, 418, 1283, 914))
band.save(os.path.join(M, "detail_feet_noshadow.jpg"), "JPEG", quality=93, optimize=True)
print("detail_feet_noshadow", band.size)
