# -*- coding: utf-8 -*-
"""合成 ImageStage 各关键状态图（本地合成，不使用生成式模型）。

产出（imagecompose-site/media/）：
  source_original.png      去水印后的原始照片（章节 02 之前的主图）
  stage_compose.png        主体放入新环境，但未打光、无接触阴影
  stage_ground.png         加入接触阴影与水面倒影
  stage_final.png          整体调色 / 颗粒统一的成片
"""
import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

M = r"G:\myself\作业\生产实习\imagecompose-site\media"
ERASED = r"G:\myself\作业\生产实习\processed_image_img89134063-7a77d2d2-90d0-4e58-b7a8-35d4e2eb2816_1.png"

# ---------- 1. 原始照片 ----------
orig = Image.open(ERASED).convert("RGB")
orig.save(os.path.join(M, "source_original.png"))
print("source_original", orig.size)

# ---------- 2. 主体 ----------
sub = Image.open(os.path.join(M, "subject_alpha.png")).convert("RGBA")
sub = sub.crop(sub.getchannel("A").getbbox())
print("subject cropped", sub.size)

# ---------- 3. 环境 ----------
scene = Image.open(os.path.join(M, "scene_lakeside.png")).convert("RGB")
W, H = scene.size
print("scene", W, H)

SH = 700                      # 主体高度
SW = int(sub.width * SH / sub.height)
s = sub.resize((SW, SH), Image.LANCZOS)

FEET_Y = 872                  # 脚底落点
CX = 700                      # 主体中心 x


def grade(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """从左侧夕阳打来的暖光：左暖右冷，并压低整体亮度以融入暮色。"""
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


def paste(base: Image.Image, layer: Image.Image, x: int, y: int) -> Image.Image:
    base.paste(layer, (x, y), layer)
    return base


px, py = int(CX - SW / 2), int(FEET_Y - SH)


def build(with_shadow=True, with_grade=True, grain=3.0) -> Image.Image:
    canvas = scene.copy()
    s2 = grade(s, 0.85 if with_grade else 0.0)

    if with_shadow:
        # 水面倒影
        refl = s2.transpose(Image.FLIP_TOP_BOTTOM)
        refl = refl.resize((SW, int(SH * 0.34)), Image.LANCZOS)
        ra = np.asarray(refl).astype(np.float32)
        ra[:, :, 3] *= 0.22
        refl = Image.fromarray(ra.astype(np.uint8), "RGBA") \
            .filter(ImageFilter.GaussianBlur(4))
        paste(canvas, refl, px, FEET_Y - 6)

        # 接触阴影（脚下椭圆，软边）
        sh = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(sh)
        d.ellipse([CX - SW * 0.75, FEET_Y - 26, CX + SW * 0.75, FEET_Y + 20], fill=120)
        sh = sh.filter(ImageFilter.GaussianBlur(16))
        dark = Image.new("RGB", (W, H), (22, 26, 30))
        canvas = Image.composite(Image.blend(canvas, dark, 0.42), canvas, sh)

    paste(canvas, s2, px, py)

    if grain > 0:
        a = np.asarray(canvas).astype(np.float32)
        rng = np.random.default_rng(7)
        a += rng.normal(0, grain, a.shape)
        canvas = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    return canvas


build(with_shadow=False, with_grade=False, grain=0).save(
    os.path.join(M, "stage_compose.png"))
print("stage_compose ok")

build(with_shadow=True, with_grade=True, grain=0).save(
    os.path.join(M, "stage_ground.png"))
print("stage_ground ok")

final = build(with_shadow=True, with_grade=True, grain=3.0)
final.save(os.path.join(M, "stage_final.png"))
print("stage_final ok", final.size)

# 主体（透明底）单独导出成便于网页使用的版本
s.resize((int(sub.width * 1100 / sub.height), 1100), Image.LANCZOS).save(
    os.path.join(M, "subject_transparent.png"))
print("subject_transparent ok")
