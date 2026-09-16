# -*- coding: utf-8 -*-
"""为 v2 设计稿生成「真实媒体」素材（全部来自真实照片的合成 / 裁切，不用生成式模型）。

产出到 imagecompose-site/media/：
  stage_relight.jpg   05 RELIGHT —— 更强的左侧暖光 + 主体左缘轮廓光
  detail_feet.jpg     06 SHADOW  —— 脚部与水面接触阴影的特写条
  macro_cloud.jpg     08 ENHANCE —— 云层肌理
  macro_water.jpg     08 ENHANCE —— 水面波光
  macro_hair.jpg      08 ENHANCE —— 发丝边缘（逆光）
"""
import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

M = r"G:\myself\作业\生产实习\imagecompose-site\media"


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
    return Image.fromarray(np.concatenate([rgb, a[:, :, 3:4]], axis=2).astype(np.uint8), "RGBA")


def rim_light(img, shift=9, gain=95):
    """在主体左缘叠加一条暖色轮廓光。"""
    a = np.asarray(img).astype(np.float32)
    al = a[:, :, 3:4] / 255.0
    edge = np.clip(al - np.roll(al, shift, axis=1), 0, 1)
    edge = np.asarray(
        Image.fromarray((edge[:, :, 0] * 255).astype(np.uint8), "L")
        .filter(ImageFilter.GaussianBlur(1.6))
    ).astype(np.float32)[:, :, None] / 255.0
    warm = np.array([255.0, 226.0, 178.0], dtype=np.float32)[None, None, :]
    a[:, :, :3] = np.clip(a[:, :, :3] + edge * warm * gain / 255.0, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


# ---------- 05 RELIGHT ----------
scene = Image.open(os.path.join(M, "scene_lakeside.png")).convert("RGB")
W, H = scene.size
sub = Image.open(os.path.join(M, "subject_alpha.png")).convert("RGBA")
sub = sub.crop(sub.getchannel("A").getbbox())
SH = 700
SW = int(sub.width * SH / sub.height)
s = sub.resize((SW, SH), Image.LANCZOS)
FEET_Y, CX = 872, 700
px, py = int(CX - SW / 2), int(FEET_Y - SH)

rel = grade(s, 1.45)
rel = rim_light(rel, 9, 130)

canvas = scene.copy()
# 倒影
refl = rel.transpose(Image.FLIP_TOP_BOTTOM).resize((SW, int(SH * 0.34)), Image.LANCZOS)
ra = np.asarray(refl).astype(np.float32)
ra[:, :, 3] *= 0.24
refl = Image.fromarray(ra.astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(4))
canvas.paste(refl, (px, FEET_Y - 6), refl)
# 接触阴影
sh = Image.new("L", (W, H), 0)
ImageDraw.Draw(sh).ellipse([CX - SW * 0.8, FEET_Y - 26, CX + SW * 0.8, FEET_Y + 22], fill=130)
sh = sh.filter(ImageFilter.GaussianBlur(16))
canvas = Image.composite(Image.blend(canvas, Image.new("RGB", (W, H), (20, 24, 28)), 0.46), canvas, sh)
canvas.paste(rel, (px, py), rel)
# 整体再强化一次左侧高光
arr = np.asarray(canvas).astype(np.float32)
k1 = np.clip(np.linspace(0, 1, W)[None, :] * 1.3, 0, 1)      # (1, W)
arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + 0.22 * (1 - k1)), 0, 255)
arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - 0.10 * (1 - k1)), 0, 255)
rng = np.random.default_rng(11)
arr += rng.normal(0, 2.6, arr.shape)
canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
canvas.save(os.path.join(M, "stage_relight.jpg"), "JPEG", quality=92, optimize=True)
print("stage_relight", canvas.size)

# ---------- 06 SHADOW 特写条 ----------
gr = Image.open(os.path.join(M, "stage_ground.png")).convert("RGB")
band = gr.crop((253, 418, 1283, 914))          # 1030 x 496
band = band.resize((1030, 496), Image.LANCZOS)
band.save(os.path.join(M, "detail_feet.jpg"), "JPEG", quality=93, optimize=True)
print("detail_feet", band.size)

# ---------- 08 ENHANCE 三张局部特写 ----------
def macro(src, box, out, size=(640, 800), sharpen=1.6):
    im = Image.open(os.path.join(M, src)).convert("RGB").crop(box)
    im = im.resize(size, Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=2.0, percent=int(sharpen * 90), threshold=3))
    im = ImageEnhance.Contrast(im).enhance(1.06)
    im = im.filter(ImageFilter.GaussianBlur(0.4))
    im.save(os.path.join(M, out), "JPEG", quality=92, optimize=True)
    print(out, im.size)


macro("stage_final.jpg", (300, 40, 700, 540), "macro_cloud.jpg")
macro("stage_final.jpg", (920, 380, 1320, 880), "macro_water.jpg")
macro("source_original.png", (452, 62, 672, 337), "macro_hair.jpg", sharpen=2.1)
