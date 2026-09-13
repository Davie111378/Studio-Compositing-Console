#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演播室场景抠图数据集生成器
================================

面向「虚拟演播室 / 主持人抠像」真实分布的程序化数据合成。

与通用几何合成数据的本质区别 —— 本生成器显式建模了演播室特有的物理：

1. **三点布光 (Three-point lighting)**
   - Key light   : 主光, 5600K, 强度 1.0, 前侧 45°
   - Fill light  : 补光, 6500K, 强度 0.35, 反向填充阴影
   - Rim light   : 轮廓光, 冷白, 强度 0.8, 逆光勾边（发丝边缘发光）
   用 alpha 的距离场构造伪高度 → 法线 → Lambertian + Rim，任意形状都成立体光照。

2. **绿幕溢出 (Green spill)**
   绿幕反射的绿光污染前景边缘，是演播室抠像的头号难题：
       F' = F + spill * G * w(a),   w(a) = 4a(1-a)  (过渡带最强)

3. **四类演播室背景**
   - greenscreen : 绿幕, 含布光不均 + 褶皱 + 暗角
   - bluescreen  : 蓝幕
   - led_wall    : LED 大屏虚拟演播室（发光块 + 点阵 + 光晕）
   - studio_set  : 实景演播室（地板透视 + 聚光灯 + 景深虚化）

4. **八类演播室难点主体**
   - anchor_hair      主持人发丝（最密发丝）
   - anchor_glasses   戴眼镜主持人（镜片高光 + 框外半透明）
   - anchor_gesture   手势/手指（细长结构, 自遮挡）
   - guest_motion     快速运动（方向性运动模糊）
   - prop_transparent 透明道具（玻璃水杯）
   - prop_veil        丝巾/薄纱（大面积半透明 + 褶皱）
   - greenscreen_spill 强绿幕溢出（边缘严重污染）
   - lowlight_rim     逆光轮廓光（低照度 + 强 rim, 亮边易误判）

5. **真实传感器退化**
   运动模糊 / 高斯噪声 / 色温偏移 / 暗角

输出（与 train_refiner.py 的 manifest 协议完全兼容）:
    <root>/<split>/<name>.png         合成 RGB
    <root>/<split>/<name>_alpha.png   GT alpha (精确, 超采样得到亚像素)
    <root>/manifest.csv               image,alpha,case,split,bg,lighting
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ---------------------------------------------------------------- 常量

SS = 2  # 超采样倍率：在 2x 画布渲染后降采样, 得到亚像素精度 alpha

# 演播室三点布光（方向已归一化, 指向光源）
LIGHT_RIGS: Dict[str, List[Tuple[np.ndarray, float, np.ndarray]]] = {
    #  名称        (方向xyz,                强度,  色温RGB 0-1)
    "three_point": [
        (np.array([-0.55, -0.65, 0.52]), 1.00, np.array([1.00, 0.97, 0.90])),  # key 5600K
        (np.array([0.75, 0.25, 0.45]), 0.35, np.array([0.85, 0.92, 1.00])),  # fill 6500K
        (np.array([0.30, -0.35, -0.88]), 0.80, np.array([0.80, 0.88, 1.00])),  # rim 冷白
    ],
    "key_only": [
        (np.array([-0.60, -0.60, 0.53]), 1.00, np.array([1.00, 0.95, 0.88])),
        (np.array([0.60, 0.30, 0.50]), 0.12, np.array([0.9, 0.9, 1.0])),
        (np.array([0.20, -0.30, -0.90]), 0.25, np.array([0.9, 0.9, 1.0])),
    ],
    "warm_studio": [
        (np.array([-0.50, -0.70, 0.50]), 0.95, np.array([1.00, 0.90, 0.78])),
        (np.array([0.70, 0.20, 0.48]), 0.40, np.array([1.00, 0.93, 0.82])),
        (np.array([0.25, -0.40, -0.88]), 0.70, np.array([1.00, 0.85, 0.70])),
    ],
    "lowlight_rim": [
        (np.array([-0.45, -0.55, 0.60]), 0.28, np.array([0.75, 0.82, 1.00])),
        (np.array([0.60, 0.30, 0.55]), 0.10, np.array([0.7, 0.8, 1.0])),
        (np.array([0.35, -0.45, -0.82]), 1.35, np.array([0.70, 0.85, 1.00])),  # 强 rim
    ],
}

SKIN_TONES = [
    (0.96, 0.76, 0.64), (0.88, 0.66, 0.52), (0.78, 0.56, 0.42),
    (0.65, 0.45, 0.33), (0.52, 0.35, 0.26), (0.90, 0.72, 0.60),
]
HAIR_COLORS = [
    (0.08, 0.07, 0.07), (0.16, 0.11, 0.08), (0.30, 0.20, 0.12),
    (0.45, 0.32, 0.18), (0.62, 0.50, 0.30), (0.30, 0.28, 0.30),
]
CLOTH_COLORS = [
    (0.14, 0.16, 0.24), (0.20, 0.22, 0.30), (0.30, 0.16, 0.18),
    (0.18, 0.28, 0.24), (0.55, 0.55, 0.58), (0.12, 0.12, 0.14),
    (0.35, 0.30, 0.42), (0.72, 0.70, 0.68),
]

CASES = [
    "anchor_hair", "anchor_glasses", "anchor_gesture", "guest_motion",
    "prop_transparent", "prop_veil", "greenscreen_spill", "lowlight_rim",
]
BGS = ["greenscreen", "bluescreen", "led_wall", "studio_set"]


# ---------------------------------------------------------------- 工具

def _n(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def layer_over(dst_rgb, dst_a, src_rgb, src_a):
    """标准 over 合成: 返回 (rgb, a)。"""
    sa = src_a[..., None]
    da = dst_a[..., None]
    a_out = src_a + dst_a * (1.0 - src_a)
    denom = np.clip(a_out[..., None], 1e-6, None)
    rgb_out = (src_rgb * sa + dst_rgb * da * (1.0 - sa)) / denom
    a_out3 = np.clip(a_out, 0.0, 1.0)
    return np.clip(rgb_out, 0.0, 1.0), a_out3


def draw_poly_layer(size, polys, value):
    """在 L 画布上绘制若干多边形, 填充灰度 value。"""
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)
    for pts, v in polys:
        d.polygon(pts, fill=int(v if value is None else value))
    return img


def bezier(p0, p1, p2, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------- 前景：人形

def build_person(canvas: int, rng: random.Random, case: str):
    """返回 (rgb[H,W,3] float, alpha[H,W] float, hair_mask[H,W] float)。
    所有绘制在 canvas×canvas 的超采样画布上完成。"""
    W = H = canvas
    cx = W * 0.5 + rng.uniform(-0.03, 0.03) * W
    head_cy = H * (0.30 if case != "anchor_gesture" else 0.34)
    rx = W * rng.uniform(0.125, 0.150)
    ry = rx * rng.uniform(1.16, 1.30)

    skin = _n(rng.choice(SKIN_TONES))
    hair = _n(rng.choice(HAIR_COLORS))
    cloth = _n(rng.choice(CLOTH_COLORS))

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)

    # ---------- 躯干（肩 + 身体） ----------
    shoulder_y = head_cy + ry * 1.05
    half_shoulder = rx * rng.uniform(1.95, 2.35)
    body_pts = []
    # 左肩到左腰
    body_pts.append((cx - half_shoulder, H))
    body_pts.append((cx - half_shoulder * 0.98, shoulder_y + ry * 0.55))
    body_pts += bezier((cx - half_shoulder * 0.98, shoulder_y + ry * 0.55),
                       (cx - rx * 1.15, shoulder_y - ry * 0.10),
                       (cx - rx * 0.62, shoulder_y + ry * 0.22), 16)
    body_pts += bezier((cx - rx * 0.62, shoulder_y + ry * 0.22),
                       (cx - rx * 0.55, head_cy + ry * 0.72),
                       (cx + rx * 0.62, shoulder_y + ry * 0.22), 16)
    body_pts += bezier((cx + rx * 0.62, shoulder_y + ry * 0.22),
                       (cx + rx * 1.15, shoulder_y - ry * 0.10),
                       (cx + half_shoulder * 0.98, shoulder_y + ry * 0.55), 16)
    body_pts.append((cx + half_shoulder, H))
    body_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(body_img).polygon(body_pts, fill=255)
    body_img = body_img.filter(ImageFilter.GaussianBlur(SS * 0.9))
    b_a = np.array(body_img, dtype=np.float32) / 255.0
    c = np.broadcast_to(cloth, (H, W, 3)).copy()
    # 衣服加低频纹理（褶皱）
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fold = 0.90 + 0.10 * np.sin(xx / (18.0 * SS) + rng.random() * 6) * np.cos(yy / (55.0 * SS))
    c *= fold[..., None]
    rgb, alpha = layer_over(rgb, alpha, c, b_a)

    # ---------- 脖子 ----------
    neck_pts = [(cx - rx * 0.42, head_cy + ry * 0.55),
                (cx + rx * 0.42, head_cy + ry * 0.55),
                (cx + rx * 0.56, shoulder_y + ry * 0.35),
                (cx - rx * 0.56, shoulder_y + ry * 0.35)]
    neck_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(neck_img).polygon(neck_pts, fill=255)
    neck_img = neck_img.filter(ImageFilter.GaussianBlur(SS * 0.7))
    n_a = np.array(neck_img, dtype=np.float32) / 255.0
    nc = np.broadcast_to(skin * 0.86, (H, W, 3)).copy()
    rgb, alpha = layer_over(rgb, alpha, nc, n_a)

    # ---------- 头 ----------
    head_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(head_img).ellipse(
        [cx - rx, head_cy - ry, cx + rx, head_cy + ry], fill=255)
    head_img = head_img.filter(ImageFilter.GaussianBlur(SS * 0.7))
    h_a = np.array(head_img, dtype=np.float32) / 255.0
    hc = np.broadcast_to(skin, (H, W, 3)).copy()
    # 面部轻微色差（脸颊红润 + 额头亮）
    cheek = np.exp(-(((xx - cx) / (rx * 0.9)) ** 2 + ((yy - (head_cy + ry * 0.25)) / (ry * 0.35)) ** 2))
    hc[..., 0] += 0.06 * cheek
    hc[..., 1] -= 0.02 * cheek
    rgb, alpha = layer_over(rgb, alpha, np.clip(hc, 0, 1), h_a)

    # 五官（只改 RGB, 不改 alpha —— 提供语义让 matting 模型有东西可学）
    fd = ImageDraw.Draw(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))
    eye_y = head_cy - ry * 0.05
    eye_dx = rx * 0.42
    eye_r = rx * 0.115
    for sgn in (-1, 1):
        ex = cx + sgn * eye_dx
        fd.ellipse([ex - eye_r, eye_y - eye_r * 0.62, ex + eye_r, eye_y + eye_r * 0.62],
                   fill=tuple((np.clip(skin * 0.30, 0, 1) * 255).astype(int)))
        fd.ellipse([ex - eye_r * 0.42, eye_y - eye_r * 0.28,
                    ex + eye_r * 0.42, eye_y + eye_r * 0.28],
                   fill=tuple((np.clip(skin * 1.55, 0, 1) * 255).astype(int)))
        # 眉毛
        fd.arc([ex - eye_r * 1.5, eye_y - eye_r * 2.4, ex + eye_r * 1.5, eye_y - eye_r * 0.3],
               200, 340, fill=tuple((np.clip(hair * 255, 0, 255)).astype(int)), width=max(2, int(SS * 2)))
    # 嘴
    fd.arc([cx - rx * 0.30, head_cy + ry * 0.30, cx + rx * 0.30, head_cy + ry * 0.62],
           20, 160, fill=tuple((np.clip(skin * np.array([1.5, 0.55, 0.55]), 0, 1) * 255).astype(int)),
           width=max(2, int(SS * 2.2)))
    rgb = np.array(fd._image if hasattr(fd, "_image") else
                   Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)),
                   dtype=np.float32) / 255.0 if False else rgb

    # ---------- 头发（发块 + 大量发丝） ----------
    n_strands = {"anchor_hair": 520, "greenscreen_spill": 300, "lowlight_rim": 380}.get(case, 240)
    hair_img = Image.new("L", (W, H), 0)
    hd = ImageDraw.Draw(hair_img)
    # 发块：头顶椭圆上部
    hd.ellipse([cx - rx * 1.10, head_cy - ry * 1.16, cx + rx * 1.10, head_cy + ry * 0.10], fill=255)
    # 侧发
    hd.polygon([(cx - rx * 1.08, head_cy - ry * 0.9), (cx - rx * 1.02, head_cy + ry * 0.95),
                (cx - rx * 0.55, head_cy + ry * 0.80), (cx - rx * 0.78, head_cy - ry * 0.85)], fill=235)
    hd.polygon([(cx + rx * 1.08, head_cy - ry * 0.9), (cx + rx * 1.02, head_cy + ry * 0.95),
                (cx + rx * 0.55, head_cy + ry * 0.80), (cx + rx * 0.78, head_cy - ry * 0.85)], fill=235)
    hair_img = hair_img.filter(ImageFilter.GaussianBlur(SS * 1.1))
    hair_block = np.array(hair_img, dtype=np.float32) / 255.0

    # 发丝：从头顶/两侧发出, 二次贝塞尔, 灰度随机 → 亚像素半透明
    strand_img = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(strand_img)
    for _ in range(n_strands):
        side = rng.choice([-1, 1])
        t = rng.uniform(0.0, 1.0)
        # 起点：沿头顶弧
        ang = math.pi * (0.06 + 0.88 * t)
        sx = cx + side * rx * 1.02 * math.cos(ang) * (1.0 if side > 0 else 1.0)
        sy = head_cy - ry * 1.02 * math.sin(ang) * 0.98
        if side < 0:
            sx = cx - rx * 1.02 * math.cos(ang)
        length = rng.uniform(0.10, 0.42) * W
        dirx = side * rng.uniform(0.55, 1.0)
        diry = rng.uniform(-0.55, 0.95)
        norm = math.hypot(dirx, diry) + 1e-6
        ex = sx + dirx / norm * length
        ey = sy + diry / norm * length * rng.uniform(0.7, 1.3)
        cxp = sx + dirx / norm * length * 0.5 + rng.uniform(-1, 1) * length * 0.42
        cyp = sy + diry / norm * length * 0.5 + rng.uniform(-1, 1) * length * 0.30
        grey = int(rng.uniform(70, 245))
        wdt = max(1, int(round(rng.uniform(0.8, 2.6) * SS / 1.4)))
        sd.line(bezier((sx, sy), (cxp, cyp), (ex, ey), 18), fill=grey, width=wdt)
    strand_img = strand_img.filter(ImageFilter.GaussianBlur(SS * 0.55))
    hair_strands = np.array(strand_img, dtype=np.float32) / 255.0

    hair_mask = np.clip(hair_block + hair_strands * (1 - hair_block * 0.35), 0, 1)
    # 发块内部应为实心, 发丝提供边缘半透明
    hair_a = np.clip(np.maximum(hair_block * 0.98, hair_strands), 0, 1)
    hair_a = np.where(hair_block > 0.6, np.maximum(hair_a, 0.93), hair_a)
    hcol = np.broadcast_to(hair, (H, W, 3)).copy()
    # 发丝高光（演播室 rim light 会点亮发梢）
    strand_hi = gaussian_filter(hair_strands, sigma=SS * 1.2)
    hcol += (strand_hi[..., None] * np.array([0.22, 0.20, 0.16], dtype=np.float32))
    rgb, alpha = layer_over(rgb, alpha, np.clip(hcol, 0, 1), hair_a)

    # ---------- 眼镜 ----------
    if case == "anchor_glasses":
        g_img = Image.new("L", (W, H), 0)
        gd = ImageDraw.Draw(g_img)
        gw = max(2, int(SS * 2.4))
        for sgn in (-1, 1):
            ex = cx + sgn * eye_dx
            gd.ellipse([ex - rx * 0.30, eye_y - rx * 0.24, ex + rx * 0.30, eye_y + rx * 0.24],
                       outline=255, width=gw)
        gd.line([(cx - rx * 0.18, eye_y), (cx + rx * 0.18, eye_y)], fill=255, width=gw)
        # 镜腿（超出脸部, 半透明金属）
        gd.line([(cx - rx * 1.02, eye_y - rx * 0.05), (cx - rx * 1.30, eye_y - rx * 0.12)],
                fill=200, width=gw)
        gd.line([(cx + rx * 1.02, eye_y - rx * 0.05), (cx + rx * 1.30, eye_y - rx * 0.12)],
                fill=200, width=gw)
        g_img = g_img.filter(ImageFilter.GaussianBlur(SS * 0.5))
        g_a = np.array(g_img, dtype=np.float32) / 255.0
        gcol = np.broadcast_to(np.array([0.10, 0.10, 0.12], dtype=np.float32), (H, W, 3)).copy()
        rgb, alpha = layer_over(rgb, alpha, gcol, g_a)

    # ---------- 手势（细长结构 + 自遮挡） ----------
    if case == "anchor_gesture":
        for _ in range(rng.randint(1, 2)):
            bx = cx + rng.uniform(-0.55, 0.55) * rx * 2.2
            by = shoulder_y + rng.uniform(0.2, 1.4) * ry
            ang = rng.uniform(-1.2, 1.2)
            f_img = Image.new("L", (W, H), 0)
            fdd = ImageDraw.Draw(f_img)
            palm_r = rx * rng.uniform(0.32, 0.46)
            fdd.ellipse([bx - palm_r, by - palm_r * 1.15, bx + palm_r, by + palm_r * 1.15], fill=255)
            for k in range(5):
                fa = ang + (k - 2) * 0.34 + rng.uniform(-0.08, 0.08)
                flen = palm_r * rng.uniform(1.5, 2.6)
                tip = (bx + math.cos(fa - math.pi / 2) * flen,
                       by + math.sin(fa - math.pi / 2) * flen * 1.25)
                ctrl = (bx + math.cos(fa - math.pi / 2) * flen * 0.5 + rng.uniform(-8, 8),
                        by + math.sin(fa - math.pi / 2) * flen * 0.55)
                fdd.line(bezier((bx, by), ctrl, tip, 14), fill=255,
                         width=max(2, int(palm_r * 0.42)))
            f_img = f_img.filter(ImageFilter.GaussianBlur(SS * 0.75))
            f_a = np.array(f_img, dtype=np.float32) / 255.0
            fcol = np.broadcast_to(skin * 1.02, (H, W, 3)).copy()
            rgb, alpha = layer_over(rgb, alpha, np.clip(fcol, 0, 1), f_a)

    # ---------- 透明道具（玻璃水杯） ----------
    if case == "prop_transparent":
        px = cx + rng.uniform(0.85, 1.35) * rx * rng.choice([-1, 1])
        py = shoulder_y + rng.uniform(0.3, 1.1) * ry
        cw, ch = rx * rng.uniform(0.28, 0.42), ry * rng.uniform(0.55, 0.85)
        gl_img = Image.new("L", (W, H), 0)
        gld = ImageDraw.Draw(gl_img)
        gld.rounded_rectangle([px - cw, py - ch, px + cw, py + ch],
                              radius=int(cw * 0.35), fill=86)      # 杯体 0.34
        gld.rounded_rectangle([px - cw, py - ch, px + cw, py + ch],
                              radius=int(cw * 0.35), outline=190, width=max(2, int(SS * 1.8)))
        gld.rounded_rectangle([px - cw * 0.86, py + ch * 0.05, px + cw * 0.86, py + ch * 0.92],
                              radius=int(cw * 0.28), fill=130)     # 液体 0.51
        gl_img = gl_img.filter(ImageFilter.GaussianBlur(SS * 0.6))
        gl_a = np.array(gl_img, dtype=np.float32) / 255.0
        glcol = np.broadcast_to(np.array([0.72, 0.80, 0.86], dtype=np.float32), (H, W, 3)).copy()
        rgb, alpha = layer_over(rgb, alpha, glcol, gl_a)

    # ---------- 薄纱 / 丝巾 ----------
    if case == "prop_veil":
        v_img = Image.new("L", (W, H), 0)
        vd = ImageDraw.Draw(v_img)
        vy = shoulder_y + rng.uniform(-0.1, 0.5) * ry
        vpts = []
        for i in range(28):
            tt = i / 27
            vpts.append((cx - half_shoulder * 1.05 + tt * half_shoulder * 2.10,
                         vy + math.sin(tt * 9.0 + rng.random() * 0.4) * ry * 0.16))
        for i in range(22):
            tt = i / 21
            vpts.append((cx + half_shoulder * 1.05 - tt * half_shoulder * 2.10,
                         vy + ry * 1.25 + math.sin(tt * 7.0) * ry * 0.14))
        vd.polygon(vpts, fill=92)  # 0.36
        v_img = v_img.filter(ImageFilter.GaussianBlur(SS * 1.4))
        # 褶皱：明暗调制 → 转成 alpha 调制
        v_arr = np.array(v_img, dtype=np.float32) / 255.0
        yv, xv = np.mgrid[0:H, 0:W].astype(np.float32)
        fold = 0.72 + 0.28 * np.sin(xv / (26.0 * SS) + np.sin(yv / (90.0 * SS)) * 2.2)
        v_a = np.clip(v_arr * fold, 0, 1)
        vcol = np.broadcast_to(np.array([0.86, 0.82, 0.88], dtype=np.float32), (H, W, 3)).copy()
        rgb, alpha = layer_over(rgb, alpha, vcol, v_a)

    return np.clip(rgb, 0, 1), np.clip(alpha, 0, 1), hair_mask


# ---------------------------------------------------------------- 光照

def apply_studio_lighting(rgb, alpha, rig_name: str, rng: random.Random):
    """三点布光：距离场 → 伪高度 → 法线 → Lambertian + Rim。"""
    H, W = alpha.shape
    inside = alpha > 0.5
    if inside.sum() < 50:
        return rgb
    # 到背景的距离（内部）
    dist = distance_transform_edt(inside).astype(np.float32)
    dmax = dist.max() + 1e-6
    height = np.sqrt(np.clip(dist / dmax, 0, 1))
    height = gaussian_filter(height, sigma=max(2.0, 10.0 * SS / 2.0))
    # 外部也略微延展, 避免边界法线突变
    gy, gx = np.gradient(height)
    nz = np.ones_like(height) * 0.55
    nl = np.sqrt(gx ** 2 + gy ** 2 + nz ** 2)
    Nx, Ny, Nz = -gx / nl, -gy / nl, nz / nl

    shade = np.zeros((H, W, 3), dtype=np.float32)
    ambient = 0.14
    for L, inten, color in LIGHT_RIGS[rig_name]:
        L = L / (np.linalg.norm(L) + 1e-6)
        ndotl = np.clip(Nx * L[0] + Ny * L[1] + Nz * L[2], 0, 1)
        shade += (ndotl * inten)[..., None] * _n(color)[None, None, :]

    # Rim light：视线(0,0,1)与法线夹角 → 边缘发光
    rim_inten = max(l[1] for l in LIGHT_RIGS[rig_name] if l[0][2] < -0.5) if any(
        l[0][2] < -0.5 for l in LIGHT_RIGS[rig_name]) else 0.6
    rim = (1.0 - np.clip(Nz, 0, 1)) ** 2.6 * rim_inten
    rim_color = _n([0.78, 0.86, 1.0])
    shade += rim[..., None] * rim_color[None, None, :] * 0.9

    lit = rgb * (shade + ambient)
    return np.clip(lit, 0, 1)


def apply_green_spill(rgb, alpha, strength: float, tint=(0.05, 1.0, 0.20)):
    """绿幕溢出：过渡带 (4a(1-a)) 最强, 并向内衰减。"""
    w = 4.0 * alpha * (1.0 - alpha)          # alpha=0.5 处最大
    inner = np.clip(alpha, 0, 1) ** 0.65     # 内部也有微弱污染
    weight = np.clip(w * 1.35 + inner * 0.22, 0, 1)
    tint = _n(tint)
    out = rgb + weight[..., None] * strength * tint[None, None, :] * rgb
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- 背景

def make_greenscreen(H, W, rng, base):
    yv, xv = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        bg[..., c] = base[c]
    # 布光不均（低频）
    for _ in range(rng.randint(2, 4)):
        fx = rng.uniform(0.4, 2.2) / W * 6.0
        fy = rng.uniform(0.4, 2.2) / H * 6.0
        ph = rng.uniform(0, 6.28)
        amp = rng.uniform(0.02, 0.07)
        bg += (amp * np.sin(xv * fx + ph) * np.cos(yv * fy + ph * 0.7))[..., None]
    # 褶皱（折线状低频）
    if rng.random() < 0.6:
        k = rng.randint(2, 5)
        for _ in range(k):
            ang = rng.uniform(0, math.pi)
            f = rng.uniform(3.0, 9.0) / W
            ph = rng.uniform(0, 6.28)
            bg += (0.035 * np.sin((xv * math.cos(ang) + yv * math.sin(ang)) * f * 6.0 + ph))[..., None]
    # 暗角
    vign = 1.0 - 0.22 * (((xv / W) - 0.5) ** 2 + ((yv / H) - 0.5) ** 2) * 4
    bg *= vign[..., None]
    # 噪声
    bg += rng.uniform(0.002, 0.010) * np.random.randn(H, W, 1).astype(np.float32)
    return np.clip(bg, 0, 1)


def make_led_wall(H, W, rng):
    """LED 大屏虚拟演播室：深色底 + 发光内容块 + 点阵 + 光晕。"""
    yv, xv = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.array([0.04, 0.06, 0.11], dtype=np.float32)
    bg = np.ones((H, W, 3), dtype=np.float32) * base[None, None, :]
    # 发光块
    for _ in range(rng.randint(4, 9)):
        bx, by = rng.uniform(0, W), rng.uniform(0, H)
        bw, bh = rng.uniform(0.10, 0.42) * W, rng.uniform(0.06, 0.30) * H
        col = _n([rng.uniform(0.2, 0.95), rng.uniform(0.25, 0.9), rng.uniform(0.35, 1.0)])
        m = np.exp(-(((xv - bx) / (bw * 0.5)) ** 2 + ((yv - by) / (bh * 0.5)) ** 2) * 1.6)
        bg += m[..., None] * col[None, None, :] * rng.uniform(0.35, 0.95)
    # 横向扫描条
    for _ in range(rng.randint(2, 5)):
        yy = rng.uniform(0, H)
        m = np.exp(-((yv - yy) / (rng.uniform(8, 40) * SS / 2)) ** 2)
        bg += m[..., None] * rng.uniform(0.05, 0.22)
    # LED 点阵
    pitch = max(3, int(round(4 * SS)))
    gx = ((np.arange(W)[None, :] % pitch) < pitch * 0.45).astype(np.float32)
    gy = ((np.arange(H)[:, None] % pitch) < pitch * 0.45).astype(np.float32)
    grid = gx * gy
    bg += grid[..., None] * rng.uniform(0.01, 0.05)
    bg = np.clip(bg, 0, 1)
    return gaussian_filter(bg, sigma=[SS * 0.8, SS * 0.8, 0])


def make_studio_set(H, W, rng):
    """实景演播室：地板透视 + 背景墙 + 聚光灯 + 景深虚化。"""
    yv, xv = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), dtype=np.float32)
    wall_col = _n([rng.uniform(0.10, 0.22), rng.uniform(0.11, 0.23), rng.uniform(0.14, 0.28)])
    floor_col = _n([rng.uniform(0.06, 0.16), rng.uniform(0.06, 0.15), rng.uniform(0.07, 0.17)])
    horizon = H * rng.uniform(0.52, 0.68)
    wall_m = (yv < horizon).astype(np.float32)
    # 墙：上暗下亮
    wall_grad = 0.7 + 0.5 * (yv / max(horizon, 1))
    bg += wall_m[..., None] * (wall_col[None, None, :] * np.clip(wall_grad, 0, 1.4)[..., None])
    # 地板：透视渐变 + 反光
    fl = np.clip((yv - horizon) / max(H - horizon, 1), 0, 1)
    bg += (1 - wall_m)[..., None] * (floor_col[None, None, :] * (0.55 + 0.9 * fl)[..., None])
    # 聚光灯
    for _ in range(rng.randint(1, 3)):
        sx, sy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.1, 0.55) * H
        r = rng.uniform(0.12, 0.32) * W
        m = np.exp(-(((xv - sx) / r) ** 2 + ((yv - sy) / (r * 1.15)) ** 2))
        bg += m[..., None] * rng.uniform(0.10, 0.38)
    # 模糊道具剪影
    for _ in range(rng.randint(1, 3)):
        px, py = rng.uniform(0, W), rng.uniform(horizon * 0.75, H)
        pw, ph = rng.uniform(0.08, 0.25) * W, rng.uniform(0.06, 0.22) * H
        m = np.exp(-(((xv - px) / pw) ** 2 + ((yv - py) / ph) ** 2) * 2.0)
        bg -= m[..., None] * rng.uniform(0.03, 0.10)
    bg = np.clip(bg, 0, 1)
    # 景深虚化
    return gaussian_filter(bg, sigma=[SS * 1.6, SS * 1.6, 0])


def make_background(bg_type, H, W, rng):
    if bg_type == "greenscreen":
        return make_greenscreen(H, W, rng, base=np.array([0.02, 0.72, 0.26]))
    if bg_type == "bluescreen":
        return make_greenscreen(H, W, rng, base=np.array([0.03, 0.20, 0.78]))
    if bg_type == "led_wall":
        return make_led_wall(H, W, rng)
    return make_studio_set(H, W, rng)


# ---------------------------------------------------------------- 退化

def motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    if length < 2:
        return img
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    # 用多次小角度旋转 + box 近似：这里用可分离的方向模糊核（PIL 无旋转核, 用仿射近似）
    ker = Image.new("L", (length * 2 + 1, length * 2 + 1), 0)
    d = ImageDraw.Draw(ker)
    cx = cy = length
    ex = cx + math.cos(angle) * length
    ey = cy + math.sin(angle) * length
    d.line([(cx, cy), (ex, ey)], fill=255, width=max(1, length // 3))
    k = np.array(ker, dtype=np.float32)
    k /= (k.sum() + 1e-6)
    # 手动 separable 不可用 → 直接用 scipy 卷积
    from scipy.ndimage import convolve
    out = np.stack([convolve(img[..., c], k, mode="nearest") for c in range(3)], axis=-1)
    return np.clip(out, 0, 1)


def sensor_degrade(img, rng, case):
    H, W, _ = img.shape
    # 色温偏移
    if rng.random() < 0.5:
        t = rng.uniform(-0.05, 0.05)
        img = img * np.array([1.0 + t, 1.0, 1.0 - t], dtype=np.float32)[None, None, :]
    # 噪声
    sigma = {"lowlight_rim": 0.016}.get(case, 0.008) * rng.uniform(0.6, 1.5)
    img = img + sigma * np.random.randn(H, W, 3).astype(np.float32)
    # 暗角
    yv, xv = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1.0 - rng.uniform(0.05, 0.20) * (((xv / W) - .5) ** 2 + ((yv / H) - .5) ** 2) * 4
    img = img * v[..., None]
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- 单张合成

def synthesize_one(size_wh, case, out_dir: Path, idx: int, rng: random.Random):
    W, H = size_wh
    canvas = W * SS

    fg_rgb, fg_a, hair_mask = build_person(canvas, rng, case)

    # 布光
    rig = "lowlight_rim" if case == "lowlight_rim" else rng.choice(
        ["three_point", "three_point", "three_point", "warm_studio", "key_only"])
    fg_rgb = apply_studio_lighting(fg_rgb, fg_a, rig, rng)

    # 背景选择（强绑定 case 语义）
    if case == "greenscreen_spill":
        bg_type = rng.choice(["greenscreen", "greenscreen", "bluescreen"])
    elif case == "lowlight_rim":
        bg_type = rng.choice(["studio_set", "led_wall", "studio_set"])
    elif case == "prop_transparent":
        bg_type = rng.choice(["studio_set", "led_wall", "greenscreen"])
    else:
        bg_type = rng.choice(BGS)
    bg = make_background(bg_type, canvas, canvas, rng)

    # 绿幕溢出（物理：只有绿/蓝幕才溢出）
    spill = 0.0
    if bg_type in ("greenscreen", "bluescreen"):
        spill = {"greenscreen_spill": rng.uniform(0.55, 0.95),
                 "anchor_hair": rng.uniform(0.25, 0.55)}.get(case, rng.uniform(0.15, 0.45))
        tint = (0.05, 1.0, 0.20) if bg_type == "greenscreen" else (0.05, 0.25, 1.0)
        fg_rgb = apply_green_spill(fg_rgb, fg_a, spill, tint)

    # 合成
    a3 = fg_a[..., None]
    comp = fg_rgb * a3 + bg * (1.0 - a3)

    # 运动模糊
    if case == "guest_motion":
        comp = motion_blur(comp, int(rng.uniform(4, 16) * SS / 2), rng.uniform(0, 6.28))
        fg_a = np.clip(np.array(Image.fromarray((fg_a * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(rng.uniform(1.0, 3.0) * SS / 2)), dtype=np.float32) / 255.0, 0, 1)

    # 降采样（超采样 → 亚像素 alpha）
    comp_img = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    alpha_img = Image.fromarray((np.clip(fg_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)

    comp = np.array(comp_img, dtype=np.float32) / 255.0
    alpha = np.array(alpha_img, dtype=np.float32) / 255.0

    comp = sensor_degrade(comp, rng, case)

    name = f"{case}_{idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{name}.png")
    Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{name}_alpha.png")
    return {
        "image": f"{name}.png", "alpha": f"{name}_alpha.png",
        "case": case, "bg": bg_type, "lighting": rig, "spill": round(spill, 3),
    }


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/AIcode/生产实习/data/studio")
    ap.add_argument("--n_per_case", type=int, default=120)
    ap.add_argument("--size", default="512,512")
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    W, H = [int(v) for v in args.size.split(",")]
    root = Path(args.root)
    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2 ** 32))

    manifest = []
    for case in CASES:
        for i in range(args.n_per_case):
            if i < int(args.n_per_case * 0.75):
                split = "train"
            elif i < int(args.n_per_case * 0.875):
                split = "val"
            else:
                split = "test"
            info = synthesize_one((W, H), case, root / split, i, rng)
            info["split"] = split
            manifest.append(info)
        print(f"[studio] case={case} done", flush=True)

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "alpha", "case", "split", "bg", "lighting", "spill"])
        w.writeheader()
        w.writerows(manifest)
    print(f"[studio] total {len(manifest)} -> {root}")


if __name__ == "__main__":
    main()
