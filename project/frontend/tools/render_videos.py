# -*- coding: utf-8 -*-
"""用现有真实状态帧渲染两段视频：

  1) relight.mp4        —— 环境光进入 → 人物被重新打光（阶段 05 RELIGHT）
  2) critic_reroll.mp4  —— 逐项自检 → SHADOW 未达标 → 退回 GROUND → 重跑 → PASS（阶段 09 CRITIC）

每一段都出两个版本：
  *_clean.mp4  只有画面变化（给前端 <video> 用）
  *.mp4        画面 + 设计稿里的 UI 叠加（给评审看节奏用）

所有像素来自真实状态帧（原图 → 抠像 → 合成 → 打光 → 阴影），不做程序化插画。
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

M = r"G:\myself\作业\生产实习\imagecompose-site\media"
OUT = os.path.join(M, "video")
os.makedirs(OUT, exist_ok=True)

FPS = 20
W, H = 1952, 1240          # 976 x 620 @2x，与设计稿的 ImageStage 完全同坐标
SC = 2.0

C_INK = (32, 39, 43)
C_OK = (126, 167, 180)
C_WARN = (227, 178, 122)
C_TXT = (228, 231, 232)
C_MUTED = (124, 135, 139)
C_WHITE = (255, 246, 230)

F_MONO = r"C:\Windows\Fonts\consola.ttf"
F_MONOB = r"C:\Windows\Fonts\consolab.ttf"
F_CJK = r"C:\Windows\Fonts\msyh.ttc"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(F_MONO, size)


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3          # easeOutCubic


def ease_io(x):
    x = min(1.0, max(0.0, x))
    return 3 * x * x - 2 * x * x * x


def clamp01(x):
    return min(1.0, max(0.0, x))


def cover(im, w=W, h=H):
    iw, ih = im.size
    s = max(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - w) // 2, (nh - h) // 2
    return im.crop((l, t, l + w, t + h))


def load(p):
    return cover(Image.open(os.path.join(M, p)).convert("RGB"))


# ---------------------------------------------------------------- 关键帧
comp = load("stage_compose.jpg")
rel = load("stage_relight.jpg")
fin = load("stage_final.jpg")
ns = load("stage_final_noshadow.jpg")

A_COMP = np.asarray(comp).astype(np.float32)
A_REL = np.asarray(rel).astype(np.float32)
A_FIN = np.asarray(fin).astype(np.float32)
A_NS = np.asarray(ns).astype(np.float32)


# 主体左缘轮廓光（真实发光层，不是画上去的高光）
def build_rim():
    sub = Image.open(os.path.join(M, "subject_transparent.png")).convert("RGBA")
    big = Image.new("RGBA", (1536, 914), (0, 0, 0, 0))
    big.paste(sub, (700 - sub.width // 2, 872 - 700), sub)
    # 先柔化 alpha 再差分：抠像边缘的毛发级锯齿会变成抖动亮线
    al = Image.fromarray(np.asarray(big)[:, :, 3], "L").filter(ImageFilter.GaussianBlur(6))
    a = np.asarray(al).astype(np.float32)[:, :, None] / 255.0
    edge = np.clip(a - np.roll(a, 14, axis=1), 0, 1)
    edge = np.asarray(
        Image.fromarray((edge[:, :, 0] * 255).astype(np.uint8), "L")
        .filter(ImageFilter.GaussianBlur(5))
    ).astype(np.float32)[:, :, None] / 255.0
    edge = edge * 0.8
    warm = np.array([255.0, 227.0, 180.0], dtype=np.float32)[None, None, :]
    rgb = np.clip(edge * warm, 0, 255).repeat(3, axis=2)[:, :, :3]
    rgba = np.concatenate([rgb, (edge * 255)], axis=2).astype(np.uint8)
    return np.asarray(cover(Image.fromarray(rgba, "RGBA"))).astype(np.float32)


RIM = build_rim()

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
XXn = XX / W
YYn = YY / H

f_mono_9 = font(F_MONO, int(9 * SC))
f_mono_11 = font(F_MONO, int(11 * SC))
f_mono_7 = font(F_MONO, int(23 * SC / 2))


def rounded(draw, box, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def put(draw, xy, text, f, fill):
    draw.text(xy, text, font=f, fill=fill)


# ---------------------------------------------------------------- RELIGHT
LIGHT = (194 * SC, 162 * SC)                      # 光源中心
RAY_A = (198 * SC, 184 * SC)
RAY_B = (494 * SC, 420 * SC)
CAP = (40 * SC, 566 * SC)
TAG = (40 * SC, 610 * SC)


def relight_frame(t, ui=True):
    base = A_COMP * (1 - ease((t - 0.16) / 0.60)) + A_REL * ease((t - 0.16) / 0.60)

    # 光还没进来：整体压暗（更明确的一拍）
    d = 0.55 * (1 - ease(t / 0.34))
    base = base * (1 - d) + np.array(C_INK, dtype=np.float32) * d

    # 光从左上扩散
    sweep = clamp01((t - 0.10) / 0.55)
    if sweep > 0:
        dd = np.sqrt(((XX - LIGHT[0]) / W) ** 2 + ((YY - LIGHT[1]) / H) ** 2)
        mask = np.clip(1 - dd / (0.10 + 0.80 * sweep), 0, 1) ** 1.4
        warm = np.array([255.0, 214.0, 150.0], dtype=np.float32)[None, None, :]
        base = base + warm * mask[:, :, None] * (0.26 * sweep)

    # 主体轮廓光
    rk = ease((t - 0.40) / 0.42)
    if rk > 0:
        base = base + RIM[:, :, :3] * (RIM[:, :, 3:4] / 255.0) * rk

    base = np.clip(base, 0, 255)
    rng = np.random.default_rng(int(t * 9973))
    base = np.clip(base + rng.normal(0, 2.0, base.shape), 0, 255)
    img = Image.fromarray(base.astype(np.uint8), "RGB")

    if not ui:
        return img

    dr = ImageDraw.Draw(img, "RGBA")

    # 光源：环 + 芯，t=0.10 起放大出现
    g = ease((t - 0.10) / 0.22)
    if g > 0:
        r1 = 44 * SC * (0.55 + 0.45 * g)
        r2 = 20 * SC * (0.55 + 0.45 * g)
        al = int(255 * g)
        dr.ellipse([LIGHT[0] - r1, LIGHT[1] - r1, LIGHT[0] + r1, LIGHT[1] + r1],
                   outline=(247, 220, 168, al), width=max(1, int(1.2 * SC)))
        dr.ellipse([LIGHT[0] - r2, LIGHT[1] - r2, LIGHT[0] + r2, LIGHT[1] + r2],
                   fill=(255, 236, 194, al))

    # 方向虚线：长度随滚动生长
    lg = ease((t - 0.16) / 0.40)
    if lg > 0:
        ex = RAY_A[0] + (RAY_B[0] - RAY_A[0]) * lg
        ey = RAY_A[1] + (RAY_B[1] - RAY_A[1]) * lg
        L = math.hypot(ex - RAY_A[0], ey - RAY_A[1])
        n = max(1, int(L / (11 * SC)))
        for i in range(n):
            if i % 2:
                continue
            u0, u1 = i / n, min((i + 0.6) / n, 1)
            dr.line([RAY_A[0] + (ex - RAY_A[0]) * u0, RAY_A[1] + (ey - RAY_A[1]) * u0,
                     RAY_A[0] + (ex - RAY_A[0]) * u1, RAY_A[1] + (ey - RAY_A[1]) * u1],
                    fill=(247, 220, 168, 210), width=max(1, int(1.2 * SC)))
        if lg > 0.9:
            dr.ellipse([ex - 3.5 * SC, ey - 3.5 * SC, ex + 3.5 * SC, ey + 3.5 * SC],
                       fill=(247, 220, 168, 230))

    # 台词
    cg = ease((t - 0.55) / 0.24)
    if cg > 0:
        a = int(255 * cg)
        put(dr, CAP, "我把环境光重新分配到人物身上。", font(F_CJK, int(15 * SC)), C_WHITE + (a,))
        put(dr, TAG, "RELIGHT · 0.52 → 0.70 · 全页唯一视觉高潮", f_mono_9, (255, 246, 230, int(180 * cg)))
    return img


# ---------------------------------------------------------------- CRITIC
PN = [460 * SC, 150 * SC, 952 * SC, 472 * SC]      # 面板
ROWS = [("LIGHTING", 94, 0.10), ("SHADOW", 78, 0.20), ("COLOR", 95, 0.30), ("EDGE", 97, 0.40)]


def critic_frame(t, ui=True):
    # 画面后退 / 重跑
    if t < 0.58:
        k = 0.0
    elif t < 0.70:
        k = ease((t - 0.58) / 0.12)
    elif t < 0.74:
        k = 1.0
    elif t < 0.90:
        k = 1 - ease((t - 0.74) / 0.16)
    else:
        k = 0.0
    base = A_FIN * (1 - k) + A_NS * k
    # 贴合失败阶段：降饱和，让"浮起来"一眼可读
    if k > 0.01:
        g = base.mean(axis=2, keepdims=True)
        base = base * (1 - 0.20 * k) + g * (0.20 * k)

    dim = 0.38 * ease(t / 0.10)
    base = base * (1 - dim) + np.array(C_INK, dtype=np.float32) * dim

    # 重跑：竖直刷新扫描
    if 0.72 < t < 0.92:
        sp = (t - 0.72) / 0.20
        band = np.exp(-((YYn - sp) ** 2) / (2 * 0.035 ** 2))
        base = base + band[:, :, None] * np.array([120, 150, 165], dtype=np.float32) * 0.35
    base = np.clip(base, 0, 255)
    rng = np.random.default_rng(int(t * 7717))
    base = np.clip(base + rng.normal(0, 2.2, base.shape), 0, 255)
    img = Image.fromarray(base.astype(np.uint8), "RGB")

    if not ui:
        return img

    dr = ImageDraw.Draw(img, "RGBA")
    pa = int(217 * ease(t / 0.12))
    if pa <= 2:
        return img
    rounded(dr, PN, int(16 * SC), fill=(18, 22, 24, pa), outline=(58, 65, 69, pa),
            width=max(1, int(1 * SC)))
    inner = PN[0] + 26 * SC
    put(dr, (inner, PN[1] + 24 * SC), "AGENT SELF-CHECK", f_mono_9, (124, 135, 139, pa))
    hw = dr.textlength("THRESHOLD 80", font=f_mono_9)
    put(dr, (PN[2] - 26 * SC - hw, PN[1] + 24 * SC), "THRESHOLD 80", f_mono_9, (124, 135, 139, pa))

    y0 = PN[1] + 74 * SC
    for i, (name, val, ts) in enumerate(ROWS):
        a = ease((t - ts) / 0.08)
        if a <= 0.01:
            continue
        y = y0 + i * 52 * SC / 2 * 2
        y = y0 + i * 52 * SC
        failed = (name == "SHADOW") and t < 0.90
        col = C_WARN if failed else C_TXT
        # 修正阶段 SHADOW 从 78 数到 93
        show = val
        if name == "SHADOW":
            if 0.86 <= t:
                show = int(78 + (93 - 78) * ease((t - 0.86) / 0.10))
            if t >= 0.90:
                col = C_TXT
        put(dr, (inner, y), name, font(F_MONO, int(11.5 * SC)), col + (int(255 * a),))
        sv = str(show)
        sw = dr.textlength(sv, font=font(F_MONO, int(11.5 * SC)))
        put(dr, (PN[2] - 26 * SC - sw, y), sv, font(F_MONO, int(11.5 * SC)), col + (int(255 * a),))

    # 贴合失败：脚部标注（让"浮起来"这件事被看见）
    if 0.04 < k < 0.99:
        a = int(235 * min(1, k * 2.4))
        bx0, by0, bx1, by1 = 792, 1080, 980, 1224
        dr.rectangle([bx0, by0, bx1, by1], outline=(227, 178, 122, a),
                     width=max(1, int(1.4 * SC)))
        put(dr, (bx0, by0 - 26), "SHADOW MISSING", f_mono_9, (227, 178, 122, a))

    # 判定 / 回退条
    if t >= 0.50:
        a = ease((t - 0.50) / 0.10)
        chip = [inner, PN[1] + 196 * SC, PN[2] - 26 * SC, PN[1] + 196 * SC + 34 * SC]
        rounded(dr, chip, int(34 * SC / 2), fill=(46, 54, 57, int(230 * a)))
        if t < 0.90:
            txt, col = "SHADOW 78 < 80　→　reroll T04 回到 GROUND 重跑", C_WARN
        else:
            txt, col = "SHADOW 93 ≥ 80　→　PASS　·　OVERALL 95 / 100", C_OK
        put(dr, (inner + 14 * SC, chip[1] + 8 * SC), txt, f_mono_7, col + (int(255 * a),))
    return img


# ---------------------------------------------------------------- 输出
def write(path, frames):
    import imageio
    w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=None,
                           macro_block_size=1,
                           ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p",
                                          "-movflags", "+faststart", "-g", str(FPS // 4)])
    for f in frames:
        w.append_data(np.asarray(f))
    w.close()
    print(path, os.path.getsize(path) // 1024, "KB")


REL_N = int(3.4 * FPS)
CRI_N = int(5.6 * FPS)
print("relight frames", REL_N, "critic frames", CRI_N)

for tag, fn, n in (("relight", relight_frame, REL_N), ("critic_reroll", critic_frame, CRI_N)):
    for ui in (True, False):
        name = tag + ("" if ui else "_clean") + ".mp4"
        write(os.path.join(OUT, name), [fn(i / (n - 1), ui) for i in range(n)])
print("done")
