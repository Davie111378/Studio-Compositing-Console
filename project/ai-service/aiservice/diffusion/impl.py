"""T02 background_generate —— 千问文生图真实档 + 程序化渐变 mock 回落。

真实档：T2I_MODEL + DASHSCOPE_API_KEY 配置后走千问文生图（diffusion/t2i.py）；
任何失败（未配置/超时/安全拦截/网络）自动回落本文件的关键词渐变 mock。
契约不变：bg_png + depth_png（T03/T05/T07 依赖），失败抛 ToolFailure。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from aiservice.common import ensure_out_dir, save_png, seeded_random
from aiservice.diffusion import t2i

logger = logging.getLogger("aiservice.t02")

# 关键词 -> (上色, 下色, 光晕色)
_PALETTES: list[tuple[tuple[str, ...], tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]] = [
    (("咖啡", "咖啡馆", "cafe"), ((72, 48, 34), (140, 96, 62), (255, 196, 120))),
    (("傍晚", "黄昏", "夕阳", "dusk", "sunset"), ((64, 46, 84), (238, 140, 74), (255, 170, 90))),
    (("夜晚", "夜景", "夜色", "night"), ((8, 10, 30), (40, 52, 90), (120, 150, 220))),
    (("海边", "海滩", "沙滩", "beach", "sea"), ((30, 90, 140), (212, 190, 150), (255, 230, 180))),
    (("雪", "snow"), ((150, 170, 195), (235, 240, 248), (255, 255, 255))),
    (("森林", "树林", "草地", "forest", "grass"), ((20, 60, 34), (90, 140, 70), (200, 235, 160))),
    (("赛博", "霓虹", "cyber"), ((20, 8, 40), (90, 30, 110), (90, 220, 255))),
    (("太空", "星空", "space"), ((5, 5, 18), (30, 22, 70), (170, 150, 255))),
]
_DEFAULT = ((52, 66, 84), (150, 168, 186), (250, 250, 240))


def _pick(prompt: str):
    for keys, pal in _PALETTES:
        if any(k in prompt for k in keys):
            return pal
    return _DEFAULT


def _light_side(prompt: str) -> float:
    """光晕水平位置 0(左)~1(右)。"""
    if any(k in prompt for k in ("左", "left")):
        return 0.18
    if any(k in prompt for k in ("右", "right")):
        return 0.82
    return 0.5


def run(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    prompt = str(inputs.get("prompt", ""))
    size = inputs.get("size") or {}
    quality = options.get("quality", "normal")
    base = 512 if quality == "draft" else 1024
    w = int(size.get("width", base))
    h = int(size.get("height", base))

    # 真实档优先：千问文生图；失败回落程序化渐变（L3 降级，契约不变）
    if t2i.configured():
        try:
            return t2i.generate(prompt, w, h, out_dir, root)
        except Exception as e:  # noqa: BLE001 —— 任何失败都降级，不阻塞合成链
            logger.warning("千问文生图失败，回落程序化渐变 mock: %s", e)

    rng = seeded_random("bg", prompt, quality)

    top, bottom, glow = _pick(prompt)
    lx = _light_side(prompt)

    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    # 光晕
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    gr = int(min(w, h) * (0.55 if quality == "fine" else 0.4))
    gd.ellipse([w * lx - gr, h * 0.28 - gr, w * lx + gr, h * 0.28 + gr], fill=glow)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(min(w, h) // 6))
    img = Image.blend(img, Image.composite(glow_layer, img, glow_layer.convert("L").point(lambda v: v // 2)), 0.35)
    # 噪点纹理
    noise = Image.effect_noise((w, h), 18).convert("RGB")
    img = Image.blend(img, noise, 0.06)
    # 暗角
    vig = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse([-w // 3, -h // 3, w + w // 3, h + h // 3], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(min(w, h) // 5))
    dark = img.point(lambda v: int(v * 0.72))
    img = Image.composite(img, dark, vig)

    # 深度图：水平 + 径向渐变（近亮远暗的近似）
    depth = Image.new("L", (w, h))
    dp = depth.load()
    for y in range(h):
        for x in range(w):
            d = 1.0 - 0.55 * (y / max(1, h - 1)) - 0.2 * math.hypot(x / w - lx, y / h - 0.3)
            dp[x, y] = int(max(0.0, min(1.0, d)) * 255)
    depth = depth.filter(ImageFilter.GaussianBlur(8))

    return {
        "bg_png": save_png(img.convert("RGB"), out_dir, root, "bg.png"),
        "depth_png": save_png(depth, out_dir, root, "depth.png"),
    }
