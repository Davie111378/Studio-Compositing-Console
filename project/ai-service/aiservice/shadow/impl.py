"""T05 shadow_generate —— 引擎分层实现（程序化接触阴影 + 合成）。

B 组引擎（bsrc/shadow/shadow_tool.py）：alpha 高斯模糊 + 背光偏移 + 背景压暗（darken），
    偏移方向/长度由 A 组 light_dir（azimuth/polar）换算；随后用 B 组 alpha_over 把
    重打光前景合成到含阴影背景上（fg + shadow + bg -> composited，契约 N7）。
A 组确定性引擎（离线兜底）：包围盒接触椭圆 + 高斯软化。
两种引擎输出契约不变：shadow_png（独立阴影层）+ composited_png（fg+shadow+bg）。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from aiservice import bsrc_loader as bl
from aiservice.common import clamp, ensure_out_dir, load_rgb, load_rgba, resolve_uri, save_png

logger = logging.getLogger("aiservice.shadow")


def _cover(bg: Image.Image, size: tuple[int, int]) -> Image.Image:
    """背景 cover 到前景画布尺寸（对齐策略；真实实现对齐属 B5 pipeline 职责）。"""
    w, h = size
    bw, bh = bg.size
    scale = max(w / bw, h / bh)
    nw, nh = int(bw * scale + 0.5), int(bh * scale + 0.5)
    bg = bg.resize((nw, nh))
    left = (nw - w) // 2
    top = int((nh - h) * 0.35)
    return bg.crop((left, top, left + w, top + h))


# ---- B 组引擎 ----

def _bsrc_run(fg: Image.Image, bg: Image.Image, mask: Image.Image, ld: dict,
              strength: float, quality: str) -> tuple[Image.Image, Image.Image, dict] | None:
    try:
        generate_shadow, apply_shadow_to_bg = bl.shadow_funcs()
        alpha_over, _feather = bl.composite_funcs()
    except Exception as e:
        logger.warning("B 组阴影引擎不可用（%s）；走本地引擎", e)
        return None

    w, h = fg.size
    bg_cover = np.array(_cover(bg, (w, h)))
    alpha01 = np.array(fg.split()[-1].resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
    if alpha01.shape != (h, w):  # 尺寸异常防御
        return None

    az = math.radians(float(ld.get("azimuth", 0.0)))
    polar = clamp(float(ld.get("polar", 45.0)), 0.0, 90.0)

    ys, xs = np.where(alpha01 > 0.3)
    fh = float(ys.max() - ys.min()) if len(ys) else float(h)
    # 影子投向光源反方向；顶光（polar->90）影子变短、收拢到脚下
    mag = fh * (0.12 + 0.28 * (1.0 - polar / 90.0))
    offset = (int(round(-math.cos(az) * mag)), int(round(math.sin(az) * mag)))
    blur = int(max(6, fh * 0.05) * (1.6 if quality == "fine" else 1.0))
    opacity = clamp(0.15 + 0.6 * strength, 0.05, 0.75)

    shadow = generate_shadow(alpha01, blur_radius=blur, opacity=opacity, offset=offset)
    bg_with_shadow = apply_shadow_to_bg(bg_cover, shadow)

    fg_rgb = np.array(fg.convert("RGB"))
    composited = alpha_over(fg_rgb, alpha01, bg_with_shadow)
    shadow_layer = Image.fromarray((shadow * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    meta = {"offset": offset, "blur_radius": blur, "opacity": round(opacity, 3),
            "method": "bsrc-procedural"}
    return shadow_layer, Image.fromarray(composited), meta


# ---- A 组确定性引擎（离线兜底）----

def _legacy_run(fg: Image.Image, bg: Image.Image, mask: Image.Image, ld: dict,
                strength: float, quality: str) -> tuple[Image.Image, Image.Image]:
    w, h = fg.size
    bgc = _cover(bg, (w, h))
    alpha_bbox = mask.getbbox()
    canvas = bgc.convert("RGBA")
    az = math.radians(float(ld.get("azimuth", 0)))
    polar = float(ld.get("polar", 45))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if alpha_bbox and strength > 0.01:
        x0, y0, x1, y1 = alpha_bbox
        fw, fh = x1 - x0, y1 - y0
        ground_y = min(y1 + int(fh * 0.02), h - 2)
        # 光从 az 方向来 -> 影子向反方向偏移；极角越高影子越短
        shift = -math.cos(az) * fw * 0.45 * (1 - clamp(polar, 0, 90) / 100.0)
        squash = clamp(0.16 + polar / 90 * 0.14, 0.1, 0.4)
        sw_, sh_ = int(fw * (1.35 + 0.3 * (1 - clamp(polar, 0, 90) / 90.0))), max(6, int(fh * squash))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse(
            [x0 + fw / 2 + shift - sw_ / 2, ground_y - sh_ / 2, x0 + fw / 2 + shift + sw_ / 2, ground_y + sh_ / 2],
            fill=(10, 8, 6, int(200 * strength)),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(3, int(fh * 0.05)) * (2 if quality == "fine" else 1)))

    canvas = Image.alpha_composite(canvas, shadow)
    canvas = Image.alpha_composite(canvas, fg)
    shadow_flat = shadow.convert("RGB")
    return shadow_flat, canvas.convert("RGB")


def run(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    fg = load_rgba(inputs["foreground_png"], root)
    bg = load_rgb(inputs["background_png"], root)
    mask = Image.open(resolve_uri(inputs["mask_png"], root)).convert("L").resize(fg.size, Image.BILINEAR)
    ld = inputs["light_dir"]
    strength = clamp(float(inputs.get("shadow_strength", 0.6)), 0.0, 1.0)
    quality = options.get("quality", "normal")

    shadow_layer, composited, meta = None, None, None
    if bl.use_bsrc():
        try:
            shadow_layer, composited, meta = (
                _bsrc_run(fg, bg, mask, ld, strength, quality) or (None, None, None))
        except Exception as e:
            logger.warning("B 组阴影引擎失败（%s）；走本地引擎", e)
            shadow_layer, composited, meta = None, None, None
    if composited is None:
        shadow_layer, composited = _legacy_run(fg, bg, mask, ld, strength, quality)

    out = {
        "shadow_png": save_png(shadow_layer, out_dir, root, "shadow.png"),
        "composited_png": save_png(composited, out_dir, root, "composited.png"),
    }
    if meta:
        out["engine"] = "bsrc"
        out["shadow_meta"] = meta
    return out
