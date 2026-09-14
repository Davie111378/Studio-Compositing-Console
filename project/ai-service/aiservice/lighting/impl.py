"""T03 lighting_estimate + T04 relight —— 引擎分层实现。

T03（B 组引擎）：Lambert 半球近似光照估计（bsrc/lighting/lighting_estimate.py），
    输出按 A 组契约换算为数值：light_dir.azimuth/polar（度）、color_temp（开尔文）、
    intensity（0~2，1=标准）、sh_coeff[9]（B 组最小二乘真投影）。
T04（B 组引擎）：方向性重打光（亮度渐变场 + 色温偏移 + 低频软光弥散），
    由 A 组的 azimuth/polar/开尔文换算为 B 的 (dx,dy) 方向与 warm/cool 标签，保留 RGBA alpha。
无 cv2 或 IMC_ENGINE=mock 时回落 A 组确定性引擎，契约不变。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from aiservice import bsrc_loader as bl
from aiservice.common import clamp, ensure_out_dir, load_rgb, load_rgba, resolve_uri, save_png

logger = logging.getLogger("aiservice.lighting")


# ---- T03 lighting_estimate ----

def _luma(rgb: tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _legacy_estimate(img: Image.Image) -> dict:
    """A 组确定性引擎：亮度质心 -> 方向；暖冷色相 -> 色温。"""
    w, h = img.size
    small = img.resize((min(64, w), min(64, h)))
    sw, sh = small.size
    px = small.load()

    total = 0.0
    sx = 0.0
    sy = 0.0
    sr = sg = sb = 0.0
    for y in range(sh):
        for x in range(sw):
            r, g, b = px[x, y]
            l = _luma((r, g, b)) + 1.0
            total += l
            sx += l * (x / (sw - 1))
            sy += l * (y / (sh - 1))
            sr += r
            sg += g
            sb += b
    cx = sx / total
    cy = sy / total
    mr, mg, mb = sr / (sw * sh), sg / (sw * sh), sb / (sw * sh)

    # 方位角：0=从右 90=从上 180=从左 270=从下（图像平面约定，写进 docs/api-spec）
    dx, dy = cx - 0.5, cy - 0.5
    azimuth = (math.degrees(math.atan2(-dy, dx))) % 360.0
    elevation = clamp(1.0 - math.hypot(dx, dy) * 2, 0.0, 1.0)  # 质心越靠边->越低
    polar = 15 + elevation * 60

    # 色温：暖(R>B) -> 低 K
    warm = (mr - mb) / 255.0
    color_temp = round(clamp(5800 - warm * 3200, 2400, 8000), 1)
    intensity = round(clamp((mr + mg + mb) / (3 * 255.0) * 1.6, 0.2, 1.8), 3)

    # SH9：环境项 + 一阶方向项（L0, L1x, L1y, L1z + 4 阶近似为 0）
    lum = (mr + mg + mb) / (3 * 255.0)
    kx = math.cos(math.radians(azimuth)) * intensity
    ky = -math.sin(math.radians(azimuth)) * intensity
    sh = [
        round(0.28 * lum, 4), round(0.28 * lum, 4), round(0.30 * lum, 4),
        round(0.12 * kx, 4), round(0.12 * ky, 4), round(0.16 * elevation * intensity, 4),
        0.0, 0.0, round(0.05 * lum, 4),
    ]
    return {
        "sh_coeff": sh,
        "light_dir": {"azimuth": round(azimuth, 1), "polar": round(polar, 1)},
        "color_temp": color_temp,
        "intensity": intensity,
    }


def _bsrc_estimate(bg_path: Path, quality: str) -> dict | None:
    """B 组 Lambert 半球估计 -> A 组数值契约。失败返回 None。"""
    try:
        estimate = bl.lighting_estimate()
        est = estimate(str(bg_path), quality)
    except Exception as e:
        logger.warning("B 组光照估计失败（%s）；走本地引擎", e)
        return None

    dx, dy = est["light_dir"]
    azimuth = math.degrees(math.atan2(-dy, dx)) % 360.0
    theta_deg, phi_deg = est.get("theta_phi", [0.0, 45.0])
    polar = clamp(float(phi_deg), 0.0, 90.0)

    return {
        "azimuth": round(azimuth, 1),
        "polar": round(polar, 1),
        "theta_phi": [round(float(theta_deg), 1), round(float(phi_deg), 1)],
        "temp_label": est.get("color_temp"),
        "contrast": est.get("intensity"),
        "sh_coeff": est["sh_coeff"],
        "method": est.get("method", "lambert-hemisphere-approx"),
    }


def run_estimate(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    quality = options.get("quality", "normal")
    bg_path = resolve_uri(inputs["bg_png"], root)

    bsrc = _bsrc_estimate(bg_path, quality) if bl.use_bsrc() else None
    if bsrc is not None:
        img = load_rgb(inputs["bg_png"], root)
        arr = np.asarray(img.resize((64, 64)), dtype=np.float32)
        r_mean, b_mean = float(arr[..., 0].mean()), float(arr[..., 2].mean())
        luma = float((0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean())
        out = _legacy_estimate(img)  # 复用其 intensity/色温口径
        return {
            "sh_coeff": bsrc["sh_coeff"],
            "light_dir": {"azimuth": bsrc["azimuth"], "polar": bsrc["polar"]},
            "color_temp": bl.kelvin_from_channels(r_mean, b_mean),
            "intensity": out["intensity"],
            "engine": "bsrc",
            "method": bsrc["method"],
            "theta_phi": bsrc["theta_phi"],
            "temp_label": bsrc["temp_label"],
            "contrast": bsrc["contrast"],
        }

    img = load_rgb(inputs["bg_png"], root)
    return _legacy_estimate(img)


# ---- T04 relight ----

def _tint_for_temp(color_temp: float) -> tuple[float, float, float]:
    """色温 -> RGB 增益（暖低K偏红，冷高K偏蓝）。"""
    t = clamp((color_temp - 3500) / 4000.0, -1.0, 1.0)  # -1 暖 ~ 1 冷
    return (1.0 - 0.18 * t), (1.0 - abs(t) * 0.04), (1.0 + 0.20 * t)


def _legacy_relight(fg: Image.Image, ld: dict, color_temp: float, intensity: float,
                    quality: str) -> Image.Image:
    """A 组确定性引擎：按光向生成明暗渐变 + 色温染色。"""
    w, h = fg.size
    az = math.radians(float(ld.get("azimuth", 0)))

    # 光方向单位向量（图像平面：右为 +x，下为 +y）
    lx = math.cos(az)
    ly = -math.sin(az)
    tr, tg, tb = _tint_for_temp(color_temp)

    src = fg.copy()
    sp = src.load()
    cx, cy = w / 2, h / 2
    radius = math.hypot(cx, cy)
    # strength 随距离衰减，形成方向明暗
    base = 0.86 + 0.14 * clamp(intensity, 0.5, 1.5)
    amp = 0.24 * clamp(intensity, 0.3, 1.8)
    if quality == "draft":
        amp *= 0.8
    for y in range(h):
        for x in range(w):
            r, g, b, a = sp[x, y]
            if a == 0:
                continue
            ux, uy = (x - cx) / radius, (y - cy) / radius
            d = ux * lx + uy * ly  # -1 背光 ~ 1 向光
            k = base + amp * d
            sp[x, y] = (
                int(clamp(r * k * tr, 0, 255)),
                int(clamp(g * k * tg, 0, 255)),
                int(clamp(b * k * tb, 0, 255)),
                a,
            )
    return src


def _bsrc_relight(fg: Image.Image, ld: dict, color_temp: float, intensity: float,
                  quality: str, bg_rgb: np.ndarray | None) -> Image.Image | None:
    """B 组方向性重打光（保留 alpha）。失败返回 None。"""
    try:
        tool_cls = bl.relight_tool()
    except Exception as e:
        logger.warning("B 组重打光不可用（%s）；走本地引擎", e)
        return None
    try:
        import cv2  # noqa: F401  引擎依赖自检
        rgb = np.array(fg.convert("RGB"))
        alpha_ch = np.array(fg.split()[-1])
        az = float(ld.get("azimuth", 0.0))
        polar = clamp(float(ld.get("polar", 45.0)), 0.0, 90.0)
        dx, dy = bl.az_polar_to_direction(az, polar)

        label, ct_strength = bl.kelvin_to_temp_hint(color_temp)
        gain = 0.30 * clamp(intensity, 0.5, 1.5) * (1.0 - 0.45 * polar / 90.0)
        if quality == "draft":
            gain *= 0.8
        hint = {
            "direction": [dx, dy],
            "color_temp": label or "neutral",
            # neutral -> 强度 0（B 分支安全）；暖冷按开尔文偏离 5500 的比例给强度
            "color_temp_strength": 0.0 if label == "neutral" else ct_strength,
        }
        bg_ref = bg_rgb if bg_rgb is not None else rgb
        tool = tool_cls(method="directional")
        out_rgb = tool.directional_relight(rgb, bg_ref, hint, gain=gain)
        rgba = Image.fromarray(out_rgb).convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha_ch))
        return rgba
    except Exception as e:
        logger.warning("B 组重打光失败（%s）；走本地引擎", e)
        return None


def run_relight(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    fg = load_rgba(inputs["rgba_png"], root)
    ld = inputs["light_dir"]
    color_temp = float(inputs.get("color_temp", 5500))
    intensity = clamp(float(inputs.get("intensity", 1.0)), 0.0, 2.0)
    quality = options.get("quality", "normal")

    bg_rgb = None
    bg_hint = inputs.get("bg_hint")
    if bg_hint:
        try:
            bg_img = load_rgb(bg_hint, root)
            if bg_img.size != fg.size:
                bg_img = bg_img.resize(fg.size, Image.BILINEAR)
            bg_rgb = np.array(bg_img)
        except Exception as e:
            logger.info("bg_hint 不可用（%s），重打光仅按 light_dir", e)

    out = None
    engine = "legacy"
    if bl.use_bsrc():
        out = _bsrc_relight(fg, ld, color_temp, intensity, quality, bg_rgb)
        if out is not None:
            engine = "bsrc"
    if out is None:
        out = _legacy_relight(fg, ld, color_temp, intensity, quality)
    return {"relit_png": save_png(out, out_dir, root, "relit.png"), "engine": engine}
