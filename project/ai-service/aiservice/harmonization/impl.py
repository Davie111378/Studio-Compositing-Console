"""T06 harmonize + T07 enhance —— 引擎分层实现。

T06（B 组引擎，bsrc/harmonization/harmonize_tool.py harmonize_v2）：
    Lab 色度对齐（限幅）+ 低频亮度迁移 + 高频细节保留 + 肤色保护 + FDR 防压黑。
    A 组契约输入是 composite_png + mask_png：背景统计先用 cv2.inpaint 抠掉前景再估计，
    结果按 mask 软边界贴回，背景像素保持不动。
T07（B 组引擎）：深度感知景深虚化（近清晰/远虚化，修复旧引擎的远近反置）+ 细节增强，
    并接入 B 组特效链 fx（spotlight/bokeh/fog/vignette/color_temp/depth_blur，支持区域选区），
    经 inputs.effects 传入（T07.json 已补充该可选字段）。
无 cv2 或 IMC_ENGINE=mock 时回落 A 组确定性引擎，契约不变。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from aiservice import bsrc_loader as bl
from aiservice.common import clamp, ensure_out_dir, fail, load_rgb, resolve_uri, save_png

logger = logging.getLogger("aiservice.harmonization")


# ---- T06 harmonize ----

def _stats(img: Image.Image, mask: Image.Image | None):
    return ImageStat.Stat(img, mask)


def _legacy_harmonize(comp: Image.Image, mask: Image.Image, strength: float) -> Image.Image:
    """A 组确定性引擎：mask 区域内均值/方差颜色迁移（Reinhard 思路）。"""
    fg_region = Image.composite(comp, Image.new("RGB", comp.size, (0, 0, 0)), mask)
    bg_region = Image.composite(Image.new("RGB", comp.size, (0, 0, 0)), comp, mask)
    fs = _stats(fg_region, mask)
    bs = _stats(bg_region, Image.eval(mask, lambda v: 255 - v))

    out = comp.copy()
    op = out.load()
    mp = mask.load()
    w, h = comp.size
    for y in range(h):
        for x in range(w):
            m = mp[x, y]
            if m < 8:
                continue
            r, g, b = op[x, y]
            w_t = (m / 255.0) * strength
            nr = (r - fs.mean[0]) * (bs.stddev[0] / max(fs.stddev[0], 1)) + bs.mean[0]
            ng = (g - fs.mean[1]) * (bs.stddev[1] / max(fs.stddev[1], 1)) + bs.mean[1]
            nb = (b - fs.mean[2]) * (bs.stddev[2] / max(fs.stddev[2], 1)) + bs.mean[2]
            op[x, y] = (
                int(clamp(r + (nr - r) * w_t, 0, 255)),
                int(clamp(g + (ng - g) * w_t, 0, 255)),
                int(clamp(b + (nb - b) * w_t, 0, 255)),
            )
    return out


def _bsrc_harmonize(comp: Image.Image, mask: Image.Image, strength: float):
    """B 组 v2 和谐化：inpaint 出纯净背景做统计，v2 变换后按软 mask 贴回。"""
    import cv2

    comp_rgb = np.array(comp)
    w, h = comp.size
    mask01 = np.array(mask.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0

    # 纯净背景估计：前景区域膨胀后 inpaint，避免前景像素污染背景统计
    hard = (mask01 > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    inpaint_mask = (cv2.dilate(hard, k) * 255).astype(np.uint8)
    bg_only = cv2.inpaint(comp_rgb, inpaint_mask, 5, cv2.INPAINT_TELEA)

    harmonize_v2 = bl.harmonize_v2()
    harm_rgb, info = harmonize_v2(comp_rgb, bg_only, mask01, strength=clamp(strength + 0.1, 0.0, 1.0))

    m = (mask01 * clamp(strength * 1.15, 0.0, 1.0))[..., None]
    out = comp_rgb.astype(np.float32) * (1.0 - m) + harm_rgb.astype(np.float32) * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), info


def run_harmonize(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    comp = load_rgb(inputs["composite_png"], root)
    mask = Image.open(resolve_uri(inputs["mask_png"], root)).convert("L").resize(comp.size)
    strength = clamp(float(inputs.get("strength", 0.7)), 0.0, 1.0)

    out, info = None, None
    if bl.use_bsrc():
        try:
            out, info = _bsrc_harmonize(comp, mask, strength)
        except Exception as e:
            logger.warning("B 组和谐化失败（%s）；走本地引擎", e)
            out = None
    if out is None:
        out = _legacy_harmonize(comp, mask, strength)

    result = {"harmonized_png": save_png(out, out_dir, root, "harmonized.png")}
    if info:
        result["engine"] = "bsrc"
        result["harmony_meta"] = info
    return result


# ---- T07 enhance ----

def _legacy_enhance(img: Image.Image, strength: float, depth_uri: str | None, root: Path) -> Image.Image:
    out = img
    if depth_uri:
        depth = Image.open(resolve_uri(depth_uri, root)).convert("L").resize(img.size)
        blurred = img.filter(ImageFilter.GaussianBlur(1.5 + 6 * strength))
        # 深度小(远) -> 更虚；用反相深度作为锐利区域 mask
        sharp_mask = Image.eval(depth, lambda v: 255 - v).point(lambda v: int(v * strength))
        out = Image.composite(img, blurred, sharp_mask)
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(40 + 120 * strength), threshold=3))


def _bsrc_enhance(img: Image.Image, strength: float, depth_uri: str | None, root: Path):
    """B 组引擎：深度感知虚化（近清晰/远虚化）+ 细节增强。返回 (PIL 图, 深度是否生效)。"""
    import cv2

    rgb = np.array(img)
    depth_used = False
    if depth_uri:
        depth = Image.open(resolve_uri(depth_uri, root)).convert("L").resize(img.size)
        depth01 = np.array(depth).astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(rgb, (0, 0), 1.5 + 6 * strength)
        m = (depth01 * clamp(strength * 1.3, 0.0, 1.0))[..., None]  # 深度高(近)保持清晰
        rgb = blurred.astype(np.float32) * (1.0 - m) + rgb.astype(np.float32) * m
        depth_used = True
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(40 + 120 * strength), threshold=3)), depth_used


def _apply_fx_chain(base: Image.Image, effects: list, out_dir: Path, root: Path):
    """B 组特效链：[{"effect","params"?,"region"?}, ...]，返回 (图, 已应用清单)。"""
    if not bl.use_bsrc():
        raise fail("E_INVALID_INPUT", "特效引擎不可用（需要 opencv-python）")
    try:
        fx_cls, valid = bl.fx_tool()
    except Exception as e:
        raise fail("E_INVALID_INPUT", f"特效引擎不可用: {e}") from e
    fx = fx_cls()
    cur = out_dir / "fx_00_base.png"
    base.save(cur, format="PNG")
    applied = []
    for i, e in enumerate(effects):
        if not isinstance(e, dict) or e.get("effect") not in valid:
            raise fail("E_INVALID_INPUT", f"未知特效: {e!r}")
        step = out_dir / f"fx_{i + 1:02d}_{e['effect']}.png"
        r = fx.apply(str(cur), str(step), e["effect"], e.get("params"), e.get("region"))
        if "error" in r:
            raise fail("E_ENHANCE_FAILED", f"特效 {e['effect']} 执行失败: {r['error'].get('message')}")
        applied.append(e["effect"])
        cur = step
    return Image.open(cur).convert("RGB"), applied


def run_enhance(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    img = load_rgb(inputs["image"], root)
    strength = clamp(float(inputs.get("strength", 0.5)), 0.0, 1.0)
    depth_uri = inputs.get("depth_map")
    effects = inputs.get("effects") or []

    out, engine, depth_used = None, "legacy", False
    if bl.use_bsrc():
        try:
            out, depth_used = _bsrc_enhance(img, strength, depth_uri, root)
            engine = "bsrc"
        except Exception as e:
            logger.warning("B 组增强失败（%s）；走本地引擎", e)
            out = None
    if out is None:
        out = _legacy_enhance(img, strength, depth_uri, root)
        depth_used = depth_uri is not None

    applied = []
    if effects:
        out, applied = _apply_fx_chain(out, effects, out_dir, root)
        engine = "bsrc"

    result = {"enhanced_png": save_png(out, out_dir, root, "enhanced.png"), "engine": engine}
    if depth_used:
        result["depth_applied"] = True
    if applied:
        result["effects_applied"] = applied
    return result
