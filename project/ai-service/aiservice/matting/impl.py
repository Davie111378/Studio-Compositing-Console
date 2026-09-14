"""T01 matting —— 引擎分层实现。

引擎链（见 aiservice/bsrc_loader.py，IMC_ENGINE 切换）：
    1. point/box/trimap 提示 + cv2 -> GrabCut 交互分割（以色彩先验做种子）
    2. B 组真实抠图链（BiRefNet -> carvekit -> SimplifiedBiRefNet，需 torch；
       权重经 BIREFNET_WEIGHT_PATH 注入）
    3. rembg 离线神经网络抠图（onnxruntime + u2net 系模型，REMBG_MODEL 可选型号）
    4. GrabCut 双候选自动分割（肤色锚点 + 天空先验 + 边界对齐度评分择优）
    5. A 组确定性色彩距离引擎（离线兜底，L1 降级基线）

输入输出契约不变（T01.json）：rgba_png / alpha_png / mask_png。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image, ImageFilter

from aiservice import bsrc_loader as bl
from aiservice.common import (
    clamp,
    ensure_out_dir,
    load_rgb,
    resolve_uri,
    save_png,
)

logger = logging.getLogger("aiservice.matting")

_EDGE_BAND = 40  # 色距 0..~441 映射到 alpha 的过渡带宽


def _bg_color(img: Image.Image) -> tuple[int, int, int]:
    w, h = img.size
    corners = [img.getpixel((x, y)) for x, y in ((2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3))]
    rs = sorted(c[0] for c in corners)
    gs = sorted(c[1] for c in corners)
    bs = sorted(c[2] for c in corners)
    return (rs[1] + rs[2]) // 2, (gs[1] + gs[2]) // 2, (bs[1] + bs[2]) // 2


def _color_distance_alpha(src: Image.Image, quality: str,
                          max_size: int | None = None) -> Image.Image:
    """确定性色彩距离 alpha（A 组 L1 兜底引擎）。max_size 限定时在缩略图上计算再放大
    （供 GrabCut 种子/占比判断用，省去全分辨率逐像素循环）。"""
    w, h = src.size
    calc = src
    if max_size and max(w, h) > max_size:
        s = max_size / max(w, h)
        calc = src.resize((max(32, int(w * s + 0.5)), max(32, int(h * s + 0.5))), Image.BILINEAR)
    cw, ch = calc.size
    bg = _bg_color(calc)
    alpha = Image.new("L", (cw, ch))
    px = calc.load()
    pa = alpha.load()
    br, bgc, bb = bg
    for y in range(ch):
        for x in range(cw):
            r, g, b = px[x, y]
            dist = ((r - br) ** 2 + (g - bgc) ** 2 + (b - bb) ** 2) ** 0.5
            pa[x, y] = int(clamp((dist - 30) / _EDGE_BAND, 0.0, 1.0) * 255)
    # 平滑 + 细化（fine 档多一轮边缘羽化，模拟高精度）
    alpha = alpha.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(1.2))
    if quality == "fine":
        alpha = alpha.filter(ImageFilter.GaussianBlur(0.6))
    if calc.size != (w, h):
        alpha = alpha.resize((w, h), Image.BILINEAR)
    return alpha


# ---- 引擎 1：GrabCut 分割（提示驱动 / 自动中心初始化）----

def _boundary_score(img_small, alpha_small) -> float:
    """候选 mask 质量代理：mask 边界落在图像强边缘上的平均强度。
    主体轮廓（发丝/肩线）对齐强边缘；误吞的背景块边界处于平滑区，得分低。"""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    hard = (alpha_small > 128).astype(np.uint8)
    contour = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    sel = contour > 0
    if not sel.any():
        return 0.0
    return float(grad[sel].mean())


def _skin_seeds(img_small) -> np.ndarray | None:
    """肤色像素 -> 确定前景种子（YCrCb 规则，与 B 组 harmonize 肤色保护同口径）。
    人物照片脸部/手部是可靠的主体锚点，可阻止 GrabCut 把色调相近的背景吞入前景。"""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    ycrcb = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)
    cb, cr = ycrcb[..., 1].astype(np.float32), ycrcb[..., 2].astype(np.float32)
    skin = ((cb > 77) & (cb < 135) & (cr > 133) & (cr < 180)).astype(np.uint8)
    if skin.mean() < 0.005:  # 几乎无肤色像素（非人物照）不做种子
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin = cv2.dilate(skin, k)
    return skin


def _grabcut_alpha(src: Image.Image, prior: Image.Image, box=None, point=None,
                   trimap: Image.Image | None = None,
                   center_init: bool = False) -> Image.Image | None:
    """GrabCut 精修 alpha。背景只用图像边框带定义（先验失真时会整幅污染，
    故不再从先验标定背景）；结果退化（前景占比异常）时返回 None。"""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    w, h = src.size
    scale = min(1.0, 512.0 / max(w, h))
    sw, sh = max(32, int(w * scale + 0.5)), max(32, int(h * scale + 0.5))
    img = np.array(src.resize((sw, sh), Image.BILINEAR))
    pr = np.array(prior.resize((sw, sh), Image.BILINEAR)).astype(np.uint8)

    gm = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    bw = max(2, int(min(sw, sh) * 0.03))
    gm[:bw, :] = cv2.GC_BGD
    gm[-bw:, :] = cv2.GC_BGD
    gm[:, :bw] = cv2.GC_BGD
    gm[:, -bw:] = cv2.GC_BGD
    if center_init or (box is None and point is None and trimap is None):
        # 自动模式：肤色锚点为确定前景（人物照的主体可靠证据）
        skin = _skin_seeds(img)
        if skin is not None:
            gm[skin > 0] = cv2.GC_FGD
        # 天空先验：与边框连通的高亮区（云/天）是强背景证据，阻止 GrabCut 吞入亮背景
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bright = (gray > 225).astype(np.uint8)
        ff = bright.copy()
        fmask = np.zeros((sh + 2, sw + 2), np.uint8)
        cv2.floodFill(ff, fmask, (0, 0), 2)
        border_bright = (ff == 2) & (bright > 0)
        if border_bright.mean() < 0.45:  # 防御：整图皆亮（雪地/白墙）时不做该先验
            gm[border_bright] = cv2.GC_PR_BGD
    if center_init:
        # 经典初始化：画面中央矩形为候选前景（主体照片的稳健假设）
        x1, x2 = int(sw * 0.18), int(sw * 0.82)
        y1, y2 = int(sh * 0.04), int(sh * 0.96)
        gm[y1:y2, x1:x2] = cv2.GC_PR_FGD
    elif trimap is not None:
        tm = np.array(trimap.resize((sw, sh), Image.BILINEAR))
        gm[(tm > 50) & (tm < 200)] = cv2.GC_PR_FGD
        gm[tm >= 200] = cv2.GC_PR_FGD
    elif box and len(box) == 4:
        x1, y1, x2, y2 = box
        x1 = int(clamp(x1 * scale, 0, sw - 2)); x2 = int(clamp(x2 * scale, x1 + 2, sw))
        y1 = int(clamp(y1 * scale, 0, sh - 2)); y2 = int(clamp(y2 * scale, y1 + 2, sh))
        gm[y1:y2, x1:x2] = cv2.GC_PR_FGD
    elif point:
        cx, cy = int(point.get("x", 0) * scale), int(point.get("y", 0) * scale)
        r = max(8, max(sw, sh) // 12)
        cv2.circle(gm, (cx, cy), r, cv2.GC_PR_FGD, -1)
    else:
        gm[pr > 200] = cv2.GC_PR_FGD  # 先验健康时作为前景种子

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)  # grabCut 的 GMM 初始化用全局 RNG，固定种子保证确定性
        cv2.grabCut(img, gm, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        logger.warning("grabcut failed: %s", e)
        return None

    a = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, k)
    # 去白边：先收缩 1px 再羽化，避免原图背景色残留进软边
    a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    fg_frac = float((a > 128).mean())
    if not (0.015 <= fg_frac <= 0.92):
        return None  # 分割退化，交回下一级引擎
    return Image.fromarray(a).resize((w, h), Image.BILINEAR)


# ---- 引擎 2a：B 组真实抠图链（torch BiRefNet）----

def _bsrc_real_alpha(image_path: Path, out_alpha: Path, out_fg: Path) -> tuple[Image.Image, str] | None:
    try:
        build = bl.matting_backend()
    except Exception as e:
        logger.info("B 组抠图链不可用（%s）；走本地引擎", type(e).__name__)
        return None
    weight = os.environ.get("BIREFNET_WEIGHT_PATH") or None
    try:
        tool, name = build(variant="general", device=None, weight_path=weight)
        tool.predict(str(image_path), str(out_alpha), str(out_fg))
        return Image.open(out_alpha).convert("L"), str(name)
    except Exception as e:
        logger.warning("B 组抠图链失败（%s）；走本地引擎", e)
        return None


# ---- 引擎 2b：rembg 离线神经网络抠图（onnxruntime，无需 torch）----

_rembg_cache: tuple[str, object] | None = None


def _rembg_alpha(image_path: Path) -> tuple[Image.Image, str] | None:
    global _rembg_cache
    try:
        from rembg import new_session, remove
    except Exception:
        return None
    model = os.environ.get("REMBG_MODEL", "u2net_human_seg")  # 人物合成场景默认人像分割模型
    try:
        if _rembg_cache is None or _rembg_cache[0] != model:
            _rembg_cache = (model, new_session(model))
        out = remove(Image.open(image_path).convert("RGB"), session=_rembg_cache[1])
        return out.split()[-1], f"rembg-{model}"
    except Exception as e:
        logger.warning("rembg 抠图失败（%s）；走本地引擎", e)
        return None


def run(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    src_path = resolve_uri(inputs["image"], root)
    src = load_rgb(inputs["image"], root)
    quality = options.get("quality", "normal")

    prior = _color_distance_alpha(src, quality, max_size=512)

    engine_used = "color-distance"
    alpha: Image.Image | None = None

    # 1) 提示模式：GrabCut 交互分割（cv2）
    box = inputs.get("box")
    point = inputs.get("point")
    trimap_uri = inputs.get("trimap")
    trimap_img = None
    if trimap_uri:
        trimap_img = Image.open(resolve_uri(trimap_uri, root)).convert("L")
    if box or point or trimap_img is not None:
        gc = _grabcut_alpha(src, prior, box=box, point=point, trimap=trimap_img)
        if gc is not None:
            alpha = gc
            engine_used = "grabcut-hinted"

    # 2) 自动模式（质量优先级从高到低，前一档可用即停）：
    #    B 组 torch BiRefNet（需权重）-> rembg 离线 NN -> GrabCut 双候选 -> 色彩先验
    if alpha is None and bl.use_bsrc():
        real = _bsrc_real_alpha(src_path, out_dir / "_b_real_alpha.png", out_dir / "_b_real_fg.png")
        if real is not None:
            alpha, engine_used = real

    if alpha is None:
        rb = _rembg_alpha(src_path)
        if rb is not None:
            alpha, engine_used = rb

    if alpha is None:
        try:
            import numpy as np
            prior_frac = float((np.array(prior) > 128).mean())
            degenerate = prior_frac > 0.60 or prior_frac < 0.02
            cand_priors = [False] if not degenerate else [False, True]
            candidates = []
            for ci in cand_priors:
                gc = _grabcut_alpha(src, prior, center_init=ci)
                if gc is not None:
                    candidates.append((gc, ci))
            if candidates:
                scale = min(1.0, 512.0 / max(src.size))
                sw, sh = max(32, int(src.size[0] * scale + 0.5)), max(32, int(src.size[1] * scale + 0.5))
                img_small = np.array(src.resize((sw, sh), Image.BILINEAR))
                scored = []
                for gc, _ci in candidates:
                    a_small = np.array(gc.resize((sw, sh), Image.BILINEAR))
                    area = float((a_small > 128).mean())
                    scored.append((_boundary_score(img_small, a_small), area, gc))
                top = max(s for s, _a, _g in scored)
                pool = [x for x in scored if x[0] >= top * 0.92]
                # 分数接近时取面积小者（blob 只增面积、不加边缘支持）
                alpha = min(pool, key=lambda x: x[1])[2]
                engine_used = "grabcut-auto"
        except Exception as e:
            logger.warning("grabcut-auto 失败: %s", e)

    # 3) 兜底：色彩距离引擎（全分辨率重算）
    if alpha is None:
        alpha = _color_distance_alpha(src, quality)

    mask_soft = alpha.point(lambda v: 255 if v > 128 else 0)
    rgba = src.convert("RGBA")
    rgba.putalpha(alpha)
    return {
        "rgba_png": save_png(rgba, out_dir, root, "rgba.png"),
        "alpha_png": save_png(alpha, out_dir, root, "alpha.png"),
        "mask_png": save_png(mask_soft, out_dir, root, "mask.png"),
        "engine": engine_used,
        "bsrc_rev": bl.upstream_rev() if engine_used.startswith(("birefnet", "carvekit", "Simplified")) else None,
    }
