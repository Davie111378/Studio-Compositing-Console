# -*- coding: utf-8 -*-
"""
region_fx.py — C: mask 驱动的区域特效 (聚光灯 / 局部滤镜 / 区域调色 / 背景虚化)

与 fx_library.local_effect(矩形 box) 的区别: 本模块接受**任意形状 mask**
(来自 COCO segmentation / BiRefNet alpha / 色度键 / 手绘区域), 支持:
  - spotlight_on   : 主体打亮 + 周围压暗 (聚光灯)
  - spotlight_off  : 主体压暗 + 周围保持 (隐私/去焦点)
  - region_filter  : 仅主体(或仅背景)套用滤镜
  - region_color   : 仅区域内调色/调温
  - bg_blur        : 背景虚化 (主体锐利, 景深感)
  - region_glow    : 区域内发光/辉光
  - edge_highlight : mask 边缘描边发光 (抠图质检可视化 / 风格化)

用法:
  from region_fx import apply_region_fx
  out, meta = apply_region_fx(img_rgb, mask_u8, "spotlight_on", {"intensity": 0.7})

CLI:
  python region_fx.py --image in.png --mask m.png --effect spotlight_on --out out.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "ai-service" / "src" / "fx") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai-service" / "src" / "fx"))

import cv2  # noqa: E402


# ---------------------------------------------------------------- 工具
def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _feather(mask_u8: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """mask → 羽化后的 float32 (0-1)。sigma<=0 时按尺寸自适应。"""
    m = mask_u8.astype(np.float32) / 255.0
    if sigma <= 0:
        sigma = max(2.0, min(mask_u8.shape) * 0.012)
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0, 1)


def _fit_color(img: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
    return _u8(img.astype(np.float32) * (1 - strength) + target.astype(np.float32) * strength)


# ---------------------------------------------------------------- 单特效
def fx_spotlight_on(img, m, p):
    """主体打亮 + 周围压暗 (经典聚光灯)。"""
    inten = float(p.get("intensity", 0.65))
    warm = float(p.get("warm", 0.0))
    lit = img.astype(np.float32) * (1 + 0.45 * inten)
    if warm > 0:
        lit[..., 0] *= 1 + 0.12 * warm
        lit[..., 2] *= 1 - 0.10 * warm
    dark = img.astype(np.float32) * (1 - 0.55 * inten)
    # 背景轻微降饱和
    if p.get("desat_bg", True):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
        dark = dark * (1 - 0.35 * inten) + g * (0.35 * inten)
    a = m[..., None]
    return _u8(lit * a + dark * (1 - a))


def fx_spotlight_off(img, m, p):
    """主体压暗/模糊 (隐私保护), 周围不变。"""
    inten = float(p.get("intensity", 0.8))
    mode = p.get("mode", "darken")
    base = img.astype(np.float32)
    if mode == "blur":
        r = max(3, int(min(img.shape[:2]) * 0.03 * inten))
        eff = cv2.GaussianBlur(base, (0, 0), r)
    elif mode == "mosaic":
        h, w = img.shape[:2]
        cell = max(6, int(min(h, w) * 0.035))
        small = cv2.resize(base, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_LINEAR)
        eff = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    else:
        eff = base * (1 - 0.72 * inten)
    a = m[..., None]
    return _u8(eff * a + base * (1 - a))


def fx_region_filter(img, m, p):
    """仅主体(或仅背景)套用滤镜。filter: 滤镜 key 或中文。invert=True 表示作用于背景。"""
    import fx_library as F
    key = p.get("filter") or p.get("name") or "teal_orange"
    key = _resolve_filter(key)
    fn = F.FILTERS.get(key)
    if fn is None:
        fn = F.STYLES.get(key)
    if fn is None:
        fn = F.STYLES.get({"oil_painting": "oil_painting", "cartoon": "cartoon_comic"}.get(key, key))
    if fn is None:
        raise ValueError(f"未知滤镜 {key}; 可用: {sorted(F.FILTERS)[:12]}...")
    try:
        eff = fn(img.astype(np.float32))
    except TypeError:
        eff = fn(img.astype(np.float32), {})
    eff = np.asarray(eff, np.float32)
    if p.get("blend", 1.0) < 1.0:
        eff = img.astype(np.float32) * (1 - p["blend"]) + eff * p["blend"]
    a = m[..., None]
    if p.get("invert"):
        a = 1 - a
    return _u8(eff * a + img.astype(np.float32) * (1 - a))


def fx_region_color(img, m, p):
    """仅区域内调色: warm>0 暖 / cool>0 冷; brightness/saturation 可选。"""
    base = img.astype(np.float32)
    warm = float(p.get("warm", 0.0))
    cool = float(p.get("cool", 0.0))
    eff = base.copy()
    if warm > 0:
        eff[..., 0] *= 1 + 0.16 * warm
        eff[..., 1] *= 1 + 0.04 * warm
        eff[..., 2] *= 1 - 0.14 * warm
    if cool > 0:
        eff[..., 2] *= 1 + 0.18 * cool
        eff[..., 0] *= 1 - 0.14 * cool
    b = float(p.get("brightness", 0.0))
    if b:
        eff = eff + b * 60
    sat = float(p.get("saturation", 0.0))
    if sat:
        g = cv2.cvtColor(_u8(base), cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
        eff = eff * (1 + sat) - g * sat
    a = m[..., None]
    if p.get("invert"):
        a = 1 - a
    return _u8(eff * a + base * (1 - a))


def fx_bg_blur(img, m, p):
    """背景虚化: 主体锐利, 背景高斯模糊 (景深感)。"""
    inten = float(p.get("intensity", 0.7))
    r = max(3, int(min(img.shape[:2]) * 0.02 * (0.5 + inten)))
    bg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), r)
    if p.get("darken_bg"):
        bg *= (1 - 0.18 * inten)
    a = m[..., None]
    return _u8(img.astype(np.float32) * a + bg * (1 - a))


def fx_region_glow(img, m, p):
    """区域内辉光: 亮部扩散叠加。"""
    inten = float(p.get("intensity", 0.7))
    r = max(4, int(min(img.shape[:2]) * 0.05))
    bright = np.clip(img.astype(np.float32) - 110, 0, None)
    glow = cv2.GaussianBlur(bright, (0, 0), r)
    tint = p.get("tint", [1.0, 0.95, 0.8])
    glow = glow * np.array(tint, np.float32)
    a = m[..., None]
    out = img.astype(np.float32) + glow * (1.1 * inten)
    return _u8(out * a + img.astype(np.float32) * (1 - a))


def fx_edge_highlight(img, m, p):
    """mask 边缘描边发光 (抠图质检 / 风格化)。"""
    inten = float(p.get("intensity", 0.8))
    color = p.get("color", [80, 210, 255])
    band = max(1, int(p.get("width", 3)))
    k = np.ones((band * 2 + 1, band * 2 + 1), np.uint8)
    inner = cv2.erode((m * 255).astype(np.uint8), k)
    outer = cv2.dilate((m * 255).astype(np.uint8), k)
    edge = ((outer.astype(np.float32) - inner.astype(np.float32)) / 255.0)[..., None]
    out = img.astype(np.float32) * (1 - edge * inten) + \
        np.array(color, np.float32) * (edge * inten)
    return _u8(out)


EFFECTS = {
    "spotlight_on": fx_spotlight_on,
    "spotlight_off": fx_spotlight_off,
    "region_filter": fx_region_filter,
    "region_color": fx_region_color,
    "bg_blur": fx_bg_blur,
    "region_glow": fx_region_glow,
    "edge_highlight": fx_edge_highlight,
}

EFFECT_CN = {
    "spotlight_on": "主体聚光灯(打亮主体压暗背景)",
    "spotlight_off": "主体遮罩(压暗/模糊/马赛克主体)",
    "region_filter": "局部滤镜(仅主体或仅背景)",
    "region_color": "区域调色(暖/冷/明度/饱和)",
    "bg_blur": "背景虚化(景深感)",
    "region_glow": "主体辉光",
    "edge_highlight": "边缘描边发光",
}


# ---------------------------------------------------------------- 主入口
def apply_region_fx(img_rgb: np.ndarray, mask_u8: np.ndarray,
                    effect: str, params: dict | None = None):
    """img_rgb: HxWx3 uint8; mask_u8: HxW uint8 (0-255, 非零=区域)。
    返回 (out_rgb, meta)。mask 自动羽化。"""
    p = dict(params or {})
    if effect not in EFFECTS:
        raise ValueError(f"未知区域特效 {effect}; 可用: {list(EFFECTS)}")
    H, W = img_rgb.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = np.array(Image.fromarray(mask_u8).resize((W, H), Image.NEAREST))
    sigma = float(p.get("feather", 0))
    m = _feather(mask_u8, sigma)
    out = EFFECTS[effect](img_rgb, m, p)
    meta = {"effect": effect, "cn": EFFECT_CN[effect],
            "area_ratio": round(float((mask_u8 > 127).mean()), 4),
            "feather_sigma": round(sigma if sigma > 0 else max(2.0, min(H, W) * 0.012), 2),
            "params": {k: v for k, v in p.items() if k != "feather"}}
    return out, meta


def mask_from_coco_ann(ann_path: str, image_id: int, cat_name: str | None = None) -> np.ndarray:
    """从 COCO 标注构造某图某类的 mask (供测试/演示)。"""
    import json
    from PIL import ImageDraw
    d = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    im = next(i for i in d["images"] if i["id"] == image_id)
    W, H = im["width"], im["height"]
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    for a in d["annotations"]:
        if a["image_id"] != image_id or a.get("iscrowd"):
            continue
        if cat_name and cats[a["category_id"]] != cat_name:
            continue
        if not isinstance(a.get("segmentation"), list):
            continue
        for poly in a["segmentation"]:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 3:
                dr.polygon(pts, fill=255)
    return np.array(m, dtype=np.uint8)


def _resolve_filter(key: str) -> str:
    import fx_library as F
    if key in F.FILTERS or key in F.STYLES:
        return key
    cn = {"黑白": "bw_noir", "青橙": "teal_orange", "复古": "vintage_1970s",
          "赛博": "cyberpunk", "油画": "oil_painting", "漫画": "cartoon_comic",
          "水彩": "watercolor", "素描": "sketch_pencil", "热成像": "thermal",
          "双色调": "duotone", "暗黑": "moody", "红外": "bw_ir"}
    for k, v in cn.items():
        if k in key:
            return v
    return key


# ---------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True, help="mask 图或 'coco:image_id[:cat]'")
    ap.add_argument("--effect", default="spotlight_on", choices=list(EFFECTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="{}", help="JSON 参数, 如 {\"intensity\":0.8}")
    a = ap.parse_args()
    import json
    img = np.array(Image.open(a.image).convert("RGB"))
    if a.mask.startswith("coco:"):
        rest = a.mask.split(":")[1:]
        iid = int(rest[0]); cat = rest[1] if len(rest) > 1 else None
        ann = str(ROOT / "data" / "coco_stuff" / "annotations" / "instances_val2017.json")
        m = mask_from_coco_ann(ann, iid, cat)
    else:
        m = np.array(Image.open(a.mask).convert("L"))
    out, meta = apply_region_fx(img, m, a.effect, json.loads(a.params))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(a.out)
    print(f"OK -> {a.out} | {meta}")


if __name__ == "__main__":
    _main()
