# -*- coding: utf-8 -*-
"""
frame_engine.py — 把图片引擎逐帧接入视频 (视频管线 Step2)
每帧: 复用 studio_cli 的色度键/去溢出 + fx_library 特效/贴纸/水印/滤镜。

核心工厂:
  make_frame_fn(cfg) -> fn(frame_rgb, frame_idx) -> frame_rgb'
    cfg 见 video_pipeline 文档:
      {
        "green": {"bg": "path或(None整帧换)"},   # 视频换背景(键控)
        "filter": "teal_orange",                # 全图滤镜
        "fx": [{"effect":..,"params":..,"region":..}],  # 特效(静态,锚定)
        "watermark": {"text":"..","opacity":0.4},       # 水印
        "beautify": "natural",                  # 美颜
        "region_box": (x1,y1,x2,y2)             # 特效区域(圈选锁定, 首帧)
      }
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # -> ai-service
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "fx"))

import fx_library as FX   # noqa


def _lazy_studio():
    # studio_cli 顶部 import 副作用较大, 延迟到首次需要时
    import importlib
    sys.path.insert(0, str(ROOT.parent / "training" / "scripts"))  # chroma helpers
    from ai_material_eval import despill
    # 用 scene 键控 (绿幕任意位置) —— 从 studio_cli 借用则避免循环; 这里独立实现轻量色度键
    return despill


# 轻量场景色度键 (逐帧用, 不依赖 studio_cli import 副作用)
def chroma_key_scene(img: np.ndarray) -> np.ndarray:
    import cv2
    f = img.astype(np.float32)
    excess = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    thr = max(float(np.percentile(excess, 78)), 15.0)
    greenish = excess > thr
    if greenish.sum() < 200:
        return np.ones(img.shape[:2], np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[greenish], axis=0)
    d = np.linalg.norm(lab - ref, axis=-1)
    t1 = float(np.percentile(d[greenish], 88)) + 5.0
    t2 = t1 + 30.0
    alpha = np.clip((d - t1) / max(t2 - t1, 1.0), 0, 1)
    alpha = np.where(alpha < 0.1, 0.0, np.where(alpha > 0.85, 1.0, alpha))
    # 连通域清理
    from scipy.ndimage import label
    bg = alpha < 0.5
    lb, n = label(bg)
    if n > 1:
        sizes = np.bincount(lb.ravel()); sizes[0] = 0
        keep = sizes.argmax()
        alpha[(bg) & (lb != keep)] = np.maximum(alpha[(bg) & (lb != keep)], 0.9)
    return alpha


def despill_strong(img: np.ndarray, alpha: np.ndarray, cap=14) -> np.ndarray:
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    g2 = np.minimum(g, (r + b) / 2 + cap)
    return np.clip(np.stack([r, g2, b], -1), 0, 255).astype(np.uint8)


def fit_bg_cover(bg, tw, th):
    """背景等比 cover 裁到 tw x th。bg: HxWx3 uint8。返回 tw x th。"""
    import cv2
    h, w = bg.shape[:2]
    sc = max(tw / w, th / h)
    nw, nh = max(tw, int(round(w * sc))), max(th, int(round(h * sc)))
    res = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    x, y = (nw - tw) // 2, (nh - th) // 2
    return res[y:y + th, x:x + tw]


def make_frame_fn(cfg: dict, bg_cache: dict = None):
    """返回 fn(frame_rgb, idx)->frame_rgb'。bg_cache 复用背景预处理结果。"""
    bg_cache = bg_cache if bg_cache is not None else {}
    # 预解析背景(整图) -> cover 到首帧尺寸再缓存
    import cv2
    _bg = None
    _bg_key = None

    def _get_bg(H, W):
        nonlocal _bg, _bg_key
        g = cfg.get("green") or {}
        bgp = g.get("bg")
        key = (bgp, H, W)
        if _bg_key == key and _bg is not None:
            return _bg
        if bgp:
            bg = np.array(Image.open(bgp).convert("RGB"))
            _bg = fit_bg_cover(bg, W, H)
        else:
            # 无背景图: 统一色
            _bg = np.full((H, W, 3), g.get("color", (30, 80, 120)), np.uint8)
        _bg_key = key
        return _bg

    def fn(frame, idx):
        out = frame.copy()
        # ---- 1) 视频换背景 (绿幕键控) ----
        if cfg.get("green"):
            g = cfg["green"]
            if g.get("enabled", True):
                a = chroma_key_scene(out)
                fg = despill_strong(out, a)
                bg = _get_bg(out.shape[0], out.shape[1])
                a3 = a[..., None]
                out = np.clip(fg.astype(np.float32) * a3 + bg.astype(np.float32) * (1 - a3), 0, 255).astype(np.uint8)
        # ---- 2) 全图滤镜 ----
        filt = cfg.get("filter")
        if filt:
            out = FX.fx_auto.place(out, filt)[0]
        # ---- 3) 美颜 ----
        bf = cfg.get("beautify")
        if bf and bf in FX.BEAUTY:
            out = FX.BEAUTY[bf](out)
        # ---- 4) 静态特效/贴纸/水印 (region 若给则锚定, 每帧同框) ----
        fx_list = cfg.get("fx") or []
        for fx in fx_list:
            eff = fx["effect"]; p = fx.get("params") or {}
            region = fx.get("region")
            out = FX.fx_auto.place(out, eff, region, p)[0]
        # ---- 5) 水印 (可放最上层) ----
        wm = cfg.get("watermark")
        if wm:
            if wm.get("style") == "diag":
                out = FX._diag_watermark(out, wm.get("text", "WATERMARK"),
                                         int(out.shape[1] * wm.get("size", 0.06)),
                                         wm.get("opacity", 0.25), (255, 255, 255))
            else:
                # 单角落文字水印
                box = (0, 0, int(out.shape[1] * 0.4), int(out.shape[0] * 0.1))
                out = FX._text_layer(out, wm.get("text", "© 2026"),
                                     box, int(out.shape[1] * wm.get("size", 0.04)),
                                     (255, 255, 255), wm.get("opacity", 0.5))
        return out
    return fn


def process(cfg: dict, in_path, out_path, progress_cb=None):
    """便捷入口: 单帧引擎处理整段视频。"""
    from video import video_io
    fn = make_frame_fn(cfg)
    return video_io.process_video(in_path, out_path, fn, progress_cb=progress_cb)
