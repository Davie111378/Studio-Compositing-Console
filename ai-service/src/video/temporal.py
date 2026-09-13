# -*- coding: utf-8 -*-
"""
temporal.py — 视频时序一致性模块 (视频管线 Step3)
解决逐帧独立处理导致的闪烁/抖动问题。

核心机制:
  1) StaticAnchor: 静态参数首帧确定后跨帧沿用 (贴纸位置/圈选框/水印字/特效参数)
  2) KeyerReferenceStabilizer: 键控参考色时间稳定 (首帧估计后用 EMA 慢漂移更新,
        防止逐帧重估引起的绿幕边缘抖动)
  3) TemporalSmoother: 亮/色时间低通 (相邻帧约束, 防曝光闪烁)
"""
from __future__ import annotations
import collections
import numpy as np
import cv2
from PIL import Image


class StaticAnchor:
    """静态参数锚定: 首帧确定后跨帧沿用, 不再变化。
    用法:  anchor.set("heart_box", (x1,y1,x2,y2)) -> 所有帧都拿这个 box 用
    """
    def __init__(self):
        self._store = {}

    def set(self, key, value):
        self._store[key] = value

    def get(self, key, default=None):
        return self._store.get(key, default)

    def has(self, key):
        return key in self._store

    def update(self, key, value):
        """仅当 key 不存在时设置(首帧锚定语义)。"""
        if key not in self._store:
            self._store[key] = value

    def clear(self):
        self._store.clear()


class KeyerReferenceStabilizer:
    """色度键参考色时间稳定: 首帧估计, 后续帧仅做 EMA 慢漂移更新 (lr=0.05)。
    用法:
        stab = KeyerReferenceStabilizer(lr=0.05)
        for frame in frames:
            alpha = key_with_stable_ref(frame, stab)
    """
    def __init__(self, lr=0.05, max_drift=12.0):
        self.lr = float(lr)
        self.max_drift = float(max_drift)  # 单帧最大漂移 (Lab 距离)
        self.ref_lab = None
        self.ema_lab = None
        self.ema_n = 0

    def update(self, ref_lab_new):
        ref_lab_new = np.asarray(ref_lab_new, np.float32)
        if self.ema_lab is None:
            self.ema_lab = ref_lab_new.copy()
            self.ref_lab = ref_lab_new.copy()
            self.ema_n = 1
        else:
            # 单步限制 (避免某帧大跳跃)
            d = float(np.linalg.norm(ref_lab_new - self.ema_lab))
            if d > self.max_drift:
                ref_lab_new = self.ema_lab + (ref_lab_new - self.ema_lab) * (self.max_drift / (d + 1e-6))
            self.ema_lab = (1 - self.lr) * self.ema_lab + self.lr * ref_lab_new
            self.ref_lab = self.ema_lab.copy()
            self.ema_n += 1
        return self.ref_lab

    def get(self):
        return self.ref_lab


def stabilized_chroma_key(img_bgr, stabilizer, prev_alpha=None, t1_pad=4.0, t2_pad=30.0):
    """带参考色时间稳定的场景色度键 (BGR in / alpha out)。
    首帧或无 prev 时按 chroma_key_scene 估计参考色, 后续帧做 EMA 慢更新。
    prev_alpha (可选) 用于稳定 alpha mask (前帧 alpha 平滑)。
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    f = img.astype(np.float32)
    excess = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    thr = max(float(np.percentile(excess, 78)), 15.0)
    greenish = excess > thr
    if greenish.sum() < 200:
        # 无绿幕
        a = np.ones(img.shape[:2], np.float32)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_now = np.median(lab[greenish], axis=0)
        ref = stabilizer.update(ref_now)            # 跨帧稳定化
        d = np.linalg.norm(lab - ref, axis=-1)
        # t1 基于全图 d 自分位 (与本帧内容自适应)
        d_thr = max(float(np.percentile(d[greenish], 88)) + t1_pad, t1_pad + 4)
        t2 = d_thr + t2_pad
        a = np.clip((d - d_thr) / max(t2 - d_thr, 1.0), 0, 1)
        a = np.where(a < 0.1, 0.0, np.where(a > 0.85, 1.0, a))
        from scipy.ndimage import label
        bg = a < 0.5
        lb, n = label(bg)
        if n > 1:
            sizes = np.bincount(lb.ravel()); sizes[0] = 0
            keep = sizes.argmax()
            a[(bg) & (lb != keep)] = np.maximum(a[(bg) & (lb != keep)], 0.9)
    # 跨帧 alpha 时序平滑 (解决边缘帧间跳动)
    if prev_alpha is not None:
        a = 0.7 * a + 0.3 * prev_alpha
    return a, img


class TemporalSmoother:
    """亮/色时间低通 (防曝光闪烁 / 边缘抖动)。
    维护前一帧的色度均值, 与当前帧做 EMA 平滑, 强度 0~1 (越高越平滑)。
    """
    def __init__(self, strength=0.2):
        self.strength = float(strength)
        self.prev = None  # (mean_lab, mean_hsv) 元组

    def smooth(self, frame_rgb, frame_idx):
        # 仅在视频长过程中, 跳过首帧
        if frame_idx == 0 or self.strength <= 0:
            self.prev = frame_rgb.copy()
            return frame_rgb
        # 像素级 EMA (低强度才不影响内容)
        if self.prev is None:
            self.prev = frame_rgb.copy()
        s = self.strength
        out = (frame_rgb.astype(np.float32) * (1 - s) + self.prev.astype(np.float32) * s)
        out = np.clip(out, 0, 255).astype(np.uint8)
        self.prev = out.copy()
        return out

    def reset(self):
        self.prev = None


# ---- 快捷工厂
def build_stabilized_processor(cfg, stabilizer_lr=0.05, smoother_strength=0.15,
                                anchor_keys=None):
    """根据 cfg 返回 (frame_fn, state_dict), state_dict 跨帧保持。
    frame_fn(rgb, idx) -> rgb'
    """
    state = {
        "anchor": StaticAnchor(),
        "keyer": KeyerReferenceStabilizer(lr=stabilizer_lr),
        "smoother": TemporalSmoother(strength=smoother_strength),
        "prev_alpha": None,
    }
    return state


def stabilized_frame_apply(frame_rgb, idx, cfg, state):
    """单帧稳定化: 时间低通 -> 绿幕键控(参考色 EMA) -> 特效/滤镜/水印。
    state 跨帧持续。
    """
    smoother = state["smoother"]
    # 1) 亮度/色度时间低通
    f = smoother.smooth(frame_rgb, idx)
    # 2) 绿幕键控(用稳定化参考色)
    if cfg.get("green", {}).get("enabled", True) and cfg.get("green", {}).get("bg"):
        bg_path = cfg["green"]["bg"]
        a, rgb = stabilized_chroma_key(cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                                        state["keyer"],
                                        prev_alpha=state["prev_alpha"])
        state["prev_alpha"] = a
        import fx_library as FX
        from frame_engine import despill_strong, fit_bg_cover
        fg = despill_strong(rgb, a)
        bg = fit_bg_cover(np.array(Image.open(bg_path).convert("RGB")), rgb.shape[1], rgb.shape[0])
        a3 = a[..., None]
        f = np.clip(fg.astype(np.float32) * a3 + bg.astype(np.float32) * (1 - a3), 0, 255).astype(np.uint8)
    # 3) 全图滤镜
    if cfg.get("filter"):
        import fx_library as FX
        f = FX.fx_auto.place(f, cfg["filter"])[0]
    # 4) 静态特效/水印 (region 锚定: 首帧位置)
    fx_list = cfg.get("fx") or []
    for fx in fx_list:
        import fx_library as FX
        eff = fx["effect"]; p = fx.get("params") or {}
        region = fx.get("region")
        f = FX.fx_auto.place(f, eff, region, p)[0]
    return f
