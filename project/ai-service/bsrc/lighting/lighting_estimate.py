# -*- coding: utf-8 -*-
"""
lighting_estimate.py — T03 lighting_estimate 独立工具 (B 组新补, v0.9)
从背景图估计主光方向 / 色温 / 强度 / SH9 系数。

方法 (Lambert 半球近似):
  1) 背景降采样 → 灰度 L (亮度), 把像素位置映射为上半球方向 (等距圆柱近似):
     方位角 θ = 2π*(x/W), 仰角 φ = π/2*(1 - y/H)  → 单位向量 n
  2) 主光方向 = 亮度加权平均方向 normalize(Σ L*n) → 反推出 (θ,φ) 与 2D (dx,dy)
  3) 色温: R/B 通道均值比 → warm/cool/neutral
  4) 强度: 亮度归一化标准差 (0~1)
  5) SH9: 前 2 面球谐基 (9 系数) 对 亮度*Lambert 权重做最小二乘投影
输出: {light_dir:[dx,dy], theta_phi:[θdeg,φdeg], color_temp, intensity, sh_coeff[9], method}
"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path


def _band1_dirs(nx, ny, nz):
    """SH 前 2 面基函数 (l=0 全部 + l=1 三项), 返回 (N,4)。"""
    return np.stack([np.ones_like(nx) * 0.282095,
                     0.488603 * ny,
                     0.488603 * nz,
                     0.488603 * nx], axis=-1)


def _imread_unicode(path: str):
    """cv2.imread 不支持中文路径 (Windows), 用 fromfile+imdecode 替代。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def estimate(bg_path: str, quality: str = "draft") -> dict:
    p = Path(bg_path)
    if not p.exists():
        raise FileNotFoundError(f"背景不存在: {bg_path}")
    img = _imread_unicode(str(p))
    if img is None:
        raise ValueError(f"背景无法读取: {bg_path}")
    size = 128 if quality in ("draft", "normal") else 256
    h, w = img.shape[:2]
    scale = size / max(h, w)
    img = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                     interpolation=cv2.INTER_AREA)
    b, g, r = [c.astype(np.float32) for c in cv2.split(img)]
    lum = 0.299 * r + 0.587 * g + 0.114 * b          # 0~255
    H, W = lum.shape

    # --- 像素 → 上半球方向 (等距圆柱近似: 图像平面贴到半球) ---
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = (xs / max(W - 1, 1)) * np.pi              # 0..π (左→右)
    phi = (1.0 - ys / max(H - 1, 1)) * (np.pi / 2)    # π/2(顶)→0(底)
    nx = np.sin(phi) * np.cos(theta)
    ny = np.cos(phi)                                  # 向上
    nz = np.sin(phi) * np.sin(theta)

    # --- 主光方向: 亮度加权平均方向 ---
    wl = np.clip(lum / 255.0, 0, 1) ** 2
    sx = float((wl * nx).sum()); sy = float((wl * ny).sum()); sz = float((wl * nz).sum())
    v = np.array([sx, sy, sz], np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:                                       # 均匀光照 → 顶光
        v = np.array([0.0, 1.0, 0.0]); n = 1.0
    v /= n
    theta_deg = float(np.degrees(np.arctan2(v[2], v[0])))       # 方位角
    phi_deg = float(np.degrees(np.arcsin(np.clip(v[1], -1, 1))))  # 仰角
    # 2D 屏幕方向: 光从 (θ) 方向来 → 阴影/渐变朝反向; 归一化 (dx,dy), y 向下为正
    dx = float(np.cos(np.radians(theta_deg)))
    dy = float(-np.sin(np.radians(theta_deg)) * max(0.2, abs(v[1])))
    dn = max(np.hypot(dx, dy), 1e-6)
    light_dir = [round(dx / dn, 4), round(dy / dn, 4)]

    # --- 色温 ---
    rm, bm = float(r.mean()), float(b.mean())
    ratio = rm / max(bm, 1e-3)
    color_temp = "warm" if ratio > 1.08 else ("cool" if ratio < 0.92 else "neutral")

    # --- 强度: 亮度标准差归一 ---
    intensity = round(float(lum.std() / 127.5), 4)

    # --- SH9 (前 2 面共 4 基 → 9 系数: 这里输出 9 维, 前 4 维为真投影, 后 5 维置 0 占位 l=2) ---
    flat_dirs = _band1_dirs(nx.ravel(), ny.ravel(), nz.ravel())
    w_flat = (wl.ravel() * lum.ravel())
    coef, *_ = np.linalg.lstsq(flat_dirs, w_flat, rcond=None)
    norm = float(np.abs(w_flat).sum()) + 1e-6
    sh4 = (coef / norm).astype(np.float64)
    sh_coeff = [round(float(x), 6) for x in np.concatenate([sh4, np.zeros(5)])]

    return {"light_dir": light_dir,
            "theta_phi": [round(theta_deg, 2), round(phi_deg, 2)],
            "color_temp": color_temp,
            "intensity": intensity,
            "sh_coeff": sh_coeff,
            "method": "lambert-hemisphere-approx",
            "version": "0.9"}


if __name__ == "__main__":
    import sys, json
    bg = sys.argv[1] if len(sys.argv) > 1 else "data/ai_generated/studio_bg/bg_02_interview.png"
    print(json.dumps(estimate(bg), ensure_ascii=False, indent=2))
