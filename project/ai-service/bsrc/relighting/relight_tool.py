"""
T04 Relighting Tool
- 主方案：MKL color transfer (CPU 友好，无需 GPU)
- 进阶：基于环境光的亮度/色温调整
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageEnhance


def estimate_light(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """估算图像的主光：返回 (主光颜色 RGB ∈ [0,1], 平均亮度 ∈ [0,1])."""
    # 主光颜色：图像边缘的平均颜色（背景区域）
    h, w = img.shape[:2]
    edge = np.concatenate([
        img[:5].reshape(-1, 3),
        img[-5:].reshape(-1, 3),
        img[:, :5].reshape(-1, 3),
        img[:, -5:].reshape(-1, 3),
    ], axis=0).mean(axis=0)
    light_color = edge / 255.0
    light_brightness = float(img.mean() / 255.0)
    return light_color, light_brightness


def mkl_transfer(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reinhard-style MKL color transfer."""
    from scipy import linalg
    src = src.astype(np.float32) / 255.0
    ref = ref.astype(np.float32) / 255.0
    # log 域
    src_log = np.log1p(src)
    ref_log = np.log1p(ref)
    # 统计
    src_mu = src_log.mean(axis=(0, 1))
    ref_mu = ref_log.mean(axis=(0, 1))
    src_cov = np.cov(src_log.reshape(-1, 3), rowvar=False)
    ref_cov = np.cov(ref_log.reshape(-1, 3), rowvar=False)
    # 白化 + 染色
    src_whiten = (src_log - src_mu)
    eigvals, eigvecs = linalg.eigh(src_cov)
    eigvals = np.maximum(eigvals, 1e-6)
    whiten_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    colored_mat = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    src_w = src_whiten @ whiten_mat
    eigvals_r, eigvecs_r = linalg.eigh(ref_cov)
    eigvals_r = np.maximum(eigvals_r, 1e-6)
    target_w = src_w @ (eigvecs_r @ np.diag(np.sqrt(eigvals_r)) @ eigvecs_r.T)
    out_log = target_w + ref_mu
    out = np.expm1(out_log)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


class RelightTool:
    """T04 Relighting tool.
    method="directional" (默认, 2026-09-09): 方向性光照 (亮度渐变场 + 色温), 与色彩迁移正交,
        修复旧版 T04==T06 问题; 光方向自动从背景亮度分布估计, 或由 light_hint 指定。
    method="mkl": 旧版 MKL 统计迁移 (B6 对比实验保留)。"""

    def __init__(self, method: str = "directional"):
        self.method = method

    @staticmethod
    def estimate_direction(bg_rgb: np.ndarray) -> tuple[float, float]:
        """从背景亮度分布估计光方向: 更亮的一侧即光源方向 (归一化向量)。"""
        g = bg_rgb.astype(np.float32).mean(-1)
        h, w = g.shape
        dx = g[:, w // 2:].mean() - g[:, :w // 2].mean()
        dy = g[h // 2:, :].mean() - g[:h // 2, :].mean()
        n = max(np.hypot(dx, dy), 1e-3)
        return float(dx / n), float(dy / n)

    def directional_relight(self, fg_rgb: np.ndarray, bg_rgb: np.ndarray,
                            light_hint: Optional[Dict[str, Any]] = None,
                            gain: float = 0.30) -> np.ndarray:
        """方向性重打光:
        - shading(x,y) = 1 + gain * (dx*nx + dy*ny)   线性亮度渐变场 (光源侧更亮)
        - 色温: warm -> R+ B- / cool -> R- B+ (可由 light_hint['color_temp'] 指定,
          默认按背景色温自动: 背景偏暖 -> 暖光)
        - 额外弥散: 大核高斯的 shading 低频版叠加, 模拟软光过渡"""
        if light_hint and light_hint.get("direction"):
            dx, dy = light_hint["direction"]
            n = max(np.hypot(dx, dy), 1e-3)
            dx, dy = dx / n, dy / n
        else:
            dx, dy = self.estimate_direction(bg_rgb)
        h, w = fg_rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        nx = (xx / w) * 2 - 1
        ny = (yy / h) * 2 - 1
        grad = (dx * nx + dy * ny)
        shade = 1.0 + gain * grad
        shade_soft = 1.0 + (gain * 0.6) * cv2.GaussianBlur(grad.astype(np.float32), (0, 0), 60)
        shade = (shade * 0.5 + shade_soft * 0.5)[..., None]

        out = fg_rgb.astype(np.float32) * shade
        ct = (light_hint or {}).get("color_temp")
        if ct is None:
            ct = "warm" if bg_rgb.astype(np.float32)[..., 0].mean() > bg_rgb.astype(np.float32)[..., 2].mean() else "cool"
        amt = float((light_hint or {}).get("color_temp_strength", 0.10))
        amt = max(0.0, min(amt, 0.25))   # 色温偏移上限 25%: 更大值会把人像染成蓝/橙怪物 (2026-09-12 实测)
        if ct == "warm":
            out[..., 0] *= 1 + amt; out[..., 2] *= 1 - amt
        else:
            out[..., 2] *= 1 + amt; out[..., 0] *= 1 - amt
        return np.clip(out, 0, 255).astype(np.uint8)

    def relight(self, fg_path: str, bg_path: str, out_path: str,
                light_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        fg = np.array(Image.open(fg_path).convert("RGB"))
        bg_img = Image.open(bg_path).convert("RGB")
        if fg.shape[1] != bg_img.size[0] or fg.shape[0] != bg_img.size[1]:
            bg_img = bg_img.resize((fg.shape[1], fg.shape[0]), Image.BILINEAR)
        bg = np.array(bg_img)
        if self.method == "directional":
            out = self.directional_relight(fg, bg, light_hint)
            meta = {"direction": [round(v, 3) for v in self.estimate_direction(bg)]}
        elif self.method == "mkl":
            out = mkl_transfer(fg, bg)
            meta = {}
        else:
            out = fg
            meta = {}
        # light_hint 二次调整（intensity = 重打光混合强度; 颜色系数兼容旧接口）
        # 2026-09-12 修复: intensity 之前是整体乘法(out *= 0.6 直接压暗 40%),
        # 语义应为 "重打光结果与原图的线性混合", 与 tier_intensity (draft 0.6) 的意图一致
        if light_hint:
            intensity = max(0.0, min(float(light_hint.get("intensity", 1.0)), 1.0))
            color = light_hint.get("color", [1.0, 1.0, 1.0])
            out_f = out.astype(np.float32)
            for c in range(3):
                out_f[..., c] *= float(color[c])
            out_f = fg.astype(np.float32) * (1.0 - intensity) + out_f * intensity
            out = np.clip(out_f, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(out_path)
        return {"relit_fg_path": out_path, "method": self.method, **meta}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tool = RelightTool()
    print(tool.relight(args.fg, args.bg, args.out))
