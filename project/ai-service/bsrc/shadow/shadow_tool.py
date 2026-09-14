"""
T05 Shadow Generation Tool
- 程序化阴影：基于 alpha + 高斯模糊 + 偏移
- 可参数化方向 / 强度 / 模糊半径
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image, ImageFilter


def generate_shadow(alpha: np.ndarray, direction: tuple = (20, 20),
                    blur_radius: int = 25, opacity: float = 0.5,
                    offset: tuple = (15, 15)) -> np.ndarray:
    """生成阴影 mask：模糊 + 偏移 + 衰减。

    返回阴影 alpha (H, W)，与 bg 同尺寸。
    """
    a_img = Image.fromarray((alpha * 255).astype(np.uint8))
    # 模糊
    shadow = a_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow_arr = np.array(shadow).astype(np.float32) / 255.0
    # 偏移（用 numpy roll 模拟）
    dx, dy = offset
    H, W = shadow_arr.shape
    out = np.zeros_like(shadow_arr)
    src_x_start = max(0, dx)
    src_x_end = min(W, W + dx)
    dst_x_start = max(0, -dx)
    dst_x_end = min(W, W - dx)
    src_y_start = max(0, dy)
    src_y_end = min(H, H + dy)
    dst_y_start = max(0, -dy)
    dst_y_end = min(H, H - dy)
    out[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = shadow_arr[src_y_start:src_y_end, src_x_start:src_x_end]
    out *= opacity
    return np.clip(out, 0, 1)


def apply_shadow_to_bg(bg: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """将阴影叠加到背景上（darken）。"""
    bg_f = bg.astype(np.float32)
    factor = 1.0 - shadow * 0.6
    factor = np.clip(factor[..., None], 0.4, 1.0)
    out = bg_f * factor
    return np.clip(out, 0, 255).astype(np.uint8)


class ShadowTool:
    """T05 Shadow tool."""

    def __init__(self, method: str = "procedural"):
        self.method = method

    def generate(self, alpha_path: str, bg_path: str, out_path: str,
                 shadow_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        alpha = np.array(Image.open(alpha_path).convert("L")).astype(np.float32) / 255.0
        bg_img = Image.open(bg_path).convert("RGB")
        bg = np.array(bg_img)
        hint = shadow_hint or {}
        direction = tuple(hint.get("direction", [20, 20]))
        offset = tuple(hint.get("offset", [15, 15]))
        blur_radius = int(hint.get("blur_radius", 25))
        opacity = float(hint.get("opacity", 0.5))
        # 尺寸对齐：alpha 与 bg 分辨率可能不同（如 alpha 384 / bg 512），统一到 bg 尺寸
        if alpha.shape != bg.shape[:2]:
            alpha = np.array(Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (bg.shape[1], bg.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0
        shadow = generate_shadow(alpha, direction, blur_radius, opacity, offset)
        # 应用到 bg
        bg_with_shadow = apply_shadow_to_bg(bg, shadow)
        Image.fromarray(bg_with_shadow).save(out_path)
        return {
            "shadow_path": out_path,
            "method": self.method,
            "opacity": opacity,
            "blur_radius": blur_radius,
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(ShadowTool().generate(args.alpha, args.bg, args.out))
