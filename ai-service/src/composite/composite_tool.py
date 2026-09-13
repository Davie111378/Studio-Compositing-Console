"""
T07 Composite Tool
- alpha over 基础合成
- 边界 feathering（可选）
- Poisson blending（可选，调用 scipy）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image, ImageFilter


def feather_alpha(alpha: np.ndarray, radius: int = 2) -> np.ndarray:
    """边界羽化。"""
    a_img = Image.fromarray((alpha * 255).astype(np.uint8))
    a_img = a_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(a_img).astype(np.float32) / 255.0


def alpha_over(fg: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """标准 alpha over。"""
    a = alpha[..., None]
    return (fg.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)


def poisson_blend(fg: np.ndarray, alpha: np.ndarray, bg: np.ndarray, feather: int = 3) -> np.ndarray:
    """简化版泊松融合：基于梯度的无缝融合。
    完整 Poisson 求解需迭代；这里用 alpha over + 边界羽化作为快速近似。
    """
    a = feather_alpha(alpha, radius=feather)
    return alpha_over(fg, a, bg)


class CompositeTool:
    """T07 Composite tool."""

    def __init__(self, method: str = "alpha_over", feather_radius: int = 2):
        self.method = method
        self.feather_radius = feather_radius

    def composite(self, fg_path: str, alpha_path: str, bg_path: str,
                  out_path: str, shadow_path: Optional[str] = None) -> Dict[str, Any]:
        fg = np.array(Image.open(fg_path).convert("RGB"))
        bg = np.array(Image.open(bg_path).convert("RGB"))
        if fg.shape != bg.shape:
            fg_img = Image.open(fg_path).convert("RGB").resize((bg.shape[1], bg.shape[0]), Image.BILINEAR)
            fg = np.array(fg_img)
        alpha = np.array(Image.open(alpha_path).convert("L")).astype(np.float32) / 255.0
        if alpha.shape != bg.shape[:2]:
            alpha_img = Image.open(alpha_path).convert("L").resize((bg.shape[1], bg.shape[0]), Image.BILINEAR)
            alpha = np.array(alpha_img).astype(np.float32) / 255.0
        # 阴影叠加到 bg
        if shadow_path:
            shadow_img = Image.open(shadow_path).convert("RGB")
            if shadow_img.size != (bg.shape[1], bg.shape[0]):
                shadow_img = shadow_img.resize((bg.shape[1], bg.shape[0]), Image.BILINEAR)
            bg = np.minimum(bg, np.array(shadow_img))
        # 合成
        if self.method == "poisson":
            out = poisson_blend(fg, alpha, bg, feather=self.feather_radius)
        else:
            out = alpha_over(fg, alpha, bg)
        Image.fromarray(out).save(out_path)
        return {"output_path": out_path, "method": self.method}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg", required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shadow", default=None)
    args = ap.parse_args()
    print(CompositeTool().composite(args.fg, args.alpha, args.bg, args.out, args.shadow))
