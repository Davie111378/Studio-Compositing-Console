# -*- coding: utf-8 -*-
"""
T06 Harmonization Tool
- method="v2" (默认, 2026-09-09): Lab 色度对齐 + 低频亮度迁移 + 高频细节保留 + FDR 防压黑
  修复旧版问题: MKL 全图统计迁移把前景亮度硬拉向暗背景 -> 前景压黑 (B6 网格图实锤)
- method="mkl": 旧版 MKL (保留用于对比实验)
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "relighting"))
from relight_tool import mkl_transfer  # noqa: E402  (旧版保留)


def _lab(img_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def harmonize_v2(fg_rgb: np.ndarray, bg_rgb: np.ndarray,
                 alpha: Optional[np.ndarray] = None,
                 strength: float = 0.75) -> tuple[np.ndarray, Dict[str, Any]]:
    """防压黑和谐化:
    - a/b 色度: 前景统计向背景对齐 (均值偏移限幅 +-14, 方差比限幅 [0.75,1.35])
    - L 亮度:   只迁移低频(<~25px)分量, 高频细节(发丝/五官)完整保留
    - FDR 保护: sigma(out)/sigma(in) 超出 [0.7,1.3] 自动降 strength 重算
    返回 (uint8 RGB, info dict)"""
    fg_l = _lab(fg_rgb)
    bg_l = _lab(bg_rgb)
    if alpha is None:
        alpha = np.ones(fg_rgb.shape[:2], np.float32)
    else:
        alpha = cv2.resize(alpha, (fg_rgb.shape[1], fg_rgb.shape[0])).astype(np.float32) \
            if alpha.shape[:2] != fg_rgb.shape[:2] else alpha.astype(np.float32)
    am = (alpha > 0.3).astype(np.float32)

    out = fg_l.copy()
    # --- a/b 色度对齐 (限幅) ---
    for c in (1, 2):
        f_mu, b_mu = fg_l[..., c][am > 0].mean(), bg_l[..., c].mean()
        f_sd = max(fg_l[..., c][am > 0].std(), 1e-3)
        b_sd = max(bg_l[..., c].std(), 1e-3)
        d_mu = np.clip((b_mu - f_mu) * strength, -14, 14)
        d_sd = float(np.clip(b_sd / f_sd, 0.75, 1.35))
        shifted = (fg_l[..., c] - f_mu) * d_sd + f_mu + d_mu
        out[..., c] = fg_l[..., c] * (1 - strength) + shifted * strength

    # --- 肤色保护 (2026-09-10): 冷背景会把肤色拉成青绿 → 皮肤区保留 65% 原色度 ---
    try:
        ycrcb = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        cb, cr = ycrcb[..., 1], ycrcb[..., 2]
        skin = ((cb > 77) & (cb < 135) & (cr > 133) & (cr < 180)).astype(np.float32)
        skin = cv2.GaussianBlur(skin, (0, 0), 5)      # 软边缘
        skin *= am                                     # 仅前景区内生效
        keep = 0.80                                     # 肤色区保留 80% 原色度 (强保护)
        for c in (1, 2):
            out[..., c] = out[..., c] * (1 - skin * keep) + fg_l[..., c] * (skin * keep)
    except Exception:
        pass  # 肤色检测失败不影响主流程

    # --- L 亮度: 低频迁移 + 细节保留 ---
    L = fg_l[..., 0]
    low_f = cv2.GaussianBlur(L, (0, 0), 25)
    low_b = cv2.GaussianBlur(bg_l[..., 0], (0, 0), 25)
    detail = L - low_f
    ratio = float(np.clip(low_b[am > 0].mean() / max(low_f[am > 0].mean(), 1e-3), 0.8, 1.25))
    L_new = detail + low_f * (1 + (ratio - 1) * strength)
    out[..., 0] = np.clip(L_new, 0, 255)

    rgb = _rgb(out)

    # --- FDR 防压黑保护: sigma(L_out)/sigma(L_in) 超界则向原前景回退 ---
    def fdr_of(lab_l: np.ndarray) -> float:
        return lab_l[am > 0].std() / (L[am > 0].std() + 1e-6)

    orig = fg_rgb.astype(np.float32)
    fdr = fdr_of(_lab(rgb)[..., 0])
    for _ in range(4):
        if 0.70 <= fdr <= 1.30:
            break
        blend = 0.35
        mixed = rgb.astype(np.float32) * (1 - blend) + orig * blend
        rgb = mixed.astype(np.uint8)
        strength *= 0.7
        fdr = fdr_of(_lab(rgb)[..., 0])
    info = {"method": "v2", "strength": round(float(strength), 3),
            "fdr": round(float(fdr), 3),
            "fdr_ok": bool(0.70 <= fdr <= 1.30),
            "l_ratio": round(ratio, 3)}
    return rgb, info


class HarmonizeTool:
    """T06 Harmonization tool. method: v2 (默认/防压黑) | mkl (旧版对比用)"""

    def __init__(self, method: str = "v2"):
        self.method = method

    def harmonize(self, fg_path: str, bg_path: str, out_path: str,
                  alpha_path: Optional[str] = None) -> Dict[str, Any]:
        fg = np.array(Image.open(fg_path).convert("RGB"))
        bg_img = Image.open(bg_path).convert("RGB")
        if fg.shape[1] != bg_img.size[0] or fg.shape[0] != bg_img.size[1]:
            bg_img = bg_img.resize((fg.shape[1], fg.shape[0]), Image.BILINEAR)
        bg = np.array(bg_img)
        alpha = np.array(Image.open(alpha_path).convert("L")).astype(np.float32) / 255.0 \
            if alpha_path and Path(alpha_path).exists() else None
        if self.method == "v2":
            out, info = harmonize_v2(fg, bg, alpha)
            info["harmonized_fg_path"] = out_path
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(out_path)
            return info
        out = mkl_transfer(fg, bg)
        Image.fromarray(out).save(out_path)
        return {"harmonized_fg_path": out_path, "method": "mkl"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", default=None)
    ap.add_argument("--method", default="v2")
    args = ap.parse_args()
    print(HarmonizeTool(method=args.method).harmonize(args.fg, args.bg, args.out, args.alpha))
