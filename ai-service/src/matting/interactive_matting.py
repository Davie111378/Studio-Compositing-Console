# -*- coding: utf-8 -*-
"""
交互式圈选抠图（T03 交互档, 2026-09-09 新增）
- GrabCut: 用户给 box / polygon / 正负点 -> 粗 mask (OpenCV, CPU, 零依赖)
- 可选接 AlphaRefiner 精修边缘 (与两阶段抠图同一精修头)

prompt 格式 (与 A 组选区 schema 对齐):
  {"type":"box",      "xyxy":[x1,y1,x2,y2]}
  {"type":"polygon",  "points":[[x,y],...]}        # 圈选: 多边形内部=前景
  {"type":"points",   "positive":[[x,y],...], "negative":[[x,y],...]}
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image


def _prompt_to_mask(prompt: Dict[str, Any], shape) -> np.ndarray:
    """用户选区 -> (sure_fg, maybe_fg) 两级 bool mask。"""
    h, w = shape[:2]
    sure = np.zeros((h, w), np.uint8)
    maybe = np.zeros((h, w), np.uint8)
    t = prompt.get("type")
    if t == "box":
        x1, y1, x2, y2 = [int(max(0, v)) for v in prompt["xyxy"]]
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        sure[max(0, y1 + int(0.12 * (y2 - y1))):max(0, y2 - int(0.12 * (y2 - y1))),
             max(0, x1 + int(0.12 * (x2 - x1))):max(0, x2 - int(0.12 * (x2 - x1)))] = 1
        maybe[y1:y2, x1:x2] = 1
    elif t == "polygon":
        pts = np.array(prompt["points"], np.int32)
        cv2.fillPoly(maybe, [pts], 1)
        er = cv2.erode(maybe, np.ones((15, 15), np.uint8))
        sure = (er > 0).astype(np.uint8)
    elif t == "points":
        for x, y in prompt.get("positive", []):
            cv2.circle(sure, (int(x), int(y)), 12, 1, -1)
            cv2.circle(maybe, (int(x), int(y)), 28, 1, -1)
        if not prompt.get("positive"):
            raise ValueError("points prompt needs 'positive'")
    else:
        raise ValueError(f"unsupported prompt type: {t}")
    return sure, maybe


def grabcut_matte(image_rgb: np.ndarray, prompt: Dict[str, Any],
                  iters: int = 5) -> np.ndarray:
    """GrabCut 圈选抠图, 返回 float alpha (0-1)。"""
    img = image_rgb.copy()
    h, w = img.shape[:2]
    gd = cv2.GC_BGD * np.ones((h, w), np.uint8)
    sure, maybe = _prompt_to_mask(prompt, img.shape)
    gd[sure > 0] = cv2.GC_FGD
    gd[(maybe > 0) & (sure == 0)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.grabCut(img, gd, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iters, cv2.GC_INIT_WITH_MASK)
    m = np.where((gd == cv2.GC_FGD) | (gd == cv2.GC_PR_FGD), 1.0, 0.0)
    # 平滑到软边缘
    m = cv2.GaussianBlur(m, (5, 5), 1.2)
    return np.clip(m, 0, 1)


class InteractiveMatting:
    """圈选抠图工具。refine=True 时接 AlphaRefiner 精修 (需 GPU/权重)。"""

    def __init__(self, refiner_weight: Optional[str] = None, device: Optional[str] = None):
        self.refiner_weight = refiner_weight
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    def matte(self, image_path: str, out_alpha_path: str,
              prompt: Dict[str, Any], refine: bool = False) -> Dict[str, Any]:
        img = np.array(Image.open(image_path).convert("RGB"))
        alpha = grabcut_matte(img, prompt)
        model = "grabcut"
        if refine and self.refiner_weight and Path(self.refiner_weight).exists():
            # 接 AlphaRefiner 精修: 粗 alpha 为起点, 学残差提边缘 (与两阶段抠图同款精修头)
            import sys as _sys
            import torch as _torch
            _sys.path.insert(0, str(ROOT_TS := Path(__file__).resolve().parents[3] / "training" / "scripts"))
            from train_refiner import AlphaRefiner
            net = AlphaRefiner(base=32).to(self.device)
            ck = _torch.load(self.refiner_weight, map_location=self.device, weights_only=False)
            net.load_state_dict(ck["model"]); net.eval()
            S = 384
            rgb_t = _torch.from_numpy(
                np.array(Image.fromarray(img).resize((S, S))).astype("float32") / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)
            co_t = _torch.from_numpy(
                cv2.resize(alpha, (S, S)).astype("float32")
            ).unsqueeze(0).unsqueeze(0).to(self.device)
            with _torch.no_grad():
                pred = net(rgb_t, co_t)
            a_s = pred[0, 0].float().clamp(0, 1).cpu().numpy()
            alpha = cv2.resize(a_s, (img.shape[1], img.shape[0]))
            model = "grabcut+refiner"
        Path(out_alpha_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((alpha * 255).astype(np.uint8)).save(out_alpha_path)
        fg = (img.astype(np.float32) * alpha[..., None] + 255.0 * (1 - alpha[..., None])).astype(np.uint8)
        out_fg = str(Path(out_alpha_path).with_name(Path(out_alpha_path).stem + "_fg.png"))
        Image.fromarray(fg).save(out_fg)
        return {"alpha_path": out_alpha_path, "fg_path": out_fg, "model": model,
                "prompt": prompt, "fg_ratio": round(float((alpha > 0.5).mean()), 3)}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", required=True, help="x1,y1,x2,y2")
    a = ap.parse_args()
    box = [float(v) for v in a.box.split(",")]
    print(json.dumps(InteractiveMatting().matte(
        a.image, a.out, {"type": "box", "xyxy": box}), ensure_ascii=False, indent=2))
