# -*- coding: utf-8 -*-
"""
T-FX 特效工具（2026-09-09 新增, 任务表"添加特效"落地）
支持"区域特效": 用户圈选背景某区域 -> 特效只作用于该区域 (region mask 机制)。

特效清单 (每个均带 intensity 参数, 且支持 region):
  spotlight   聚光灯/舞台光斑 (椭圆羽化亮斑, screen 混合)
  bokeh       光斑散景 (高光提取 + 圆盘叠加)
  fog         舞台雾效/体积光 (噪声 + 径向衰减)
  vignette    暗角
  color_temp  色温滤镜 (warm/cool)
  depth_blur  区域景深虚化 (圈选区外模糊)

region 格式 (与 A 组选区 schema 对齐):
  None / {"type":"full"}
  {"type":"box",      "xyxy":[x1,y1,x2,y2]}
  {"type":"polygon",  "points":[[x,y],...]}
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

EFFECTS = ("spotlight", "bokeh", "fog", "vignette", "color_temp", "depth_blur")


def build_region_mask(shape, region: Optional[Dict[str, Any]],
                      feather: int = 41) -> np.ndarray:
    """选区 -> 羽化 float mask (0-1)。None/full => 全图。"""
    h, w = shape[:2]
    m = np.ones((h, w), np.float32) if not region or region.get("type") == "full" else np.zeros((h, w), np.float32)
    if region and region.get("type") == "box":
        x1, y1, x2, y2 = [int(v) for v in region["xyxy"]]
        m[y1:y2, x1:x2] = 1.0
    elif region and region.get("type") == "polygon":
        pts = np.array(region["points"], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
    if feather > 1:
        k = feather | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0, 1)


def spotlight(img: np.ndarray, mask: np.ndarray, intensity: float = 0.7,
              color=None, falloff: float = 2.2) -> np.ndarray:
    """聚光: mask 内椭圆亮斑, 中心最亮向外衰减, screen 混合。"""
    ys, xs = np.where(mask > 0.05)
    if len(xs) == 0:
        return img
    cx, cy = xs.mean(), ys.mean()
    rx, ry = max(np.ptp(xs) / 2, 8), max(np.ptp(ys) / 2, 8)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    glow = np.clip(1.0 - d, 0, 1) ** falloff
    glow = glow * mask * intensity
    col = np.ones(3, np.float32) if not color else np.array(color, np.float32)
    layer = np.clip(glow[..., None] * 255 * col, 0, 255)
    out = 255 - (255 - img.astype(np.float32)) * (255 - layer) / 255.0   # screen
    return np.clip(out, 0, 255)


def bokeh(img: np.ndarray, mask: np.ndarray, intensity: float = 0.6,
          n_dots: int = 60, seed: int = 3) -> np.ndarray:
    """散景: 提取高光点, 在 mask 内叠加柔边圆盘。"""
    rng = np.random.default_rng(seed)
    gray = img.astype(np.float32).mean(-1)
    hot = (gray > np.percentile(gray[mask > 0.05], 92) if (mask > 0.05).any()
           else gray > 200).astype(np.float32)
    layer = np.zeros_like(img, np.float32)
    h, w = img.shape[:2]
    for _ in range(n_dots):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        if mask[y, x] < 0.2:
            continue
        r = int(rng.uniform(6, 26))
        b = float(rng.uniform(0.25, 0.8)) * intensity * (0.4 + 0.6 * hot[y, x])
        tint = rng.uniform(0.9, 1.1, 3)
        cv2.circle(layer, (x, y), r, b * 255 * tint, -1, lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 6)
    layer *= mask[..., None]
    out = 255 - (255 - img.astype(np.float32)) * (255 - layer) / 255.0
    return np.clip(out, 0, 255)


def fog(img: np.ndarray, mask: np.ndarray, intensity: float = 0.45,
        seed: int = 7) -> np.ndarray:
    """舞台雾: 分形噪声白雾, mask 内径向衰减。"""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    small = rng.normal(0.5, 0.18, (h // 16 + 1, w // 16 + 1)).astype(np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.clip((noise - noise.min()) / (np.ptp(noise) + 1e-6), 0, 1)
    ys, xs = np.where(mask > 0.05)
    cy, cx = (ys.mean(), xs.mean()) if len(xs) else (h / 2, w / 2)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    fogm = np.clip(1.15 - dist, 0, 1) * noise * mask * intensity
    fog_rgb = np.stack([fogm * 235, fogm * 240, fogm * 255], -1)
    out = img.astype(np.float32) * (1 - fog_rgb) + fog_rgb
    return np.clip(out, 0, 255)


def vignette(img: np.ndarray, mask: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = 1 - np.clip(d - 0.55, 0, 1) ** 1.8 * intensity
    out = img.astype(np.float32) * (vig * mask + (1 - mask))[..., None]
    return np.clip(out, 0, 255)


def color_temp(img: np.ndarray, mask: np.ndarray, intensity: float = 0.5,
               mode: str = "warm") -> np.ndarray:
    out = img.astype(np.float32).copy()
    if mode == "warm":
        out[..., 0] += 38 * intensity; out[..., 2] -= 22 * intensity
    else:
        out[..., 2] += 38 * intensity; out[..., 0] -= 22 * intensity
    return np.clip(out * (1 - 0) , 0, 255) if mask is None else np.clip(
        img.astype(np.float32) * (1 - mask[..., None]) + np.clip(out, 0, 255) * mask[..., None], 0, 255)


def depth_blur(img: np.ndarray, mask: np.ndarray, intensity: float = 0.6,
               keep_region: bool = False) -> np.ndarray:
    """mask 区域保持清晰(keep_region=True)或虚化(=False), 其余相反。"""
    blurred = cv2.GaussianBlur(img, (0, 0), 9 + 12 * intensity)
    sharp_m = mask if keep_region else 1 - mask
    out = img.astype(np.float32) * sharp_m[..., None] + blurred.astype(np.float32) * (1 - sharp_m[..., None])
    return np.clip(out, 0, 255)


_DISPATCH = {"spotlight": spotlight, "bokeh": bokeh, "fog": fog,
             "vignette": vignette, "color_temp": color_temp, "depth_blur": depth_blur}


class FxTool:
    """特效工具: apply(image, effect, params, region)。可串联多个特效。"""

    def apply(self, image_path: str, out_path: str, effect: str,
              params: Optional[Dict[str, Any]] = None,
              region: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if effect not in EFFECTS:
            return {"error": {"code": "E_FX_UNKNOWN", "message": f"unknown effect {effect}", "retryable": False}}
        params = params or {}
        img = np.array(Image.open(image_path).convert("RGB"))
        mask = build_region_mask(img.shape, region)
        intensity = float(params.get("intensity", 0.6))
        out = _DISPATCH[effect](img, mask, intensity=intensity, **{k: v for k, v in params.items() if k != "intensity"})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out.astype(np.uint8)).save(out_path)
        return {"out_path": out_path, "effect": effect,
                "params": {**params, "intensity": intensity}, "region": region or "full"}

    def chain(self, image_path: str, out_path: str,
              effects: list, work_dir: str) -> Dict[str, Any]:
        """特效链: [{"effect":..., "params":{...}, "region":{...}}, ...]"""
        cur = image_path
        trace = []
        for i, e in enumerate(effects):
            cur_i = str(Path(work_dir) / f"fx_{i:02d}_{e['effect']}.png")
            r = self.apply(cur, cur_i, e["effect"], e.get("params"), e.get("region"))
            if "error" in r:
                return r
            trace.append(r)
            cur = cur_i
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.open(cur).convert("RGB").save(out_path)
        return {"out_path": out_path, "chain": trace}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effect", default="spotlight")
    ap.add_argument("--intensity", type=float, default=0.7)
    ap.add_argument("--box", default="", help="x1,y1,x2,y2 可选区域")
    a = ap.parse_args()
    region = ({"type": "box", "xyxy": [float(v) for v in a.box.split(",")]} if a.box else None)
    print(FxTool().apply(a.image, a.out, a.effect, {"intensity": a.intensity}, region))
