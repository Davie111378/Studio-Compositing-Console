# -*- coding: utf-8 -*-
"""
studio_regression.py — 专用 studio 域回归检查（大样本）

eval_real_refiner 里的回归只抽 24 张, 方差大不可靠。本脚本用全部 studio test
逐张比较多个 refiner 权重的 L1, 并给出"逐张劣化占比"(比均值更能反映风险)。

用法: python studio_regression.py --weights refiner_studio refiner_real_v3 [--size 384]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
from train_refiner import AlphaRefiner  # noqa: E402

ST = ROOT / "data" / "studio"
CC = ST / "coarse_cache" / "test"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", nargs="+", default=["refiner_studio", "refiner_real_v3"])
ap.add_argument("--size", type=int, default=384)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

items = []
for p in sorted((ST / "test").glob("*.png")):
    if p.stem.endswith("_alpha"):
        continue
    gtp = ST / "test" / f"{p.stem}_alpha.png"
    cp = CC / f"{p.stem}.png"
    if gtp.exists() and cp.exists():
        items.append((p, gtp, cp))
if args.limit:
    items = items[: args.limit]
print(f"[studio-regression] n={len(items)} size={args.size}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(weight: str):
    net = AlphaRefiner(base=32).to(dev)
    ck = torch.load(ROOT / "training" / "checkpoints" / weight / "best.pt",
                    map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()
    l1s, co_l1s = [], []
    for ip, gp, cp in items:
        img = Image.open(ip).convert("RGB")
        co = Image.open(cp).convert("L")
        W, H = img.size
        rgb = torch.from_numpy(np.array(img.resize((args.size, args.size)), np.float32) / 255.0
                               ).permute(2, 0, 1)[None].to(dev)
        cot = torch.from_numpy(np.array(co.resize((args.size, args.size)), np.float32) / 255.0
                               )[None, None].to(dev)
        pred = net(rgb, cot)[0, 0].float().clamp(0, 1).cpu().numpy()
        a = np.array(Image.fromarray((pred * 255).astype("uint8")).resize((W, H), Image.BILINEAR),
                     np.float32) / 255.0
        gt = np.array(Image.open(gp).convert("L"), np.float32) / 255.0
        l1s.append(float(np.abs(a - gt).mean()))
        co_l1s.append(float(np.abs(np.array(co.resize((W, H), Image.BILINEAR), np.float32) / 255.0 - gt).mean()))
    return np.array(l1s), np.array(co_l1s)


res = {}
for w in args.weights:
    l1, co = run(w)
    res[w] = l1
    print(f"  {w:22s} L1 mean={l1.mean()*100:6.3f}%  median={np.median(l1)*100:6.3f}%  "
          f"p90={np.percentile(l1,90)*100:6.3f}%   (coarse 基线 {co.mean()*100:.3f}%)", flush=True)

base = args.weights[0]
for w in args.weights[1:]:
    d = (res[w] - res[base]) / np.maximum(res[base], 1e-9) * 100
    print(f"  {w} vs {base}: 均值{(res[w].mean()-res[base].mean())/res[base].mean()*100:+.1f}%  "
          f"劣化张数={(d > 5).sum()}/{len(d)}  严重劣化(>20%)={(d > 20).sum()}")
