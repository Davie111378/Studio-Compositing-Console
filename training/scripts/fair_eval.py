# -*- coding: utf-8 -*-
"""
fair_eval.py — 公平对照评测: 消除"推理分辨率"带来的评测偏差

问题: coarse 由 BiRefNet@512 产出后上采样到原图尺寸, 而 refiner 是 384 训练/推理后
      上采样。Grad(梯度误差)/Conn(连通性) 对边缘锐度极敏感, 直接用两者比较会让
      refiner 因"分辨率更低"而非"模型更差"被判负。
做法: 让 coarse 也走一次 384 往返 (512->384->原尺寸), 与 refiner 输出同路径比较。

用法: python fair_eval.py --refined preds_v3_384
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
import evaluate_matting as EM  # noqa: E402

DATA = ROOT / "data" / "matting_real"
EVALD = ROOT / "experiments" / "real_refiner_eval"

ap = argparse.ArgumentParser()
ap.add_argument("--refined", default="preds_v3_384")
ap.add_argument("--size", type=int, default=384)
args = ap.parse_args()

names = [x.strip() for x in (DATA / "test_names.txt").read_text(encoding="utf-8").split("\n") if x.strip()]
stems = [Path(x).stem for x in names]

# coarse 走与 refiner 同样的 size 往返
out = EVALD / f"preds_coarse_{args.size}rt"
out.mkdir(parents=True, exist_ok=True)
for s in stems:
    cp = DATA / "coarse_cache" / "test" / f"{s}.png"
    if not cp.exists():
        continue
    op = out / f"{s}.png"
    if op.exists():
        continue
    im = Image.open(cp).convert("L")
    W, H = im.size
    im.resize((args.size, args.size), Image.BILINEAR).resize((W, H), Image.BILINEAR).save(op)

cm = EM.load_case_map(DATA)


def rep(d, label):
    rows, mean, pc = EM.evaluate(DATA / "test", Path(d), "_alpha", stems, cm)
    print(f"{label:24s} SAD={mean[0]:9.1f} MSE={mean[1]:.5f} Grad={mean[2]:8.1f} Conn={mean[3]:.4f}", flush=True)
    return mean


m1 = rep(DATA / "coarse_cache" / "test", "coarse@512(原始)")
m2 = rep(out, f"coarse@{args.size}往返")
m3 = rep(EVALD / args.refined, f"{args.refined}@{args.size}")

print()
print(f"refined vs coarse@512  : SAD {(m3[0]-m1[0])/m1[0]*100:+.1f}%  "
      f"Grad {(m3[2]-m1[2])/m1[2]*100:+.1f}%  Conn {(m3[3]-m1[3])/m1[3]*100:+.1f}%")
print(f"refined vs coarse@{args.size}rt : SAD {(m3[0]-m2[0])/m2[0]*100:+.1f}%  "
      f"Grad {(m3[2]-m2[2])/m2[2]*100:+.1f}%  Conn {(m3[3]-m2[3])/m2[3]*100:+.1f}%")
