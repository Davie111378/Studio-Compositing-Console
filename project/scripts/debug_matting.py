"""调试 T01 双候选 GrabCut：输出两候选的面积/边界分并保存 PNG。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "ai-service"))
sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image

from aiservice.matting.impl import (
    _boundary_score,
    _color_distance_alpha,
    _grabcut_alpha,
)

OUT = PROJECT / "data" / "artifacts" / "debug_matting"
OUT.mkdir(parents=True, exist_ok=True)

img_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT.parent / "imagecompose-site" / "media" / "source_original.png")
src = Image.open(img_path).convert("RGB")
src.save(OUT / "src.png")

prior = _color_distance_alpha(src, "normal", max_size=512)
prior.save(OUT / "prior.png")
frac = float((np.array(prior) > 128).mean())
print(f"prior fg_frac = {frac:.3f}  degenerate = {frac > 0.60 or frac < 0.02}")

scale = min(1.0, 512.0 / max(src.size))
sw, sh = max(32, int(src.size[0] * scale + 0.5)), max(32, int(src.size[1] * scale + 0.5))
img_small = np.array(src.resize((sw, sh), Image.BILINEAR))

for name, ci in (("prior_seeded", False), ("center_init", True)):
    gc = _grabcut_alpha(src, prior, center_init=ci)
    if gc is None:
        print(f"{name}: INVALID (sanity gate)")
        continue
    a_small = np.array(gc.resize((sw, sh), Image.BILINEAR))
    area = float((a_small > 128).mean())
    score = _boundary_score(img_small, a_small)
    gc.save(OUT / f"cand_{name}.png")
    print(f"{name}: area={area:.3f} boundary_score={score:.2f}")

# 与实现里一致的择优结果
scored = []
for nm, ci in (("prior_seeded", False), ("center_init", True)):
    gc = _grabcut_alpha(src, prior, center_init=ci)
    if gc is None:
        continue
    a_small = np.array(gc.resize((sw, sh), Image.BILINEAR))
    scored.append((_boundary_score(img_small, a_small), float((a_small > 128).mean()), nm, gc))
if scored:
    top = max(s for s, _a, _n, _g in scored)
    pool = [x for x in scored if x[0] >= top * 0.92]
    winner = min(pool, key=lambda x: x[1])
    print(f"WINNER: {winner[2]}  (score {winner[0]:.2f}, area {winner[1]:.3f})")
    winner[3].save(OUT / "winner.png")
