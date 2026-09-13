# -*- coding: utf-8 -*-
"""T2 加图全流水线测试：主播台 -> 两张虚拟背景
复用 S0 BiRefNet alpha（单图测试中的最优抠图），跑 T06 harmonize + T04 relight + T05 shadow + T07 composite
对照组：直接 alpha_over（不调色不重光照），量化 harmonize/relight 的增益
"""
import sys, time, json, shutil
from pathlib import Path

ROOT = Path("D:/AIcode/生产实习")
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))

import os
os.chdir(ROOT / "ai-service")

from PIL import Image
import numpy as np
from pipeline import run_pipeline

T = ROOT / "experiments" / "studio" / "single_image_test"
IMG = str(ROOT / "演播室图片生成需求.png")
ALPHA = str(T / "alpha_coarse.png")

bgs = {
    "led_studio": T / "backgrounds" / "Empty_modern_TV_news_studio_in_2026-09-09T07-35-31.png",
    "city_sunset": T / "backgrounds" / "Panoramic_city_skyline_at_gold_2026-09-09T07-35-58.png",
}
# 统一到输入图尺寸（1600x2880? 实际看图），背景 resize 成前景同尺寸
src_size = Image.open(IMG).size
for name, p in bgs.items():
    dst = T / f"bg_{name}.png"
    Image.open(p).convert("RGB").resize(src_size, Image.LANCZOS).save(dst)

results = {}
for name in bgs:
    bg = str(T / f"bg_{name}.png")
    # A) 完整流水线
    t0 = time.time()
    r_full = run_pipeline(IMG, bg, str(T / f"comp_{name}_full"),
                          precomputed_alpha=ALPHA, device="cuda")
    t_full = time.time() - t0
    # B) 裸合成对照（只 matting + composite，无 harmonize/relight/shadow）
    t0 = time.time()
    r_raw = run_pipeline(IMG, bg, str(T / f"comp_{name}_raw"),
                         precomputed_alpha=ALPHA, device="cuda",
                         enable_relight=False, enable_shadow=False, enable_harmonize=False)
    t_raw = time.time() - t0
    results[name] = {"full": str(r_full["final"]), "raw": str(r_raw["final"]),
                     "t_full_s": round(t_full, 1), "t_raw_s": round(t_raw, 1),
                     "stages_full": {s["name"]: round(s["time_ms"]) for s in r_full["stages"]}}
    print(f"[{name}] full={t_full:.1f}s raw={t_raw:.1f}s -> {r_full['final']}")

(T / "t2_pipeline.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("[T2] DONE")
