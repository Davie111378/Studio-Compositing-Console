# -*- coding: utf-8 -*-
"""
AI 素材端到端评测 (B6 口径 + 新增 FDR 防压黑 + 特效组)
四组: A 直接合成 / B +Harmonize(v2) / C +Relight(directional) / D 全流程 / D+fx 特效
指标: dL / dAB (前后景 Lab 差) / BNR (边界带梯度) / FDR (前景细节保持率)

用法:
  python ai_e2e_eval.py --alpha_dir experiments/ai_materials/refined \
      --n_bg 3 --out_dir experiments/ai_materials/e2e
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service"))

CLEAN_DIR = ROOT / "experiments" / "ai_materials" / "preview"
BG_DIR = ROOT / "data" / "ai_generated" / "studio_bg"


def lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def metrics_pair(comp: np.ndarray, fg_orig: np.ndarray, alpha: np.ndarray) -> dict:
    """dL / dAB / BNR / FDR"""
    a = cv2.resize(alpha, (comp.shape[1], comp.shape[0])).astype(np.float32)
    fg_m = a > 0.5
    bg_m = a < 0.2
    lc, lo = lab(comp), lab(fg_orig)
    dL = abs(lc[..., 0][fg_m].mean() - lc[..., 0][bg_m].mean()) if fg_m.any() and bg_m.any() else -1
    dAB = (abs(lc[..., 1][fg_m].mean() - lc[..., 1][bg_m].mean())
           + abs(lc[..., 2][fg_m].mean() - lc[..., 2][bg_m].mean())) if fg_m.any() and bg_m.any() else -1
    # BNR: alpha 边缘带内梯度幅值均值
    band = ((a > 0.05) & (a < 0.95)).astype(np.uint8)
    band = cv2.dilate(band, np.ones((5, 5), np.uint8)) > 0
    gx = np.abs(np.diff(comp.astype(np.float32).mean(-1), axis=1))
    gy = np.abs(np.diff(comp.astype(np.float32).mean(-1), axis=0))
    g = gx[:-1, :] + gy[:, :-1]
    band_c = band[1:, 1:] | band[:-1, 1:] | band[1:, :-1] | band[:-1, :-1]
    bnr = float(g[band_c].mean()) if band_c.any() else -1
    # FDR: 前景区域 Lab-L 标准差比 (原前景 vs 合成后)
    fdr = float(lc[..., 0][fg_m].std() / (lo[..., 0][fg_m].std() + 1e-6)) if fg_m.any() else -1
    return {"dL": round(float(dL), 2), "dAB": round(float(dAB), 2),
            "BNR": round(bnr, 3), "FDR": round(fdr, 3),
            "fdr_ok": bool(0.70 <= fdr <= 1.30)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_dir", default=str(ROOT / "experiments" / "ai_materials" / "refined"))
    ap.add_argument("--n_bg", type=int, default=3)
    ap.add_argument("--out_dir", default=str(ROOT / "experiments" / "ai_materials" / "e2e"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    from pipeline import run_pipeline
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cleans = sorted(CLEAN_DIR.glob("*_clean.png"))
    bgs = sorted(BG_DIR.glob("*.png"))[: args.n_bg]
    if not cleans or not bgs:
        raise FileNotFoundError("缺少 clean 人像或背景素材")

    all_metrics, combos = {}, []
    grid_rows = []
    for cp in cleans:
        stem = cp.stem.replace("_clean", "")
        alpha_p = Path(args.alpha_dir) / f"{stem}.png"
        if not alpha_p.exists():
            print(f"[skip] no alpha for {stem}")
            continue
        fg_orig = np.array(Image.open(cp).convert("RGB"))
        for bp in bgs:
            key = f"{stem}__{bp.stem}"
            row = []
            # 统一画布: 背景尺寸
            bg0 = Image.open(bp).convert("RGB")
            fg_resized = np.array(Image.open(cp).convert("RGB").resize(bg0.size, Image.BILINEAR))
            configs = [
                ("A", dict(enable_relight=False, enable_shadow=False, enable_harmonize=False)),
                ("B", dict(enable_relight=False, enable_shadow=False, enable_harmonize=True)),
                ("C", dict(enable_relight=True, enable_shadow=False, enable_harmonize=False)),
                ("D", dict(enable_relight=True, enable_shadow=True, enable_harmonize=True)),
            ]
            for name, cfg in configs:
                o = out / key / name
                r = run_pipeline(str(cp), str(bp), str(o), device=device,
                                 precomputed_alpha=str(alpha_p), **cfg)
                comp = np.array(Image.open(r["final"]).convert("RGB"))
                m = metrics_pair(comp, fg_resized, np.array(Image.open(alpha_p).convert("L")) / 255.0)
                all_metrics[f"{key}/{name}"] = m
                row.append(comp)
                print(f"[{key}/{name}] {m}", flush=True)
            # D + fx: 聚光灯(右上) + 暗角
            o = out / key / "Dfx"
            fx = [
                {"effect": "spotlight", "params": {"intensity": 0.55},
                 "region": {"type": "box", "xyxy": [int(bg0.size[0]*0.55), 0, bg0.size[0], int(bg0.size[1]*0.6)]}},
                {"effect": "vignette", "params": {"intensity": 0.35}},
            ]
            r = run_pipeline(str(cp), str(bp), str(o), device=device,
                             precomputed_alpha=str(alpha_p),
                             enable_relight=True, enable_shadow=True, enable_harmonize=True,
                             fx_chain=fx)
            comp = np.array(Image.open(r["final"]).convert("RGB"))
            m = metrics_pair(comp, fg_resized, np.array(Image.open(alpha_p).convert("L")) / 255.0)
            all_metrics[f"{key}/Dfx"] = m
            row.append(comp)
            combos.append(key)
            print(f"[{key}/Dfx] {m}", flush=True)
            # 网格行: 原图 | A | B | C | D | D+fx
            cell = (256, 170)
            def rs(im): return np.array(Image.fromarray(im).resize(cell))
            grid_rows.append([rs(fg_resized)] + [rs(x) for x in row])

    # 网格
    pad = 6
    cw, ch = 256, 170
    canvas = np.full((len(grid_rows) * (ch + pad) + pad, 6 * (cw + pad) + pad, 3), 28, np.uint8)
    for i, row in enumerate(grid_rows):
        for j, im in enumerate(row):
            y, x = pad + i * (ch + pad), pad + j * (cw + pad)
            canvas[y:y + ch, x:x + cw] = im
    Image.fromarray(canvas).save(out / "e2e_grid.png")
    print(f"[grid] wrote {out / 'e2e_grid.png'}")

    # 汇总
    def mean_of(suffix):
        vals = [m for k, m in all_metrics.items() if k.endswith(suffix)]
        return {k: round(float(np.mean([v[k] for v in vals])), 3)
                for k in ("dL", "dAB", "BNR", "FDR")}
    summary = {s: mean_of("/" + s) for s in ("A", "B", "C", "D", "Dfx")}
    summary["n"] = len(combos)
    (out / "e2e_metrics.json").write_text(
        json.dumps({"per_case": all_metrics, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
