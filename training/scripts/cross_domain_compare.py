#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨域对比实验：S0 BiRefNet baseline / S1 通用域 Refiner / S2 演播室域 Refiner
=============================================================

在演播室域 test 上评估, 量化"域内训练"带来的增益。
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))

# 通用数据用的粗 alpha 缓存(已生成的通用 matting coarse cache)
GEN_DATA = ROOT / "data" / "matting"
GEN_TEST = GEN_DATA / "test"
GEN_COARSE = GEN_DATA / "coarse_cache" / "test"

# 演播室数据
STUDIO_DATA = ROOT / "data" / "studio"
STUDIO_TEST = STUDIO_DATA / "test"
STUDIO_COARSE = STUDIO_DATA / "coarse_cache" / "test"


def load_alpha(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def resize_alpha(a: np.ndarray, target_hw):
    if a.shape == target_hw:
        return a
    return np.array(Image.fromarray((a * 255).astype(np.uint8))
                    .resize((target_hw[1], target_hw[0]), Image.BILINEAR),
                    dtype=np.float32) / 255.0


def sad(pr, gt): return float(np.abs(pr - gt).sum())
def mse(pr, gt): return float(((pr - gt) ** 2).mean())
def gradient_loss(pr, gt):
    dx_p, dy_p = np.gradient(pr); dx_g, dy_g = np.gradient(gt)
    return float(((dx_p - dx_g) ** 2 + (dy_p - dy_g) ** 2).sum() / pr.size)
def connectivity_loss(pr, gt, t=0.5):
    # 简化版：阈值化差异度量
    pm, gm = pr > t, gt > t
    inter = (pm & gm).sum(); uni = (pm | gm).sum() + 1e-6
    return float(1.0 - inter / uni)


def evaluate(name: str, gts: Path, preds_dir: Path, gt_suffix: str, test_names: list):
    rows = []
    sads, mses, grads, conns = [], [], [], []
    for n in test_names:
        gt_p = gts / f"{n}{gt_suffix}.png"
        pr_p = preds_dir / f"{n}.png"
        if not gt_p.exists() or not pr_p.exists():
            continue
        gt = load_alpha(gt_p)
        pr = resize_alpha(load_alpha(pr_p), gt.shape)
        s = sad(pr, gt); m = mse(pr, gt)
        g = gradient_loss(pr, gt); c = connectivity_loss(pr, gt)
        rows.append((n, s, m, g, c))
        sads.append(s); mses.append(m); grads.append(g); conns.append(c)
    return rows, (float(np.mean(sads)), float(np.mean(mses)),
                  float(np.mean(grads)), float(np.mean(conns))), len(rows)


def make_grid(inputs, gts, rows, paths, out_path, gt_suffix):
    """3-列对比图：原图 / GT alpha / 各 model alpha"""
    import matplotlib.pyplot as plt
    n = min(12, len(rows))
    fig, axes = plt.subplots(n, 5, figsize=(20, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    model_names = [p[0] for p in paths]
    for i, (name, *_rest) in enumerate(rows[:n]):
        axes[i, 0].imshow(Image.open(inputs / f"{name}.png").convert("RGB"))
        axes[i, 1].imshow(Image.open(gts / f"{name}{gt_suffix}.png").convert("L"), cmap="gray")
        for j, (_, pdir) in enumerate(paths):
            axes[i, 2 + j].imshow(Image.open(pdir / f"{name}.png").convert("L"), cmap="gray")
        axes[i, 0].set_ylabel(name[:18], fontsize=7, rotation=0, labelpad=42, va="center")
        for c in range(5):
            axes[i, c].axis("off")
        if i == 0:
            for c, t in enumerate(["Input", "GT alpha"] + model_names):
                axes[i, c].set_title(t, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"  grid -> {out_path}")


def write_csv(rows, mean, label, out_csv):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name," + label + "_SAD," + label + "_MSE," + label + "_Grad," + label + "_Conn\n")
        for n, s, m, g, c in rows:
            f.write(f"{n},{s:.4f},{m:.6f},{g:.4f},{c:.6f}\n")
        f.write(f"# mean,{mean[0]:.4f},{mean[1]:.6f},{mean[2]:.4f},{mean[3]:.6f}\n")
    print(f"  csv  -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_refined_dir", default=str(ROOT / "experiments/fine_tuned/preds_refined"),
                    help="通用域 Refiner 在 *通用 test* 的 preds（无法直接用，但可说明）")
    ap.add_argument("--studio_coarse_dir", default=str(STUDIO_COARSE),
                    help="BiRefNet 在演播室 test 上的粗 alpha（也用作 baseline）")
    ap.add_argument("--studio_refined_general_dir", default=None,
                    help="通用域 Refiner 在演播室 test 上的 preds（S1 跨域）")
    ap.add_argument("--studio_refined_studio_dir", default=None,
                    help="演播室域 Refiner 在演播室 test 上的 preds（S2 域内）")
    ap.add_argument("--out_root", default=str(ROOT / "experiments/cross_domain"))
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # 演播室 test 名单（manifest 权威）
    import csv as _csv
    test_names = []
    with open(STUDIO_DATA / "manifest.csv", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r["split"] == "test":
                test_names.append(r["image"].rsplit(".", 1)[0])

    print(f"[cross] studio test n = {len(test_names)}")

    # 加载所有 preds 路径
    paths = []  # [(label, dir)]
    paths.append(("S0_BiRefNet512", Path(args.studio_coarse_dir)))
    if args.studio_refined_general_dir:
        paths.append(("S1_GeneralRefiner", Path(args.studio_refined_general_dir)))
    if args.studio_refined_studio_dir:
        paths.append(("S2_StudioRefiner", Path(args.studio_refined_studio_dir)))

    results = {}
    for label, pdir in paths:
        if not pdir.exists():
            print(f"[WARN] missing {label} -> {pdir}")
            continue
        print(f"[cross] eval {label} from {pdir}")
        t0 = time.time()
        rows, mean, n = evaluate(label, STUDIO_TEST, pdir, "_alpha", test_names)
        print(f"  -> mean SAD={mean[0]:.2f}  MSE={mean[1]:.5f}  "
              f"Grad={mean[2]:.2f}  Conn={mean[3]:.4f}  n={n}  ({time.time()-t0:.1f}s)")
        write_csv(rows, mean, label, out_root / f"eval_{label}.csv")
        results[label] = (mean, rows)

    # 写汇总 markdown
    md = ["# 跨域对比实验报告\n",
          f"评估集: studio/test ({len(test_names)} 张)\n", "\n",
          "## 0. 实验设置\n",
          "- S0: BiRefNet 直出（粗 alpha, 512 输入, 权重来自 hf-mirror）\n",
          "- S1: BiRefNet + 通用域 AlphaRefiner（跨域, 在 data/matting 上训练）\n",
          "- S2: BiRefNet + 演播室域 AlphaRefiner（域内, 在 data/studio 上训练）\n",
          "\n## 1. 指标对比\n",
          "\n| Model | SAD | MSE | Grad | Conn | 相对 S0 SAD 增益 |\n",
          "|---|---|---|---|---|---|\n"]
    s0 = results.get("S0_BiRefNet512", (None,))[0]
    for label, (mean, _) in results.items():
        rel = ""
        if s0 is not None and label != "S0_BiRefNet512":
            rel = f"{((s0[0] - mean[0]) / s0[0] * 100):+.2f}%"
        md.append(f"| {label} | {mean[0]:.2f} | {mean[1]:.5f} | {mean[2]:.2f} | "
                  f"{mean[3]:.4f} | {rel} |\n")
    md.append("\n## 2. 视觉对比\n"
              "![cross domain](cross_domain_grid.png)\n")
    (out_root / "report.md").write_text("".join(md), encoding="utf-8")

    # 对比图
    if len(paths) >= 2:
        inputs_demo = ROOT / "experiments/demo/inputs"
        # demo 的输入是 512，但 test 是 512，不一定一一对应；改用 test 自身
        try:
            make_grid(STUDIO_TEST, STUDIO_TEST,
                      results[paths[0][0]][1], paths,
                      out_root / "cross_domain_grid.png", "_alpha")
        except Exception as e:
            print(f"[WARN] grid failed: {e}")


if __name__ == "__main__":
    main()