"""
B4 - 生成抠图视觉对比图：输入原图 | GT alpha | Baseline alpha | Ours alpha | 差异热图
输出 ≥20 组对比图 + 一张汇总网格图。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB").resize((256, 256)))


def load_a(p):
    return np.array(Image.open(p).convert("L").resize((256, 256))).astype(np.float32) / 255.0


def alpha_to_rgb(a):
    """alpha 转彩色热力（前景=白，背景=黑），便于肉眼判断边缘。"""
    return (np.stack([a] * 3, -1) * 255).astype(np.uint8)


def diff_heat(pred, gt):
    """差异热图：蓝=OK，红=误差大。"""
    d = np.abs(pred - gt)
    h = np.zeros((*d.shape, 3), dtype=np.uint8)
    h[..., 2] = np.clip((1 - d) * 255, 0, 255)          # 蓝：正确
    h[..., 0] = np.clip(d * 255 * 2, 0, 255)            # 红：错误
    h[..., 1] = np.clip((1 - d) * 120, 0, 255)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="原图目录")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gt_suffix", default="_alpha")
    ap.add_argument("--names_file", default="")
    args = ap.parse_args()

    inp, gt_dir = Path(args.inputs), Path(args.gts)
    bl, ou = Path(args.baseline), Path(args.ours)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.names_file and Path(args.names_file).exists():
        names = [Path(x.strip()).stem for x in
                 Path(args.names_file).read_text(encoding="utf-8").split("\n") if x.strip()]
    else:
        names = sorted([p.stem for p in bl.glob("*.png")])

    # 按 baseline 误差排序，挑最有代表性的（含好/中/差）
    scored = []
    for n in names:
        g = gt_dir / f"{n}{args.gt_suffix}.png"
        b = bl / f"{n}.png"
        o = ou / f"{n}.png"
        if not (g.exists() and b.exists() and o.exists()):
            continue
        ga, ba, oa = load_a(g), load_a(b), load_a(o)
        scored.append((n, float(np.abs(ba - ga).mean()), float(np.abs(oa - ga).mean())))
    if not scored:
        print("[vis] 无可用样本")
        return
    scored.sort(key=lambda x: -x[1])
    # 均匀采样：最难的一半 + 中等的一部分
    step = max(1, len(scored) // args.n)
    picked = scored[::step][: args.n]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = 2, 3
    grid_items = []
    for n, e_b, e_o in picked:
        ip = inp / f"{n}.png"
        if not ip.exists():
            ip = inp / f"{n}.jpg"
        img = load_rgb(ip) if ip.exists() else np.zeros((256, 256, 3), np.uint8)
        ga = load_a(gt_dir / f"{n}{args.gt_suffix}.png")
        ba = load_a(bl / f"{n}.png")
        oa = load_a(ou / f"{n}.png")

        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes[0, 0].imshow(img); axes[0, 0].set_title("Input", fontsize=10)
        axes[0, 1].imshow(alpha_to_rgb(ga), cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("GT alpha", fontsize=10)
        axes[0, 2].imshow(img); axes[0, 2].imshow(ba, cmap="jet", alpha=0.45)
        axes[0, 2].set_title("Input + Baseline alpha", fontsize=10)
        axes[1, 0].imshow(alpha_to_rgb(ba), cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title(f"Baseline (err={e_b:.4f})", fontsize=10)
        axes[1, 1].imshow(alpha_to_rgb(oa), cmap="gray", vmin=0, vmax=255)
        axes[1, 1].set_title(f"Ours (err={e_o:.4f})", fontsize=10)
        axes[1, 2].imshow(diff_heat(oa, ga))
        axes[1, 2].set_title("|Ours - GT| (red=bad)", fontsize=10)
        for a in axes.ravel():
            a.axis("off")
        plt.suptitle(f"{n}", fontsize=11)
        plt.tight_layout()
        fig.savefig(out / f"cmp_{n}.png", dpi=100)
        plt.close(fig)
        grid_items.append((n, img, ga, ba, oa, e_b, e_o))

    # 汇总网格：每行一个样本（input | GT | baseline | ours）
    k = len(grid_items)
    fig, axes = plt.subplots(k, 4, figsize=(12, 3 * k))
    if k == 1:
        axes = axes[None, :]
    for i, (n, img, ga, ba, oa, e_b, e_o) in enumerate(grid_items):
        axes[i, 0].imshow(img); axes[i, 0].set_ylabel(n, fontsize=7, rotation=0, labelpad=45)
        axes[i, 1].imshow(alpha_to_rgb(ga), cmap="gray", vmin=0, vmax=255)
        axes[i, 2].imshow(alpha_to_rgb(ba), cmap="gray", vmin=0, vmax=255)
        axes[i, 3].imshow(alpha_to_rgb(oa), cmap="gray", vmin=0, vmax=255)
        for j in range(4):
            axes[i, j].axis("off")
        if i == 0:
            axes[i, 0].set_title("Input", fontsize=9)
            axes[i, 1].set_title("GT", fontsize=9)
            axes[i, 2].set_title("Baseline", fontsize=9)
            axes[i, 3].set_title("Ours", fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "_grid_overview.png", dpi=90)
    plt.close(fig)

    # 误差下降统计
    eb = np.mean([x[4] for x in grid_items]); eo = np.mean([x[5] for x in grid_items])
    print(f"[vis] {k} comparisons -> {out}")
    print(f"[vis] mean |err| baseline={eb:.5f} ours={eo:.5f} "
          f"rel_drop={(eb-eo)/max(1e-9,eb)*100:.2f}%")


if __name__ == "__main__":
    main()
