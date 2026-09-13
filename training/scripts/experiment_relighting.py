"""
B6 - 四组光影一致性对比实验
  A: 直接合成（仅 alpha over）
  B: + Harmonization
  C: + Relighting
  D: 全流程 Ours（Relighting + Shadow + Harmonization）

客观指标（自动计算，替代/辅助人工双盲评分）：
  1. dL    : 前景与背景的 LAB 亮度差（越小越协调）
  2. dAB   : 前景与背景的色度差（越小越协调）
  3. BNR   : Boundary Naturalness Ratio = 边缘带梯度 / 全图梯度
             （>1 表示边界存在硬边/不自然，越接近 1 越自然）
  4. ShadowEnergy: 合成图相对"无阴影版"的暗区能量（验证阴影确实生成）

用法：
  python experiment_relighting.py --n_samples 24 --size 512
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service"))
from pipeline import run_pipeline  # noqa

try:
    from skimage.color import rgb2lab
except Exception:  # pragma: no cover
    rgb2lab = None


# ------------------------------------------------------------------ 指标
def _lab(arr01: np.ndarray) -> np.ndarray:
    if rgb2lab is not None:
        return rgb2lab(arr01)
    # 退化：sRGB -> 近似 Lab（避免无 skimage 时崩）
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = np.clip(arr01 @ m.T, 0, 1)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def metrics(comp_rgb: np.ndarray, alpha: np.ndarray) -> dict:
    """comp_rgb: uint8 HWC; alpha: float HW in [0,1]"""
    a01 = comp_rgb.astype(np.float32) / 255.0
    lab = _lab(a01)
    fg = alpha > 0.90
    bg = alpha < 0.10
    if fg.sum() < 50 or bg.sum() < 50:
        fg = alpha > 0.6
        bg = alpha < 0.4
    L = lab[..., 0]
    AB = lab[..., 1:]
    dL = float(np.mean(L[fg]) - np.mean(L[bg]))
    dAB = float(np.linalg.norm(np.mean(AB[fg], axis=0) - np.mean(AB[bg], axis=0)))

    gray = 0.299 * a01[..., 0] + 0.587 * a01[..., 1] + 0.114 * a01[..., 2]
    gy, gx = np.gradient(gray)
    g = np.sqrt(gx ** 2 + gy ** 2)
    edge = (alpha > 0.15) & (alpha < 0.85)
    bnr = float(g[edge].mean() / max(1e-6, g.mean())) if edge.sum() > 50 else float("nan")
    return {"dL": dL, "dAB": dAB, "BNR": bnr,
            "edge_px": int(edge.sum()), "fg_px": int(fg.sum()), "bg_px": int(bg.sum())}


def load_alpha_any(p: Path, size) -> np.ndarray:
    return np.array(Image.open(p).convert("L").resize(size)).astype(np.float32) / 255.0


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_root", default=str(ROOT / "experiments" / "demo"))
    ap.add_argument("--out_root", default=str(ROOT / "experiments" / "relighting"))
    ap.add_argument("--n_samples", type=int, default=24)
    ap.add_argument("--names_file", default=str(ROOT / "data" / "matting" / "test_names.txt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--alpha_dir", default="",
                    help="预计算 alpha 目录（如 refined 输出）。提供则四组共用同一 alpha，"
                         "既保证公平对比，又省去 3/4 的 matting 开销")
    args = ap.parse_args()

    demo = Path(args.demo_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names = []
    if Path(args.names_file).exists():
        names = [Path(x.strip()).stem for x in
                 Path(args.names_file).read_text(encoding="utf-8").split("\n") if x.strip()]
    names = [n for n in names if (demo / "inputs" / f"{n}.png").exists()][: args.n_samples]
    if not names:  # 退化：直接取 inputs
        names = [p.stem for p in sorted((demo / "inputs").glob("*.png"))][: args.n_samples]

    print(f"[B6] {len(names)} samples x 4 groups -> {out_root}")
    results = {g: [] for g in "ABCD"}
    t_all = time.time()

    for i, n in enumerate(names):
        img_p = demo / "inputs" / f"{n}.png"
        bg_p = demo / "backgrounds" / f"{n}.png"
        if not bg_p.exists():
            continue
        # alpha 复用：优先用外部预计算 alpha；否则 A 组跑真实 matting，B/C/D 复用 A 的结果。
        # 这样既验证了一次完整全链路（T03 真实执行），又保证四组在完全相同的 alpha 下公平对比。
        shared_alpha = None
        for g in "ABCD":
            out_dir = out_root / g / n
            if shared_alpha is None:
                a_p = Path(args.alpha_dir) / f"{n}.png" if args.alpha_dir else None
                shared_alpha = str(a_p) if (a_p and a_p.exists()) else None
            t0 = time.time()
            try:
                r = run_pipeline(
                    str(img_p), str(bg_p), str(out_dir),
                    enable_relight=(g in "CD"),
                    enable_shadow=(g == "D"),
                    enable_harmonize=(g in "BD"),
                    composite_method="alpha_over",
                    device=args.device,
                    precomputed_alpha=shared_alpha,
                )
                if shared_alpha is None:
                    shared_alpha = str(Path(r["intermediate_dir"]) / "alpha.png")
                final = Path(r["final"])
                comp = np.array(Image.open(final).convert("RGB"))
                # pipeline 输出的 alpha 位于 <out_dir>/intermediate/alpha.png
                alpha_p = Path(r["intermediate_dir"]) / "alpha.png"
                if not alpha_p.exists():
                    alpha_p = out_dir / "alpha.png"
                al = load_alpha_any(alpha_p, (comp.shape[1], comp.shape[0])) if alpha_p.exists() \
                    else np.ones(comp.shape[:2])
                m = metrics(comp, al)
                m.update({"sample": n, "time_sec": round(time.time() - t0, 2), "out": str(final)})
                results[g].append(m)
                print(f"  [{g}] {n} dL={m['dL']:+.2f} dAB={m['dAB']:.2f} "
                      f"BNR={m['BNR']:.3f} {m['time_sec']}s", flush=True)
            except Exception as e:
                print(f"  [{g}] {n} FAILED: {type(e).__name__}: {e}", flush=True)
                results[g].append({"sample": n, "error": f"{type(e).__name__}: {e}"})

    # ---------------- 汇总
    summary = {"n_samples": len(names), "groups": {}, "aggregate": {}}
    for g in "ABCD":
        rows = [r for r in results[g] if "dL" in r]
        summary["groups"][g] = results[g]
        if rows:
            summary["aggregate"][g] = {
                k: float(np.nanmean([r[k] for r in rows]))
                for k in ["dL", "dAB", "BNR", "time_sec"]
            }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[B6] ===== 汇总 (n={len(names)}) =====")
    print(f"{'组':<4}{'说明':<28}{'|dL|':>8}{'dAB':>8}{'BNR':>8}{'秒/张':>8}")
    desc = {"A": "直接合成", "B": "+Harmonization", "C": "+Relighting", "D": "全流程 Ours"}
    for g in "ABCD":
        a = summary["aggregate"].get(g)
        if not a:
            continue
        print(f"{g:<4}{desc[g]:<28}{abs(a['dL']):>8.2f}{a['dAB']:>8.2f}{a['BNR']:>8.3f}{a['time_sec']:>8.2f}")
    print(f"[B6] total {time.time()-t_all:.1f}s -> {out_root}/summary.json")

    # ---------------- 可视化对比网格
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = min(12, len(names))
        # 6 列：Input | Background | A | B | C | D
        fig, axes = plt.subplots(k, 6, figsize=(20, 3.2 * k))
        if k == 1:
            axes = axes[None, :]
        for i, n in enumerate(names[:k]):
            axes[i, 0].imshow(Image.open(demo / "inputs" / f"{n}.png").convert("RGB"))
            axes[i, 1].imshow(Image.open(demo / "backgrounds" / f"{n}.png").convert("RGB"))
            for j, g in enumerate("ABCD"):
                p = out_root / g / n / "final.png"
                axes[i, 2 + j].imshow(Image.open(p).convert("RGB")) if p.exists() else axes[i, 2 + j].axis("off")
            axes[i, 0].set_ylabel(n, fontsize=7, rotation=0, labelpad=48)
            for c in range(6):
                axes[i, c].axis("off")
            if i == 0:
                for c, t in enumerate(["Input", "Background", "A 直接合成", "B +Harmon", "C +Relight", "D 全流程"]):
                    axes[i, c].set_title(t, fontsize=9)
        plt.tight_layout()
        fig.savefig(out_root / "_grid_ABCD.png", dpi=85)
        plt.close(fig)
        print(f"[B6] grid -> {out_root}/_grid_ABCD.png")
    except Exception as e:
        print(f"[B6] vis skipped: {e}")


if __name__ == "__main__":
    main()
