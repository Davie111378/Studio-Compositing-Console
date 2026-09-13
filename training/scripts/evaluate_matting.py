"""
Matting 评估脚本（B4）：在测试集上对比 baseline 与 fine-tuned。

指标（与 Composition-1k / P3M 标准一致）：
- SAD  : Sum of Absolute Differences（越小越好）
- MSE  : Mean Squared Error（越小越好）
- Grad : 梯度域误差（越小越好）
- Conn : 连通性误差 Connectivity error（越小越好）

用法：
  python evaluate_matting.py --gts DIR --preds DIR --out CSV --label NAME \
      --gt_suffix _alpha --names_file data/matting/test_names.txt
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve


def load_alpha(path: Path) -> np.ndarray:
    """读 alpha。RGBA/LA 的真值是 alpha 通道本身 (如 HM-1k matting 为前景抠图),
    convert('L') 会错误地取前景颜色亮度 -> 必须先取 A 通道再转灰度。"""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        return np.array(im.convert("RGBA"))[..., 3].astype(np.float32) / 255.0
    return np.array(im.convert("L")).astype(np.float32) / 255.0


def sad(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred - gt).sum())


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(((pred - gt) ** 2).mean())


def gradient_loss(pred: np.ndarray, gt: np.ndarray) -> float:
    """梯度域误差（Laplacian）。"""
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    gp = np.abs(convolve(pred, k))
    gg = np.abs(convolve(gt, k))
    return float(((gp - gg) ** 2).sum())


def connectivity_loss(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    """连通性误差（简化版，参考 Composition-1k 定义）。"""
    h, w = pred.shape
    loss = 0.0
    for thr in np.arange(step, 1.0, step):
        p_bin = (pred > thr).astype(np.float32)
        g_bin = (gt > thr).astype(np.float32)
        loss += np.abs(p_bin - g_bin).sum()
    return float(loss / (h * w))


def load_case_map(root: Path):
    """name_stem -> case 类别（用于难例子集分组统计）。"""
    mf = root / "manifest.csv"
    m = {}
    if mf.exists():
        with open(mf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m[Path(r["image"]).stem] = r.get("case", "unknown")
    return m


def evaluate(gts_dir: Path, preds_dir: Path, gt_suffix: str = "",
             names=None, case_map=None):
    rows = []
    per_case = {}
    sads, mses, grads, conns = [], [], [], []
    if names is None:
        names = sorted([p.stem[: -len(gt_suffix)] if gt_suffix else p.stem
                        for p in gts_dir.glob(f"*{gt_suffix}.png")])
    for raw in names:
        name = Path(raw).stem
        g_p = gts_dir / f"{name}{gt_suffix}.png"
        p_p = preds_dir / f"{name}.png"
        if not g_p.exists() or not p_p.exists():
            continue
        gt = load_alpha(g_p)
        pr = load_alpha(p_p)
        if pr.shape != gt.shape:
            pr = np.array(Image.open(p_p).convert("L").resize(
                gt.shape[::-1], Image.BILINEAR)).astype(np.float32) / 255.0
        s = sad(pr, gt); m = mse(pr, gt); g = gradient_loss(pr, gt); c = connectivity_loss(pr, gt)
        case = (case_map or {}).get(name, "unknown")
        rows.append((name, case, s, m, g, c))
        per_case.setdefault(case, []).append((s, m, g, c))
        sads.append(s); mses.append(m); grads.append(g); conns.append(c)
    return rows, (np.mean(sads), np.mean(mses), np.mean(grads), np.mean(conns)), per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--gt_suffix", default="", help="GT alpha 文件名后缀，如 _alpha")
    ap.add_argument("--names_file", default="", help="按 manifest 隔离的样本名清单")
    args = ap.parse_args()

    gts = Path(args.gts)
    names = None
    if args.names_file and Path(args.names_file).exists():
        names = [x.strip() for x in Path(args.names_file).read_text(encoding="utf-8").split("\n") if x.strip()]
    case_map = load_case_map(gts.parent if gts.name == "test" else gts)

    rows, mean, per_case = evaluate(gts, Path(args.preds), args.gt_suffix, names, case_map)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("name,case,SAD,MSE,Grad,Conn\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.6f},{r[4]:.4f},{r[5]:.4f}\n")
        f.write(f"# MEAN,all,{mean[0]:.4f},{mean[1]:.6f},{mean[2]:.4f},{mean[3]:.4f}\n")
        for c, vals in sorted(per_case.items()):
            a = np.array(vals)
            f.write(f"# CASE,{c},{a[:,0].mean():.4f},{a[:,1].mean():.6f},{a[:,2].mean():.4f},{a[:,3].mean():.4f}\n")

    print(f"[{args.label}] n={len(rows)} MEAN SAD={mean[0]:.2f} MSE={mean[1]:.6f} "
          f"Grad={mean[2]:.2f} Conn={mean[3]:.4f}")
    for c, vals in sorted(per_case.items()):
        a = np.array(vals)
        print(f"    {c:<12} n={len(vals):<4} SAD={a[:,0].mean():.2f} MSE={a[:,1].mean():.6f} "
              f"Grad={a[:,2].mean():.2f} Conn={a[:,3].mean():.4f}")
    print(f"[{args.label}] wrote {out}")
    # 机器可读摘要，供对比脚本合并
    summary = {"label": args.label, "n": len(rows),
               "SAD": mean[0], "MSE": mean[1], "Grad": mean[2], "Conn": mean[3],
               "per_case": {c: {"n": len(v), "SAD": float(np.array(v)[:, 0].mean()),
                                "MSE": float(np.array(v)[:, 1].mean()),
                                "Grad": float(np.array(v)[:, 2].mean()),
                                "Conn": float(np.array(v)[:, 3].mean())}
                            for c, v in per_case.items()}}
    (out.parent / f"summary_{args.label}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
