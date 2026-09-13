"""
B8 - 消融实验 + 失败案例分析
  消融：在 test 集挑若干样本，对比 full / no_shadow / no_relight / no_harmonize 四种配置
  失败案例：按 case 类别从 test 集挑 1 张跑全流程，分析归因
  输出：JSON 报告 + 失败案例对比图 + 消融对比图
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import csv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service"))
from pipeline import run_pipeline  # noqa


def load_split(split: str):
    mf = ROOT / "data" / "matting" / "manifest.csv"
    names = []
    if mf.exists():
        with open(mf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    names.append((r["image"], r.get("case", "unknown")))
    return names


def metrics(comp_rgb: np.ndarray, alpha: np.ndarray) -> dict:
    """同 experiment_relighting 里的指标"""
    a01 = comp_rgb.astype(np.float32) / 255.0
    from skimage.color import rgb2lab
    lab = rgb2lab(a01)
    fg = alpha > 0.90
    bg = alpha < 0.10
    L = lab[..., 0]
    AB = lab[..., 1:]
    dL = float(np.mean(L[fg]) - np.mean(L[bg]))
    dAB = float(np.linalg.norm(np.mean(AB[fg], axis=0) - np.mean(AB[bg], axis=0)))
    gray = 0.299 * a01[..., 0] + 0.587 * a01[..., 1] + 0.114 * a01[..., 2]
    gy, gx = np.gradient(gray)
    g = np.sqrt(gx ** 2 + gy ** 2)
    edge = (alpha > 0.15) & (alpha < 0.85)
    bnr = float(g[edge].mean() / max(1e-6, g.mean())) if edge.sum() > 50 else float("nan")
    return {"dL": dL, "dAB": dAB, "BNR": bnr}


def pick_one_per_case(split="test"):
    """每个 case 类别选 1 个样本（用于失败案例）"""
    chosen = {}
    for img, case in load_split(split):
        chosen.setdefault(case, img)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_dir", default="",
                    help="预计算 alpha 目录（refined 输出）；提供则全部走预计算")
    ap.add_argument("--out_root", default=str(ROOT / "experiments" / "failure"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "experiments" / "demo" / "inputs"
    bg = ROOT / "experiments" / "demo" / "backgrounds"
    alpha_dir = Path(args.alpha_dir) if args.alpha_dir else None

    # ---------------- 1) 消融（test 集挑 4 张） ----------------
    ablation_out = out / "ablation"
    ablation_out.mkdir(parents=True, exist_ok=True)
    # 选 4 个有代表性的 case：hair / glass / normal / complex_bg
    test_pick = [("hair", "hair_0108"), ("glass", "glass_0108"),
                 ("normal", "normal_0108"), ("complex_bg", "complex_bg_0108")]
    # 实际取 case 的第一个
    by_case = {}
    for img, case in load_split("test"):
        by_case.setdefault(case, img)
    targets = [(c, Path(by_case[c]).stem) for c in ["hair", "glass", "normal", "complex_bg"] if by_case.get(c)]
    configs = [
        ("full",          True,  True,  True),
        ("no_shadow",     True,  False, True),
        ("no_relight",    False, True,  True),
        ("no_harmonize",  True,  True,  False),
    ]
    ablation_data = {"samples": [], "configs": [c[0] for c in configs], "results": {}}
    shared_alphas = {}
    for case, stem in targets:
        img_p = inp / f"{stem}.png"
        bg_p = bg / f"{stem}.png"
        if not img_p.exists() or not bg_p.exists():
            print(f"[ablation] skip {case} (demo missing)")
            continue
        # 预计算 alpha：来自 alpha_dir；否则第一次跑真实 matting 取出
        pre_alpha = None
        if alpha_dir:
            a = alpha_dir / f"{stem}.png"
            if a.exists():
                pre_alpha = str(a)
        ablation_data["samples"].append({"case": case, "stem": stem})
        for cfg, er, es, eh in configs:
            t0 = time.time()
            try:
                r = run_pipeline(
                    str(img_p), str(bg_p), str(ablation_out / cfg / f"{case}_{stem}"),
                    enable_relight=er, enable_shadow=es, enable_harmonize=eh,
                    composite_method="alpha_over", device=args.device,
                    precomputed_alpha=pre_alpha,
                )
                # 取 alpha（首次真实 matting 则记下供后续组复用）
                if pre_alpha is None:
                    a_p = Path(r["intermediate_dir"]) / "alpha.png"
                    pre_alpha = str(a_p)
                comp = np.array(Image.open(r["final"]).convert("RGB"))
                a = np.array(Image.open(pre_alpha).convert("L").resize(
                    (comp.shape[1], comp.shape[0]))).astype(np.float32) / 255.0
                m = metrics(comp, a)
                key = f"{case}_{cfg}"
                ablation_data["results"][key] = {
                    "case": case, "cfg": cfg, "metrics": m,
                    "time_sec": round(time.time() - t0, 2), "out": r["final"],
                }
                print(f"  [{cfg}] {case} dL={m['dL']:+.2f} dAB={m['dAB']:.2f} BNR={m['BNR']:.3f}", flush=True)
            except Exception as e:
                ablation_data["results"][f"{case}_{cfg}"] = {
                    "case": case, "cfg": cfg, "error": f"{type(e).__name__}: {e}",
                }
                print(f"  [{cfg}] {case} FAILED: {e}", flush=True)

    # ---------------- 2) 失败案例 ----------------
    fail_out = out / "failure_cases"
    fail_out.mkdir(parents=True, exist_ok=True)
    fail_data = {}
    for case, stem in pick_one_per_case("test").items():
        stem = Path(stem).stem  # 去后缀
        img_p = inp / f"{stem}.png"
        bg_p = bg / f"{stem}.png"
        if not img_p.exists() or not bg_p.exists():
            continue
        # 评估指标：用 GT alpha 对比 base vs refined
        gt_a_p = ROOT / "data" / "matting" / "test" / f"{stem}_alpha.png"
        for_model = alpha_dir or (ROOT / "data" / "matting" / "coarse_cache" / "test")
        a_p = for_model / f"{stem}.png"
        analysis = {"case": case, "stem": stem}
        if gt_a_p.exists() and a_p.exists():
            gt = np.array(Image.open(gt_a_p).convert("L")).astype(np.float32) / 255.0
            pr = np.array(Image.open(a_p).convert("L").resize(gt.shape[::-1])) / 255.0
            analysis["mean_abs_err"] = float(np.abs(pr - gt).mean())
            # 边缘带
            edge = (gt > 0.15) & (gt < 0.85)
            if edge.sum() > 0:
                analysis["edge_err"] = float(np.abs(pr[edge] - gt[edge]).mean())
            # 硬区（0/1）准确度
            bin_pred = (pr > 0.5).astype(np.float32)
            bin_gt = (gt > 0.5).astype(np.float32)
            analysis["iou_binary"] = float((bin_pred * bin_gt).sum() /
                                           max(1, (bin_pred + bin_gt - bin_pred * bin_gt).sum()))
        try:
            r = run_pipeline(
                str(img_p), str(bg_p), str(fail_out / f"{case}_{stem}"),
                enable_relight=True, enable_shadow=True, enable_harmonize=True,
                composite_method="alpha_over", device=args.device,
                precomputed_alpha=str(a_p) if a_p.exists() else None,
            )
            analysis["pipeline_output"] = r["final"]
            analysis["total_time_ms"] = sum(s["time_ms"] for s in r["stages"])
        except Exception as e:
            analysis["pipeline_error"] = f"{type(e).__name__}: {e}"
        fail_data[case] = analysis
        print(f"  [fail] {case:<12} mean_err={analysis.get('mean_abs_err','?'):.4f}  "
              f"iou={analysis.get('iou_binary','?')}", flush=True)

    # 归因说明
    NOTES = {
        "hair": "hair 边缘极细（亚像素），512 推理 + refiner 可提升，但极细丝仍易断。",
        "glass": "半透明/反光区域 alpha 本身具有歧义性（labeler 间 IoU 通常 <0.7）。",
        "veil": "半透明薄纱：即使真人标注也难与'原图保留'区分。",
        "transparent": "透明物体在 RGB 几乎无信号，需额外输入（depth / polarisation）。",
        "semi": "半透明：alpha 与色彩耦合，是 matting 的本质困难。",
        "fine_edge": "细密边缘：受推理分辨率 + 上采样插值影响，refiner 残差学习可缓解。",
        "complex_bg": "前景与背景颜色相近时，BiRefNet 倾向于吞掉部分前景。",
        "normal": "常规类：应当几乎完美；仍差时多为合成图本身的 GT 噪声。",
    }
    for c, n in fail_data.items():
        n["note"] = NOTES.get(c, "")

    # ---------------- 报告 + 可视化 ----------------
    report = {"ablation": ablation_data, "failures": fail_data, "notes": NOTES}
    (out / "failure_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[failure] saved report -> {out}/failure_report.json")

    # 消融图：4 个样本 × 4 个配置 = 16 张 grid
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cases_done = [s["case"] for s in ablation_data["samples"]]
        n = len(cases_done)
        fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
        if n == 1:
            axes = axes[None, :]
        for i, sd in enumerate(ablation_data["samples"]):
            for j, cfg in enumerate(ablation_data["configs"]):
                p = ablation_out / cfg / f"{sd['case']}_{sd['stem']}" / "final.png"
                axes[i, j].imshow(Image.open(p).convert("RGB")) if p.exists() else axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(cfg, fontsize=9)
            axes[i, 0].set_ylabel(sd["case"], fontsize=8, rotation=0, labelpad=45)
            for c in range(4):
                axes[i, c].axis("off")
        plt.tight_layout()
        fig.savefig(out / "ablation_grid.png", dpi=85)
        plt.close(fig)

        # 失败案例 grid
        fail_cases = sorted(fail_data.keys())
        nf = len(fail_cases)
        fig, axes = plt.subplots(nf, 4, figsize=(14, 3.0 * nf))
        if nf == 1:
            axes = axes[None, :]
        for i, c in enumerate(fail_cases):
            d = fail_data[c]
            axes[i, 0].imshow(Image.open(inp / f"{d['stem']}.png").convert("RGB"))
            axes[i, 1].imshow(Image.open(ROOT / "data" / "matting" / "test" /
                                          f"{d['stem']}_alpha.png").convert("L"), cmap="gray")
            a_shown = (alpha_dir / f"{d['stem']}.png") if alpha_dir else (
                ROOT / "data" / "matting" / "coarse_cache" / "test" / f"{d['stem']}.png")
            axes[i, 2].imshow(Image.open(a_shown).convert("L"), cmap="gray") if a_shown.exists() else axes[i, 2].axis("off")
            p = Path(d.get("pipeline_output", ""))
            axes[i, 3].imshow(Image.open(p).convert("RGB")) if p.exists() else axes[i, 3].axis("off")
            axes[i, 0].set_ylabel(c, fontsize=8, rotation=0, labelpad=45)
            for k in range(4):
                axes[i, k].axis("off")
            if i == 0:
                for k, t in enumerate(["Input", "GT", "Pred alpha", "Ours composited"]):
                    axes[i, k].set_title(t, fontsize=9)
        plt.tight_layout()
        fig.savefig(out / "failure_grid.png", dpi=85)
        plt.close(fig)
        print(f"[failure] saved {out}/ablation_grid.png, failure_grid.png")
    except Exception as e:
        print(f"[failure] vis skipped: {e}")


if __name__ == "__main__":
    main()
