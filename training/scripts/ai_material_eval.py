# -*- coding: utf-8 -*-
"""
AI 素材评测集构建 + 现有模型跨域检查（2026-09-09 新实验线）

流程:
  1) 绿幕人像预处理: 修复右下角"AI生成"水印(用绿幕中值色填充)
  2) 色度键提取 GT alpha (软阈值 + 最大连通域清理 + 闭运算填洞)
  3) BiRefNet 粗 alpha (base)  与  BiRefNet + StudioRefiner (refined) 双路推理
  4) 四指标 SAD/MSE/Grad/Conn 对比 base vs refined (以色度键 GT 为参照)
  5) 输出 4x4 对比网格 + metrics JSON

用法:
  python ai_material_eval.py --green_dir data/ai_generated/green_fg \
      --out_dir experiments/ai_materials --device cuda
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

from evaluate_matting import sad, mse, gradient_loss, connectivity_loss  # 复用四指标

# ---------- 1. 水印修复 ----------
WM_BOX = None  # (r0, c0) 若右下角矩形以绿幕为主则记录, 供 GT 兜底清零

def fix_watermark(img: np.ndarray, margin_r: int = 340, margin_b: int = 150) -> np.ndarray:
    """右下角水印修复:
    - 矩形内绿幕占比 > 50%  => 整个矩形填全图绿幕中值色 (水印整体消失, 记录 WM_BOX)
    - 否则 (前景为主, 如台面) => 只替换绿色占优像素, 前景不动"""
    global WM_BOX
    h, w = img.shape[:2]
    r0, c0 = h - margin_b, w - margin_r
    f = img.astype(np.float32)
    excess = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    green_all = excess > 25
    if green_all.sum() < 100:
        return img
    med = np.median(img[green_all], axis=0).astype(np.uint8)
    if green_all[r0:, c0:].mean() > 0.5:
        img[r0:, c0:] = med
        WM_BOX = (r0, c0)
    else:
        m = excess[r0:, c0:] > 25
        img[r0:, c0:][m] = med
        WM_BOX = None
    return img

# ---------- 2. 色度键 GT ----------
def chroma_key_alpha(img: np.ndarray) -> np.ndarray:
    """绿色优势度软阈值色度键; 返回 float32 alpha (0-1)。"""
    r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    excess = g - np.maximum(r, b)
    alpha = np.clip((60.0 - excess) / (60.0 - 15.0), 0, 1)
    # 自适应: 若整图绿幕偏暗, 阈值下移
    bg_excess = np.median(excess[excess > 30]) if (excess > 30).sum() > 100 else 60.0
    if bg_excess < 40:
        alpha = np.clip((bg_excess * 0.75 - excess) / max(bg_excess * 0.75 - 8.0, 1.0), 0, 1)
    # 背景清理: 所有非前景连通域(不与最大前景域重叠的 alpha>0.1 区域)全部清零
    try:
        from scipy.ndimage import label, binary_closing, binary_fill_holes
        solid = alpha > 0.5
        lab, n = label(solid)
        if n >= 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            keep = sizes.argmax()
        else:
            keep = 0
        fg_any = alpha > 0.1
        lab2, n2 = label(fg_any)
        keep_ids = set(np.unique(lab2[solid & (lab == keep)])) if keep else set()
        keep_ids.discard(0)
        kill = fg_any & ~np.isin(lab2, list(keep_ids)) if keep_ids else fg_any & (lab2 > 0) & (keep == 0)
        alpha[kill] = 0.0
        # 水印兜底: 右下角矩形(绿幕为主) alpha 强制归零
        if WM_BOX is not None:
            r0, c0 = WM_BOX
            alpha[r0:, c0:] = 0.0
        solid2 = binary_closing(alpha > 0.2, np.ones((5, 5), bool))
        holes = binary_fill_holes(solid2) & ~solid2
        alpha[holes] = 1.0
    except ImportError:
        pass
    return alpha

def despill(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """绿幕溢出压回: g 超出 max(r,b) 的部分压平。"""
    out = img.astype(np.float32).copy()
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    spill = np.clip(g - np.maximum(r, b), 0, None) * (alpha > 0.05)
    out[..., 1] = g - spill
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- 3. 模型推理 ----------
def infer_coarse(tool, img_path: Path, out_path: Path, infer_size: int):
    tool.predict(str(img_path), str(out_path), original_size=True)

def infer_refined(net, device, img_path: Path, coarse_path: Path, out_path: Path,
                  refine_size: int, guard: float = 0.05):
    """精修 + 置信门控: refined 与 coarse 平均偏差超 guard 则回退 coarse (防越修越坏)。
    返回 (fallback: bool)。"""
    import torch
    img = Image.open(img_path).convert("RGB")
    co = Image.open(coarse_path).convert("L").resize((refine_size, refine_size), Image.BILINEAR)
    rgb = img.resize((refine_size, refine_size), Image.BILINEAR)
    rgb_t = torch.from_numpy(np.array(rgb).astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    co_t = torch.from_numpy(np.array(co).astype("float32") / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = net(rgb_t, co_t)
    a = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    a_full = np.array(Image.fromarray(a).resize(img.size, Image.BILINEAR))
    co_full = np.array(Image.open(coarse_path).convert("L").resize(img.size, Image.BILINEAR))
    fallback = False
    # 门控信号: 大变化像素占比 (劣化时 refined 会在局部区域大幅偏离 coarse,
    # 全图均值被稀释; >5% 像素变化超过 0.2 即判定不稳定, 回退 coarse)
    big_ratio = float((np.abs(a_full.astype(np.float32) - co_full.astype(np.float32)) > 51).mean())
    if guard and big_ratio > guard:
        a_full = co_full
        fallback = True
    Image.fromarray(a_full).save(out_path)
    return fallback

# ---------- 5. 网格 ----------
def make_grid(rows, out_path: Path):
    """rows: list of [input, GT, coarse, refined] uint8 arrays (同尺寸)。"""
    pad = 8
    h, w = rows[0][0].shape[:2]
    canvas = np.full((rows.__len__() * (h + pad) + pad, 4 * (w + pad) + pad, 3), 30, np.uint8)
    titles = ["Input", "GT(chroma)", "Coarse(BiRefNet)", "Refined(Studio)"]
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            y, x = pad + i * (h + pad), pad + j * (w + pad)
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            canvas[y:y + h, x:x + w] = im
    Image.fromarray(canvas).save(out_path)
    print(f"[grid] wrote {out_path}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--green_dir", default=str(ROOT / "data" / "ai_generated" / "green_fg"))
    ap.add_argument("--out_dir", default=str(ROOT / "experiments" / "ai_materials"))
    ap.add_argument("--ckpt", default=str(ROOT / "training" / "checkpoints" / "refiner_studio" / "best.pt"))
    ap.add_argument("--infer_size", type=int, default=512)
    ap.add_argument("--refine_size", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    import torch
    from matting.matting_backend import BiRefNetMatting
    from train_refiner import AlphaRefiner

    out = Path(args.out_dir)
    for sub in ("gt", "coarse", "refined", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # 加载两路模型
    print(f"[load] BiRefNet @512 device={device}")
    tool = BiRefNetMatting(device=device, seg_mask_size=args.infer_size)
    net = None
    ck_p = Path(args.ckpt)
    if ck_p.exists():
        net = AlphaRefiner(base=32).to(device)
        ck = torch.load(ck_p, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"]); net.eval()
        print(f"[load] StudioRefiner from {ck_p.name} (epoch={ck.get('epoch')}, best={ck.get('best'):.5f})")
    else:
        print(f"[warn] refiner 权重不存在, 仅评 base: {ck_p}")

    greens = sorted(Path(args.green_dir).glob("*.png"))
    rows, metrics, n_fallback = [], {}, []
    for p in greens:
        stem = p.stem
        t0 = time.time()
        img = np.array(Image.open(p).convert("RGB"))
        img = fix_watermark(img)
        gt = chroma_key_alpha(img)

        # 落盘: 干净输入(水印已修) + GT
        clean_p = out / "preview" / f"{stem}_clean.png"
        Image.fromarray(img).save(clean_p)
        Image.fromarray((gt * 255).astype(np.uint8)).save(out / "gt" / f"{stem}.png")

        # base 粗抠
        co_p = out / "coarse" / f"{stem}.png"
        infer_coarse(tool, clean_p, co_p, args.infer_size)
        # refined 精修
        if net is not None:
            re_p = out / "refined" / f"{stem}.png"
            n_fallback.append(infer_refined(net, device, clean_p, co_p, re_p, args.refine_size))

        # 指标
        def L(path):
            return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        m = {"SAD": sad(L(co_p), gt), "MSE": mse(L(co_p), gt),
             "Grad": gradient_loss(L(co_p), gt), "Conn": connectivity_loss(L(co_p), gt)}
        if net is not None:
            m2 = {"SAD": sad(L(re_p), gt), "MSE": mse(L(re_p), gt),
                  "Grad": gradient_loss(L(re_p), gt), "Conn": connectivity_loss(L(re_p), gt)}
        fg_ratio = float((gt > 0.5).mean())
        metrics[stem] = {"coarse": m, "refined": (m2 if net is not None else None),
                         "fg_ratio": round(fg_ratio, 3), "sec": round(time.time() - t0, 1)}
        print(f"[{stem}] fg={fg_ratio*100:.0f}%  coarse SAD={m['SAD']:.0f}"
              + (f"  refined SAD={m2['SAD']:.0f}" if net is not None else ""))

        # 网格行: 输入 | GT | coarse | refined
        row = [np.array(Image.open(clean_p).convert("RGB").resize((512, 512))),
               np.array(Image.open(out / "gt" / f"{stem}.png").convert("L").resize((512, 512))),
               np.array(Image.open(co_p).convert("L").resize((512, 512)))]
        if net is not None:
            row.append(np.array(Image.open(re_p).convert("L").resize((512, 512))))
        else:
            row.append(row[-1])
        rows.append(row)

    make_grid(rows, out / "matting_grid.png")
    (out / "matting_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # 汇总
    def mean_of(key):
        vals = [metrics[s][key] for s in metrics if metrics[s][key]]
        return {k: round(float(np.mean([v[k] for v in vals])), 4) for k in ("SAD", "MSE", "Grad", "Conn")}
    summary = {"n": len(metrics), "guard_fallbacks": int(sum(n_fallback)), "coarse_MEAN": mean_of("coarse")}
    if net is not None:
        summary["refined_MEAN"] = mean_of("refined")
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
