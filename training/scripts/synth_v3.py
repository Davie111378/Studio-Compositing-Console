# -*- coding: utf-8 -*-
"""
synth_v3.py — v3 训练数据增强（修"深色衣服被挖"carve 行为）

背景: v2 在 P3M-NP 上仍把深色服装从前景中移除。诊断结论:
  1) v2 合成集的 HM fg 未 despill → 绿边伪影教出坏边缘先验;
  2) 真实衣服 on 锐利背景 的样本只有 700 (P3M源), 被 1100 绿幕源淹没;
  3) coarse 误差方向单一 (BiRefNet 在各域的固定偏见) → 模型学成
     "信任/不信任 coarse"的启发式, 而非真实边界语义。

产出 (全部进 train/, manifest 追加):
  A) sy3_{src}_{stem}: HM(本次 despill) 700 + P3M(真实衣服) 800 → COCO 实景背景
     coarse 由 BiRefNet 现算
  B) sy2_{stem}: 抽 1800 个既有 train 项, 图/GT 复用(硬链接), coarse 换成
     随机扰动版 (dilate/erode/blur/噪声/阈值偏移) → 教"任意 coarse 误差下回归真边界"

用法: python synth_v3.py [--n_hm 700] [--n_p3m 800] [--n_perturb 1800]
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

DATA = ROOT / "data" / "matting_real"
TR = DATA / "train"
BG_DIR = ROOT / "data" / "coco_assets" / "bg"

rng = random.Random(777)


def cover_resize(bg: Image.Image, W: int, H: int) -> Image.Image:
    s = max(W / bg.width, H / bg.height)
    b = bg.resize((max(1, int(bg.width * s) + 1), max(1, int(bg.height * s) + 1)), Image.BILINEAR)
    x0, y0 = rng.randint(0, b.width - W), rng.randint(0, b.height - H)
    return b.crop((x0, y0, x0 + W, y0 + H))


def synth_comp(img: np.ndarray, alpha: np.ndarray, bg_pool) -> np.ndarray:
    H, W = alpha.shape
    bg = cover_resize(Image.open(rng.choice(bg_pool)).convert("RGB"), W, H).astype(np.float32)
    if rng.random() < 0.7:
        bg *= rng.uniform(0.75, 1.25)
    if rng.random() < 0.3:
        import cv2
        bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.5, 1.5))
    a = alpha[..., None]
    comp = a * img.astype(np.float32) + (1 - a) * bg
    return np.clip(comp + np.random.normal(0, 1.5, comp.shape), 0, 255).astype(np.uint8)


def _link_or_copy(src: Path, dst: Path) -> None:
    """硬链接优先(零拷贝), 跨卷/已存在时退化为复制。"""
    import os
    import shutil
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_gt(p: Path) -> np.ndarray:
    """GT 读取: RGBA/LA 取 alpha 通道 (HM matting 为前景抠图), 其余转灰度。"""
    im = Image.open(p)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        return np.array(im.convert("RGBA"))[..., 3]
    return np.array(im.convert("L"))


def perturb_coarse(co: np.ndarray) -> np.ndarray:
    """模拟任意方向的 coarse 误差 (bool->uint8 0/255)。"""
    import cv2
    a = (co > 127).astype(np.uint8)
    r = rng.random()
    if r < 0.45:      # 膨胀: 模拟 over-inclusion (把背景并进来)
        k = rng.choice([3, 5, 9, 15, 25])
        a = cv2.dilate(a, np.ones((k, k), np.uint8))
    elif r < 0.7:     # 侵蚀: 模拟 under-inclusion (丢头发/丢衣物)
        k = rng.choice([3, 5, 9])
        a = cv2.erode(a, np.ones((k, k), np.uint8))
    f = a.astype(np.float32)
    if rng.random() < 0.6:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(2, 8))
    f = np.clip(f + np.random.normal(0, rng.uniform(0.05, 0.2), f.shape), 0, 1)
    f = np.clip(f + (co.astype(np.float32) / 255. - f) * rng.uniform(0, 0.3), 0, 1)  # 混一点原 coarse
    return (f * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hm", type=int, default=700)
    ap.add_argument("--n_p3m", type=int, default=800)
    ap.add_argument("--n_perturb", type=int, default=1800)
    ap.add_argument("--infer_size", type=int, default=512)
    args = ap.parse_args()

    mf = DATA / "manifest.csv"
    with open(mf, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f)][1:]
    have = {r[0] for r in rows}

    # ---- A) despill 合成 ----
    from ai_material_eval import despill
    hm = sorted(TR.glob("hm_*.jpg"))
    p3m = [p for p in sorted(TR.glob("p3m_*.*")) if not p.stem.endswith("_alpha")]
    used_hm = {p.stem for p in TR.glob("syn_hm_*.jpg")}
    used_p3m = {p.stem.replace("syn_p3m_", "") for p in TR.glob("syn_p3m_*.jpg")}
    hm_pool = [p for p in hm if p.stem.replace("hm_", "") not in
               {s.replace("syn_hm_", "") for s in used_hm}]
    p3m_pool = [p for p in p3m if p.stem not in used_p3m]
    rng.shuffle(hm_pool)
    rng.shuffle(p3m_pool)
    picks = [("hm", p) for p in hm_pool[:args.n_hm]] + [("p3m", p) for p in p3m_pool[:args.n_p3m]]
    bg_pool = sorted(str(p) for p in BG_DIR.glob("*.jpg"))
    print(f"[v3A] despill合成: hm={min(args.n_hm, len(hm_pool))} p3m={min(args.n_p3m, len(p3m_pool))}", flush=True)

    new_a = []
    t0 = time.time()
    for src, ip in picks:
        stem = f"sy3_{src}_{ip.stem}"
        if f"{stem}{ip.suffix}" in have:
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = load_gt(TR / f"{ip.stem}_alpha.png")
        if src == "hm":                                   # 绿幕源压绿边 (v2 教训)
            img = despill(img, gt.astype(np.float32) / 255.)
        comp = synth_comp(img, gt.astype(np.float32) / 255., bg_pool)
        Image.fromarray(comp).save(TR / f"{stem}{ip.suffix}", quality=92)
        Image.fromarray(gt).save(TR / f"{stem}_alpha.png")
        new_a.append([f"{stem}{ip.suffix}", "train", "syn"])
        if len(new_a) % 200 == 0:
            print(f"  A {len(new_a)}/{len(picks)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- B) coarse 扰动变体 ----
    cands = [r[0] for r in rows if r[2] in ("p3m", "hm", "syn")]
    rng.shuffle(cands)
    picks_b = cands[:args.n_perturb]
    print(f"[v3B] 扰动变体: {len(picks_b)}", flush=True)
    new_b = []
    for img_name in picks_b:
        ip = TR / img_name
        stem = f"sy2_{Path(img_name).stem}"
        ext = ip.suffix
        if f"{stem}{ext}" in have:
            continue
        co_src = DATA / "coarse_cache" / "train" / f"{Path(img_name).stem}.png"
        if not co_src.exists() or not (TR / f"{Path(img_name).stem}_alpha.png").exists():
            continue
        # 图/GT 复用: 硬链接 (零拷贝, 不额外占盘)
        dst_i, dst_g = TR / f"{stem}{ext}", TR / f"{stem}_alpha.png"
        _link_or_copy(ip, dst_i)
        _link_or_copy(TR / f"{Path(img_name).stem}_alpha.png", dst_g)
        co = np.array(Image.open(co_src).convert("L"))
        Image.fromarray(perturb_coarse(co)).save(DATA / "coarse_cache" / "train" / f"{stem}.png")
        new_b.append([f"{stem}{ext}", "train", "syn"])
        if len(new_b) % 400 == 0:
            print(f"  B {len(new_b)}/{len(picks_b)}", flush=True)

    # ---- coarse (仅 A 的新图需要 BiRefNet) ----
    if new_a:
        from matting.matting_backend import BiRefNetMatting
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        tool = BiRefNetMatting(device=device, seg_mask_size=args.infer_size)
        cc = DATA / "coarse_cache" / "train"
        t0 = time.time()
        for i, r in enumerate(new_a):
            stem = Path(r[0]).stem
            cp = cc / f"{stem}.png"
            if cp.exists():
                continue
            tool.predict(str(TR / r[0]), str(cp))
            if (i + 1) % 200 == 0:
                print(f"  coarse {i+1}/{len(new_a)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- manifest ----
    new_rows = new_a + new_b
    if new_rows:
        with open(mf, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(new_rows)
        with open(DATA / "train_names.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(r[0] for r in new_rows))
    print(f"[v3] +A={len(new_a)} +B={len(new_b)} manifest 更新完毕", flush=True)


if __name__ == "__main__":
    main()
