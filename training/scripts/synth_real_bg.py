# -*- coding: utf-8 -*-
"""
synth_real_bg.py — 训练方案 §4.3 背景合成增广（修 P3M-NP 锐利背景 OOD）

背景: v1 训练集三种背景全为 虚化(P3M)/绿幕(HM)/演播室(studio)，P3M-500-NP
      (锐利自然背景) 上精修 SAD +60%。本脚本把真实前景(以 GT alpha 提取)
      重合成到 COCO 实景照片(锐利自然背景)上, 补齐该域。

做法: I' = a*I + (1-a)*bg'  (GT alpha 不变, 仍为精确监督; 仅替换背景区)
      bg 来自 data/coco_assets/bg (120 张实景), 亮度/模糊抖动, JPEG 落盘带自然压缩
      BiRefNet@512 对新合成图现算 coarse (真实误差分布, 不复用旧 coarse)
产出: data/matting_real/train/syn_{src}_{stem}.jpg + _alpha.png
      data/matting_real/coarse_cache/train/syn_{src}_{stem}.png
      manifest.csv 追加 (case=syn)  —— 幂等, 重跑跳过已存在

用法: python synth_real_bg.py [--n_hm 1100] [--n_p3m 700]
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
BG_DIR = ROOT / "data" / "coco_assets" / "bg"

rng = random.Random(123)


def cover_resize(bg: Image.Image, W: int, H: int) -> Image.Image:
    """裁切填充到 WxH (保持比例, 中心裁)。"""
    s = max(W / bg.width, H / bg.height)
    nw, nh = int(bg.width * s) + 1, int(bg.height * s) + 1
    b = bg.resize((nw, nh), Image.BILINEAR)
    x0 = rng.randint(0, nw - W)
    y0 = rng.randint(0, nh - H)
    return b.crop((x0, y0, x0 + W, y0 + H))


def synth_one(img: np.ndarray, alpha: np.ndarray, bg_pool) -> np.ndarray:
    H, W = alpha.shape
    bg = Image.open(rng.choice(bg_pool)).convert("RGB")
    bg = cover_resize(bg, W, H)
    bg = np.array(bg, np.float32)
    if rng.random() < 0.7:                                   # 亮度抖动
        bg *= rng.uniform(0.75, 1.25)
    if rng.random() < 0.3:                                   # 少量轻微虚化 (混合景深)
        import cv2
        bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.5, 1.5))
    a = alpha[..., None]
    comp = a * img.astype(np.float32) + (1 - a) * bg         # 仅替换背景区
    return np.clip(comp + np.random.normal(0, 1.5, comp.shape), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hm", type=int, default=1100)
    ap.add_argument("--n_p3m", type=int, default=700)
    ap.add_argument("--infer_size", type=int, default=512)
    args = ap.parse_args()

    tr = DATA / "train"
    mf = DATA / "manifest.csv"
    with open(mf, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    have = {r[0] for r in rows[1:]}

    # 候选: train 里已有 GT alpha 的 p3m / hm 项
    p3m = sorted(tr.glob("p3m_*.*"))
    p3m = [p for p in p3m if not p.stem.endswith("_alpha")]
    hm = sorted(tr.glob("hm_*.jpg"))
    rng.shuffle(p3m)
    rng.shuffle(hm)
    picks = [("p3m", p) for p in p3m[:args.n_p3m]] + [("hm", p) for p in hm[:args.n_hm]]
    bg_pool = sorted(str(p) for p in BG_DIR.glob("*.jpg"))
    assert bg_pool, "coco_assets/bg 为空"
    print(f"[synth] fg={len(picks)} (p3m={min(args.n_p3m, len(p3m))} hm={min(args.n_hm, len(hm))}) "
          f"bg_pool={len(bg_pool)}", flush=True)

    new_rows, n_new = [], 0
    t0 = time.time()
    for src, ip in picks:
        stem = f"syn_{src}_{ip.stem}"
        out_img = tr / f"{stem}{ip.suffix}"
        out_gt = tr / f"{stem}_alpha.png"
        if out_img.exists() and out_gt.exists() and stem in have:
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = np.array(Image.open(tr / f"{ip.stem}_alpha.png").convert("L"))
        comp = synth_one(img, gt.astype(np.float32) / 255.0, bg_pool)
        Image.fromarray(comp).save(out_img, quality=92)      # JPEG 自带轻微压缩自然感
        Image.fromarray(gt).save(out_gt)
        new_rows.append([f"{stem}{ip.suffix}", "train", "syn"])
        n_new += 1
        if n_new % 200 == 0:
            print(f"  composited {n_new}/{len(picks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[synth] composited {n_new} new pairs ({time.time()-t0:.0f}s)", flush=True)

    # BiRefNet coarse (只对新图)
    if new_rows:
        from matting.matting_backend import BiRefNetMatting
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        tool = BiRefNetMatting(device=device, seg_mask_size=args.infer_size)
        cc = DATA / "coarse_cache" / "train"
        cc.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, r in enumerate(new_rows):
            stem = Path(r[0]).stem
            cp = cc / f"{stem}.png"
            if cp.exists():
                continue
            tool.predict(str(tr / r[0]), str(cp))
            if (i + 1) % 200 == 0:
                print(f"  coarse {i+1}/{len(new_rows)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[synth] coarse done ({time.time()-t0:.0f}s)", flush=True)
        # manifest 追加
        with open(mf, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(new_rows)
        names = [r[0] for r in new_rows]
        with open(DATA / "train_names.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(names))
        print(f"[synth] manifest +{len(new_rows)} (case=syn)", flush=True)
    print("[synth] DONE", flush=True)


if __name__ == "__main__":
    main()
