# -*- coding: utf-8 -*-
"""
coco_synth.py — A: 用 COCO 2017 instances segmentation 生成真实域合成样本

流程:
  1) 读 instances_val2017.json, 按 category 筛可用前景物体 (person/chair/table/bottle...)
  2) 用 polygon segmentation 生成 GT alpha (精确, 非抠图猜测)
  3) 合成到演播室背景 (复用 studio_bg 系列) + 自动软边/光照/接触阴影
  4) 输出 (comp, alpha) 对 + manifest.csv, 供 Refiner 训练与评测

与 studio_dataset.py 的区别: 前景来自**真实照片**(COCO)而非程序化几何体, 域更接近真实。

用法:
  python coco_synth.py --n 600 --size 512,512 --out data/coco_synth
"""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
COCO = ROOT / "data" / "coco_stuff"
BG_DIR = ROOT / "data" / "ai_generated" / "studio_bg"

# 适合做"人或物体前景"的类别 (排除过小/过杂的)
FG_CATS = {
    "person", "chair", "dining table", "bottle", "cup", "bowl", "book",
    "potted plant", "handbag", "backpack", "suitcase", "vase", "wine glass",
    "laptop", "tv", "keyboard", "cell phone", "microwave", "oven", "sink",
    "refrigerator", "clock", "teddy bear", "bicycle", "motorcycle", "car",
    "bench", "umbrella", "dog", "cat", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "surfboard", "skateboard",
    "tennis racket", "baseball bat", "sports ball", "frisbee", "kite",
    "donut", "cake", "pizza", "banana", "apple", "orange", "broccoli",
    "carrot", "sandwich", "hot dog", "toilet", "bed", "couch", "remote",
    "mouse", "toaster", "hair drier", "toothbrush", "scissors", "teddy bear",
    "knife", "spoon", "fork", "tie", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "boat", "airplane", "train", "bus", "truck",
}

BG_FILES = sorted(BG_DIR.glob("bg_*.png"))


# ---------------------------------------------------------------- 标注加载
def load_coco(ann_path: Path, min_area_ratio: float = 0.015, max_objs: int = 6):
    """返回 [{image, w, h, objs:[{cat, poly, area_ratio}]}]"""
    d = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        name = cats[a["category_id"]]
        if name not in FG_CATS or not a.get("segmentation"):
            continue
        if not isinstance(a["segmentation"], list):
            continue
        by_img.setdefault(a["image_id"], []).append(a)

    out = []
    for iid, anns in by_img.items():
        im = imgs[iid]
        W, H = im["width"], im["height"]
        area_img = W * H
        objs = []
        for a in anns:
            x, y, w, h = a["bbox"]
            ratio = (w * h) / area_img
            if ratio < min_area_ratio or ratio > 0.85:
                continue
            poly = a["segmentation"]
            if not poly or len(poly[0]) < 6:
                continue
            objs.append({"cat": cats[a["category_id"]], "poly": poly,
                         "bbox": a["bbox"], "area": a["area"]})
        if not objs:
            continue
        # 优先大物体, 最多 max_objs 个
        objs.sort(key=lambda o: -o["area"])
        out.append({"id": iid, "file_name": im["file_name"], "W": W, "H": H,
                    "objs": objs[:max_objs]})
    return out


def poly_to_alpha(objs: list, W: int, H: int, blur: float = 1.2) -> np.ndarray:
    """多边形 → 软边 alpha (0-255 uint8)。"""
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    for o in objs:
        for poly in o["poly"]:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 3:
                dr.polygon(pts, fill=255)
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    return np.array(m, dtype=np.uint8)


# ---------------------------------------------------------------- 合成
def make_contact_shadow(sub_a, pad: int):
    """由 alpha (PIL L 图) 生成接触阴影 (下方偏移 + 模糊)。返回 float32 (0-1)。"""
    a = np.array(sub_a, np.float32) / 255.0
    H, W = a.shape
    sh = np.zeros((H + pad * 2, W + pad * 2), np.float32)
    sh[pad * 2:, pad:pad + W] += a * 0.55            # 下移偏移
    sh = np.array(Image.fromarray((np.clip(sh, 0, 1) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(8, pad * 1.5))), np.float32) / 255.0
    return sh


def composite_one(item: dict, img_dir: Path, bg_path: Path, rng: random.Random,
                  out_size: int = 512, with_shadow: bool = True,
                  with_harmonize: bool = True) -> dict | None:
    """把 COCO 图中选中物体抠出, 缩放/移动后合成到演播室背景。"""
    ip = img_dir / item["file_name"]
    if not ip.exists():
        return None
    src = Image.open(ip).convert("RGB")
    W0, H0 = src.size
    alpha_full = poly_to_alpha(item["objs"], W0, H0, blur=1.0)
    if (alpha_full > 127).sum() < (W0 * H0 * 0.01):
        return None

    # 裁到前景包围盒 (带 margin)
    ys, xs = np.where(alpha_full > 127)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    mw, mh = int((x1 - x0) * 0.06) + 8, int((y1 - y0) * 0.06) + 8
    x0, y0 = max(0, x0 - mw), max(0, y0 - mh)
    x1, y1 = min(W0, x1 + mw), min(H0, y1 + mh)
    fg = src.crop((x0, y0, x1, y1))
    fa = Image.fromarray(alpha_full[y0:y1, x0:x1])
    if fa.filter(ImageFilter.GaussianBlur(0.8)):
        pass
    fw, fh = fg.size

    # 背景
    bg = Image.open(bg_path).convert("RGB").resize((out_size, out_size), Image.LANCZOS)

    # 前景缩放: 占画布高度 0.35~0.92
    scale = rng.uniform(0.35, 0.92) * out_size / fh
    nw, nh = max(16, int(fw * scale)), max(16, int(fh * scale))
    # 不要超过画布
    if nw > out_size * 0.98:
        r = out_size * 0.98 / nw
        nw, nh = int(nw * r), int(nh * r)
    fg_s = fg.resize((nw, nh), Image.LANCZOS)
    fa_s = fa.resize((nw, nh), Image.LANCZOS)

    # ---- 光照对齐 (关键): 把前景亮度向背景靠拢, 避免 FDR 越界/前景贴纸感 ----
    # 背景亮度统计 (排除将要被覆盖区, 简化为全背景中位)
    bg_gray = np.array(bg.convert("L"), np.float32)
    bg_lum = float(np.median(bg_gray))
    fg_arr = np.array(fg_s, np.float32)
    fa_np = np.array(fa_s, np.float32) / 255.0
    fg_mask = fa_np > 0.5
    fg_lum = float(np.median(fg_arr[fg_mask])) if fg_mask.sum() > 50 else bg_lum
    # 目标亮度: 背景亮度 * 0.75~1.25 随机 (保留合理差异, 但不失控)
    target = bg_lum * rng.uniform(0.85, 1.20)
    gain = np.clip(target / max(fg_lum, 1.0), 0.55, 1.6)
    fg_arr = fg_arr * gain
    # 把背景环境色(色温)轻微渗入前景, 提升融合感
    bg_mean = np.array(bg, np.float32).reshape(-1, 3).mean(0)
    fg_mean = fg_arr[fg_mask].mean(0) if fg_mask.sum() > 50 else bg_mean
    tint = np.clip(bg_mean / np.maximum(fg_mean, 1.0), 0.9, 1.1)
    tint_strength = rng.uniform(0.25, 0.55)
    fg_arr = fg_arr * (1 + (tint - 1) * tint_strength)
    if with_harmonize:      # 轻微色彩抖动, 模拟不同来源
        fg_arr = fg_arr * rng.uniform(0.96, 1.04)
    fg_s = Image.fromarray(np.clip(fg_arr, 0, 255).astype(np.uint8))

    # 位置: 底部对齐 (站姿) 或随机
    place = rng.choice(["bottom", "bottom", "center", "random"])
    if place == "bottom":
        px = rng.randint(int(-nw * 0.05), max(0, out_size - int(nw * 0.95)))
        py = out_size - nh - rng.randint(0, max(1, int(out_size * 0.08)))
    elif place == "center":
        px, py = (out_size - nw) // 2, (out_size - nh) // 2
    else:
        px = rng.randint(0, max(0, out_size - nw))
        py = rng.randint(0, max(0, out_size - nh))

    canvas = bg.copy()
    pa = np.array(fa_s, np.float32) / 255.0

    # 接触阴影
    if with_shadow:
        pad = max(10, out_size // 32)
        sh = make_contact_shadow(fa_s, pad)
        sh_img = Image.fromarray((sh * 255).astype(np.uint8))
        sx, sy = px - pad, py - pad * 2
        bx0, by0 = max(0, -sx), max(0, -sy)
        bx1, by1 = min(sh_img.width, out_size - sx), min(sh_img.height, out_size - sy)
        if bx1 > bx0 and by1 > by0:
            canvas = np.array(canvas, np.float32)
            region = canvas[max(0, sy):max(0, sy) + (by1 - by0), max(0, sx):max(0, sx) + (bx1 - bx0)]
            sm = np.array(sh_img.crop((bx0, by0, bx1, by1)), np.float32) / 255.0
            if region.shape[:2] == sm.shape:
                canvas[max(0, sy):max(0, sy) + sm.shape[0],
                       max(0, sx):max(0, sx) + sm.shape[1]] = region * (1 - sm[..., None] * 0.75)
            canvas = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    # 前景 alpha 合成 (裁剪越界)
    cx0, cy0 = max(0, px), max(0, py)
    cx1, cy1 = min(out_size, px + nw), min(out_size, py + nh)
    fx0, fy0 = cx0 - px, cy0 - py
    fx1, fy1 = fx0 + (cx1 - cx0), fy0 + (cy1 - cy0)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    sub = np.array(fg_s.crop((fx0, fy0, fx1, fy1)), np.float32)
    sa = pa[fy0:fy1, fx0:fx1]
    cvs = np.array(canvas, np.float32)
    reg = cvs[cy0:cy1, cx0:cx1]
    cvs[cy0:cy1, cx0:cx1] = sub * sa[..., None] + reg * (1 - sa[..., None])
    comp = Image.fromarray(np.clip(cvs, 0, 255).astype(np.uint8))

    # 全图 alpha (用于监督)
    full_a = np.zeros((out_size, out_size), np.uint8)
    full_a[cy0:cy1, cx0:cx1] = (np.clip(sa, 0, 1) * 255).astype(np.uint8)

    return {"comp": comp, "alpha": full_a,
            "cats": "+".join(sorted({o["cat"] for o in item["objs"]})),
            "n_obj": len(item["objs"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(COCO / "annotations" / "instances_val2017.json"))
    ap.add_argument("--img_dir", default=str(COCO / "val2017"))
    ap.add_argument("--out", default=str(ROOT / "data" / "coco_synth"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    out_root = Path(args.out)
    img_dir = Path(args.img_dir)

    items = load_coco(Path(args.ann))
    print(f"[coco_synth] 可用图片 {len(items)} 张", flush=True)
    rng.shuffle(items)
    if not BG_FILES:
        raise SystemExit("找不到演播室背景素材")

    manifest, ok, i = [], 0, 0
    while ok < args.n and i < len(items) * 3:
        it = items[i % len(items)]
        i += 1
        bg = rng.choice(BG_FILES)
        r = composite_one(it, img_dir, bg, rng, out_size=args.size,
                          with_shadow=rng.random() < 0.8,
                          with_harmonize=rng.random() < 0.7)
        if not r:
            continue
        # split 划分 80/10/10
        frac = ok / max(1, args.n)
        split = "train" if frac < 0.8 else ("val" if frac < 0.9 else "test")
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        name = f"coco_{ok:05d}"
        r["comp"].save(d / f"{name}.png")
        Image.fromarray(r["alpha"]).save(d / f"{name}_alpha.png")
        manifest.append({"image": f"{name}.png", "alpha": f"{name}_alpha.png",
                         "split": split, "cats": r["cats"], "n_obj": r["n_obj"],
                         "bg": bg.name, "src": it["file_name"]})
        ok += 1
        if ok % 100 == 0:
            print(f"[coco_synth] {ok}/{args.n}", flush=True)

    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "alpha", "split", "cats", "n_obj", "bg", "src"])
        w.writeheader()
        w.writerows(manifest)
    print(f"[coco_synth] 完成 {ok} 张 -> {out_root}")


if __name__ == "__main__":
    main()
