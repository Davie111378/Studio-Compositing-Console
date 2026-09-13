# -*- coding: utf-8 -*-
"""
coco_assets.py — B: 从 COCO 2017 抽取可用作背景/前景的素材, 建立中文语义索引。

产物:
  data/coco_assets/bg/     —— 适合做"背景"的图 (风景/室内/街道/自然, 无明显主体)
  data/coco_assets/fg/     —— 适合做"前景"的抠出物体 (透明底 PNG, 来自 segmentation)
  data/coco_assets/assets.json —— 索引: [{file, kind, cn, cat, w, h, src}]

中文索引用于 agent 的语义检索 (assets.py 的 find_background / 前景挑选)。

用法:
  python coco_assets.py --n_bg 120 --n_fg 200
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COCO = ROOT / "data" / "coco_stuff"

# 场景类类别 → 可作"背景" (取自 COCO 场景相关类别)
SCENE_CATS = {
    "street", "sidewalk", "building", "sky", "tree", "grass", "field",
    "mountain", "water", "river", "lake", "sea", "ocean", "beach", "sand",
    "snow", "forest", "hill", "road", "bridge", "railing", "fence",
    "window", "wall", "floor", "ceiling", "house", "room", "table",
    "curtain", "shelf", "counter", "platform", "pier", "dock",
}

# 类别 → 中文描述 (前景物体)
CAT_CN = {
    "person": "人物", "chair": "椅子", "dining table": "桌子", "bottle": "瓶子",
    "cup": "杯子", "bowl": "碗", "book": "书本", "potted plant": "盆栽",
    "handbag": "手提包", "backpack": "双肩包", "suitcase": "行李箱",
    "wine glass": "高脚杯", "laptop": "笔记本电脑", "tv": "电视",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "sink": "水槽", "refrigerator": "冰箱", "clock": "时钟",
    "teddy bear": "毛绒玩具", "bicycle": "自行车", "motorcycle": "摩托车",
    "car": "汽车", "bench": "长椅", "umbrella": "雨伞", "dog": "狗",
    "cat": "猫", "bird": "鸟", "horse": "马", "sheep": "绵羊", "cow": "牛",
    "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿",
    "surfboard": "冲浪板", "skateboard": "滑板", "tennis racket": "网球拍",
    "baseball bat": "棒球棒", "sports ball": "运动球", "frisbee": "飞盘",
    "kite": "风筝", "donut": "甜甜圈", "cake": "蛋糕", "pizza": "披萨",
    "banana": "香蕉", "apple": "苹果", "orange": "橙子", "broccoli": "西兰花",
    "carrot": "胡萝卜", "sandwich": "三明治", "hot dog": "热狗",
    "toilet": "马桶", "bed": "床", "couch": "沙发", "remote": "遥控器",
    "mouse": "鼠标", "toaster": "烤面包机", "hair drier": "吹风机",
    "toothbrush": "牙刷", "scissors": "剪刀", "knife": "刀", "spoon": "勺子",
    "fork": "叉子", "tie": "领带", "traffic light": "红绿灯",
    "fire hydrant": "消防栓", "stop sign": "停车牌", "parking meter": "停车计费器",
    "boat": "船", "airplane": "飞机", "train": "火车", "bus": "公交车",
    "truck": "卡车", "vase": "花瓶",
}

FG_CATS = set(CAT_CN.keys())


def load(ann_path: Path):
    d = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    return d, cats, imgs


def poly_mask(seg, W, H) -> np.ndarray:
    from PIL import ImageDraw
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    for poly in seg:
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        if len(pts) >= 3:
            dr.polygon(pts, fill=255)
    return np.array(m, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(COCO / "annotations" / "instances_val2017.json"))
    ap.add_argument("--img_dir", default=str(COCO / "val2017"))
    ap.add_argument("--out", default=str(ROOT / "data" / "coco_assets"))
    ap.add_argument("--n_bg", type=int, default=120)
    ap.add_argument("--n_fg", type=int, default=200)
    ap.add_argument("--min_cat_count", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "bg").mkdir(parents=True, exist_ok=True)
    (out / "fg").mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    d, cats, imgs = load(Path(args.ann))

    # ---- 按图片聚合: 每图有哪些类 ----
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        if a.get("iscrowd") or not isinstance(a.get("segmentation"), list):
            continue
        by_img.setdefault(a["image_id"], []).append(a)

    import collections
    cat_count = collections.Counter()
    for a in d["annotations"]:
        cat_count[cats[a["category_id"]]] += 1

    index = []

    # ---- 前景: 每个"足够常见"的类别抽几张 ----
    per_cat_target = max(2, args.n_fg // max(1, len([c for c in FG_CATS if cat_count[c] >= args.min_cat_count])))
    taken = collections.Counter()
    img_ids = list(by_img.keys())
    rng.shuffle(img_ids)
    fg_ok = 0
    for iid in img_ids:
        if fg_ok >= args.n_fg:
            break
        im = imgs[iid]
        W, H = im["width"], im["height"]
        ip = img_dir / im["file_name"]
        if not ip.exists():
            continue
        for a in by_img[iid]:
            name = cats[a["category_id"]]
            if name not in FG_CATS or cat_count[name] < args.min_cat_count:
                continue
            if taken[name] >= per_cat_target:
                continue
            x, y, w, h = a["bbox"]
            ratio = (w * h) / (W * H)
            if ratio < 0.02 or ratio > 0.8 or w < 24 or h < 24:
                continue
            taken[name] += 1
            m = poly_mask(a["segmentation"], W, H)
            if (m > 127).sum() < 400:
                continue
            src = Image.open(ip).convert("RGB")
            arr = np.array(src)
            rgba = np.dstack([arr, m])
            ys, xs = np.where(m > 127)
            pad = 6
            box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
                   min(W, xs.max() + pad), min(H, ys.max() + pad))
            crop = Image.fromarray(rgba, "RGBA").crop(box)
            fn = f"fg_{fg_ok:04d}_{name.replace(' ', '_')}.png"
            crop.save(out / "fg" / fn)
            index.append({"file": f"fg/{fn}", "kind": "fg",
                          "cn": f"{CAT_CN.get(name, name)}(COCO 真实照片抠出)",
                          "cat": name, "w": crop.width, "h": crop.height,
                          "src": im["file_name"]})
            fg_ok += 1
            break

    # ---- 背景: 无/少显著物体的图片 (标注数少 + 面积小), 且画面有场景感 ----
    bg_ok = 0
    for iid in img_ids:
        if bg_ok >= args.n_bg:
            break
        anns = by_img.get(iid, [])
        im = imgs[iid]
        W, H = im["width"], im["height"]
        ip = img_dir / im["file_name"]
        if not ip.exists():
            continue
        # 判据: 该图没有大型前景物体 (任一 bbox 面积 > 25% 则跳过)
        bigs = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if (w * h) / (W * H) > 0.25:
                bigs.append(cats[a["category_id"]])
        if bigs:
            continue
        if not anns:
            # 纯无标注图也可做背景 (但样本少)
            pass
        src = Image.open(ip).convert("RGB")
        # 缩到长边 1280 以内
        scale = min(1.0, 1280 / max(W, H))
        if scale < 1.0:
            src = src.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        fn = f"bg_{bg_ok:04d}.jpg"
        src.save(out / "bg" / fn, quality=90)
        # 中文描述: 用出现的小物体做提示 (无则给通用)
        if anns:
            names = sorted({cats[a["category_id"]] for a in anns})
            cn = "COCO 实景照片(含" + "、".join(CAT_CN.get(n, n) for n in names[:4]) + "等元素)"
        else:
            cn = "COCO 实景照片(空旷场景, 适合做背景)"
        index.append({"file": f"bg/{fn}", "kind": "bg", "cn": cn,
                      "cat": "scene", "w": src.width, "h": src.height,
                      "src": im["file_name"]})
        bg_ok += 1

    (out / "assets.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coco_assets] fg={fg_ok} bg={bg_ok} -> {out}")
    # 打印类别覆盖
    print("前景类别:", collections.Counter(i["cat"] for i in index if i["kind"] == "fg").most_common(20))


if __name__ == "__main__":
    main()
