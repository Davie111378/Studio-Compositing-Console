# -*- coding: utf-8 -*-
"""
stuff_semantic.py — D: COCO-Stuff 语义分割应用

利用 stuff_val2017_pixelmaps (每像素类别 ID 图) 实现:
  1. 按语义类别提取区域 mask (天空/建筑/树/地面/人等)
  2. 场景构成分析: 各类别占比统计
  3. 语义驱动的区域特效: 只改天空 / 只虚化背景建筑 / 只调整地面
  4. 与 region_fx 联动: 把语义 mask 直接喂给区域特效

像素图编码: PNG 单通道, 像素值 = category_id (0=未标注)

用法:
  python stuff_semantic.py --analyze 000000000139
  python stuff_semantic.py --effect sky_replace --image <src> --out <dst>
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STUFF = ROOT / "data" / "coco_stuff" / "annotations"
PM_DIR = STUFF / "stuff_val2017_pixelmaps"
JSON = STUFF / "stuff_val2017.json"
IMG_DIR = ROOT / "data" / "coco_stuff" / "val2017"
sys.path.insert(0, str(ROOT / "ai-service" / "src" / "fx"))


def load_cats() -> dict[int, str]:
    d = json.loads(JSON.read_text(encoding="utf-8"))
    return {c["id"]: c["name"] for c in d["categories"]}


# 语义分组: 中文名 -> COCO-Stuff 类别名 (含 thing 类)
SEMANTIC_GROUPS = {
    "天空": ["sky-other", "sky-other-merged"],
    "建筑": ["building-other", "building-other-merged", "house", "skyscraper", "bridge", "wall-brick"],
    "树木植物": ["tree", "tree-merged", "bush", "bush-merged", "branch", "leaves", "flower",
                 "grass", "grass-merged", "plant-other", "potted plant"],
    "地面": ["floor-wood", "floor-other-merged", "floor-stone", "floor-marble", "floor-tile",
             "rug-merged", "road", "road-merged", "sidewalk", "sidewalk-merged",
             "gravel", "dirt-merged", "sand", "platform"],
    "山石": ["mountain", "mountain-merged", "rock", "rock-merged", "stone"],
    "水": ["water-other", "waterdrops", "sea", "river", "lake", "waterfall", "swimming pool"],
    "雪": ["snow", "snow-merged"],
    "人物": ["person", "player"],
    "车辆": ["car", "car-merged", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"],
    "家具": ["chair", "couch", "bed", "dining table", "bench", "cabinet", "shelf", "desk-stuff",
             "table-merged", "lamp", "mirror-stuff", "curtain", "blanket", "pillow"],
    "电子设备": ["tv", "computer", "laptop", "cell phone", "keyboard", "monitor", "screen",
                 "projector"],
    "食物": ["food-other", "fruit", "vegetable", "bread", "cake", "pizza", "bowl", "cup",
             "bottle", "wine glass"],
    "文字标识": ["banner", "sign", "poster", "billboard", "book", "paper", "textile-other"],
    "室内空间": ["ceiling-merged", "wall-other-merged", "wall-concrete", "wall-panel",
                 "wall-wood", "window-other", "door-stuff", "carpet", "tile"],
    "户外环境": ["field", "field-merged", "pavement-merged", "railing", "fence", "pier",
                 "boardwalk", "stage", "bench", "bleachers", "sculpture"],
}


def build_group_index() -> dict[str, list[int]]:
    cats = load_cats()
    idx = {k: [] for k in SEMANTIC_GROUPS}
    for cid, name in cats.items():
        for grp, names in SEMANTIC_GROUPS.items():
            if name in names:
                idx[grp].append(cid)
    return idx


def load_pixmap(image_id: str) -> np.ndarray | None:
    p = PM_DIR / f"{image_id}.png"
    if not p.exists():
        return None
    return np.array(Image.open(p))


def group_mask(image_id: str, group: str, include_thing: bool = True) -> np.ndarray | None:
    """返回某语义组的二值 mask (uint8 0/255)。"""
    pm = load_pixmap(image_id)
    if pm is None:
        return None
    idx = build_group_index()
    ids = idx.get(group, [])
    if not ids:
        return None
    m = np.isin(pm, ids)
    return (m * 255).astype(np.uint8)


def analyze(image_id: str, top: int = 15) -> dict:
    """场景构成分析: 类别占比 + 语义组占比。"""
    pm = load_pixmap(image_id)
    if pm is None:
        return {"error": f"未找到 {image_id} 的 pixelmap"}
    cats = load_cats()
    total = pm.size
    cnt = Counter(pm.ravel().tolist())
    cnt.pop(0, None)
    cls = [{"id": cid, "name": cats.get(cid, f"?{cid}"),
            "ratio": round(n / total, 4)} for cid, n in cnt.most_common(top)]
    # 语义组汇总
    idx = build_group_index()
    groups = {}
    for grp, ids in idx.items():
        n = sum(cnt.get(i, 0) for i in ids)
        if n:
            groups[grp] = round(n / total, 4)
    groups = dict(sorted(groups.items(), key=lambda x: -x[1]))
    return {"image_id": image_id, "size": list(pm.shape), "classes": cls, "groups": groups}


def to_image_space(mask: np.ndarray, image_id: str) -> np.ndarray:
    """pixelmap 与图片同尺寸 (COCO-Stuff 保证), 若不同则 resize。"""
    src = IMG_DIR / f"{image_id}.jpg"
    if not src.exists():
        return mask
    W, H = Image.open(src).size
    if mask.shape[:2] == (H, W):
        return mask
    return np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))


def apply_semantic_fx(image_id: str, group: str, effect: str,
                      params: dict | None = None, out_path: str | Path | None = None,
                      invert: bool = False) -> dict:
    """语义组 mask + region_fx 区域特效。"""
    src = IMG_DIR / f"{image_id}.jpg"
    if not src.exists():
        raise FileNotFoundError(f"原图不存在: {src}")
    m = group_mask(image_id, group)
    if m is None:
        raise ValueError(f"无法得到语义组 {group} 的 mask")
    m = to_image_space(m, image_id)
    if invert:
        m = 255 - m
    if (m > 127).mean() < 0.005:
        raise ValueError(f"语义组 {group} 占比过小 (<0.5%), 跳过")
    img = np.array(Image.open(src).convert("RGB"))
    import region_fx as RF
    p = dict(params or {})
    out, meta = RF.apply_region_fx(img, m, effect, p)
    meta["semantic_group"] = group
    meta["image_id"] = image_id
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(out_path)
        meta["path"] = str(out_path)
    return out, meta


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", help="分析某图场景构成 (image_id)")
    ap.add_argument("--ids", help="批量列出有某语义组的图, 格式 'group:N'")
    ap.add_argument("--effect", help="语义区域特效, 如 '天空'")
    ap.add_argument("--image", help="image_id (不带 .jpg)")
    ap.add_argument("--fx", default="region_color", help="region_fx 特效名")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out", help="输出路径")
    a = ap.parse_args()

    if a.analyze:
        r = analyze(a.analyze)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    if a.ids:
        grp, n = a.ids.split(":")
        n = int(n)
        idx = build_group_index()
        ids = idx.get(grp, [])
        hits = []
        for p in sorted(PM_DIR.glob("*.png")):
            pm = np.array(Image.open(p))
            r = float(np.isin(pm, ids).mean())
            if r > 0.02:
                hits.append((p.stem, round(r, 3)))
            if len(hits) >= n:
                break
        for i, r in hits:
            print(f"{i}  {grp} {r*100:.1f}%")
        return
    if a.effect and a.image:
        out, meta = apply_semantic_fx(a.image, a.effect, a.fx, json.loads(a.params), a.out)
        print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    _main()
