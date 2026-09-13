# -*- coding: utf-8 -*-
"""
coco_pipeline_check.py — A/B/C/D 四项产物的统一验证与报告

检查项:
  A. coco_synth:   样本数/split/manifest/alpha 有效性/FDR 合理性
  B. coco_assets:  bg/fg 数量/类别覆盖/中文索引完整性
  C. region_fx:    7 个特效在 COCO 真实 mask 上全部可跑 + mask 覆盖率
  D. coco_stuff:   stuff 标注是否就位 (未完成则标记 PENDING)

用法: python training/scripts/coco_pipeline_check.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src" / "fx"))


def check_A() -> dict:
    root = ROOT / "data" / "coco_synth"
    rep = {"module": "A. coco_synth", "ok": False, "items": []}
    if not (root / "manifest.csv").exists():
        rep["error"] = "manifest.csv 缺失"
        return rep
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    splits = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    rep["items"].append(f"样本 {len(rows)} 张, split={splits}")

    # 抽检 alpha 有效性
    bad, fdrs = 0, []
    import random
    random.seed(0)
    for r in random.sample(rows, min(60, len(rows))):
        ip = root / r["split"] / r["image"]
        ap = root / r["split"] / r["alpha"]
        if not (ip.exists() and ap.exists()):
            bad += 1
            continue
        a = np.array(Image.open(ap).convert("L"), np.float32)
        ratio = (a > 127).mean()
        if ratio < 0.005 or ratio > 0.95:
            bad += 1
        # FDR: 前景/背景平均亮度比 (合理区间 0.7-1.3)
        img = np.array(Image.open(ip).convert("L"), np.float32)
        fg = img[a > 200].mean() if (a > 200).sum() > 50 else None
        bg = img[a < 30].mean() if (a < 30).sum() > 50 else None
        if fg and bg and bg > 1:
            fdrs.append(fg / bg)
    rep["items"].append(f"alpha 抽检 60: 异常 {bad} 张")
    if fdrs:
        arr = np.array(fdrs)
        in_range = ((arr >= 0.7) & (arr <= 1.3)).mean()
        rep["items"].append(f"FDR 中位 {np.median(arr):.3f}, ∈[0.7,1.3] 占比 {in_range*100:.0f}%")
    rep["ok"] = len(rows) > 0 and bad <= 3
    return rep


def check_B() -> dict:
    rep = {"module": "B. coco_assets", "ok": False, "items": []}
    p = ROOT / "data" / "coco_assets" / "assets.json"
    if not p.exists():
        rep["error"] = "assets.json 缺失"
        return rep
    idx = json.loads(p.read_text(encoding="utf-8"))
    bg = [i for i in idx if i["kind"] == "bg"]
    fg = [i for i in idx if i["kind"] == "fg"]
    rep["items"].append(f"背景 {len(bg)} 张 / 前景 {len(fg)} 个透明底物体")
    miss = [i["file"] for i in idx if not (p.parent / i["file"]).exists()]
    rep["items"].append(f"文件缺失: {len(miss)}")
    cats = sorted({i["cat"] for i in fg})
    rep["items"].append(f"前景类别 {len(cats)} 种: {', '.join(cats[:12])}...")
    no_cn = [i["file"] for i in idx if not i.get("cn")]
    rep["items"].append(f"缺中文描述: {len(no_cn)}")
    rep["ok"] = len(bg) > 0 and len(fg) > 0 and not miss and not no_cn
    return rep


def check_C() -> dict:
    rep = {"module": "C. region_fx", "ok": False, "items": []}
    try:
        import region_fx as RF
    except Exception as e:
        rep["error"] = f"import 失败: {e}"
        return rep
    ann = ROOT / "data" / "coco_stuff" / "annotations" / "instances_val2017.json"
    img_dir = ROOT / "data" / "coco_stuff" / "val2017"
    if not ann.exists():
        rep["error"] = "COCO 标注缺失"
        return rep
    d = json.loads(ann.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    target = None
    for a in d["annotations"]:
        if cats[a["category_id"]] == "person" and not a.get("iscrowd"):
            im = next(i for i in d["images"] if i["id"] == a["image_id"])
            x, y, w, h = a["bbox"]
            if (w * h) / (im["width"] * im["height"]) > 0.15:
                target = im
                break
    if not target:
        rep["error"] = "找不到测试图"
        return rep
    img = np.array(Image.open(img_dir / target["file_name"]).convert("RGB"))
    mask = RF.mask_from_coco_ann(str(ann), target["id"], "person")
    ok = 0
    for eff in RF.EFFECTS:
        try:
            out, meta = RF.apply_region_fx(img, mask, eff, {"intensity": 0.75})
            assert out.shape == img.shape and out.dtype == np.uint8
            ok += 1
        except Exception as e:
            rep["items"].append(f"  FAIL {eff}: {e}")
    rep["items"].append(f"特效 {ok}/{len(RF.EFFECTS)} 通过 (测试图 {target['file_name']})")
    rep["items"].append(f"mask 覆盖 {round(float((mask>127).mean())*100,1)}%")
    rep["ok"] = ok == len(RF.EFFECTS)
    return rep


def check_D() -> dict:
    rep = {"module": "D. coco_stuff", "ok": False, "items": []}
    root = ROOT / "data" / "coco_stuff"
    cands = list(root.glob("stuff*/**/*.json")) + list(root.glob("**/stuff*_val2017.json"))
    if not cands:
        rep["items"].append("PENDING: stuff 标注尚未就位")
        return rep
    p = cands[0]
    d = json.loads(p.read_text(encoding="utf-8"))
    rep["items"].append(f"标注文件 {p.name}: images={len(d.get('images', []))}, "
                        f"annotations={len(d.get('annotations', []))}, "
                        f"categories={len(d.get('categories', []))}")
    rep["ok"] = len(d.get("annotations", [])) > 0
    return rep


def main():
    reps = [check_A(), check_B(), check_C(), check_D()]
    print("=" * 66)
    for r in reps:
        flag = "PASS" if r["ok"] else ("PENDING" if "PENDING" in str(r.get("items")) else "FAIL")
        print(f"[{flag}] {r['module']}")
        for it in r["items"]:
            print(f"    {it}")
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
    print("=" * 66)
    out = ROOT / "experiments" / "coco_pipeline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {out}")


if __name__ == "__main__":
    main()
