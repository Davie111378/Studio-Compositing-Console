# -*- coding: utf-8 -*-
"""
ingest_p3m_hm.py — P3M-10k + HM(archive 1) 真实人像数据入库（训练方案 Step1+Step2）

输入:  数据集/P3M-10k.zip   (官方 train 9421 对 + validation P3M-500-NP/P)
       数据集/archive (1).zip (HM 绿幕序列 148 scene / 34426 帧, 30.8GB, 全解压不划算)
输出:  data/matting_real/
         raw/                     选择性解压原图 (zip 内相对路径保持, 供硬链接)
         {train,val,test}/        {stem}.{jpg|png} + {stem}_alpha.png  (硬链接, 零拷贝)
         coarse_cache/{split}/    studio 域复用既有缓存硬链接; P3M/HM 留给 --prepare
         manifest.csv (+manifest_real.csv) / {split}_names.txt
       冒烟报告 -> experiments/matting_real_smoke/ (计数+对齐抽查+可视化)

切分原则 (防泄漏, 训练方案 §5):
  P3M train: 官方 9421 对随机采样 P3M_CAP=3000 (seed42, 会话预算; --p3m_cap 0=全量)
  P3M val  : P3M-500-NP 抽 96 + P3M-500-P 抽 48 (训练期 val loss)
  P3M test : P3M-500-NP / P3M-500-P 全量 500+500 (官方基准, 评估用)
  HM       : 按 scene 切分 (同 clip 帧绝不跨 split) — train 20 scene x<=80 帧,
             val 2 scene x16, test 2 scene x20; seed42
  studio   : 既有演播室域 train 720 + val 32 硬链接混入 (防灾难性遗忘)
"""
from __future__ import annotations
import argparse, csv, random, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Z_P3M = ROOT / "数据集" / "P3M-10k.zip"
Z_HM = ROOT / "数据集" / "archive (1).zip"
OUT = ROOT / "data" / "matting_real"
SMOKE = ROOT / "experiments" / "matting_real_smoke"
STUDIO = ROOT / "data" / "studio"

P3M_CAP = 3000
HM_TRAIN_SCENES = 20
HM_PER_SCENE = 80
P3M_VAL_NP, P3M_VAL_P = 96, 48
STUDIO_VAL_N = 32

rng = random.Random(42)


def link(src: Path, dst: Path):
    """硬链接 (同卷零拷贝), 失败退回复制。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        import os
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def extract_needed(zf: zipfile.ZipFile, member: str):
    """选择性解压: 目标已存在则跳过 (可断点重跑)。返回是否新解压。"""
    dst = OUT / "raw" / member
    if dst.exists() and dst.stat().st_size > 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    zf.extract(member, OUT / "raw")
    return True


# ---------------- P3M 计划 ----------------
def plan_p3m(cap: int):
    print("== P3M-10k ==", flush=True)
    zf = zipfile.ZipFile(Z_P3M)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    tr_img = sorted(n for n in names
                    if n.startswith("P3M-10k/train/blurred_image/") and n.endswith(".jpg"))
    tr_msk = {Path(n).stem: n for n in names
              if n.startswith("P3M-10k/train/mask/") and n.endswith(".png")}
    stems = [Path(n).stem for n in tr_img]
    missing = [s for s in stems if s not in tr_msk]
    assert not missing, f"P3M train 缺 mask {len(missing)} 例: {missing[:3]}"
    picked = rng.sample(stems, cap) if 0 < cap < len(stems) else stems
    print(f"  train: {len(stems)} 官方对, 采样 {len(picked)}", flush=True)

    def set_pairs(set_name: str, cap_n: int):
        pre = f"P3M-10k/validation/{set_name}/"
        imgs = [n for n in names if n.startswith(pre) and n.endswith(".jpg")
                and n.count("/") >= 4]                       # original_image/
        # 只取 mask/ 子目录 (集合内还有 trimap/, 按 stem 建字典会覆盖真 GT!)
        msks = {Path(n).stem: n for n in names
                if n.startswith(pre) and n.endswith(".png") and "/mask/" in n
                and n.count("/") >= 4}
        pairs = [(m, msks[Path(m).stem]) for m in imgs if Path(m).stem in msks]
        assert pairs, f"{set_name}: 无配对 (img={len(imgs)} mask={len(msks)})"
        if cap_n and cap_n < len(pairs):
            pairs = rng.sample(pairs, cap_n)
        print(f"  {set_name}: 配对 {len(pairs)} (池 {len(imgs)})", flush=True)
        return pairs

    plan = {
        "train": [(f"P3M-10k/train/blurred_image/{s}.jpg", tr_msk[s]) for s in picked],
        "val": set_pairs("P3M-500-NP", P3M_VAL_NP) + set_pairs("P3M-500-P", P3M_VAL_P),
        "test": set_pairs("P3M-500-NP", 0) + set_pairs("P3M-500-P", 0),
    }
    n_new = 0
    for split in ("train", "val", "test"):
        for im, mk in plan[split]:
            n_new += extract_needed(zf, im) + extract_needed(zf, mk)
    print(f"  extract new={n_new}", flush=True)
    zf.close()
    return plan


# ---------------- HM 计划 ----------------
def plan_hm():
    print("== HM (archive) ==", flush=True)
    zf = zipfile.ZipFile(Z_HM)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    imgs = sorted(n for n in names
                  if n.startswith("clip_img/") and n.endswith(".jpg") and n.count("/") == 3)
    msks = {Path(n).stem: n for n in names
            if n.startswith("matting/") and n.endswith(".png") and n.count("/") == 3}
    by_scene = {}
    for m in imgs:
        s = Path(m).stem
        if s in msks:
            by_scene.setdefault(m.split("/")[1], []).append(m)
    scenes = sorted(s for s, v in by_scene.items() if len(v) >= 24)
    assert len(scenes) >= 24, f"HM 可用 scene 不足: {len(scenes)}"
    rng.shuffle(scenes)
    tr_sc = scenes[:HM_TRAIN_SCENES]
    va_sc = scenes[HM_TRAIN_SCENES:HM_TRAIN_SCENES + 2]
    te_sc = scenes[HM_TRAIN_SCENES + 2:HM_TRAIN_SCENES + 4]

    def pick(scene_list, per_cap):
        out = []
        for sc in scene_list:
            fr = sorted(by_scene[sc])
            take = fr if len(fr) <= per_cap else rng.sample(fr, per_cap)
            out.extend((m, msks[Path(m).stem]) for m in take)
        return out

    plan = {"train": pick(tr_sc, HM_PER_SCENE),
            "val": pick(va_sc, 16), "test": pick(te_sc, 20)}
    n_new = 0
    for split in ("train", "val", "test"):
        for im, mk in plan[split]:
            n_new += extract_needed(zf, im) + extract_needed(zf, mk)
    print(f"  scenes: train={len(tr_sc)} val={va_sc} test={te_sc} "
          f"| frames train={len(plan['train'])} (extract new={n_new})", flush=True)
    zf.close()
    return plan


# ---------------- materialize ----------------
def materialize(p3m, hm):
    print("== materialize (hardlink) ==", flush=True)
    rows = []  # (split, image_name, case)

    def put(split, stem, img_raw: str, msk_raw: str, case: str):
        img_ext = Path(img_raw).suffix
        link(OUT / "raw" / img_raw, OUT / split / f"{stem}{img_ext}")
        link(OUT / "raw" / msk_raw, OUT / split / f"{stem}_alpha.png")
        rows.append((f"{stem}{img_ext}", split, case))  # 列序与 header 一致: image,split,case

    for im, mk in p3m["train"]:
        put("train", f"p3m_{Path(im).stem}", im, mk, "p3m")
    for im, mk in p3m["val"]:
        case = "p3m_np" if "NP" in im else "p3m_p"
        put("val", f"{'vnp' if case == 'p3m_np' else 'vpp'}_{Path(im).stem}", im, mk, case)
    for im, mk in p3m["test"]:
        case = "p3m_np" if "NP" in im else "p3m_p"
        put("test", f"{'np' if case == 'p3m_np' else 'pp'}_{Path(im).stem}", im, mk, case)
    for split in ("train", "val", "test"):
        for im, mk in hm[split]:
            put(split, f"hm_{Path(im).stem}", im, mk, "hm")

    # studio 混入 (防遗忘) + 粗缓存硬链接复用
    st_tr = sorted(p for p in (STUDIO / "train").glob("*.png") if not p.stem.endswith("_alpha"))
    for p in st_tr:
        link(p, OUT / "train" / f"st_{p.name}")
        link(STUDIO / "train" / f"{p.stem}_alpha.png", OUT / "train" / f"st_{p.stem}_alpha.png")
        link(STUDIO / "coarse_cache" / "train" / f"{p.stem}.png",
             OUT / "coarse_cache" / "train" / f"st_{p.stem}.png")
        rows.append((f"st_{p.name}", "train", "studio"))
    st_va = sorted(p for p in (STUDIO / "val").glob("*.png") if not p.stem.endswith("_alpha"))
    rng.shuffle(st_va)
    for p in st_va[:STUDIO_VAL_N]:
        link(p, OUT / "val" / f"st_{p.name}")
        link(STUDIO / "val" / f"{p.stem}_alpha.png", OUT / "val" / f"st_{p.stem}_alpha.png")
        link(STUDIO / "coarse_cache" / "val" / f"{p.stem}.png",
             OUT / "coarse_cache" / "val" / f"st_{p.stem}.png")
        rows.append((f"st_{p.name}", "val", "studio"))

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "case"])
        w.writerows(rows)
    (OUT / "manifest_real.csv").write_text((OUT / "manifest.csv").read_text(encoding="utf-8"),
                                           encoding="utf-8")
    for split in ("train", "val", "test"):
        ns = [r[0] for r in rows if r[1] == split]
        (OUT / f"{split}_names.txt").write_text("\n".join(ns), encoding="utf-8")
        print(f"  {split}: {len(ns)}", flush=True)
    return rows


# ---------------- smoke (Step 2) ----------------
def smoke(rows):
    from PIL import Image
    import numpy as np
    print("== smoke ==", flush=True)
    SMOKE.mkdir(parents=True, exist_ok=True)
    stat = {}
    for img_name, split, case in rows:
        stat.setdefault(split, {}).setdefault(case, 0)
        stat[split][case] += 1
    print("  counts:", {k: dict(v) for k, v in stat.items()}, flush=True)

    bad = 0
    by_case = {}
    for img_name, split, case in rows:
        if split == "train":
            by_case.setdefault(case, []).append(img_name)
    for case, ns in by_case.items():
        for img_name in rng.sample(ns, min(4, len(ns))):
            ip, gp = OUT / "train" / img_name, OUT / "train" / f"{Path(img_name).stem}_alpha.png"
            im, gm = Image.open(ip), Image.open(gp)
            if im.size != gm.size:
                print(f"  [SIZE-MISMATCH] {img_name}: {im.size} vs {gm.size}")
                bad += 1
            a = np.array(gm.convert("L"))
            inter = ((a > 10) & (a < 245)).mean()
            if a.max() == 0 or inter > 0.9:
                print(f"  [ALPHA-ODD] {img_name}: range={a.min()},{a.max()} inter={inter:.2f}")
    print(f"  alignment check: {'OK' if bad == 0 else f'{bad} BAD'}", flush=True)

    for case, ns in by_case.items():
        img_name = ns[len(ns) // 2]
        ip, gp = OUT / "train" / img_name, OUT / "train" / f"{Path(img_name).stem}_alpha.png"
        im = np.array(Image.open(ip).convert("RGB").resize((384, 384)))
        a = np.array(Image.open(gp).convert("L").resize((384, 384)), np.float32)[..., None] / 255.0
        ov = (im * (0.45 + 0.55 * a)).astype("uint8")
        Image.fromarray(np.concatenate([im, ov], axis=1)).save(SMOKE / f"smoke_{case}.png")
    print(f"  viz -> {SMOKE}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3m_cap", type=int, default=P3M_CAP)
    args = ap.parse_args()
    t0 = time.time()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    p3m = plan_p3m(args.p3m_cap)
    hm = plan_hm()
    rows = materialize(p3m, hm)
    smoke(rows)
    print(f"[ingest] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
