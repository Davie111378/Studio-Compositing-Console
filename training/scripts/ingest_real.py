#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实演播室素材接入接口 (ingest_real)
===================================

把真实拍摄 / 第三方数据集的演播室抠像素材, 转换为本 pipeline 的统一格式:

  <output_root>/
    train/  val/  test/
      <name>.png           # RGB 合成图（人 + 演播室背景）
      <name>_alpha.png     # GT alpha (L 模式 0-255)
    manifest.csv           # image,alpha,case,split

用法:

  # 通用(假设图片对已经在同一目录, 命名 *_img.png / *_alpha.png)
  python ingest_real.py \\
      --img_dir    <DIR>          --alpha_dir    <DIR> \\
      --split      train           --case        anchor_hair \\
      --out_root   D:/AIcode/生产实习/data/studio_real

  # 从 matting 经典数据集(Composition-1k / P3M-10k / AM-2k)
  # 转换: 只要写一个简单的预归一化脚本, 调用上面的接口.

依赖:
  Pillow>=10

Note:
  - 不做数据增强(交给训练时)
  - 不做 train/val/test 切分(用一个 --ratio 参数简单分层)
  - 不做格式校验(调用方需保证 alpha L 模式 0-255)
"""
from __future__ import annotations

import argparse, csv, hashlib, re, shutil, sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load_image_pair(img_path: Path, alpha_path: Path):
    rgb = Image.open(img_path).convert("RGB")
    a = Image.open(alpha_path).convert("L")
    if rgb.size != a.size:
        # 居中裁剪 / resize 到较小尺寸
        size = (min(rgb.size[0], a.size[0]), min(rgb.size[1], a.size[1]))
        rgb = rgb.resize(size, Image.BILINEAR)
        a = a.resize(size, Image.BILINEAR)
    return rgb, a


def stable_name(case: str, idx: int, src_path: Path):
    h = hashlib.md5(str(src_path).encode("utf-8")).hexdigest()[:6]
    return f"{case}_real_{idx:04d}_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="原图目录（RGB 合成图）")
    ap.add_argument("--alpha_dir", required=True, help="alpha 目录（L 模式, 0-255）")
    ap.add_argument("--case", default="real_unknown", help="案例类别, 默认 unknown")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out_root", required=True, help="输出根目录")
    ap.add_argument("--img_glob", default="*.png", help="原图 glob")
    ap.add_argument("--alpha_glob", default="*.png", help="alpha glob")
    ap.add_argument("--strip_suffix", default="",
                    help="原图匹配 alpha 时要剥离的尾部 (如 '_img', '_rgb')")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    alpha_dir = Path(args.alpha_dir)
    out_root = Path(args.out_root)
    out_dir = out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按 stem 匹配
    images = sorted(img_dir.glob(args.img_glob))
    used = 0
    rows = []
    for i, ip in enumerate(images):
        stem = ip.stem
        if args.strip_suffix and stem.endswith(args.strip_suffix):
            stem = stem[: -len(args.strip_suffix)]
        a_p = alpha_dir / f"{stem}{args.alpha_glob[1:]}"  # 去掉 *
        if not a_p.exists():
            print(f"[skip] no alpha for {ip.name} -> expect {a_p.name}", file=sys.stderr)
            continue
        try:
            rgb, a = load_image_pair(ip, a_p)
        except Exception as e:
            print(f"[skip] failed: {ip.name}: {e}", file=sys.stderr)
            continue
        name = stable_name(args.case, used, ip)
        rgb.save(out_dir / f"{name}.png")
        a.save(out_dir / f"{name}_alpha.png")
        rows.append({"image": f"{name}.png", "alpha": f"{name}_alpha.png",
                     "case": args.case, "split": args.split,
                     "bg": "unknown", "lighting": "unknown", "spill": "0"})
        used += 1

    # 追加 manifest
    mf = out_root / "manifest.csv"
    new = not mf.exists()
    with open(mf, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "alpha", "case", "split", "bg", "lighting", "spill"])
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[ingest] {used} pairs -> {out_dir}  ({args.case} / {args.split})")


if __name__ == "__main__":
    main()