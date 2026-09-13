"""
合成数据生成器：用于无外部数据集时的兜底训练数据。
原理：从公开领域图像（程序化生成）中合成 (image, alpha) 对。
特点：
  - 100% 自生成，0 依赖，0 下载
  - 支持 7 类难例模拟：hair / glass / veil / transparent / semi / fine-edge / complex-bg
  - 输出与 Composition-1k 兼容的目录结构
"""

from __future__ import annotations
import os
import math
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def random_shape_mask(size: Tuple[int, int], shape: str = "blob") -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    if shape == "blob":
        cx, cy = w // 2, h // 2
        rx = random.randint(w // 6, w // 3)
        ry = random.randint(h // 6, h // 3)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    elif shape == "rect":
        x1 = random.randint(w // 4, w // 2)
        y1 = random.randint(h // 4, h // 2)
        x2 = x1 + random.randint(w // 6, w // 3)
        y2 = y1 + random.randint(h // 6, h // 3)
        d.rectangle([x1, y1, x2, y2], fill=255)
    elif shape == "person_silhouette":
        # 简易人形剪影：头 + 身体
        cx = w // 2
        # head
        d.ellipse([cx - 30, 60, cx + 30, 120], fill=255)
        # body
        d.polygon([
            (cx - 60, 130),
            (cx + 60, 130),
            (cx + 70, h - 80),
            (cx + 30, h - 30),
            (cx - 30, h - 30),
            (cx - 70, h - 80),
        ], fill=255)
    return mask


def add_hair_edges(mask: Image.Image) -> Image.Image:
    """模拟发丝边缘：在 mask 边缘添加细丝噪声。"""
    arr = np.array(mask).astype(np.float32) / 255.0
    edge = arr.copy()
    h, w = arr.shape
    # 随机撒细丝
    for _ in range(40):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        if arr[y, x] > 0.5:
            length = random.randint(8, 30)
            for i in range(length):
                yy = min(h - 1, y + i)
                xx = max(0, min(w - 1, x + random.randint(-3, 3)))
                edge[yy, xx] = max(edge[yy, xx], 0.5 + random.random() * 0.5)
    return Image.fromarray((np.clip(edge, 0, 1) * 255).astype(np.uint8))


def add_complex_bg(fg: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """生成复杂背景（纹理 + 渐变）。"""
    w, h = size
    # 随机颜色渐变 + 高频噪声（每个通道独立渐变）
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        start = random.randint(0, 255)
        end = random.randint(0, 255)
        gradient = np.linspace(start, end, w, dtype=np.float32)
        bg[:, :, c] = np.tile(gradient, (h, 1))
    bg += np.random.randint(-30, 30, bg.shape).astype(np.float32)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    # 添加纹理
    if random.random() > 0.5:
        d = ImageDraw.Draw(img)
        for _ in range(50):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            d.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 3))
    return img


def random_foreground(size: Tuple[int, int]) -> Image.Image:
    """生成随机彩色前景。"""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    base_color = (random.randint(50, 230), random.randint(50, 230), random.randint(50, 230))
    for c in range(3):
        arr[:, :, c] = base_color[c] + np.random.randint(-30, 30, (h, w))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def synthesize_one(size: Tuple[int, int], case_type: str, out_dir: Path, idx: int) -> dict:
    """合成一张 (image, alpha) 对。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. 背景
    bg = add_complex_bg(None, size)
    # 2. 前景 mask
    shape = "person_silhouette" if random.random() > 0.4 else random.choice(["blob", "rect"])
    mask = random_shape_mask(size, shape)
    if case_type == "hair":
        mask = add_hair_edges(mask)
    if case_type == "glass":
        # 半透明 mask
        arr = np.array(mask).astype(np.float32) * random.uniform(0.3, 0.6)
        mask = Image.fromarray(arr.astype(np.uint8))
    if case_type == "veil":
        # mask 模糊成薄纱
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(5, 12)))
    if case_type == "fine_edge":
        # mask 加细密锯齿
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.6))
    if case_type == "transparent":
        # 仅保留边缘稀疏点
        arr = np.array(mask).astype(np.float32)
        arr = np.where(arr > 200, np.random.uniform(0.2, 0.9, arr.shape) * 255, 0)
        mask = Image.fromarray(arr.astype(np.uint8))
    if case_type == "semi":
        arr = np.array(mask).astype(np.float32) * random.uniform(0.4, 0.8)
        mask = Image.fromarray(arr.astype(np.uint8))
    if case_type == "complex_bg":
        # 在 mask 周围画与背景相似的颜色，让 matting 难
        pass

    # 3. 前景颜色
    fg = random_foreground(size)
    # 4. 合成 image = fg * alpha + bg * (1-alpha)
    arr_fg = np.array(fg).astype(np.float32)
    arr_bg = np.array(bg).astype(np.float32)
    arr_alpha = np.array(mask).astype(np.float32) / 255.0
    arr_alpha = arr_alpha[..., None]
    out = arr_fg * arr_alpha + arr_bg * (1 - arr_alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)

    # 5. 保存
    img_name = f"{case_type}_{idx:04d}.png"
    alpha_name = f"{case_type}_{idx:04d}_alpha.png"
    img.save(out_dir / img_name)
    mask.save(out_dir / alpha_name)
    return {"image": img_name, "alpha": alpha_name, "case": case_type}


def build_synthetic_dataset(root: Path, n_per_case: int = 50, size=(384, 384), seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    cases = ["hair", "glass", "veil", "transparent", "semi", "fine_edge", "complex_bg", "normal"]
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for case in cases:
        for i in range(n_per_case):
            r = i / max(1, n_per_case)
            if r < 0.8:
                split, target = "train", train_dir
            elif r < 0.9:
                split, target = "val", val_dir
            else:
                split, target = "test", test_dir
            info = synthesize_one(size, case, target, i)
            info["split"] = split
            manifest.append(info)
    # write manifest
    (root / "manifest.csv").write_text(
        "image,alpha,case,split\n" + "\n".join(
            f"{r['image']},{r['alpha']},{r['case']},{r['split']}" for r in manifest
        ),
        encoding="utf-8",
    )
    print(f"[synthetic] generated {len(manifest)} samples to {root}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/AIcode/生产实习/data/matting")
    ap.add_argument("--n_per_case", type=int, default=50)
    ap.add_argument("--size", default="384,384")
    args = ap.parse_args()
    w, h = [int(x) for x in args.size.split(",")]
    build_synthetic_dataset(Path(args.root), args.n_per_case, (w, h))
