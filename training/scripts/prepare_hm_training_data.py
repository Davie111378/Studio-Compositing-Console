# -*- coding: utf-8 -*-
"""prepare_hm_training_data.py — 从 HM-1k 压缩包提取训练数据
输出格式: data/matting/hm_train/{img_name}.png + {img_name}_alpha.png
同时做 P3M 数据增强 (JPEG 压缩 + 下采样) 模拟 HM 域
"""
import zipfile, io, sys, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"D:\AIcode\生产实习")
HM_ZIP = ROOT / "数据集" / "archive (1).zip"
P3M_ZIP = ROOT / "数据集" / "P3M-10k.zip"
OUT = ROOT / "data" / "matting" / "hm_train"
OUT.mkdir(parents=True, exist_ok=True)

def save_pair(img_bytes: bytes, alpha_bytes: bytes, name: str, out_dir: Path):
    """保存 image + alpha 对"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    alpha_im = Image.open(io.BytesIO(alpha_bytes))
    # 取 RGBA 的 A 通道 (HM matting 是前景抠图, alpha 在 A 通道)
    if alpha_im.mode in ("RGBA", "LA", "PA"):
        alpha = np.array(alpha_im.convert("RGBA"))[..., 3]
    else:
        alpha = np.array(alpha_im.convert("L"))
    alpha_im = Image.fromarray(alpha.astype(np.uint8), mode="L")
    # 统一尺寸 (最长边 512, 保持比例)
    max_side = 512
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.BILINEAR)
        alpha_im = alpha_im.resize(new_size, Image.NEAREST)
    img.save(out_dir / f"{name}.png")
    alpha_im.save(out_dir / f"{name}_alpha.png")


def extract_hm(max_pairs: int = 200):
    """从 HM 压缩包提取训练数据"""
    with zipfile.ZipFile(HM_ZIP) as z:
        # 找到所有 clip_img 场景
        clip_entries = [e for e in z.infolist() if e.filename.startswith("clip_img/") and "/clip_00000000/" in e.filename]
        scenes = sorted({e.filename.split("/")[1] for e in clip_entries})
        print(f"[HM] 场景数: {len(scenes)}")
        count = 0
        for sc in scenes:
            if count >= max_pairs:
                break
            # 每个 scene 的 clip_00000000 帧
            frames = sorted([e for e in clip_entries if e.filename.startswith(f"clip_img/{sc}/clip_00000000/")], key=lambda x: x.filename)
            # 隔 5 帧取 1 帧, 避免连续帧冗余
            frames = frames[::5]
            for e in frames:
                if count >= max_pairs:
                    break
                stem = Path(e.filename).stem
                # 对应 matting
                matting_path = e.filename.replace("clip_img", "matting").replace("clip_00000000", "matting_00000000")
                # matting 文件可能是 .png 或 .jpg
                cands = [m for m in z.infolist() if m.filename == matting_path or
                         m.filename == matting_path.replace(".jpg", ".png")]
                if not cands:
                    continue
                img_bytes = z.read(e)
                alpha_bytes = z.read(cands[0])
                name = f"hm_{sc}_{stem}"
                save_pair(img_bytes, alpha_bytes, name, OUT)
                count += 1
        print(f"[HM] 提取 {count} 对 → {OUT}")


def augment_p3m(max_pairs: int = 100):
    """从 P3M 提取数据并做 JPEG 压缩 + 下采样增强, 模拟 HM 域"""
    with zipfile.ZipFile(P3M_ZIP) as z:
        imgs = [e for e in z.infolist() if e.filename.startswith("P3M-10k/train/blurred_image/")][:max_pairs]
        count = 0
        for e in imgs:
            stem = Path(e.filename).stem
            mask_path = f"P3M-10k/train/mask/{stem}.png"
            if mask_path not in z.namelist():
                continue
            img_bytes = z.read(e)
            alpha_bytes = z.read(mask_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            alpha = Image.open(io.BytesIO(alpha_bytes)).convert("L")
            # 增强1: 下采样再上采样 (模拟低分辨率)
            for quality in [30, 50]:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                aug_img = Image.open(buf).convert("RGB")
                # 下采样
                small = aug_img.resize((max(aug_img.size[0]//3, 200), max(aug_img.size[1]//3, 200)), Image.BILINEAR)
                aug_img = small.resize(img.size, Image.BILINEAR)
                alpha_small = alpha.resize(small.size, Image.NEAREST)
                aug_alpha = alpha_small.resize(img.size, Image.BILINEAR)
                name = f"p3m_aug_{stem}_q{quality}"
                aug_img.save(OUT / f"{name}.png")
                aug_alpha.save(OUT / f"{name}_alpha.png")
                count += 1
        print(f"[P3M aug] 提取 {count} 对 (增强后) → {OUT}")


def make_val_split(n_val: int = 20):
    """随机抽 n_val 对到 hm_val/"""
    val_dir = ROOT / "data" / "matting" / "hm_val"
    val_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    for f in sorted(OUT.glob("*.png")):
        if f.stem.endswith("_alpha"):
            continue
        alpha_f = OUT / f"{f.stem}_alpha.png"
        if alpha_f.exists():
            all_pairs.append((f, alpha_f))
    random.seed(42)
    random.shuffle(all_pairs)
    for img_f, alpha_f in all_pairs[:n_val]:
        img_f.rename(val_dir / img_f.name)
        alpha_f.rename(val_dir / alpha_f.name)
    print(f"[val] 移动 {min(n_val, len(all_pairs))} 对 → {val_dir}")


if __name__ == "__main__":
    extract_hm(max_pairs=200)
    augment_p3m(max_pairs=100)
    make_val_split(n_val=20)
    total = len(list(OUT.glob("*.png")))
    print(f"\n[done] hm_train: {total} 文件 ({total//2} 对)")
