"""
生成演播室图像合成所需的演示素材：
- backgrounds/ : 演播室风格背景（渐变 + 聚光 + 地面）
- inputs/      : 主体（前景形状）合成到“拍摄背景”上的输入图（供 T03 重新抠图）
- gts/         : 对应的 GT alpha（用于评估）

数据源：data/matting 下的合成难例（hair/glass/veil/transparent/semi/fine_edge/complex_bg）。
说明：受网络限制无法下载真实人像 matting 数据集，这里用合成的“难例”作为主体，
既满足任务板对七类难例的构造要求，也可用于 pipeline 全流程与四组光影实验的功能验证。
真实人像数据微调列为后续工作（见工作文档）。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_studio_bg(w: int, h: int, seed: int) -> Image.Image:
    """生成演播室风格背景：上方冷色渐变 + 中部聚光 + 下方地面。"""
    rng = random.Random(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)
    # 垂直渐变（顶亮底暗，演播室蓝灰调）
    top = np.array([rng.randint(40, 70), rng.randint(50, 80), rng.randint(80, 120)], dtype=np.float32)
    bot = np.array([rng.randint(10, 25), rng.randint(12, 30), rng.randint(20, 40)], dtype=np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        img[y, :] = top * (1 - t) + bot * t
    # 聚光（中心偏上圆形提亮）
    cx, cy = w * 0.5, h * 0.42
    Y, X = np.mgrid[0:h, 0:w]
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    spot = np.clip(1 - d / (max(w, h) * 0.55), 0, 1) ** 2
    img += spot[:, :, None] * np.array([60, 60, 55], dtype=np.float32)
    # 地面分隔线
    floor = int(h * 0.72)
    img[floor:, :] *= 0.7
    img = np.clip(img, 0, 255).astype(np.uint8)
    im = Image.fromarray(img).filter(ImageFilter.GaussianBlur(3))
    return im


def make_capture_bg(w: int, h: int, seed: int) -> Image.Image:
    """生成“拍摄时”的杂乱背景（用于合成输入图，考验抠图）。"""
    rng = random.Random(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        grad = np.linspace(rng.randint(0, 255), rng.randint(0, 255), w)
        img[:, :, c] = np.tile(grad, (h, 1))
    img += np.random.RandomState(seed).randint(-25, 25, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(40):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        col = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        d.line([(x1, y1), (x2, y2)], fill=col, width=rng.randint(1, 3))
    return im.filter(ImageFilter.GaussianBlur(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/AIcode/生产实习/data/matting/train")
    ap.add_argument("--out", default="D:/AIcode/生产实习/experiments/demo")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--size", default="512,512")
    args = ap.parse_args()
    w, h = [int(x) for x in args.size.split(",")]
    src = Path(args.src)
    out = Path(args.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "gts").mkdir(parents=True, exist_ok=True)
    (out / "backgrounds").mkdir(parents=True, exist_ok=True)

    # 合成数据集的组合图（case_XXXX.png）即带背景的输入图；对应 GT 为 case_XXXX_alpha.png
    fg_files = sorted([p for p in src.glob("*.png") if not p.name.endswith("_alpha.png")])[: args.n]
    rng = random.Random(42)
    for i, fg_path in enumerate(fg_files):
        name = fg_path.stem
        alpha_path = src / f"{name}_alpha.png"
        if not alpha_path.exists():
            continue
        # 输入图（已含合成复杂背景，作为 T03 重新抠图的输入）
        comp = Image.open(fg_path).convert("RGB").resize((w, h), Image.BILINEAR)
        comp.save(out / "inputs" / f"{name}.png")
        # GT alpha（matting 评估用）
        alpha = Image.open(alpha_path).convert("L").resize((w, h), Image.BILINEAR)
        alpha.save(out / "gts" / f"{name}.png")
        # 目标演播室背景
        bg = make_studio_bg(w, h, seed=i + 1000)
        bg.save(out / "backgrounds" / f"{name}.png")
    print(f"[demo] generated {len(fg_files)} sets into {out}")


if __name__ == "__main__":
    main()
