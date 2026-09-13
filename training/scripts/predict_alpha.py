"""
统一 alpha 推理脚本（B4 用）：按 manifest 严格取 test 集，产出两种 alpha 供对比。

  --model base    : BiRefNet 原始输出（baseline）
  --model refined : BiRefNet 粗 alpha + AlphaRefiner 精修（ours）
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))

DATA_DEFAULT = ROOT / "data" / "matting"
CACHE_DEFAULT = ROOT / "data" / "matting" / "coarse_cache"


def load_split(split: str):
    global DATA, CACHE
    mf = DATA / "manifest.csv"
    names = []
    if mf.exists():
        import csv
        with open(mf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    names.append(r["image"])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(DATA_DEFAULT), help="数据根目录，与 train 时一致")
    ap.add_argument("--model", default="base", choices=["base", "refined"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--weight", default=str(ROOT / "training" / "checkpoints" / "refiner" / "best.pt"))
    ap.add_argument("--infer_size", type=int, default=1024)
    ap.add_argument("--refine_size", type=int, default=1024, help="refiner 推理分辨率")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    global DATA, CACHE
    DATA = Path(args.data_root)
    CACHE = Path(args.data_root) / "coarse_cache"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = load_split(args.split)
    src = DATA / args.split
    print(f"[predict] model={args.model} split={args.split} n={len(names)} device={device}")

    t_all = time.time()
    if args.model == "base":
        from matting.matting_backend import BiRefNetMatting
        tool = BiRefNetMatting(device=device, seg_mask_size=args.infer_size)
        for i, n in enumerate(names):
            p = src / n
            if not p.exists():
                continue
            tool.predict(str(p), str(out / f"{Path(n).stem}.png"))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(names)}", flush=True)
    else:
        # refined: 复用 coarse 缓存（若存在）否则现算，再跑 refiner
        from matting.matting_backend import BiRefNetMatting
        from train_refiner import AlphaRefiner
        net = AlphaRefiner(base=32).to(device)
        w = Path(args.weight)
        if not w.exists():
            raise FileNotFoundError(f"未找到 refiner 权重 {w}，请先训练")
        ck = torch.load(w, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        net.eval()
        print(f"[predict] loaded refiner from {w} (epoch={ck.get('epoch')}, best={ck.get('best'):.5f})")

        need_coarse = not (CACHE / args.split).exists() or len(list((CACHE / args.split).glob("*.png"))) < len(names)
        tool = BiRefNetMatting(device=device, seg_mask_size=args.infer_size) if need_coarse else None

        S = args.refine_size
        for i, n in enumerate(names):
            p = src / n
            if not p.exists():
                continue
            stem = Path(n).stem
            co_p = CACHE / args.split / f"{stem}.png"
            if tool is not None or not co_p.exists():
                (CACHE / args.split).mkdir(parents=True, exist_ok=True)
                tool.predict(str(p), str(co_p))
            img = Image.open(p).convert("RGB")
            co = Image.open(co_p).convert("L").resize((S, S), Image.BILINEAR)
            rgb = img.resize((S, S), Image.BILINEAR)
            rgb_t = torch.from_numpy(__import__("numpy").array(rgb).astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            co_t = torch.from_numpy(__import__("numpy").array(co).astype("float32") / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred = net(rgb_t, co_t)
            a = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(a).resize(img.size, Image.BILINEAR).save(out / f"{stem}.png")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(names)}", flush=True)

    print(f"[predict] wrote {len(list(out.glob('*.png')))} alphas -> {out}  ({time.time()-t_all:.1f}s)")


if __name__ == "__main__":
    main()
