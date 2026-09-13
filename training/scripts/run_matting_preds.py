"""
在 test 集上批量跑抠图模型，产出 alpha 预测，供 evaluate_matting.py 评测。

--model birefnet : 真实 BiRefNet（本地权重）
--model simplified : 自实现 SimplifiedBiRefNet（可指定 --weight；不指定=随机初始化基线）
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]  # .../生产实习
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="birefnet")
    ap.add_argument("--weight", default=None)
    ap.add_argument("--test_dir", default="D:/AIcode/生产实习/data/matting/test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.model == "birefnet":
        from matting.matting_backend import BiRefNetMatting
        tool = BiRefNetMatting(device=device)
    else:
        from matting.matting_tool import MattingTool
        tool = MattingTool(weight_path=args.weight, device=device)

    test_dir = Path(args.test_dir)
    imgs = sorted([p for p in test_dir.glob("*.png") if not p.name.endswith("_alpha.png")])
    n = 0
    for p in imgs:
        out_alpha = out / f"{p.stem}.png"
        tool.predict(str(p), str(out_alpha))
        n += 1
    print(f"[preds] {args.model}: wrote {n} alphas to {out}")


if __name__ == "__main__":
    main()
