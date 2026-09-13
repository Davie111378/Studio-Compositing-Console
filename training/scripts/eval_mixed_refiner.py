# -*- coding: utf-8 -*-
"""eval_mixed_refiner.py — 混合域 Refiner 训练后复验
用法: python eval_mixed_refiner.py [--w1 <refiner.pt> --w2 <refiner.pt>]
默认对比 refiner_mixed/best.pt (W1) vs refiner_studio/best.pt (W2)
核心检验: 对"主播台(物体主体)"OOD 失效是否消除 + 演播室人像域是否回退
"""
import sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--w1", default=str(Path(__file__).resolve().parents[2] / "training/checkpoints/refiner_mixed/best.pt"))
ap.add_argument("--w2", default=str(Path(__file__).resolve().parents[2] / "training/checkpoints/refiner_studio/best.pt"))
ap.add_argument("--tag", default="mixed")
args_cli = ap.parse_args()

ROOT = Path("D:/AIcode/生产实习")
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

import torch
import numpy as np
from PIL import Image

OUT = ROOT / "experiments" / "studio" / "mixed_refiner_eval"
OUT.mkdir(parents=True, exist_ok=True)

from matting.matting_backend import BiRefNetMatting
from train_refiner import AlphaRefiner

device = "cuda" if torch.cuda.is_available() else "cpu"
w_new = Path(args_cli.w1)
w_base = Path(args_cli.w2)
tag = args_cli.tag

def refine(img_path: str, coarse_path: str, out_path: str, S=384):
    net = AlphaRefiner(base=32).to(device)
    ck = torch.load(w_new, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"]); net.eval()
    img = Image.open(img_path).convert("RGB")
    co = Image.open(coarse_path).convert("L").resize((S, S), Image.BILINEAR)
    rgb = img.resize((S, S), Image.BILINEAR)
    rgb_t = torch.from_numpy(np.array(rgb, np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    co_t = torch.from_numpy(np.array(co, np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        pred = net(rgb_t, co_t)
    a = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(a).resize(img.size, Image.BILINEAR).save(out_path)
    return np.array(Image.open(out_path).convert("L")) / 255.0

# ---------- 用例 1: 物体主体 (主播台, 之前的 OOD 失败例) ----------
desk = str(ROOT / "data/studio_input/desk_news.png")
# 对 desk_news.png 现算粗 alpha (与旧测试的原图尺寸不同, 必须同基准)
tool = BiRefNetMatting(device=device, seg_mask_size=1024)
coarse_desk = str(OUT / "desk_alpha_coarse.png")
tool.predict(desk, coarse_desk)
a_coarse = np.array(Image.open(coarse_desk).convert("L")) / 255.0

def refine_with(weight_path: str, out_path: str):
    net2 = AlphaRefiner(base=32).to(device)
    ck2 = torch.load(weight_path, map_location=device, weights_only=False)
    net2.load_state_dict(ck2["model"]); net2.eval()
    img = Image.open(desk).convert("RGB")
    co = Image.open(coarse_desk).convert("L").resize((384, 384), Image.BILINEAR)
    rgb = img.resize((384, 384), Image.BILINEAR)
    rgb_t = torch.from_numpy(np.array(rgb, np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    co_t = torch.from_numpy(np.array(co, np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        pred = net2(rgb_t, co_t)
    a = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(a).resize(img.size, Image.BILINEAR).save(out_path)
    return np.array(Image.open(out_path).convert("L")) / 255.0

a_mix = refine_with(str(w_new), str(OUT / f"desk_alpha_{tag}.png"))
a_old = refine_with(str(w_base), str(OUT / "desk_alpha_oldstudio.png"))
r_old = float(np.abs(a_old - a_coarse).mean())   # 基准 refiner 偏离粗 alpha 的程度
r_mix = float(np.abs(a_mix - a_coarse).mean())
fg_new = (np.array(Image.open(desk).convert("RGB"), np.float32) * a_mix[..., None])
Image.fromarray(fg_new.astype(np.uint8)).save(OUT / f"desk_fg_{tag}.png")
res1 = {"old_studio_refiner_mad_vs_coarse": round(r_old, 4),
        "mixed_refiner_mad_vs_coarse": round(r_mix, 4),
        "mixed_fg_ratio": f"{float((a_mix > 0.5).mean()) * 100:.1f}%",
        "coarse_fg_ratio": f"{float((a_coarse > 0.5).mean()) * 100:.1f}%"}
print("[desk]", json.dumps(res1))

# ---------- 用例 2: 演播室人像域不回退 (studio test 集 val_l1 抽样 24 张) ----------
sys.path.insert(0, str(ROOT / "training" / "scripts"))
os_chdir_ok = True
import os
os.chdir(ROOT / "ai-service")
import train_refiner as TR
TR.DATA = ROOT / "data" / "studio"
TR.CACHE = ROOT / "data" / "studio" / "coarse_cache"
names = TR.load_split("test")[:24]
l1s = []
for n, _ in names:
    stem = Path(n).stem
    ip = TR.DATA / "test" / n
    cp = TR.CACHE / "test" / f"{stem}.png"
    gp = TR.DATA / "test" / f"{stem}_alpha.png"
    if not (ip.exists() and cp.exists() and gp.exists()):
        continue
    am = refine(str(ip), str(cp), str(OUT / f"st_{tag}_{stem}.png"))
    gt = np.array(Image.open(gp).convert("L"), np.float32) / 255.0
    l1s.append(float(np.abs(am - gt).mean()))
res2 = {"studio_test_n": len(l1s), "mixed_refiner_val_l1": f"{np.mean(l1s)*100:.2f}%",
        "old_studio_refiner_val_l1": "0.96%"}
print("[studio]", json.dumps(res2))

(OUT / f"eval_summary_{tag}.json").write_text(
    json.dumps({"w_new": str(w_new), "desk_object_ood": res1, "studio_person_domain": res2},
               ensure_ascii=False, indent=2),
    encoding="utf-8")
print("[eval-mixed] DONE ->", OUT / f"eval_summary_{tag}.json")
