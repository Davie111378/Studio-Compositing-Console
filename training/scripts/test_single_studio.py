# -*- coding: utf-8 -*-
"""单图多任务流测试（T1）：用户提供的演播室图片
T03 抠图：BiRefNet 粗 alpha（S0） vs 演播室域 Refiner 精修（S2, ours）
"""
import sys, time, json
from pathlib import Path

ROOT = Path("D:/AIcode/生产实习")
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

import torch
import numpy as np
from PIL import Image

IMG = ROOT / "演播室图片生成需求.png"
OUT = ROOT / "experiments" / "studio" / "single_image_test"
OUT.mkdir(parents=True, exist_ok=True)

import os
os.chdir(ROOT / "ai-service")

from matting.matting_backend import BiRefNetMatting
from train_refiner import AlphaRefiner

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[T1] device={device} img={IMG.name}")

# ---------- S0: BiRefNet 粗 alpha ----------
t0 = time.time()
tool = BiRefNetMatting(device=device, seg_mask_size=1024)
r0 = tool.predict(str(IMG), str(OUT / "alpha_coarse.png"), str(OUT / "fg_coarse.png"))
t_coarse = time.time() - t0
print(f"[T1] S0 coarse done in {t_coarse:.1f}s -> {r0['alpha_path']}")

# ---------- S2: 演播室域 Refiner 精修（ours）----------
t0 = time.time()
net = AlphaRefiner(base=32).to(device)
w = ROOT / "training" / "checkpoints" / "refiner_studio" / "best.pt"
ck = torch.load(w, map_location=device, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"[T1] loaded refiner_studio epoch={ck.get('epoch')} best={ck.get('best'):.5f}")

S = 384
img = Image.open(IMG).convert("RGB")
co = Image.open(OUT / "alpha_coarse.png").convert("L").resize((S, S), Image.BILINEAR)
rgb = img.resize((S, S), Image.BILINEAR)
rgb_t = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
co_t = torch.from_numpy(np.array(co, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        pred = net(rgb_t, co_t)
a = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
Image.fromarray(a).resize(img.size, Image.BILINEAR).save(OUT / "alpha_refined.png")

# 精修前景（黑底预乘，用于加图）
arr = np.array(img).astype(np.float32)
a_full = np.array(Image.open(OUT / "alpha_refined.png").convert("L"), dtype=np.float32) / 255.0
af = a_full[..., None]
fg_black = arr * af
Image.fromarray(fg_black.astype(np.uint8)).save(OUT / "fg_refined.png")
t_refine = time.time() - t0
print(f"[T1] S2 refined done in {t_refine:.1f}s -> {OUT / 'alpha_refined.png'}")

# ---------- 差异统计 ----------
ac = np.array(Image.open(OUT / "alpha_coarse.png").convert("L"), dtype=np.float32) / 255.0
ar = np.array(Image.open(OUT / "alpha_refined.png").convert("L"), dtype=np.float32) / 255.0
diff = np.abs(ac - ar)
stats = {
    "coarse_time_s": round(t_coarse, 2), "refine_time_s": round(t_refine, 2),
    "mean_abs_diff": round(float(diff.mean()), 5),
    "changed_pixels_gt10": f"{float((diff > 0.1).mean()) * 100:.2f}%",
    "coarse_fg_ratio": f"{float((ac > 0.5).mean()) * 100:.1f}%",
    "refined_fg_ratio": f"{float((ar > 0.5).mean()) * 100:.1f}%",
}
print("[T1] stats:", json.dumps(stats, ensure_ascii=False))
(OUT / "t1_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print("[T1] DONE")
