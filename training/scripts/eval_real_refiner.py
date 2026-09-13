# -*- coding: utf-8 -*-
"""
eval_real_refiner.py — 训练方案 Step5：refiner_real 真实域评估 + studio 回归

流程:
  1) test split 逐张: coarse cache -> AlphaRefiner 精修 -> preds_{tag}/
     (coarse 基线直接引用 coarse_cache/test, 不复制)
  2) evaluate_matting.evaluate: SAD/MSE/Grad/Conn, coarse vs refined, 按 case 分组
  3) studio 域回归: studio test 抽 24 张, refiner_studio vs refiner_real 各自 L1
  4) 视觉对比: SAD 改善 TOP + 随机样例 -> side-by-side 面板

用法:
  python eval_real_refiner.py --ckpt training/checkpoints/refiner_real/best.pt --tag real
  python eval_real_refiner.py --tag studio --ckpt training/checkpoints/refiner_studio/best.pt
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))

import numpy as np
from PIL import Image
import torch

DATA = ROOT / "data" / "matting_real"
OUTD = ROOT / "experiments" / "real_refiner_eval"
INFER = 512

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "training/checkpoints/refiner_real/best.pt"))
ap.add_argument("--tag", default="real")
ap.add_argument("--n_studio", type=int, default=24)
ap.add_argument("--infer", type=int, default=0,
                help="推理边长; 0=自动取 ckpt 同级 config.json 的 size (与训练一致)")
args = ap.parse_args()

CKPT = Path(args.ckpt)
tag = args.tag

# 推理尺寸必须与训练尺寸一致, 否则 UNet 感受野尺度错配 -> 指标虚高 (实测 studio 模型
# 在自己域上 512 推理 SAD 会虚增 ~30 倍, 属评测假象而非模型问题)
INFER = args.infer
if not INFER:
    cfg_f = CKPT.parent / "config.json"
    if cfg_f.exists():
        try:
            INFER = int(json.loads(cfg_f.read_text(encoding="utf-8")).get("size", 384))
        except Exception:
            INFER = 384
    else:
        INFER = 384
print(f"[infer-size] {INFER} (训练尺寸, 取自 {CKPT.parent.name}/config.json)", flush=True)

# 预测目录带尺寸后缀: 避免不同尺寸推理结果互相复用造成脏缓存
PRE = OUTD / f"preds_{tag}_{INFER}"
PRE.mkdir(parents=True, exist_ok=True)

names = [x.strip() for x in (DATA / "test_names.txt").read_text(encoding="utf-8").split("\n") if x.strip()]
cache_test = DATA / "coarse_cache" / "test"

# ---------- 1) 精修预测 ----------
from train_refiner import AlphaRefiner

device = "cuda" if torch.cuda.is_available() else "cpu"
net = AlphaRefiner(base=32).to(device)
ck = torch.load(CKPT, map_location=device, weights_only=False)
net.load_state_dict(ck["model"])
net.eval()
print(f"[refine] {tag} <- {CKPT.name} (epoch={ck.get('epoch')}, best={ck.get('best'):.5f}) "
      f"device={device}", flush=True)

t0 = time.time()
n = 0
for raw in names:
    stem = Path(raw).stem
    cp = cache_test / f"{stem}.png"
    op = PRE / f"{stem}.png"
    if op.exists():
        continue
    if not cp.exists():
        continue
    img = Image.open(DATA / "test" / raw).convert("RGB")
    co = Image.open(cp).convert("L")
    rgb_t = torch.from_numpy(np.array(img.resize((INFER, INFER), Image.BILINEAR),
                                      np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    co_t = torch.from_numpy(np.array(co.resize((INFER, INFER), Image.BILINEAR),
                                     np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        pred = net(rgb_t, co_t)
    a = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(a).resize(img.size, Image.BILINEAR).save(op)
    n += 1
    if n % 200 == 0:
        print(f"  refined {n}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
print(f"[refine] {n} preds in {time.time()-t0:.0f}s -> {PRE}", flush=True)

# ---------- 2) 指标: coarse vs refined ----------
sys.path.insert(0, str(ROOT / "training" / "scripts"))
import evaluate_matting as EM

case_map = EM.load_case_map(DATA)
gt_names = [Path(x).stem for x in names]

def run_eval(preds_dir: Path, label: str):
    rows, mean, per_case = EM.evaluate(DATA / "test", preds_dir, "_alpha", gt_names, case_map)
    return rows, mean, per_case

rows_c, mean_c, pc_c = run_eval(cache_test, "coarse")
rows_r, mean_r, pc_r = run_eval(PRE, tag)

def fmt(mean, pc):
    s = f"SAD={mean[0]:.2f} MSE={mean[1]:.6f} Grad={mean[2]:.2f} Conn={mean[3]:.4f}"
    for c in sorted(pc):
        a = np.array(pc[c])
        s += f" | {c}: SAD={a[:,0].mean():.2f} Grad={a[:,2].mean():.2f} Conn={a[:,3].mean():.4f}"
    return s

print(f"[coarse ] {fmt(mean_c, pc_c)}", flush=True)
print(f"[{tag:6s}] {fmt(mean_r, pc_r)}", flush=True)
sad_drop = (mean_c[0] - mean_r[0]) / max(mean_c[0], 1e-9) * 100
grad_ok = mean_r[2] <= mean_c[2] * 1.02
conn_ok = mean_r[3] <= mean_c[3] * 1.02
gate = "PASS" if (sad_drop >= 10 and grad_ok and conn_ok) else "CHECK"
print(f"[gate] SAD drop={sad_drop:.1f}% Grad_ok={grad_ok} Conn_ok={conn_ok} -> {gate}", flush=True)

# ---------- 3) studio 回归 (防遗忘) ----------
st_items = []
cc = ROOT / "data" / "studio" / "coarse_cache" / "test"
for p in sorted((ROOT / "data" / "studio" / "test").glob("*.png")):
    if p.stem.endswith("_alpha"):
        continue
    gtp = ROOT / "data" / "studio" / "test" / f"{p.stem}_alpha.png"
    cp = cc / f"{p.stem}.png"
    if gtp.exists() and cp.exists():
        st_items.append((p, gtp, cp))
st_items = st_items[:args.n_studio]

def studio_l1(weight: Path) -> float:
    n2 = AlphaRefiner(base=32).to(device)
    c2 = torch.load(weight, map_location=device, weights_only=False)
    n2.load_state_dict(c2["model"])
    n2.eval()
    l1s = []
    for ip, gp, cp in st_items:
        img = Image.open(ip).convert("RGB")
        co = Image.open(cp).convert("L")
        rgb_t = torch.from_numpy(np.array(img.resize((INFER, INFER), Image.BILINEAR),
                                          np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        co_t = torch.from_numpy(np.array(co.resize((INFER, INFER), Image.BILINEAR),
                                         np.float32) / 255.0)[None, None].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pr = n2(rgb_t, co_t)
        a = (pr[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        am = np.array(Image.fromarray(a).resize(img.size, Image.BILINEAR), np.float32) / 255.0
        gt = np.array(Image.open(gp).convert("L"), np.float32) / 255.0
        l1s.append(float(np.abs(am - gt).mean()))
    return float(np.mean(l1s))

l1_old = studio_l1(ROOT / "training/checkpoints/refiner_studio/best.pt")
l1_new = studio_l1(CKPT)
degr = (l1_new - l1_old) / max(l1_old, 1e-9) * 100
print(f"[studio-regression] n={len(st_items)} refiner_studio L1={l1_old*100:.3f}% "
      f"{tag} L1={l1_new*100:.3f}% ({degr:+.1f}%)", flush=True)

# ---------- 4) 视觉对比面板 ----------
from evaluate_matting import load_alpha
sad_by = {r[0]: (r[2], r[1]) for r in rows_c}
gain = []
for r in rows_r:
    if r[0] in sad_by:
        gain.append((sad_by[r[0]][0] - r[2], r[0], r[1]))
gain.sort(reverse=True)
picks = [g[1] for g in gain[:3]] + [gain[-1][1]] if gain else []
random_extra = [g[1] for g in gain[len(gain)//2:len(gain)//2+2]]
picks += random_extra

PANEL_W, PANEL_H = 320, 320
def thumb(im: Image.Image) -> np.ndarray:
    w, h = im.size
    s = min(PANEL_W / w, PANEL_H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    t = np.array(im.convert("RGB").resize((nw, nh), Image.BILINEAR))
    canvas = np.full((PANEL_H, PANEL_W, 3), 255, np.uint8)
    canvas[(PANEL_H - nh) // 2:(PANEL_H - nh) // 2 + nh,
           (PANEL_W - nw) // 2:(PANEL_W - nw) // 2 + nw] = t
    return canvas

tiles = []
for stem in picks[:6]:
    raw = next(x for x in names if Path(x).stem == stem)
    img = thumb(Image.open(DATA / "test" / raw))
    co = thumb(Image.open(cache_test / f"{stem}.png"))
    re = thumb(Image.open(PRE / f"{stem}.png"))
    gt = thumb(Image.open(DATA / "test" / f"{stem}_alpha.png"))
    a3 = lambda x: x
    row = np.concatenate([img, a3(co), a3(re), a3(gt)], axis=1)
    tiles.append(row)
panel = np.concatenate(tiles, axis=0)
Image.fromarray(panel).save(OUTD / f"panel_{tag}.png")
print(f"[panel] -> {OUTD / f'panel_{tag}.png'} (col: img|coarse|refined|GT)", flush=True)

summary = {
    "tag": tag, "ckpt": str(CKPT), "n_test": len(names),
    "coarse": {"SAD": float(mean_c[0]), "MSE": float(mean_c[1]),
               "Grad": float(mean_c[2]), "Conn": float(mean_c[3]),
               "per_case": {c: {"SAD": float(np.array(v)[:, 0].mean()),
                                "Grad": float(np.array(v)[:, 2].mean()),
                                "Conn": float(np.array(v)[:, 3].mean())} for c, v in pc_c.items()}},
    tag: {"SAD": float(mean_r[0]), "MSE": float(mean_r[1]),
          "Grad": float(mean_r[2]), "Conn": float(mean_r[3]),
          "per_case": {c: {"SAD": float(np.array(v)[:, 0].mean()),
                           "Grad": float(np.array(v)[:, 2].mean()),
                           "Conn": float(np.array(v)[:, 3].mean())} for c, v in pc_r.items()}},
    "sad_drop_pct": round(sad_drop, 2),
    "studio_regression": {"n": len(st_items), "refiner_studio_L1": l1_old,
                          f"{tag}_L1": l1_new, "degradation_pct": round(degr, 2)},
    "acceptance_gate": gate,
}
(OUTD / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
print(f"[eval-real] DONE -> {OUTD / f'summary_{tag}.json'}", flush=True)
