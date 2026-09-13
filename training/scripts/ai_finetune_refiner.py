# -*- coding: utf-8 -*-
"""
AI 素材域微调 AlphaRefiner（2026-09-09 新实验线 · 修复真实感人像反退化）

背景: refiner_studio 在 AI 真实感人像上 SAD 反退化 2.8x (47k -> 134k)。
方案: 用 AI 生成绿幕人像 (色度键 GT) 贴到 AI 演播厅背景, 合成训练对,
      从 refiner_studio 热启动, 低 lr 微调, 修复真实感人像域。

流程:
  Phase A 合成数据: 4 fg x 6 bg x 10 变体 = 240 train + 16 val
          (despill 前景 + GT alpha + BiRefNet coarse 全部落盘)
  Phase B 微调: 热启动 + AdamW(2e-4) + 40 epoch, 产出 refiner_ai/best.pt
  四件套: config.json / train.log / metrics.csv / checkpoints

用法:
  python ai_finetune_refiner.py --stage synth   # 只合成数据
  python ai_finetune_refiner.py --stage train   # 合成(若缺)+训练
"""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

GREEN_DIR = ROOT / "data" / "ai_generated" / "green_fg"
BG_DIR = ROOT / "data" / "ai_generated" / "studio_bg"
SYNTH_DIR = ROOT / "data" / "ai_generated" / "synth"
CKPT_OUT = ROOT / "training" / "checkpoints" / "refiner_ai"
CKPT_INIT = ROOT / "training" / "checkpoints" / "refiner_studio" / "best.pt"

SIZE = 512
N_TRAIN_PER_PAIR = 10   # 4 fg x 6 bg x 10 = 240
N_VAL = 16
EPOCHS = 25
MIX_STUDIO = True
BS = 8
LR = 2e-4

# ---------- 合成 ----------
def build_sources():
    """读绿幕人像 -> despill 前景 + alpha (复用 ai_material_eval 的函数)。"""
    from ai_material_eval import fix_watermark, chroma_key_alpha, despill
    fgs = {}
    for p in sorted(GREEN_DIR.glob("*.png")):
        img = fix_watermark(np.array(Image.open(p).convert("RGB")))
        a = chroma_key_alpha(img)
        fgs[p.stem] = (despill(img, a), a.astype(np.float32))
    bgs = {}
    for p in sorted(BG_DIR.glob("*.png")):
        bgs[p.stem] = np.array(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS))
    return fgs, bgs

def jitter_fg(fg: np.ndarray, rng: random.Random) -> np.ndarray:
    """前景增强: 亮度/色温抖动。"""
    out = fg.astype(np.float32)
    out *= rng.uniform(0.85, 1.12)                                  # 亮度
    out[..., 0] += rng.uniform(-12, 12)                             # R 色温
    out[..., 2] += rng.uniform(-12, 12)                             # B 色温
    return np.clip(out, 0, 255).astype(np.uint8)

def jitter_bg(bg: np.ndarray, rng: random.Random) -> np.ndarray:
    out = bg.astype(np.float32) * rng.uniform(0.85, 1.12)
    out[..., 0] += rng.uniform(-10, 10)
    out[..., 2] += rng.uniform(-10, 10)
    return np.clip(out, 0, 255).astype(np.uint8)

def compose(fg: np.ndarray, alpha: np.ndarray, bg: np.ndarray,
            rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """前景按随机缩放/翻转/位置贴到背景, 返回 (合成图, GT alpha)。"""
    H, W = SIZE, SIZE
    scale = rng.uniform(0.55, 0.95) * H / alpha.shape[0]
    nh, nw = int(alpha.shape[0] * scale), int(alpha.shape[1] * scale)
    a_img = Image.fromarray((alpha * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    f_img = Image.fromarray(fg).resize((nw, nh), Image.BILINEAR)
    if rng.random() < 0.5:
        a_img, f_img = a_img.transpose(Image.FLIP_LEFT_RIGHT), f_img.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.array(a_img).astype(np.float32) / 255.0
    f = np.array(f_img)
    px = rng.randint(max(0, W // 2 - nw // 2 - 80), min(W - nw, W // 2 - nw // 2 + 80)) if W > nw else 0
    py = max(0, H - nh - rng.randint(0, 60))                        # 底对齐留地面
    canvas_a = np.zeros((H, W), np.float32)
    canvas_f = np.zeros((H, W, 3), np.float32)
    canvas_a[py:py + nh, px:px + nw] = a
    canvas_f[py:py + nh, px:px + nw] = f
    # 轻噪声 (模拟真实拍摄)
    noise = np.random.normal(0, rng.uniform(0, 5), (H, W, 1)).astype(np.float32)
    comp = canvas_a[..., None] * canvas_f + (1 - canvas_a[..., None]) * bg.astype(np.float32)
    comp = np.clip(comp + noise, 0, 255).astype(np.uint8)
    return comp, canvas_a

def stage_synth(device: str):
    from ai_material_eval import fix_watermark, chroma_key_alpha
    from matting.matting_backend import BiRefNetMatting
    tool = BiRefNetMatting(device=device, seg_mask_size=512)
    fgs, bgs = build_sources()
    print(f"[synth] fg={len(fgs)} bg={len(bgs)}")
    for split, n_per in (("train", N_TRAIN_PER_PAIR), ("val", N_VAL // len(fgs) + 1)):
        (SYNTH_DIR / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(42 if split == "val" else 7)
        count = 0
        for fs, (fg, alpha) in sorted(fgs.items()):
            for bs_name, bg in sorted(bgs.items()):
                for k in range(n_per if split == "train" else 1):
                    if split == "val" and count >= N_VAL:
                        break
                    bg_j = jitter_bg(bg.copy(), rng)
                    comp, gt = compose(jitter_fg(fg.copy(), rng), alpha, bg_j, rng)
                    stem = f"{fs}__{bs_name}__{k:02d}"
                    Image.fromarray(comp).save(SYNTH_DIR / split / f"{stem}.png")
                    Image.fromarray((gt * 255).astype(np.uint8)).save(SYNTH_DIR / split / f"{stem}_gt.png")
                    count += 1
        print(f"[synth] {split}: {count} pairs")
    # coarse 缓存 (BiRefNet 对合成图推理)
    for split in ("train", "val"):
        (SYNTH_DIR / split / "coarse").mkdir(exist_ok=True)
        imgs = sorted(p for p in (SYNTH_DIR / split).glob("*.png") if "_gt" not in p.name)
        t0 = time.time()
        for i, p in enumerate(imgs):
            tool.predict(str(p), str(SYNTH_DIR / split / "coarse" / f"{p.stem}.png"), original_size=True)
            if (i + 1) % 40 == 0:
                print(f"  coarse {split} {i+1}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    print("[synth] done")

# ---------- 训练 ----------
def stage_train(device: str):
    import torch
    import torch.nn.functional as F
    from train_refiner import AlphaRefiner, refiner_loss

    # 数据清单
    def load_split(split):
        items = []
        for p in sorted((SYNTH_DIR / split).glob("*.png")):
            if "_gt" in p.name:
                continue
            items.append((p, SYNTH_DIR / split / f"{p.stem}_gt.png",
                          SYNTH_DIR / split / "coarse" / f"{p.stem}.png"))
        return items
    train_items, val_items = load_split("train"), load_split("val")
    # 混合域: 纳入原演播室域数据, 防灾难性遗忘 (2026-09-09 回归实验证实纯 AI 域微调会遗忘)
    if MIX_STUDIO:
        studio_dir = ROOT / "data" / "studio"
        cc = studio_dir / "coarse_cache" / "train"
        n_st = 0
        for p in sorted((studio_dir / "train").glob("*.png")):
            if p.stem.endswith("_alpha"):
                continue
            gt = studio_dir / "train" / f"{p.stem}_alpha.png"
            co = cc / f"{p.stem}.png"
            if gt.exists() and co.exists():
                train_items.append((p, gt, co))
                n_st += 1
        ccv = studio_dir / "coarse_cache" / "val"
        n_sv = 0
        for p in sorted((studio_dir / "val").glob("*.png")):
            if p.stem.endswith("_alpha"):
                continue
            gt = studio_dir / "val" / f"{p.stem}_alpha.png"
            co = ccv / f"{p.stem}.png"
            if gt.exists() and co.exists() and n_sv < 32:
                val_items.append((p, gt, co))
                n_sv += 1
        print(f"[mix] +studio train={n_st} val={n_sv}")
    print(f"[train] n_train={len(train_items)} n_val={len(val_items)}")

    net = AlphaRefiner(base=32).to(device)
    ck = torch.load(CKPT_INIT, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"])
    print(f"[train] warm start from refiner_studio (epoch={ck.get('epoch')}, best={ck.get('best'):.5f})")

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)

    def to_t(paths):
        rgb = np.array(Image.open(paths[0]).convert("RGB").resize((SIZE, SIZE))).astype("float32") / 255.0
        co = np.array(Image.open(paths[2]).convert("L").resize((SIZE, SIZE))).astype("float32") / 255.0
        gt = np.array(Image.open(paths[1]).convert("L").resize((SIZE, SIZE))).astype("float32") / 255.0
        return rgb, co, gt

    def batch(items, bs):
        sel = random.sample(items, min(bs, len(items)))
        rgbs, cos, gts = zip(*[to_t(it) for it in sel])
        return (torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).to(device),
                torch.from_numpy(np.stack(cos)).unsqueeze(1).to(device),
                torch.from_numpy(np.stack(gts)).unsqueeze(1).to(device))

    CKPT_OUT.mkdir(parents=True, exist_ok=True)
    log_f = open(CKPT_OUT / "train.log", "w", encoding="utf-8")
    metrics_f = csv.writer(open(CKPT_OUT / "metrics.csv", "w", newline="", encoding="utf-8"))
    metrics_f.writerow(["epoch", "train_loss", "val_loss", "lr", "sec"])
    best = 1e9
    t_all = time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        random.shuffle(train_items)
        losses = []
        t0 = time.time()
        for i in range(0, len(train_items), BS):
            rgb, co, gt = batch(train_items, BS)
            pred = net(rgb, co)
            loss, _ = refiner_loss(pred, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(val_items), BS):
                rgb, co, gt = batch(val_items, BS)
                l, _ = refiner_loss(net(rgb, co), gt)
                vl.append(l.item())
        tl, v = float(np.mean(losses)), float(np.mean(vl))
        metrics_f.writerow([ep, f"{tl:.5f}", f"{v:.5f}", f"{sched.get_last_lr()[0]:.2e}", f"{time.time()-t0:.0f}"])
        log_f.write(f"epoch {ep:03d} train {tl:.5f} val {v:.5f}\n"); log_f.flush()
        state = {"model": net.state_dict(), "epoch": ep, "best": v}
        torch.save(state, CKPT_OUT / "last.pt")
        if v < best:
            best = v
            torch.save(state, CKPT_OUT / "best.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:03d} train={tl:.5f} val={v:.5f} best={best:.5f} ({time.time()-t_all:.0f}s)", flush=True)
    (CKPT_OUT / "config.json").write_text(json.dumps({
        "init": str(CKPT_INIT), "epochs": EPOCHS, "bs": BS, "lr": LR, "size": SIZE,
        "n_train": len(train_items), "n_val": len(val_items), "best_val": best,
        "loss": "L1+grad+lap (refiner_loss)", "data": "ai_generated synth"} , indent=2), encoding="utf-8")
    print(f"[train] done best_val={best:.5f} -> {CKPT_OUT / 'best.pt'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="train", choices=["synth", "train"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    global EPOCHS
    if args.epochs:
        EPOCHS = args.epochs
    if args.stage == "synth":
        stage_synth(device)
    else:
        if not (SYNTH_DIR / "train").exists() or len(list((SYNTH_DIR / "train").glob("*.png"))) < 100:
            stage_synth(device)
        stage_train(device)

if __name__ == "__main__":
    main()
