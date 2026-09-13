"""
BiRefNet 抠图训练脚本（CPU 友好）
- 解码器 + 轻量 head 微调（LoRA 风格：冻结 encoder）
- 支持断点续训
- 每 20-30 分钟存 checkpoint
- 单 epoch ≤ 24 小时
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import csv
import random
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 让脚本可直接运行
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from synthetic_dataset import build_synthetic_dataset  # noqa: E402

# 引入模型
sys.path.insert(0, str(THIS.parent.parent / "ai-service" / "src" / "matting"))
from matting_tool import SimplifiedBiRefNet  # noqa: E402


class MattingDataset(Dataset):
    """从 (image, alpha) 文件对读取的训练集。"""

    def __init__(self, root: Path, split: str = "train", size: Tuple[int, int] = (384, 384)):
        self.root = Path(root) / split
        self.size = size
        self.tx = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        self.alpha_tx = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        self.samples = []
        for img_path in sorted(self.root.glob("*.png")):
            if img_path.stem.endswith("_alpha"):
                continue
            alpha_path = self.root / f"{img_path.stem}_alpha.png"
            if alpha_path.exists():
                self.samples.append((img_path, alpha_path))
        # 过滤
        self.samples = [s for s in self.samples if s[0].exists() and s[1].exists()]
        print(f"[dataset] split={split} n={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, alpha_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        alpha = Image.open(alpha_p).convert("L")
        return self.tx(img), self.alpha_tx(alpha)


def compute_loss(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """alpha prediction loss = L1 + Laplacian (gradient) loss."""
    l1 = F.l1_loss(pred, gt)
    # 简单梯度损失（Laplacian）
    pred_dx = pred[..., 1:] - pred[..., :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    gt_dx = gt[..., 1:] - gt[..., :-1]
    gt_dy = gt[..., 1:, :] - gt[..., :-1, :]
    grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
    total = l1 + 0.5 * grad
    return total, {"l1": l1.item(), "grad": grad.item()}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    use_amp = (device == "cuda")

    # 1. 数据准备
    data_root = Path(args.data_root)
    if args.synth_n > 0 and not (data_root / "train").exists():
        print("[train] synthetic dataset not found, generating...")
        build_synthetic_dataset(data_root, n_per_case=args.synth_n)
    train_ds = MattingDataset(data_root, "train", size=(args.size, args.size))
    val_ds = MattingDataset(data_root, "val", size=(args.size, args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. 模型
    model = SimplifiedBiRefNet().to(device)
    if args.resume and Path(args.resume).exists():
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"[train] resumed from {args.resume}")

    # 3. 优化器（默认只微调解码器 + head；--train_all 时全量训练）
    encoder_params, decoder_params = [], []
    for name, p in model.named_parameters():
        if (name.startswith("enc") or name.startswith("bottleneck")) and not args.train_all:
            p.requires_grad = False  # 冻结 encoder
            encoder_params.append(p)
        else:
            decoder_params.append(p)
    opt = torch.optim.Adam(model.parameters() if args.train_all else decoder_params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 4. 训练循环
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    cfg_path = save_dir / "config.json"
    cfg_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    best_val = float("inf")
    step = 0
    t_start = time.time()
    last_ckpt_time = t_start

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (img, alpha_gt) in enumerate(train_loader):
            img = img.to(device)
            alpha_gt = alpha_gt.to(device)
            pred = model(img)
            loss, metrics = compute_loss(pred, alpha_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            now = time.time()
            if step % args.log_every == 0:
                elapsed = now - t_start
                msg = f"[epoch={epoch} step={step}] loss={loss.item():.4f} l1={metrics['l1']:.4f} grad={metrics['grad']:.4f} elapsed={elapsed:.1f}s"
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
            if now - last_ckpt_time > args.ckpt_interval_sec:
                ckpt_path = save_dir / f"checkpoint_step{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                last_ckpt_time = now
                print(f"[ckpt] saved {ckpt_path}")
        # val
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for img, alpha_gt in val_loader:
                img = img.to(device)
                alpha_gt = alpha_gt.to(device)
                pred = model(img)
                loss, _ = compute_loss(pred, alpha_gt)
                val_loss_sum += loss.item() * img.size(0)
                val_n += img.size(0)
        val_loss = val_loss_sum / max(1, val_n)
        msg = f"[epoch={epoch}] val_loss={val_loss:.4f}"
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()
        # save best / last
        torch.save(model.state_dict(), save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
            msg = f"[best] val_loss={val_loss:.4f} updated"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()
        # 早停
        if args.early_stop_epoch > 0 and epoch >= args.early_stop_epoch:
            print(f"[early stop] reached epoch {epoch}")
            break

    log_f.close()
    print(f"[train] done. best_val={best_val:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="D:/AIcode/生产实习/data/matting")
    ap.add_argument("--save_dir", default="D:/AIcode/生产实习/training/checkpoints/baseline_v1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--synth_n", type=int, default=40)  # 每类 40 张 → ~320 总
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--ckpt_interval_sec", type=int, default=1200)  # 20min
    ap.add_argument("--early_stop_epoch", type=int, default=0)
    args = ap.parse_args()
    train(args)
