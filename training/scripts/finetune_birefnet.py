# -*- coding: utf-8 -*-
"""finetune_birefnet.py — 微调真实 BiRefNet 模型以适配 HM-1k 域
策略:
  - 冻结 encoder (backbone), 只训练 decoder + head
  - 使用 HM-1k + P3M 增强数据 (JPEG压缩+下采样)
  - L1 + Laplacian gradient loss
  - 小 batch (1-2), 梯度累积
  - 保存为 finetuned.safetensors, 可直接被 BiRefNetLocal 加载
"""
from __future__ import annotations
import os, sys, time, json, argparse, random, gc
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 路径
ROOT = Path(__file__).resolve().parent.parent.parent
MATTING_SRC = ROOT / "ai-service" / "src" / "matting"
sys.path.insert(0, str(MATTING_SRC))

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class HMMattingDataset(Dataset):
    """HM + P3M 增强数据集"""
    def __init__(self, root: Path, size: int = 512, augment: bool = True):
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.samples = []
        for img_path in sorted(self.root.glob("*.png")):
            if img_path.stem.endswith("_alpha"):
                continue
            alpha_path = self.root / f"{img_path.stem}_alpha.png"
            if alpha_path.exists():
                self.samples.append((str(img_path), str(alpha_path)))
        print(f"[dataset] {self.root.name} n={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, alpha_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        alpha = Image.open(alpha_p).convert("L")
        # resize 到训练尺寸
        img = img.resize((self.size, self.size), Image.BILINEAR)
        alpha = alpha.resize((self.size, self.size), Image.BILINEAR)
        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                alpha = alpha.transpose(Image.FLIP_LEFT_RIGHT)
            # 色彩抖动
            if random.random() < 0.3:
                jitter = random.uniform(-0.1, 0.1)
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.clip(arr + jitter, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
        # 转tensor + ImageNet 归一化
        x = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        x = (x - torch.tensor(_MEAN).view(3, 1, 1)) / torch.tensor(_STD).view(3, 1, 1)
        y = torch.from_numpy(np.array(alpha).astype(np.float32) / 255.0).unsqueeze(0)
        return x, y


def compute_loss(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """L1 + Laplacian gradient loss"""
    l1 = F.l1_loss(pred, gt)
    pred_dx = pred[..., 1:] - pred[..., :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    gt_dx = gt[..., 1:] - gt[..., :-1]
    gt_dy = gt[..., 1:, :] - gt[..., :-1, :]
    grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
    # BCE 辅助 (强化二值化能力, 对付 HM 的过检/漏检)
    bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")
    total = l1 + 0.5 * grad + 0.2 * bce
    return total, {"l1": l1.item(), "grad": grad.item(), "bce": bce.item()}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[train] device={device}")

    # 1. 数据
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    train_ds = HMMattingDataset(train_dir, size=args.size, augment=True)
    val_ds = HMMattingDataset(val_dir, size=args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. 模型 — 加载真实 BiRefNet
    from birefnet_official import birefnet
    model = birefnet.BiRefNet()
    weights_path = MATTING_SRC / "birefnet_official" / "model.safetensors"
    from safetensors.torch import load_file
    sd = load_file(str(weights_path), device="cpu")
    clean = {}
    for k, v in sd.items():
        kk = k.replace("model.", "")
        clean[kk] = v.float() if v.is_floating_point() else v
    miss, unexp = model.load_state_dict(clean, strict=False)
    print(f"[model] loaded BiRefNet missing={len(miss)} unexpected={len(unexp)}")
    model.to(device)

    # 3. 冻结 encoder — 识别 backbone 参数
    # BiRefNet 的参数名通常以 encoder.layer 开头, decoder 以 decoder 开头
    encoder_params, decoder_params = [], []
    n_enc, n_dec = 0, 0
    for name, p in model.named_parameters():
        # BiRefNet backbone 通常是 feature_extraction 或 encoder
        is_backbone = any(name.startswith(prefix) for prefix in
                         ("encoder.", "backbone.", "feature_extraction.",
                          "bb.", "module.encoder", "module.bb"))
        if is_backbone and not args.train_all:
            p.requires_grad = False
            encoder_params.append(p)
            n_enc += 1
        else:
            decoder_params.append(p)
            n_dec += 1
    print(f"[model] encoder params(frozen): {n_enc}, decoder params(trainable): {n_dec}")

    # BN 层: --train_all 时不冻结 (让 BN 统计量适应新域); 冻结模式下设 eval
    if not args.train_all:
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.BatchNorm3d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        print("[model] BatchNorm layers set to eval + frozen (encoder frozen mode)")
    else:
        print("[model] BatchNorm layers trainable (full fine-tuning mode)")

    # 如果没有识别到 encoder (参数名模式不同), 按比例冻结前 70%
    if n_enc == 0 and not args.train_all:
        all_params = list(model.named_parameters())
        freeze_n = int(len(all_params) * 0.7)
        for i, (name, p) in enumerate(all_params):
            if i < freeze_n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        decoder_params = trainable
        print(f"[model] fallback: froze first {freeze_n}/{len(all_params)} params, trainable={len(trainable)}")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 4. 训练循环
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(save_dir / "train.log", "a", encoding="utf-8")

    best_val = float("inf")
    step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (img, alpha_gt) in enumerate(train_loader):
            img = img.to(device)
            alpha_gt = alpha_gt.to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(img)
                # BiRefNet forward:
                #   training: [scaled_preds(list), class_preds_lst(list)]
                #   eval: scaled_preds(list)
                # 递归找到最后一个 tensor
                def _extract_last_tensor(o):
                    """递归遍历嵌套 list/tuple, 返回最后一个 Tensor"""
                    if isinstance(o, torch.Tensor):
                        return o
                    if isinstance(o, (list, tuple)):
                        for item in reversed(o):
                            if item is not None:
                                r = _extract_last_tensor(item)
                                if r is not None:
                                    return r
                    return None
                pred = _extract_last_tensor(out)
                if pred is None:
                    print(f"[warn] step {step}: model output has no tensor, skipping", flush=True)
                    opt.zero_grad()
                    continue
                pred = torch.sigmoid(pred)
                loss, metrics = compute_loss(pred, alpha_gt)
            # 梯度累积
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                msg = f"[e={epoch} s={step}] loss={loss.item()*args.grad_accum:.4f} l1={metrics['l1']:.4f} grad={metrics['grad']:.4f} bce={metrics['bce']:.4f} {elapsed:.0f}s"
                print(msg, flush=True)
                log_f.write(msg + "\n")
                log_f.flush()

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for img, alpha_gt in val_loader:
                img = img.to(device)
                alpha_gt = alpha_gt.to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = model(img)
                    def _extract(o):
                        if isinstance(o, torch.Tensor):
                            return o
                        if isinstance(o, (list, tuple)):
                            for item in reversed(o):
                                if item is not None:
                                    r = _extract(item)
                                    if r is not None:
                                        return r
                        return None
                    pred = _extract(out)
                    pred = torch.sigmoid(pred)
                    loss, _ = compute_loss(pred, alpha_gt)
                val_loss_sum += loss.item() * img.size(0)
                val_n += img.size(0)
        val_loss = val_loss_sum / max(1, val_n)
        msg = f"[epoch={epoch}] val_loss={val_loss:.4f}"
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

        # 保存 checkpoint (state_dict, float32)
        model.cpu()
        sd = model.state_dict()
        clean_sd = {k: v.float() for k, v in sd.items()}
        torch.save(clean_sd, save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(clean_sd, save_dir / "best.pt")
            print(f"[best] val_loss={val_loss:.4f}", flush=True)
        model.to(device)
        torch.cuda.empty_cache()
        gc.collect()

    log_f.close()
    print(f"[train] done. best_val={best_val:.4f}")

    # 保存为 safetensors 格式 (供 BiRefNetLocal 加载)
    best_sd = torch.load(save_dir / "best.pt", map_location="cpu", weights_only=True)
    try:
        from safetensors.torch import save_file
        save_file(best_sd, str(save_dir / "finetuned.safetensors"))
        print(f"[train] saved finetuned.safetensors → {save_dir}")
    except Exception as e:
        print(f"[train] safetensors save failed: {e}, .pt available at {save_dir / 'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="D:/AIcode/生产实习/data/matting/hm_train")
    ap.add_argument("--val_dir", default="D:/AIcode/生产实习/data/matting/hm_val")
    ap.add_argument("--save_dir", default="D:/AIcode/生产实习/training/checkpoints/birefnet_hm")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--train_all", action="store_true", help="全量训练(不冻结)")
    args = ap.parse_args()
    train(args)
