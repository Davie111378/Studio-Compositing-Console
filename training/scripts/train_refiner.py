"""
B3 - 抠取域适配训练：BiRefNet（冻结，220M）出粗 alpha + AlphaRefiner（轻量 UNet）精修。

设计理由（RTX 4060 Laptop 4GB 显存约束）：
  - BiRefNet 全参数/解码器微调：推理已占 ~3.2GB，训练需 3-4x 显存 -> 不可行
  - 冻结 BiRefNet 仅做前向（no_grad, fp16）出粗 alpha -> 显存 ~1.2GB
  - 训练 1.9M 参数 AlphaRefiner（4ch 输入 = RGB + 粗 alpha -> 精修 alpha）
    -> 训练显存 < 1.5GB，可在同卡共存

这是工业界标准的 two-stage matting 路线（粗分割 + 精修），
也是本任务"抠图域适配"的落地形式。

用法：
  # 1) 准备粗 alpha 缓存（BiRefNet 前向，仅需跑一次）
  python train_refiner.py --prepare --infer_size 512
  # 2) 训练
  python train_refiner.py --epochs 100 --bs 4 --lr 1e-3
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))

# 默认路径（兼容旧调用）；CLI 可覆盖为演播室域或其他数据域
DATA = ROOT / "data" / "matting"
CACHE = ROOT / "data" / "matting" / "coarse_cache"
CKPT_DEFAULT = ROOT / "training" / "checkpoints" / "refiner"


# ---------------------------------------------------------------- model
class ConvBnRelu(nn.Module):
    def __init__(self, i, o, k=3, s=1):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(i, o, k, s, k // 2, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.c(x)


class AlphaRefiner(nn.Module):
    """
    轻量 UNet 精修网络。
    输入: [B, 4, H, W] = RGB(归一化) + 粗 alpha
    输出: [B, 1, H, W] 精修 alpha (sigmoid)
    base=32 -> 约 1.9M 参数
    """

    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.e1 = nn.Sequential(ConvBnRelu(4, b), ConvBnRelu(b, b))
        self.e2 = nn.Sequential(ConvBnRelu(b, b * 2), ConvBnRelu(b * 2, b * 2))
        self.e3 = nn.Sequential(ConvBnRelu(b * 2, b * 4), ConvBnRelu(b * 4, b * 4))
        self.e4 = nn.Sequential(ConvBnRelu(b * 4, b * 8), ConvBnRelu(b * 8, b * 8))
        self.pool = nn.MaxPool2d(2)
        self.bott = nn.Sequential(ConvBnRelu(b * 8, b * 8), ConvBnRelu(b * 8, b * 8))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d4 = nn.Sequential(ConvBnRelu(b * 16, b * 4), ConvBnRelu(b * 4, b * 4))
        self.d3 = nn.Sequential(ConvBnRelu(b * 8, b * 2), ConvBnRelu(b * 2, b * 2))
        self.d2 = nn.Sequential(ConvBnRelu(b * 4, b), ConvBnRelu(b, b))
        self.d1 = nn.Sequential(ConvBnRelu(b * 2, b), ConvBnRelu(b, b))
        self.head = nn.Conv2d(b, 1, 1)
        # 残差：以粗 alpha 为起点，只学残差，收敛更快更稳
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb, coarse):
        x0 = torch.cat([rgb, coarse], dim=1)
        s1 = self.e1(x0)          # H
        s2 = self.e2(self.pool(s1))  # H/2
        s3 = self.e3(self.pool(s2))  # H/4
        s4 = self.e4(self.pool(s3))  # H/8
        x = self.bott(self.pool(s4))
        x = self.d4(torch.cat([self.up(x), s4], 1))
        x = self.d3(torch.cat([self.up(x), s3], 1))
        x = self.d2(torch.cat([self.up(x), s2], 1))
        x = self.d1(torch.cat([self.up(x), s1], 1))
        delta = torch.sigmoid(self.head(x))
        # 残差连接：精修 alpha = clamp(coarse + scale * (delta - 0.5))
        out = (coarse + self.residual_scale * (delta - 0.5)).clamp(0.0, 1.0)
        return out


# ---------------------------------------------------------------- loss
def laplacian(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=x.device, dtype=x.dtype)
    k = k.view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)


def refiner_loss(pred, gt, w_l1=1.0, w_grad=1.0, w_lap=0.5):
    l1 = F.l1_loss(pred, gt)
    # 梯度域：Sobel 一阶
    gx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    gy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    gxg = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    gyg = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    grad = F.l1_loss(gx, gxg) + F.l1_loss(gy, gyg)
    # Laplacian 二阶（保边缘锐度）
    lap = F.l1_loss(laplacian(pred), laplacian(gt))
    return w_l1 * l1 + w_grad * grad + w_lap * lap, {"l1": l1.item(), "grad": grad.item(), "lap": lap.item()}


# ---------------------------------------------------------------- data
def load_split(split: str):
    """按 manifest.csv 严格划分,杜绝目录残留导致的 train/val/test 泄漏。"""
    global DATA
    mf = DATA / "manifest.csv"
    names = []
    if mf.exists():
        import csv
        with open(mf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    names.append((r["image"], r.get("case", "unknown")))
    return names


def write_split_lists():
    """导出各 split 文件名清单，供评估脚本按同一划分取 GT（保证隔离一致）。"""
    for split in ["test", "val", "train"]:
        names = load_split(split)
        out = DATA / f"{split}_names.txt"
        out.write_text("\n".join(n for n, _ in names), encoding="utf-8")
    print(f"[split] wrote *_names.txt (test={len(load_split('test'))}, "
          f"val={len(load_split('val'))}, train={len(load_split('train'))})")


class RefinerDataset(Dataset):
    def __init__(self, split: str, size: int = 384, augment: bool = True,
                 oversample: str = ""):
        self.size = size
        self.augment = augment
        img_dir = DATA / split
        coarse_dir = CACHE / split
        self.items = []
        for img_name, case in load_split(split):
            p = img_dir / img_name
            gt = img_dir / f"{Path(img_name).stem}_alpha.png"
            co = coarse_dir / f"{Path(img_name).stem}.png"
            if p.exists() and gt.exists() and co.exists():
                # 按域过采样: "studio:3,..." → 该 case 的样本重复 factor 份
                # (仅内存中重复引用, 不改 manifest/磁盘)
                factor = 1
                for kv in oversample.split(","):
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        if k.strip() == case:
                            try:
                                factor = max(1, int(v))
                            except ValueError:
                                pass
                self.items.extend([(p, gt, co, case)] * factor)
        if not self.items:
            raise RuntimeError(f"[{split}] 无样本，请先运行 --prepare 生成粗 alpha 缓存")

    def __len__(self):
        return len(self.items)

    def _load(self, ip, gp, cp):
        img = Image.open(ip).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        # GT 若为 RGBA (HM matting 为前景抠图), 真值是 alpha 通道而非颜色亮度
        g_im = Image.open(gp)
        if g_im.mode in ("RGBA", "LA", "PA") or (g_im.mode == "P" and "transparency" in g_im.info):
            g_im = Image.fromarray(np.array(g_im.convert("RGBA"))[..., 3], "L")
        gt = g_im.convert("L").resize((self.size, self.size), Image.BILINEAR)
        co = Image.open(cp).convert("L").resize((self.size, self.size), Image.BILINEAR)
        return np.array(img), np.array(gt), np.array(co)

    def __getitem__(self, i):
        ip, gp, cp, _case = self.items[i]
        img, gt, co = self._load(ip, gp, cp)
        if self.augment:
            if random.random() < 0.5:  # 水平翻转
                img, gt, co = img[:, ::-1], gt[:, ::-1], co[:, ::-1]
            if random.random() < 0.6:  # 构图增广: 随机缩小+平移, 打击"主体大且居中"先验
                s = random.uniform(0.35, 1.0)
                ns = max(16, int(self.size * s))
                x0 = random.randint(0, self.size - ns)
                y0 = random.randint(0, self.size - ns)

                def place(a, fill):
                    a = np.ascontiguousarray(a)
                    small = np.array(Image.fromarray(a).resize((ns, ns), Image.BILINEAR))
                    canvas = np.full((self.size, self.size) + a.shape[2:], fill, dtype=a.dtype)
                    canvas[y0:y0 + ns, x0:x0 + ns] = small
                    return canvas

                img = place(img, img[0, 0].tolist())   # RGB 边沿复制(延伸背景)
                gt = place(gt, 0)                      # 画布区 = 背景
                co = place(co, 0)
            if random.random() < 0.3:  # 色彩抖动（只动 RGB，不动 alpha）
                img = np.clip(img.astype(np.float32) * random.uniform(0.85, 1.15)
                              + random.uniform(-12, 12), 0, 255).astype(np.uint8)
            if random.random() < 0.2:  # 轻微模糊，模拟失焦
                import cv2
                img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.4, 1.2))
        rgb = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        gta = torch.from_numpy(gt.astype(np.float32) / 255.0).unsqueeze(0)
        coa = torch.from_numpy(co.astype(np.float32) / 255.0).unsqueeze(0)
        return rgb, coa, gta, ip.stem


# ---------------------------------------------------------------- prepare
def prepare(infer_size: int = 512, device: str = "cuda"):
    global DATA, CACHE
    from matting.matting_backend import BiRefNetMatting
    write_split_lists()
    tool = BiRefNetMatting(device=device, seg_mask_size=infer_size)
    for split in ["train", "val", "test"]:
        src = DATA / split
        dst = CACHE / split
        dst.mkdir(parents=True, exist_ok=True)
        imgs = [src / n for n, _ in load_split(split)]
        t0 = time.time()
        done = skip = 0
        for p in imgs:
            if not p.exists():
                continue
            out_p = dst / f"{p.stem}.png"
            if out_p.exists() and out_p.stat().st_size > 0:
                skip += 1          # 断点续跑 / 复用既有缓存 (如 studio 域)
                continue
            tool.predict(str(p), str(out_p))
            done += 1
            if done % 200 == 0:
                print(f"  [prepare] {split}: {done} new, {skip} reused ({time.time()-t0:.0f}s)", flush=True)
        print(f"[prepare] {split}: {done} new + {skip} reused coarse alphas "
              f"in {time.time()-t0:.1f}s -> {dst}", flush=True)


# ---------------------------------------------------------------- train
def main():
    global DATA, CACHE, CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA),
                        help="数据根目录，含 {train,val,test}/ 和 manifest.csv；默认通用 matting")
    ap.add_argument("--ckpt_root", type=str, default=str(CKPT_DEFAULT),
                        help="checkpoint 输出根目录；按域命名避免互相覆盖")
    ap.add_argument("--prepare", action="store_true", help="先用 BiRefNet 生成粗 alpha 缓存")
    ap.add_argument("--infer_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--oversample", default="", help="按域过采样, 如 studio:3")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init_from", default=None, help="热启动: 仅载入模型权重继续训练(新数据域)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_every_min", type=float, default=20.0)
    args = ap.parse_args()

    # 必须在 prepare() 之前重定向，否则 prepare 会用默认（通用）数据域
    DATA = Path(args.data_root)
    CACHE = Path(args.data_root) / "coarse_cache"
    CKPT = Path(args.ckpt_root)
    CKPT.mkdir(parents=True, exist_ok=True)
    print(f"[train] data_root={DATA}  ckpt_root={CKPT}", flush=True)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if args.prepare:
        prepare(args.infer_size)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_set = RefinerDataset("train", args.size, augment=True, oversample=args.oversample)
    except RuntimeError as e:
        print(f"[train] {e}")
        print("[train] 自动执行 --prepare ...")
        prepare(args.infer_size)
        train_set = RefinerDataset("train", args.size, augment=True, oversample=args.oversample)
    val_set = RefinerDataset("val", args.size, augment=False)

    # Windows 下 workers>0 需主模块保护；DataLoader 在 __main__ 中安全
    g = torch.Generator()
    g.manual_seed(42)
    tr = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    generator=g, pin_memory=(device == "cuda"), drop_last=True)
    va = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=0)

    model = AlphaRefiner(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_ep, best = 0, 1e9
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck.get("epoch", 0) + 1
        best = ck.get("best", 1e9)
        print(f"[train] resume from epoch {start_ep}, best={best:.5f}")
    elif args.init_from and Path(args.init_from).exists():
        # 热启动: 只载模型权重(优化器/epoch 全新), 用于换数据域继续训练
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[train] warm-start weights from {args.init_from} "
              f"(src epoch={ck.get('epoch')}, src best={ck.get('best')})")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] AlphaRefiner params={n_params:.2f}M device={device} "
          f"train={len(train_set)} val={len(val_set)} bs={args.bs}")

    log_path = CKPT / "train.log"
    csv_path = CKPT / "metrics.csv"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(args)} params={n_params:.2f}M ===\n")
    if start_ep == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_l1,lr,time_s\n")

    last_save = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        tl, nb = 0.0, 0
        for rgb, coa, gta, _ in tr:
            rgb, coa, gta = rgb.to(device), coa.to(device), gta.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(rgb, coa)
                loss, parts = refiner_loss(pred, gta)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item(); nb += 1
        sched.step()

        # ---- validate
        model.eval()
        vl, vl1 = 0.0, 0.0
        with torch.no_grad():
            for rgb, coa, gta, _ in va:
                rgb, coa, gta = rgb.to(device), coa.to(device), gta.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred = model(rgb, coa)
                    l, p_ = refiner_loss(pred, gta)
                vl += l.item(); vl1 += p_["l1"]
        vl /= max(1, len(va)); vl1 /= max(1, len(va))
        dt = time.time() - t0
        msg = (f"ep {ep:03d}/{args.epochs} train={tl/max(1,nb):.5f} val={vl:.5f} "
               f"val_l1={vl1:.5f} lr={sched.get_last_lr()[0]:.2e} {dt:.1f}s")
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{tl/max(1,nb):.6f},{vl:.6f},{vl1:.6f},{sched.get_last_lr()[0]:.2e},{dt:.1f}\n")

        # ---- checkpoint（best / last / 定时）
        now = time.time()
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                 "best": best, "args": vars(args)}
        torch.save(state, CKPT / "last.pt")
        if vl < best:
            best = vl
            state["best"] = best
            torch.save(state, CKPT / "best.pt")
            print(f"  -> new best val={best:.5f} saved", flush=True)
        if (now - last_save) / 60 >= args.save_every_min:
            torch.save(state, CKPT / f"epoch_{ep:03d}.pt")
            last_save = now

    # 保存配置三件套
    (CKPT / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] done. best val={best:.5f} ckpt={CKPT}")


if __name__ == "__main__":
    main()
