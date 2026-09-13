"""
BiRefNet 本地加载器（真实权重，来自 HF 镜像 zhengpeng7/birefnet）。

绕开 `transformers.from_pretrained` 的远程代码包导入限制：
- 直接实例化 birefnet_official.birefnet.BiRefNet()
- 用 safetensors 加载 model.safetensors

推理：resize 1024 → ImageNet 归一化 → forward → 取 [-1] → sigmoid → 还原尺寸。
"""
from __future__ import annotations

import os
import time
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

LOCAL_DIR = Path(__file__).resolve().parent / "birefnet_official"
WEIGHTS = LOCAL_DIR / "model.safetensors"
# 微调权重 (如果存在则优先使用)
FINETUNED_WEIGHTS = Path(__file__).resolve().parent.parent.parent.parent / "training" / "checkpoints" / "birefnet_hm" / "finetuned.safetensors"

# 确保 birefnet_official 包可被导入（它在 src/matting/ 下）
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class BiRefNetLocal:
    def __init__(self, device: Optional[str] = None, infer_size: int = 1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.infer_size = infer_size
        # 4GB 显存自适应
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            if total / 1024**3 <= 5.0:
                self.infer_size = min(self.infer_size, 1024)
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"BiRefNet 权重缺失: {WEIGHTS}（请从 hf-mirror 下载）")
        from birefnet_official import birefnet
        self.model = birefnet.BiRefNet()
        from safetensors.torch import load_file
        # 优先加载微调权重
        if FINETUNED_WEIGHTS.exists():
            print(f"[BiRefNet-local] 加载微调权重: {FINETUNED_WEIGHTS.name}")
            sd = load_file(str(FINETUNED_WEIGHTS), device="cpu")
            self.model_version = "BiRefNet-finetuned-HM"
        else:
            sd = load_file(str(WEIGHTS), device="cpu")
            self.model_version = "BiRefNet-general-local"
        # 兼容可能的 'model.' 前缀；权重为 F16，转 fp32 以匹配模型
        clean = {}
        for k, v in sd.items():
            kk = k.replace("model.", "")
            clean[kk] = v.float() if v.is_floating_point() else v
        miss, unexp = self.model.load_state_dict(clean, strict=False)
        print(f"[BiRefNet-local] loaded {WEIGHTS.name} missing={len(miss)} unexpected={len(unexp)}")
        self.model.to(self.device).eval()
        print(f"[BiRefNet-local] device={self.device} infer_size={self.infer_size}")

    @torch.no_grad()
    def predict(self, image_path: str, out_alpha_path: str,
                out_fg_path: Optional[str] = None, original_size: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        im = img.resize((self.infer_size, self.infer_size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im).transpose(2, 0, 1)).float() / 255.0
        x = (x - torch.tensor(_MEAN).view(3, 1, 1)) / torch.tensor(_STD).view(3, 1, 1)
        x = x.unsqueeze(0).to(self.device)
        try:
            out = self.model(x)
            pred = out[-1] if isinstance(out, list) else out
            if isinstance(pred, list):
                pred = pred[-1]
            alpha = torch.sigmoid(pred)[0, 0].cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                self.infer_size = 768
                x = F.interpolate(x, size=(768, 768), mode="bilinear", align_corners=False)
                out = self.model(x)
                pred = out[-1] if isinstance(out, list) else out
                if isinstance(pred, list):
                    pred = pred[-1]
                alpha = torch.sigmoid(pred)[0, 0].cpu().numpy()
            else:
                raise
        alpha = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        alpha_pil = Image.fromarray(alpha)
        if original_size:
            alpha_pil = alpha_pil.resize((orig_w, orig_h), Image.BILINEAR)
        alpha_pil.save(out_alpha_path)
        fg_path = out_fg_path or str(Path(out_alpha_path).with_name(Path(out_alpha_path).stem + "_fg.png"))
        arr = np.array(img).astype(np.float32)
        a = np.array(alpha_pil).astype(np.float32) / 255.0
        a = a[..., None]
        fg = arr * a + np.ones_like(arr) * 255.0 * (1 - a)
        Image.fromarray(fg.astype(np.uint8)).save(fg_path)
        torch.cuda.empty_cache()
        gc.collect()
        dt = (time.time() - t0) * 1000
        mem = torch.cuda.max_memory_allocated(0) // 1024 // 1024 if self.device == "cuda" else 0
        return {"alpha_path": out_alpha_path, "fg_path": fg_path, "latency_ms": dt,
                "memory_mb": mem, "model_version": self.model_version}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_alpha", required=True)
    ap.add_argument("--out_fg", default=None)
    args = ap.parse_args()
    tool = BiRefNetLocal()
    print(tool.predict(args.image, args.out_alpha, args.out_fg))
