"""
真实 BiRefNet 推理实现（SOTA 抠图，zhengpeng7/BiRefNet, CVPR 2024）。

使用官方 `birefnet` 包加载公开权重，支持：
- general / portrait / general-lite / dis 等多种变体
- 自动从 HuggingFace 下载权重（可配置镜像 HF_ENDPOINT）
- CUDA / CPU 自动选择，推理时显存自适应

注意：本文件替代早期的占位实现（birefnet_real.py 原简化版仅用于验证流程）。
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


# 各变体对应的 HuggingFace repo id（公开权重）
BIREFNET_REPOS = {
    "general": "zhengpeng7/birefnet",
    "general-lite": "zhengpeng7/birefnet-general-lite",
    "portrait": "zhengpeng7/birefnet-portrait",
    "matting": "zhengpeng7/birefnet-matting",
    "dis": "zhengpeng7/birefnet-dis",
}


class BiRefNetRealTool:
    """基于真实 BiRefNet 权重的抠图工具。"""

    def __init__(
        self,
        variant: str = "general",
        device: Optional[str] = None,
        weights_dir: str = "D:/AIcode/生产实习/training/checkpoints/birefnet",
        torch_dtype: str = "auto",
        infer_size: int = 1024,
    ):
        self.variant = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.infer_size = infer_size

        # 显存自适应：4GB 显存时默认降到 1024 也能跑；更大显存可 1280
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            total_gb = total / 1024**3
            if total_gb <= 5.0:
                self.infer_size = min(self.infer_size, 1024)
            else:
                self.infer_size = min(self.infer_size, 1280)

        self.dtype = torch.float32
        if torch_dtype == "auto" and self.device == "cuda":
            # 4GB 显存下 fp32 更稳，避免 fp16 数值问题；如 OOM 可切 fp16
            self.dtype = torch.float32

        self.model = self._load_model()

    def _load_model(self):
        try:
            from birefnet import BiRefNet
        except Exception as e:
            raise ImportError(
                "请先安装 birefnet 包：`pip install birefnet`。错误: " + str(e)
            )
        repo_id = BIREFNET_REPOS.get(self.variant, self.variant)
        # 优先使用本地缓存权重
        local_ckpt = self.weights_dir / f"birefnet_{self.variant}.pth"
        extra = {}
        try:
            print(f"[BiRefNet] loading '{repo_id}' on {self.device} ...")
            model = BiRefNet.from_pretrained(repo_id)
            model.to(self.device).to(self.dtype)
            model.eval()
            if local_ckpt.exists():
                sd = torch.load(local_ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(sd, strict=False)
                print(f"[BiRefNet] merged local ckpt {local_ckpt}")
            return model
        except Exception as e:
            print(f"[BiRefNet] from_pretrained failed: {e}")
            raise

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        out_alpha_path: str,
        out_fg_path: Optional[str] = None,
        original_size: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # BiRefNet 预处理：resize 到 infer_size，归一化（ImageNet 均值）
        im_t = F.interpolate(
            torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0,
            size=(self.infer_size, self.infer_size),
            mode="bilinear",
            align_corners=False,
        ).to(self.device).to(self.dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        im_t = (im_t - mean) / std

        try:
            pred = self.model(im_t)[0, 0].float().cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # 降级到更小分辨率重试
                self.infer_size = 768
                small = F.interpolate(
                    im_t, size=(768, 768), mode="bilinear", align_corners=False
                )
                pred = self.model(small)[0, 0].float().cpu().numpy()
            else:
                raise

        alpha = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        alpha_pil = Image.fromarray(alpha)
        if original_size:
            alpha_pil = alpha_pil.resize((orig_w, orig_h), Image.BILINEAR)

        alpha_pil.save(out_alpha_path)

        fg_path = out_fg_path or str(
            Path(out_alpha_path).with_name(Path(out_alpha_path).stem + "_fg.png")
        )
        arr = np.array(img).astype(np.float32)
        a = np.array(alpha_pil).astype(np.float32) / 255.0
        a = a[..., None]
        white_bg = np.ones_like(arr) * 255.0
        fg = arr * a + white_bg * (1 - a)
        Image.fromarray(fg.astype(np.uint8)).save(fg_path)

        torch.cuda.empty_cache()
        gc.collect()
        dt = (time.time() - t0) * 1000
        mem = torch.cuda.max_memory_allocated(0) // 1024 // 1024 if self.device == "cuda" else 0
        return {
            "alpha_path": out_alpha_path,
            "fg_path": fg_path,
            "latency_ms": dt,
            "memory_mb": mem,
            "model_version": f"BiRefNet-{self.variant}",
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_alpha", required=True)
    ap.add_argument("--out_fg", default=None)
    ap.add_argument("--variant", default="general")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    tool = BiRefNetRealTool(variant=args.variant, device=args.device)
    result = tool.predict(args.image, args.out_alpha, args.out_fg)
    print("[BiRefNet] done:", result)
