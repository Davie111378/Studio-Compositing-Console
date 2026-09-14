"""
Matting 后端封装（T03）。

选型（见工作文档）：
- 主模型：BiRefNet（SOTA trimap-free matting, CVPR 2024）
  - 本地加载 HF 镜像 zhengpeng7/birefnet 的 model.safetensors（权重已从 hf-mirror 下载）
  - 见 birefnet_local.py（绕开 transformers 远程代码包导入限制）
- 兜底 1：carvekit.HiInterface（默认 tracer_b7，需 huggingface.co 权重，墙内环境可能不可用）
- 兜底 2：自实现轻量 SimplifiedBiRefNet（CPU/离线可训练），用于微调实验与降级

对外暴露统一接口 predict(image_path, out_alpha, out_fg) -> dict，方便 Agent 通过 T03_matting 工具调用。
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

# 国内镜像，避免 huggingface.co 不可达（carvekit 兜底用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class BiRefNetMatting:
    """基于本地权重的真实 BiRefNet 抠图（主后端）。"""

    def __init__(self, variant: str = "general", device: Optional[str] = None,
                 seg_mask_size: int = 1024):
        self.variant = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from matting.birefnet_local import BiRefNetLocal
        self.tool = BiRefNetLocal(device=self.device, infer_size=seg_mask_size)
        self.backend = "birefnet-local"

    @torch.no_grad()
    def predict(self, image_path: str, out_alpha_path: str,
                out_fg_path: Optional[str] = None, original_size: bool = True) -> Dict[str, Any]:
        return self.tool.predict(image_path, out_alpha_path, out_fg_path, original_size)


def build_matting_tool(variant: str = "general", device: Optional[str] = None,
                       weight_path: Optional[str] = None):
    """工厂：优先真实 BiRefNet（本地权重），失败回退 carvekit，再回退 SimplifiedBiRefNet。"""
    # 1) 本地 BiRefNet
    try:
        return BiRefNetMatting(variant=variant, device=device), "BiRefNet-real"
    except Exception as e:
        print(f"[matting] BiRefNet-local 失败: {e}")
    # 2) carvekit
    try:
        from carvekit.api.high import HiInterface
        interface = HiInterface(object_type="object", device=device or "cpu",
                                seg_mask_size=1024, fp16=(device == "cuda"))
        return _CarvekitWrapper(interface), "carvekit-tracer_b7"
    except Exception as e:
        print(f"[matting] carvekit 失败: {e}")
    # 3) 简化版
    from matting.matting_tool import MattingTool
    return MattingTool(weight_path=weight_path, device=device or "cpu"), "SimplifiedBiRefNet"


class _CarvekitWrapper:
    def __init__(self, interface):
        self.interface = interface

    @torch.no_grad()
    def predict(self, image_path, out_alpha_path, out_fg_path=None, original_size=True):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        rgba = self.interface([image_path])[0].convert("RGBA")
        alpha = np.array(rgba.split()[-1]).astype(np.uint8)
        if original_size and (alpha.shape[1] != orig_w or alpha.shape[0] != orig_h):
            alpha = np.array(Image.fromarray(alpha).resize((orig_w, orig_h), Image.BILINEAR))
        alpha_pil = Image.fromarray(alpha)
        alpha_pil.save(out_alpha_path)
        fg_path = out_fg_path or str(Path(out_alpha_path).with_name(Path(out_alpha_path).stem + "_fg.png"))
        arr = np.array(img).astype(np.float32)
        a = alpha.astype(np.float32) / 255.0
        a = a[..., None]
        fg = arr * a + np.ones_like(arr) * 255.0 * (1 - a)
        Image.fromarray(fg.astype(np.uint8)).save(fg_path)
        return {"alpha_path": out_alpha_path, "fg_path": fg_path, "latency_ms": 0,
                "memory_mb": 0, "model_version": "carvekit-tracer_b7"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_alpha", required=True)
    ap.add_argument("--out_fg", default=None)
    args = ap.parse_args()
    tool, name = build_matting_tool()
    print("[matting] backend:", name)
    print(tool.predict(args.image, args.out_alpha, args.out_fg))
