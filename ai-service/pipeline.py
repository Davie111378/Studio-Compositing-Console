"""
演播室图像合成统一 Pipeline（串联 T03 → T04 → T05 → T06 → T07）
暴露给 Agent 的入口函数：run_pipeline()

自动选择：
- 如果有 BiRefNet 公开权重 → 真实 BiRefNet（GPU）
- 否则 → 简化版（CPU / 离线）
"""

from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from PIL import Image

# 让脚本可直接运行
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS / "src"))

from matting.matting_tool import MattingTool, SimplifiedBiRefNet  # noqa
from matting.matting_backend import build_matting_tool, BiRefNetMatting  # noqa
from relighting.relight_tool import RelightTool  # noqa
from shadow.shadow_tool import ShadowTool  # noqa
from harmonization.harmonize_tool import HarmonizeTool  # noqa
from composite.composite_tool import CompositeTool  # noqa
from fx.fx_tool import FxTool  # noqa: 特效模块 (区域特效/聚光灯/雾效等)


def select_matting_tool(weight_path: Optional[str], device: str) -> Any:
    """智能选择：本地 BiRefNet（真实权重）→ carvekit → 简化版。"""
    tool, name = build_matting_tool(device=device, weight_path=weight_path)
    return tool, name


def run_pipeline(
    image_path: str,
    bg_path: str,
    out_dir: str,
    matting_weight: Optional[str] = None,
    enable_relight: bool = True,
    enable_shadow: bool = True,
    enable_harmonize: bool = True,
    light_hint: Optional[Dict[str, Any]] = None,
    shadow_hint: Optional[Dict[str, Any]] = None,
    composite_method: str = "alpha_over",
    device: Optional[str] = None,
    precomputed_alpha: Optional[str] = None,
    fx_chain: Optional[list] = None,
) -> Dict[str, Any]:
    """
    完整演播室图像合成流水线。

    Args:
        image_path: 输入图像（含前景）
        bg_path: 目标背景
        out_dir: 输出目录
        matting_weight: BiRefNet 权重路径（None → 自动选择）
        enable_relight / enable_shadow / enable_harmonize: 子模块开关
        light_hint / shadow_hint: 提示参数
        composite_method: alpha_over | poisson
        device: cuda / cpu / None（自动选择）
    Returns:
        dict with all intermediate paths + final output
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    intermediate = out_dir_p / "intermediate"
    intermediate.mkdir(exist_ok=True)
    result: Dict[str, Any] = {"stages": [], "device": device}

    # T03 - Matting
    # 若上游已算好 alpha（如四组对比实验复用同一张 alpha），直接跳过推理，省 3/4 开销
    t0 = time.time()
    if precomputed_alpha and Path(precomputed_alpha).exists():
        import shutil
        alpha_path = str(intermediate / "alpha.png")
        shutil.copy(precomputed_alpha, alpha_path)
        src = Image.open(image_path).convert("RGB")
        a = Image.open(alpha_path).convert("L").resize(src.size, Image.BILINEAR)
        # 与 BiRefNetMatting.predict 保持一致的 fg 约定：白底预乘 img*a + 255*(1-a)
        arr = np.array(src).astype(np.float32)
        af = (np.array(a).astype(np.float32) / 255.0)[..., None]
        fg = arr * af + 255.0 * (1.0 - af)
        fg_path = str(intermediate / "fg.png")
        Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8)).save(fg_path)
        matting_out = {"alpha_path": alpha_path, "fg_path": fg_path}
        model_name = "precomputed"
    else:
        matting, model_name = select_matting_tool(matting_weight, device)
        matting_out = matting.predict(
            image_path,
            str(intermediate / "alpha.png"),
            str(intermediate / "fg.png"),
        )
        fg_path = matting_out["fg_path"]
        alpha_path = matting_out["alpha_path"]
    result["stages"].append({"name": "T03_matting", "time_ms": (time.time() - t0) * 1000,
                             "out": matting_out, "model": model_name})

    # T06 - Harmonize (前置调色)
    if enable_harmonize:
        t0 = time.time()
        harm = HarmonizeTool().harmonize(fg_path, bg_path, str(intermediate / "fg_harm.png"), alpha_path)
        fg_path = harm["harmonized_fg_path"]
        result["stages"].append({"name": "T06_harmonize", "time_ms": (time.time() - t0) * 1000, "out": harm})

    # T04 - Relight
    if enable_relight:
        t0 = time.time()
        relit = RelightTool().relight(fg_path, bg_path, str(intermediate / "fg_relit.png"), light_hint)
        fg_path = relit["relit_fg_path"]
        result["stages"].append({"name": "T04_relight", "time_ms": (time.time() - t0) * 1000, "out": relit})

    # T05 - Shadow
    shadow_path = None
    if enable_shadow:
        t0 = time.time()
        sh = ShadowTool().generate(alpha_path, bg_path, str(intermediate / "bg_with_shadow.png"), shadow_hint)
        shadow_path = sh["shadow_path"]
        bg_for_comp = shadow_path
        result["stages"].append({"name": "T05_shadow", "time_ms": (time.time() - t0) * 1000, "out": sh})
    else:
        bg_for_comp = bg_path

    # T07 - Composite
    # 注意：若启用了 T05，bg_for_comp 已是含阴影的背景图，composite 不再重复传 shadow_path
    t0 = time.time()
    comp = CompositeTool(method=composite_method).composite(
        fg_path, alpha_path, bg_for_comp, str(out_dir_p / "final.png"),
        shadow_path=shadow_path if not enable_shadow else None,
    )
    result["stages"].append({"name": "T07_composite", "time_ms": (time.time() - t0) * 1000, "out": comp})

    result["final"] = comp["output_path"]
    result["intermediate_dir"] = str(intermediate)
    result["total_time_ms"] = sum(s["time_ms"] for s in result["stages"])

    # T-FX 特效链 (可选): [{"effect":"spotlight","params":{...},"region":{...}}, ...]
    if fx_chain:
        t0 = time.time()
        fx_out = str(out_dir_p / "final_fx.png")
        fxr = FxTool().chain(result["final"], fx_out, fx_chain, str(intermediate))
        result["stages"].append({"name": "TFX_chain", "time_ms": (time.time() - t0) * 1000,
                                 "out": {"n_effects": len(fx_chain), **{k: v for k, v in fxr.items() if k != "chain"}}})
        result["final_fx"] = fxr.get("out_path")
        result["final"] = fxr.get("out_path", result["final"])
    return result


# Agent tool schema 定义（与 A 组对齐）
TOOL_SCHEMAS = {
    "T03_matting": {
        "name": "matting",
        "input": {"image_path": "string", "options": "object?"},
        "output": {"alpha_path": "string", "fg_path": "string", "mask_meta": "object"},
        "error_codes": {"E001": "image not found", "E002": "model weights missing"},
        "latency_ms": 1500,
    },
    "T04_relight": {
        "name": "relight",
        "input": {"fg_path": "string", "bg_path": "string", "light_hint": "object?"},
        "output": {"relit_fg_path": "string"},
        "error_codes": {"E001": "fg not found", "E002": "bg not found"},
        "latency_ms": 800,
    },
    "T05_shadow": {
        "name": "shadow",
        "input": {"alpha_path": "string", "bg_path": "string", "shadow_hint": "object?"},
        "output": {"shadow_path": "string"},
        "error_codes": {"E001": "alpha not found"},
        "latency_ms": 200,
    },
    "T06_harmonize": {
        "name": "harmonize",
        "input": {"fg_path": "string", "bg_path": "string", "alpha_path": "string?"},
        "output": {"harmonized_fg_path": "string"},
        "error_codes": {"E001": "fg not found"},
        "latency_ms": 600,
    },
    "T07_composite": {
        "name": "composite",
        "input": {"fg_path": "string", "alpha_path": "string", "bg_path": "string", "shadow_path": "string?", "method": "string"},
        "output": {"output_path": "string"},
        "error_codes": {"E001": "input not found"},
        "latency_ms": 100,
    },
}


def export_tool_schemas(out_path: str):
    """导出 tool schemas 给 Agent (A 组)."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(TOOL_SCHEMAS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[schemas] exported to {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--no_relight", action="store_true")
    ap.add_argument("--no_shadow", action="store_true")
    ap.add_argument("--no_harmonize", action="store_true")
    args = ap.parse_args()

    result = run_pipeline(
        args.image, args.bg, args.out_dir,
        matting_weight=args.weight,
        enable_relight=not args.no_relight,
        enable_shadow=not args.no_shadow,
        enable_harmonize=not args.no_harmonize,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
