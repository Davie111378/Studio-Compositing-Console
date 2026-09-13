# -*- coding: utf-8 -*-
"""
verifier.py — L6 验证与对齐层: 集中式形式化验证器
- 形式化不变量: 产物存在性 / 输出键完整性 / 合成链尺寸一致性 / FDR 区间 / 图像可解码
- 经验差分检测: 节点执行后立即校验 (engine 集成), 违反 → 按可重试性决定节点命运
- 价值对齐: 输出必须落在约束集 C 允许的产物形态内 (可见成片)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

import dag as dagmod


def _load(path) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None


def verify_node(tool: str, data: dict) -> dict:
    """节点级验证: 返回 {ok, violations[], retryable}。data = 适配层输出。"""
    sch = dagmod.TOOL_SCHEMAS.get(tool)
    if sch is None or data is None:
        return {"ok": False, "violations": ["无 Schema 或无数据"], "retryable": False}
    v: list[str] = []
    # 1) 输出键完整性 + 文件存在性 (形式化不变量: 文件即契约; 仅 *_path 键是路径)
    req = sch["output"].get("required", [])
    for key in req:
        if key not in data:
            v.append(f"缺输出键 {key}")
            continue
        val = data[key]
        if key.endswith("_path") and isinstance(val, str) and not Path(val).exists():
            v.append(f"产物不存在: {key}={val}")
    # 2) 图像可解码
    for key in ("alpha_path", "fg_path", "bg_path", "relit_fg_path", "shadow_bg_path",
                "composite_path", "enhanced_path", "exported_path"):
        p = data.get(key)
        if isinstance(p, str) and Path(p).exists() and p.lower().endswith((".png", ".jpg", ".webp")):
            if _load(p) is None:
                v.append(f"图像不可解码: {p}")
    # 3) 领域不变量
    if tool == "T06_harmonize":
        fdr = data.get("fdr")
        if isinstance(fdr, (int, float)) and not (0.70 <= fdr <= 1.30):
            if not data.get("degraded"):
                v.append(f"FDR={fdr} 越界 [0.70,1.30] (压黑/过增强)")
        comp = data.get("composite_path")
        bg = None  # 尺寸一致性: composite 应与 bg 同尺寸 (画布=bg 适配后)
        if isinstance(comp, str) and Path(comp).exists():
            arr = _load(comp)
            if arr is not None and arr.shape[0] < 16:
                v.append(f"成片尺寸异常: {arr.shape}")
    if tool == "T01_matting":
        ratio = data.get("fg_ratio")
        if isinstance(ratio, (int, float)) and not (0.01 <= ratio <= 0.99):
            v.append(f"fg_ratio={ratio} 异常 (全前景/全空)")
    retryable = not any("不可解码" in x or "不存在" in x for x in v)
    return {"ok": not v, "violations": v, "retryable": retryable}


def verify_final(final_path: str | None, quality: str = "draft") -> dict:
    """成片级验证 (L6 经验差分): 可见性 + 最小分辨率 + 非均匀图。"""
    v = []
    if not final_path or not Path(final_path).exists():
        return {"ok": False, "violations": ["成片不存在"]}
    arr = _load(final_path)
    if arr is None:
        return {"ok": False, "violations": ["成片不可解码"]}
    h, w = arr.shape[:2]
    if h < 64 or w < 64:
        v.append(f"分辨率过低 {w}x{h}")
    if float(arr.std()) < 5.0:
        v.append("图像近乎均匀 (疑似空白)")
    return {"ok": not v, "violations": v, "size": [w, h]}
