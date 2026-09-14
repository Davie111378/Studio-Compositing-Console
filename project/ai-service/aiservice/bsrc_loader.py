"""B 组算法引擎（bsrc/）加载与 A 组契约换算层。

引擎选择（环境变量 IMC_ENGINE）：
    auto (默认)  cv2 可用 -> B 组算法；否则回落 A 组 mock 引擎
    bsrc         强制 B 组算法（缺依赖直接抛错，便于发现配置问题）
    mock         强制 A 组 mock 引擎（回归对比 / 无 opencv 环境）

坐标/单位换算（两边约定不同，全部收敛在这里）：
    A 组 T03/T04/T05: light_dir = {azimuth: 度, polar: 度}
        azimuth 0=光从右 90=从上 180=从左 270=从下；polar 0=地平线 90=头顶
        color_temp 为开尔文数值（约 2400 暖 ~ 8000 冷）
    B 组: light_dir = [dx, dy]（指向光源，图像平面 y 向下为正）
        色温为 warm/cool/neutral 标签
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

from aiservice.common import clamp

logger = logging.getLogger("aiservice.bsrc")

BSRC_DIR = Path(__file__).resolve().parent.parent / "bsrc"

_upstream_rev = "b3107bc42dd6c191c6f20f54dad8f56aefb25bad"


def _try_import_cv2():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def engine_mode() -> str:
    mode = os.environ.get("IMC_ENGINE", "auto").strip().lower()
    if mode not in ("auto", "bsrc", "mock"):
        mode = "auto"
    return mode


def available() -> bool:
    """B 组引擎在此进程是否可用（cv2 + numpy 就绪）。"""
    return BSRC_DIR.is_dir() and _try_import_cv2()


def use_bsrc() -> bool:
    mode = engine_mode()
    if mode == "mock":
        return False
    if mode == "bsrc":
        return True
    return available()


def _ensure_path() -> None:
    """bsrc 根目录进 sys.path：matting_backend 等上游模块按 src 风格互导（与 B 组 pipeline.py 一致）。"""
    root = str(BSRC_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


# ---- 各算法的惰性加载（缺依赖抛 ImportError，由 impl 回落 mock）----

def relight_tool():
    from bsrc.relighting.relight_tool import RelightTool
    return RelightTool


def mkl_transfer():
    from bsrc.relighting.relight_tool import mkl_transfer as fn
    return fn


def shadow_funcs():
    from bsrc.shadow.shadow_tool import generate_shadow, apply_shadow_to_bg
    return generate_shadow, apply_shadow_to_bg


def harmonize_v2():
    from bsrc.harmonization.harmonize_tool import harmonize_v2 as fn
    return fn


def composite_funcs():
    from bsrc.composite.composite_tool import alpha_over, feather_alpha
    return alpha_over, feather_alpha


def fx_tool():
    from bsrc.fx.fx_tool import FxTool, EFFECTS
    return FxTool, EFFECTS


def lighting_estimate():
    from bsrc.lighting.lighting_estimate import estimate
    return estimate


def matting_backend():
    """真实抠图链（需要 torch/torchvision；可选权重 BIREFNET_WEIGHT_PATH）。"""
    _ensure_path()
    from matting.matting_backend import build_matting_tool
    return build_matting_tool


def upstream_rev() -> str:
    return _upstream_rev


# ---- 契约换算 ----

def az_polar_to_direction(azimuth: float, polar: float = 45.0) -> tuple[float, float]:
    """A 组方位角/仰角 -> B 组指向光源的 2D 单位向量 (y 向下为正)。"""
    az = math.radians(float(azimuth))
    dx = math.cos(az)
    dy = -math.sin(az)
    # 顶光（polar->90）时水平方向性减弱
    lateral = 1.0 - 0.6 * clamp(float(polar), 0.0, 90.0) / 90.0
    n = math.hypot(dx * lateral, dy) or 1.0
    return (dx * lateral / n, dy / n)


def direction_to_az_polar(d: tuple[float, float]) -> tuple[float, float]:
    """B 组 2D 方向向量 -> A 组 (azimuth 度, polar 度)。"""
    dx, dy = float(d[0]), float(d[1])
    azimuth = math.degrees(math.atan2(-dy, dx)) % 360.0
    # B 的 phi（仰角）未随向量带回时，用向量长度近似；此处仅从向量恢复水平角，
    # polar 由调用方结合 B 的 theta_phi 输出决定（见 lighting impl）。
    return round(azimuth, 1), 0.0


def kelvin_to_temp_hint(color_temp: float | None) -> tuple[str | None, float]:
    """开尔文 -> B 组 (色温标签, 色温偏移强度)。neutral 返回强度 0（B 分支安全）。"""
    if color_temp is None:
        return None, 0.10
    t = (float(color_temp) - 5500.0) / 2500.0  # -1.2 暖 .. +1 冷
    if t < -0.28:
        label = "warm"
    elif t > 0.28:
        label = "cool"
    else:
        label = "neutral"
    strength = clamp(abs(t) * 0.10, 0.0, 0.25)
    return label, strength


def kelvin_from_channels(r_mean: float, b_mean: float) -> float:
    """R/B 通道均值 -> 开尔文（与 A 组 mock 同一口径，保证前端徽章语义不变）。"""
    warm = (r_mean - b_mean) / 255.0
    return round(clamp(5800.0 - warm * 3200.0, 2400.0, 8000.0), 1)
