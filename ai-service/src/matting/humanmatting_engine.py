# -*- coding: utf-8 -*-
"""humanmatting_engine.py — MODNet 人像抠图引擎 (onnxruntime 版)

来源链: davilsu/HumanMatting (Android APP, AGPL-3.0) 的模型即 PaddleSeg/Matting
官方 MODNet 双 backbone 的 ncnn 移植 (param 层名 instance_norm 与 MODNet 结构吻合)。
本引擎直接采用 PaddleSeg 官方 inference model (Apache-2.0) 转 ONNX
(paddle2onnx, opset 12), 用 onnxruntime 推理——绕开本机 ncnn pip wheel 的
段错误 bug, 并保留升级 CUDA EP 的能力。

模型: modnet-hrnet_w18 (精度) / modnet-mobilenetv2 (速度), 输入 [-1,1] 归一化,
输出 sigmoid alpha (单通道)。第三方目录: third_party/HumanMatting/。
"""
from __future__ import annotations
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ONNX_DIR = ROOT / "third_party" / "HumanMatting" / "onnx"
RESOLUTION = 512                       # 对齐上游 davilsu 版 kResolution
MODELS = {
    "hrnet": "modnet_hrnet_w18",       # 精度优先
    "mobilenet": "modnet_mobilenetv2", # 速度优先
}
_LOCK = threading.Lock()
_SESSIONS: dict[str, "object"] = {}


def _get_session(model: str):
    key = MODELS.get(model)
    if key is None:
        raise ValueError(f"未知 humanmatting 模型: {model} (可选: {list(MODELS)})")
    if key not in _SESSIONS:
        p = ONNX_DIR / f"{key}.onnx"
        if not p.exists():
            raise FileNotFoundError(
                f"HumanMatting ONNX 缺失: {p} (third_party/HumanMatting/README.md 有重建步骤)")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0            # 用满物理核
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[key] = ort.InferenceSession(str(p), sess_options=so,
                                              providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def human_matting(image_bgr: np.ndarray, model: str = "hrnet",
                  return_time: bool = False):
    """人像 alpha 抠图。image_bgr: HxWx3 uint8 (OpenCV BGR)。

    返回 alpha (HxW float32 0~1)；return_time=True 时附 (alpha, infer_ms)。
    预处理对齐上游: RGB 直接 resize 512x512, x/127.5-1 ([-1,1])。
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 127.5) - 1.0
    x = x.transpose(2, 0, 1)[None].copy()      # 1x3x512x512 NCHW
    sess = _get_session(model)
    t0 = time.time()
    with _LOCK:
        out = sess.run(None, {"img": x})[0]    # (1,1,512,512) sigmoid alpha
    ms = (time.time() - t0) * 1000
    alpha = cv2.resize(out[0, 0], (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 1.0)
    return (alpha, ms) if return_time else alpha


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/ai_generated/green_fg/fg_03_glasses.png"
    model = sys.argv[2] if len(sys.argv) > 2 else "hrnet"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    alpha, ms = human_matting(img, model=model, return_time=True)
    print(f"[humanmatting:{model}] {Path(img_path).name} -> alpha {alpha.shape} "
          f"fg_ratio={(alpha > 0.5).mean():.3f} infer={ms:.0f}ms")
    out = ROOT / "outputs" / "humanmatting_test"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    cv2.imencode(".png", (alpha * 255).astype(np.uint8))[1].tofile(str(out / f"{stem}_{model}_alpha.png"))
    vis = (img * alpha[..., None]).astype(np.uint8)
    cv2.imencode(".png", vis)[1].tofile(str(out / f"{stem}_{model}_cutout.png"))
    print(f"[out] {out / (stem + '_' + model + '_cutout.png')}")
