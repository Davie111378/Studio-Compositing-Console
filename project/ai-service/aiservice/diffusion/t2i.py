# -*- coding: utf-8 -*-
"""千问文生图（DashScope 异步任务）—— T02 background_generate 的真实档。

- 启用条件：T2I_MODEL（如 wanx2.1-t2i-turbo）+ DASHSCOPE_API_KEY 均已配置。
- 流程：提交异步任务 -> 轮询 task -> 下载 PNG -> 缩放回请求画幅；
  深度图以亮度近似（autocontrast + 高斯模糊）补齐契约，T03/T05/T07 可选消费。
- 失败语义：任何环节失败抛 T2IError，由 impl.py 回落程序化渐变 mock，HTTP 契约不变。
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import httpx
from PIL import Image, ImageOps, ImageFilter

from aiservice.common import save_png

logger = logging.getLogger("aiservice.t2i")

_TASK_BASE = "https://dashscope.aliyuncs.com/api/v1"
# wanx2.1 系列支持的画幅（宽*高），按请求长宽比就近吸附
_SIZES: tuple[tuple[int, int], ...] = ((1280, 720), (1024, 1024), (720, 1280))
_POLL_INTERVAL_S = 2.0


class T2IError(RuntimeError):
    """文生图档不可用/调用失败（调用方回落 mock）。"""


def _api_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY", "").strip()


def configured() -> bool:
    return bool(os.environ.get("T2I_MODEL", "").strip() and _api_key())


def _snap_size(w: int, h: int) -> tuple[int, int]:
    return min(_SIZES, key=lambda s: (abs(s[0] / s[1] - w / h), abs(s[0] * s[1] - w * h)))


def generate(prompt: str, w: int, h: int, out_dir: Path, root: Path) -> dict:
    """生成一张背景图 + 近似深度图，返回 {bg_png, depth_png}（artifact:// URI）。"""
    key = _api_key()
    if not key:
        raise T2IError("DASHSCOPE_API_KEY 未配置")
    model = os.environ.get("T2I_MODEL", "wanx2.1-t2i-turbo").strip()
    timeout_s = float(os.environ.get("T2I_TIMEOUT_S", "120"))
    sw, sh = _snap_size(w, h)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-DashScope-Async": "enable"}
    t0 = time.perf_counter()

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{_TASK_BASE}/services/aigc/text2image/image-synthesis",
            headers=headers,
            json={
                "model": model,
                "input": {"prompt": prompt},
                "parameters": {"size": f"{sw}*{sh}", "n": 1, "prompt_extend": True,
                               "watermark": False},
            },
        )
        if resp.status_code != 200:
            raise T2IError(f"提交失败 HTTP {resp.status_code}: {resp.text[:200]}")
        task_id = (resp.json().get("output") or {}).get("task_id")
        if not task_id:
            raise T2IError(f"响应无 task_id: {resp.text[:200]}")

        url = ""
        deadline = time.monotonic() + max(30.0, timeout_s)
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_S)
            data = (client.get(f"{_TASK_BASE}/tasks/{task_id}",
                               headers={"Authorization": f"Bearer {key}"}).json()
                    .get("output") or {})
            status = data.get("task_status")
            if status == "SUCCEEDED":
                results = data.get("results") or []
                url = (results[0].get("url", "") if results else "")
                break
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise T2IError(f"任务失败 {status}: {data.get('message', data)}")
        if not url:
            raise T2IError(f"轮询超时（>{timeout_s}s）")

        img_resp = client.get(url, timeout=60.0)
        img_resp.raise_for_status()

    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    # 亮度近似深度：亮区视为近景；T03 仅可选消费，契约优先
    depth = ImageOps.autocontrast(img.convert("L")).filter(ImageFilter.GaussianBlur(8))

    logger.info("T2I %s 出图 %dx%d 用时 %.1fs", model, w, h, time.perf_counter() - t0)
    return {
        "bg_png": save_png(img, out_dir, root, "bg.png"),
        "depth_png": save_png(depth, out_dir, root, "depth.png"),
    }
