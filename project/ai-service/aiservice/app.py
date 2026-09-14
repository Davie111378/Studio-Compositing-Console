"""ai-service FastAPI 入口：统一信封 + 8 工具分发（见 agent/schema/ENVELOPE.md）。

Agent 嵌入模式会把本 app 通过 ASGITransport 进程内调用，走的完全是同一 HTTP 契约。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from aiservice import export_impl
from aiservice.common import ToolFailure, ensure_out_dir
from aiservice.diffusion import impl as t02
from aiservice.harmonization import impl as t06_t07
from aiservice.lighting import impl as t03_t04
from aiservice.matting import impl as t01
from aiservice.shadow import impl as t05

logger = logging.getLogger("aiservice")

IMPLS: dict[str, Callable[[dict, dict, Path, Path], dict]] = {
    "matting": t01.run,
    "background_generate": t02.run,
    "lighting_estimate": t03_t04.run_estimate,
    "relight": t03_t04.run_relight,
    "shadow_generate": t05.run,
    "harmonize": t06_t07.run_harmonize,
    "enhance": t06_t07.run_enhance,
    "export": export_impl.run,
}

VERSION = "0.2.0"


def artifacts_root() -> Path:
    root = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parents[2] / "data" / "artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def mock_delay_ms() -> int:
    """模拟推理耗时（演示用）；测试置 MOCK_DELAY_MS=0。quality 影响系数：draft 快 40%。"""
    return int(os.environ.get("MOCK_DELAY_MS", "120"))


def create_app() -> FastAPI:
    app = FastAPI(title="ai-service", version=VERSION, description="T01-T08 图像工具服务（B 组真实算法引擎 bsrc + A 组 mock 回落，IMC_ENGINE 切换）")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "service": "ai-service", "version": VERSION, "tools": sorted(IMPLS)}

    @app.get("/tools")
    async def tools():
        return {"tools": [{"tool": name} for name in sorted(IMPLS)]}

    @app.post("/invoke/{tool}")
    async def invoke(tool: str, request: Request) -> dict[str, Any]:
        if tool not in IMPLS:
            return _failed(request_id="?", code="E_TOOL_UNKNOWN", message=f"未知工具: {tool}", retryable=False)
        body = await request.json()
        request_id = body.get("request_id") or uuid.uuid4().hex
        inputs = body.get("inputs") or {}
        options = body.get("options") or {}
        root = artifacts_root()
        out_dir = ensure_out_dir(Path(body.get("out_dir") or f"runs/embedded/{request_id}"), root)

        started = time.perf_counter()
        delay = mock_delay_ms()
        if delay:
            factor = {"draft": 0.4, "normal": 1.0, "fine": 1.6}.get(options.get("quality", "normal"), 1.0)
            await asyncio.sleep(delay / 1000.0 * factor)
        try:
            outputs = IMPLS[tool](inputs, options, out_dir, root)
            latency = int((time.perf_counter() - started) * 1000)
            return {
                "request_id": request_id,
                "status": "success",
                "outputs": outputs,
                "error": None,
                "latency_ms": latency,
                "artifacts": [v for v in outputs.values() if isinstance(v, str) and v.startswith(("artifact://", "asset://"))],
            }
        except ToolFailure as e:
            latency = int((time.perf_counter() - started) * 1000)
            logger.warning("tool %s failed: %s %s", tool, e.code, e.message)
            return _failed(request_id, e.code, e.message, e.retryable, latency)
        except Exception as e:  # 未预期异常：不可重试，完整留栈
            logger.exception("tool %s crashed", tool)
            return _failed(request_id, "E_INTERNAL", f"{type(e).__name__}: {e}", False, int((time.perf_counter() - started) * 1000))

    return app


def _failed(request_id: str, code: str, message: str, retryable: bool, latency_ms: int = 0) -> dict:
    return {
        "request_id": request_id,
        "status": "failed",
        "outputs": {},
        "error": {"code": code, "message": message, "retryable": retryable},
        "latency_ms": latency_ms,
        "artifacts": [],
    }


app = create_app()
