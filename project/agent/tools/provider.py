"""工具调用 Provider：Agent 与模型实现之间的唯一通道（规范 N2 解耦层）。

两种形态，同一 HTTP 契约（agent/schema/ENVELOPE.md）：
    HttpProvider      独立 ai-service 进程（生产 / B 组替换真模型）
    EmbeddedProvider  进程内 ASGI 加载同一 app（单机 Demo / 兜底降级）
"""

from __future__ import annotations

import os
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from agent.errors import (
    E_AI_SERVICE_BAD_RESPONSE,
    E_AI_SERVICE_UNAVAILABLE,
    E_NOT_IMPLEMENTED,
    AgentError,
)


class ToolFailureError(AgentError):
    """工具端返回 failed/timeout 信封（错误码来自工具实现，重试性由信封携带）。"""


class ToolProvider(ABC):
    @abstractmethod
    async def invoke(self, tool: str, inputs: dict, options: dict, out_dir: str,
                     request_id: str | None = None, timeout_s: float | None = None) -> dict:
        """返回成功信封 dict；失败抛 ToolFailureError / AgentError。"""

    @abstractmethod
    async def aclose(self) -> None: ...


class HttpProvider(ToolProvider):
    def __init__(self, base_url: str, artifacts_root: Path, timeout_s: float = 120,
                 fallback: "ToolProvider | None" = None, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.artifacts_root = artifacts_root
        self.timeout_s = timeout_s
        self.fallback = fallback
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def invoke(self, tool: str, inputs: dict, options: dict, out_dir: str,
                     request_id: str | None = None, timeout_s: float | None = None) -> dict:
        payload = {
            "tool": tool,
            "version": "1.0",
            "request_id": request_id or uuid.uuid4().hex,
            "inputs": inputs,
            "options": options,
            "out_dir": out_dir,
        }
        try:
            resp = await self._client.post(f"{self.base_url}/invoke/{tool}", json=payload,
                                           timeout=timeout_s or self.timeout_s)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.StreamError) as e:
            if self.fallback is not None:
                return await self.fallback.invoke(tool, inputs, options, out_dir, request_id, timeout_s)
            raise AgentError(E_AI_SERVICE_UNAVAILABLE, f"ai-service 不可达: {e}", retryable=True) from e
        envelope = resp.json()
        return self._check(envelope)

    def _check(self, envelope: dict) -> dict:
        if not isinstance(envelope, dict) or "status" not in envelope:
            raise AgentError(E_AI_SERVICE_BAD_RESPONSE, f"信封非法: {envelope!r:.200}", retryable=False)
        if envelope["status"] == "success":
            return envelope
        err = envelope.get("error") or {}
        raise ToolFailureError(err.get("code", "E_TOOL_FAILED"), err.get("message", "工具失败"),
                               retryable=bool(err.get("retryable")), detail=envelope)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class EmbeddedProvider(ToolProvider):
    """进程内加载 ai-service（sys.path 指向 ai-service/），经 ASGI 走完整 HTTP 信封。"""

    def __init__(self, ai_service_dir: Path, artifacts_root: Path):
        self.artifacts_root = artifacts_root
        dir_str = str(ai_service_dir.resolve())
        if dir_str not in sys.path:
            sys.path.insert(0, dir_str)
        try:
            os.environ.setdefault("ARTIFACTS_DIR", str(artifacts_root))
            os.environ["ARTIFACTS_DIR"] = str(artifacts_root)  # 与 Agent 强一致
            from aiservice.app import create_app  # noqa: PLC0415 延迟导入，保持 Agent 可独立安装
        except Exception as e:
            raise AgentError(E_NOT_IMPLEMENTED, f"无法内嵌加载 ai-service: {e}") from e
        from httpx import ASGITransport  # noqa: PLC0415
        self._client = httpx.AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://embedded")

    async def invoke(self, tool: str, inputs: dict, options: dict, out_dir: str,
                     request_id: str | None = None, timeout_s: float | None = None) -> dict:
        payload = {
            "tool": tool, "version": "1.0", "request_id": request_id or uuid.uuid4().hex,
            "inputs": inputs, "options": options, "out_dir": out_dir,
        }
        try:
            resp = await self._client.post(f"/invoke/{tool}", json=payload, timeout=timeout_s or 120)
        except Exception as e:
            raise AgentError(E_AI_SERVICE_UNAVAILABLE, f"内嵌工具调用失败: {e}", retryable=True) from e
        envelope = resp.json()
        if envelope.get("status") == "success":
            return envelope
        err = envelope.get("error") or {}
        raise ToolFailureError(err.get("code", "E_TOOL_FAILED"), err.get("message", "工具失败"),
                               retryable=bool(err.get("retryable")), detail=envelope)

    async def aclose(self) -> None:
        await self._client.aclose()
