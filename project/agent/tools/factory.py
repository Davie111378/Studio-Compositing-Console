"""Provider 工厂：按配置选择 HTTP / 内嵌形态。"""

from __future__ import annotations

from agent.config import Settings
from agent.errors import E_NOT_IMPLEMENTED, AgentError
from agent.tools.provider import EmbeddedProvider, HttpProvider, ToolProvider


def build_provider(settings: Settings) -> ToolProvider:
    if settings.ai_service_url:
        fallback = None
        if settings.ai_fallback_to_embedded:
            try:
                fallback = EmbeddedProvider(settings.project_root / "ai-service", settings.artifacts_dir)
            except AgentError:
                fallback = None
        return HttpProvider(settings.ai_service_url, settings.artifacts_dir,
                            timeout_s=settings.ai_service_timeout_s, fallback=fallback)
    if settings.ai_service_embedded:
        return EmbeddedProvider(settings.project_root / "ai-service", settings.artifacts_dir)
    raise AgentError(E_NOT_IMPLEMENTED, "未配置 AI_SERVICE_URL 且内嵌模式关闭，无法调用任何工具")
