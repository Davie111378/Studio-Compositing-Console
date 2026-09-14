"""统一错误码与异常体系（规范 N1：错误码 + 可重试性标记）。

命名规范：E_<来源>_<原因>，UPPER_SNAKE_CASE（规范 N7）。
"""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Agent 侧统一异常基类。retryable 标记给 Executor 决定是否自动重试。"""

    def __init__(self, code: str, message: str, retryable: bool = False, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.detail = detail

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable, "detail": self.detail}


# ---- 通用 ----
E_INVALID_INPUT = "E_INVALID_INPUT"
E_SCHEMA_VALIDATION = "E_SCHEMA_VALIDATION"
E_TIMEOUT = "E_TIMEOUT"
E_CANCELLED = "E_CANCELLED"
E_STATE_CONFLICT = "E_STATE_CONFLICT"
E_NOT_IMPLEMENTED = "E_NOT_IMPLEMENTED"

# ---- 会话 / 运行 ----
E_SESSION_NOT_FOUND = "E_SESSION_NOT_FOUND"
E_RUN_NOT_FOUND = "E_RUN_NOT_FOUND"
E_ASSET_NOT_FOUND = "E_ASSET_NOT_FOUND"
E_ARTIFACT_NOT_FOUND = "E_ARTIFACT_NOT_FOUND"

# ---- Planner ----
E_PLANNER_LLM_UNAVAILABLE = "E_PLANNER_LLM_UNAVAILABLE"
E_PLANNER_VALIDATION = "E_PLANNER_VALIDATION"
E_PLANNER_EXHAUSTED = "E_PLANNER_EXHAUSTED"

# ---- DAG ----
E_DAG_INVALID = "E_DAG_INVALID"
E_DEP_FAILED = "E_DEP_FAILED"
E_ROLLBACK_INVALID = "E_ROLLBACK_INVALID"

# ---- 工具 ----
E_TOOL_UNKNOWN = "E_TOOL_UNKNOWN"
E_AI_SERVICE_UNAVAILABLE = "E_AI_SERVICE_UNAVAILABLE"
E_AI_SERVICE_BAD_RESPONSE = "E_AI_SERVICE_BAD_RESPONSE"
