"""全局配置：全部由环境变量驱动，代码内不给魔法数。

支持项目根目录 `.env`（复制 `.env.example` 填写），启动时自动加载；
真实环境变量优先于 `.env`，已加载的键不覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """极简 .env 加载：KEY=VALUE 每行一条，# 注释；不覆盖已存在的环境变量。"""
    path = _project_root() / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _dashscope_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY", "").strip()


# 千问 OpenAI 兼容端点（.env 只填 DASHSCOPE_API_KEY 时作为 llm/vlm 缺省，"填 key 即用"）
_DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class Settings:
    # ---- 服务 ----
    host: str = field(default_factory=lambda: os.environ.get("AGENT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("AGENT_PORT", "8000")))

    # ---- 工具调用（三级形态）----
    # ai_service_url 非空 -> 走独立 ai-service 进程（生产形态）
    # 为空且 ai_service_embedded -> 进程内加载 ai-service（单机 Demo 形态）
    # 都不行 -> 报配置错误（不允许无工具运行）
    ai_service_url: str = field(default_factory=lambda: os.environ.get("AI_SERVICE_URL", "").rstrip("/"))
    ai_service_embedded: bool = field(default_factory=lambda: _env_bool("AI_SERVICE_EMBEDDED", True))
    ai_service_timeout_s: float = field(default_factory=lambda: float(os.environ.get("AI_SERVICE_TIMEOUT_S", "120")))
    ai_fallback_to_embedded: bool = field(
        default_factory=lambda: _env_bool("AI_FALLBACK_TO_EMBEDDED", True)
    )

    # ---- 存储 ----
    project_root: Path = field(default_factory=_project_root)
    artifacts_dir: Path = field(default_factory=lambda: Path(os.environ.get("ARTIFACTS_DIR", "")) if os.environ.get("ARTIFACTS_DIR") else _project_root() / "data" / "artifacts")
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("RUNS_DIR", "")) if os.environ.get("RUNS_DIR") else _project_root() / "data" / "runs")

    # ---- Executor ----
    node_timeout_s: float = field(default_factory=lambda: float(os.environ.get("NODE_TIMEOUT_S", "120")))
    max_node_retries: int = field(default_factory=lambda: int(os.environ.get("MAX_NODE_RETRIES", "1")))
    mock_delay_ms: int = field(default_factory=lambda: int(os.environ.get("MOCK_DELAY_MS", "120")))

    # ---- Critic（规范 6.1 建议 1：阈值需 30 组样本标定，当前为占位初值）----
    critic_threshold: int = field(default_factory=lambda: int(os.environ.get("CRITIC_THRESHOLD", "85")))
    max_replans: int = field(default_factory=lambda: int(os.environ.get("MAX_REPLANS", "2")))
    critic_enabled: bool = field(default_factory=lambda: _env_bool("CRITIC_ENABLED", True))

    # ---- Planner ----
    # planner_mode: auto（配了 LLM 用 LLM，失败回落规则）| rule（只用规则）| llm（只用 LLM）
    planner_mode: str = field(default_factory=lambda: os.environ.get("PLANNER_MODE", "auto"))
    # 千问（DashScope OpenAI 兼容模式）：LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
    # 只填 DASHSCOPE_API_KEY 时，LLM 端点/模型/密钥全部取千问缺省，无需逐项配置。
    llm_api_base: str = field(default_factory=lambda: (
        os.environ.get("LLM_API_BASE", "").rstrip("/")
        or (_DASHSCOPE_COMPAT_BASE if _dashscope_key() else "")
    ))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "") or _dashscope_key())
    llm_model: str = field(default_factory=lambda: (
        os.environ.get("LLM_MODEL", "") or ("qwen-plus" if _dashscope_key() else "")
    ))
    llm_timeout_s: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_S", "20")))

    # ---- VLM Critic（缺省回落 LLM_*；视觉打分建议 qwen-vl-max）----
    vlm_api_base: str = field(default_factory=lambda: (
        os.environ.get("VLM_API_BASE", "").rstrip("/")
        or (_DASHSCOPE_COMPAT_BASE if _dashscope_key() else "")
    ))
    vlm_api_key: str = field(default_factory=lambda: (
        os.environ.get("VLM_API_KEY", "") or _dashscope_key()
    ))
    vlm_model: str = field(default_factory=lambda: (
        os.environ.get("VLM_MODEL", "") or ("qwen-vl-max" if _dashscope_key() else "")
    ))

    # ---- 日志 ----
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def uploads_dir(self) -> Path:
        return self.artifacts_dir / "uploads"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> Settings:
    """测试用：环境变量变化后重建配置。"""
    global _settings
    _settings = Settings()
    _settings.ensure_dirs()
    return _settings
