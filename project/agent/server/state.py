"""应用装配：把 Provider / Planner / Executor / Critic / Session / Recorder 接在一起。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.config import Settings, get_settings
from agent.critic import build_critic
from agent.errors import AgentError
from agent.eval.recorder import RunRecorder
from agent.executor.bus import EventBus
from agent.executor.engine import Executor
from agent.planner import build_planner
from agent.session import SessionManager
from agent.tools import build_provider


@dataclass
class AppContext:
    settings: Settings
    bus: EventBus
    sessions: SessionManager
    recorder: RunRecorder
    provider: object
    critic: object
    executor: Executor
    planner: object

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close:
            await close()


def build_context(settings: Settings | None = None) -> AppContext:
    s = settings or get_settings()
    bus = EventBus()
    sessions = SessionManager(s.artifacts_dir, s.runs_dir)
    recorder = RunRecorder(s.runs_dir)
    provider = build_provider(s)  # 可能抛 AgentError（配置缺失）
    critic = build_critic(s)
    executor = Executor(provider, critic, recorder, bus, s, sessions)
    planner = build_planner(s)
    return AppContext(settings=s, bus=bus, sessions=sessions, recorder=recorder,
                      provider=provider, critic=critic, executor=executor, planner=planner)


__all__ = ["AppContext", "build_context", "AgentError"]
