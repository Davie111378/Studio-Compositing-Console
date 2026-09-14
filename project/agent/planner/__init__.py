"""Planner 工厂：auto = LLM 优先、规则兜底（L2 降级在调用处实现）。"""

from __future__ import annotations

import logging

from agent.config import Settings
from agent.errors import AgentError
from agent.planner.base import Planner, PlanningRequest, PlanningResult
from agent.planner.llm_planner import LLMPlanner
from agent.planner.rule_planner import RulePlanner

logger = logging.getLogger("agent.planner")


class FallbackPlanner(Planner):
    """按 planner_mode 组合 LLM 与规则：llm 失败自动回落规则（规范 5.7 L2）。"""

    name = "fallback"

    def __init__(self, primary: Planner | None, rule: Planner, strict_llm: bool = False):
        self.primary = primary
        self.rule = rule
        self.strict_llm = strict_llm

    async def plan(self, req: PlanningRequest) -> PlanningResult:
        if self.primary is not None:
            try:
                return await self.primary.plan(req)
            except AgentError as e:
                if self.strict_llm:
                    raise
                logger.warning("Planner %s 失败（%s），回落规则模板（L2）", self.primary.name, e.code)
        return await self.rule.plan(req)


def build_planner(settings: Settings) -> Planner:
    rule = RulePlanner()
    mode = settings.planner_mode
    llm = None
    if mode in ("auto", "llm"):
        if LLMPlanner.configured(settings.llm_api_base, settings.llm_model):
            llm = LLMPlanner(settings.llm_api_base, settings.llm_api_key, settings.llm_model,
                             timeout_s=settings.llm_timeout_s)
        elif mode == "llm":
            raise AgentError("E_PLANNER_LLM_UNAVAILABLE", "PLANNER_MODE=llm 但 LLM_API_BASE/LLM_MODEL 未配置")
    return FallbackPlanner(llm, rule, strict_llm=(mode == "llm"))


__all__ = ["build_planner", "FallbackPlanner", "RulePlanner", "LLMPlanner", "Planner",
           "PlanningRequest", "PlanningResult"]
