"""Critic 工厂：配置了 LLM/VLM 时用 VLMCritic（内含规则回落），否则 RuleCritic。"""

from __future__ import annotations

from agent.config import Settings
from agent.critic.base import DIMS, DIM_TO_TOOL, WEIGHTS, BaseCritic
from agent.critic.rule_critic import RuleCritic
from agent.critic.vlm_critic import VLMCritic


def build_critic(settings: Settings) -> BaseCritic:
    if settings.planner_mode != "rule" and settings.llm_api_base and settings.llm_model:
        return VLMCritic(settings)
    return RuleCritic(settings)


__all__ = ["build_critic", "RuleCritic", "VLMCritic", "BaseCritic", "DIMS", "DIM_TO_TOOL", "WEIGHTS"]
