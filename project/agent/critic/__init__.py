"""Critic 工厂：配置了 LLM/VLM 时用 VLMCritic（内含规则回落），否则 RuleCritic。"""

from __future__ import annotations

from agent.config import Settings
from agent.critic.base import DIMS, DIM_TO_TOOL, WEIGHTS, BaseCritic
from agent.critic.rule_critic import RuleCritic
from agent.critic.vlm_critic import VLMCritic


def build_critic(settings: Settings) -> BaseCritic:
    # VLM_* 优先（视觉模型与文本 Planner 分开配），回落 LLM_*
    api_base = settings.vlm_api_base or settings.llm_api_base
    api_key = settings.vlm_api_key or settings.llm_api_key
    model = settings.vlm_model or settings.llm_model
    if settings.planner_mode != "rule" and api_base and model:
        return VLMCritic(settings, api_base=api_base, api_key=api_key, model=model)
    return RuleCritic(settings)


__all__ = ["build_critic", "RuleCritic", "VLMCritic", "BaseCritic", "DIMS", "DIM_TO_TOOL", "WEIGHTS"]
