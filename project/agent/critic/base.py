"""Critic 接口与维度->工具映射（规范 3.1 A4，固定）。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.dag.models import CriticResult, RunState

# 最低分维度 -> 触发重跑的工具（规范 A4 详解，不得改动）
DIM_TO_TOOL = {
    "lighting": "relight",
    "shadow": "shadow_generate",
    "color": "harmonize",
    "edge": "matting",
}
DIMS = ("lighting", "shadow", "color", "edge")
# §4.5 权重初始化 (0.3/0.25/0.2/0.25)，30 组样本标定后可调
WEIGHTS = {"lighting": 0.30, "shadow": 0.25, "color": 0.20, "edge": 0.25}


class BaseCritic(ABC):
    @abstractmethod
    def score(self, run: RunState) -> CriticResult:
        """对当前运行终态打分，并给出 pass / rerun / force_pass 决策。"""
