"""规则 Critic：确定性五维打分（节点精度档位 + 产物图像统计的客观分量）。

用途：
1. LLM/VLM 未接入时的默认 Critic（完整闭环可演示：不达标 -> 触发重跑 -> 达标）。
2. VLM 打分的客观分量（规范 5.3：光照/法线方向差等客观分与主观分融合）。

注意（规范 6.1 建议 1）：当前档位->分数映射是占位口径，09-21 前须用 30 组样本
做 Critic 分 vs 人工分标定，结果写 agent/critic/threshold.md 后更新。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from agent.config import Settings
from agent.critic.base import DIMS, DIM_TO_TOOL, WEIGHTS, BaseCritic
from agent.dag.models import CriticResult, NodeStatus, Quality, RunState

logger = logging.getLogger("agent.critic")

# 节点精度档位 -> 维度基础分（占位口径，待标定）
_BASE = {
    "draft": {"lighting": 72.0, "shadow": 69.0, "color": 73.0, "edge": 70.0},
    "normal": {"lighting": 86.0, "shadow": 80.0, "color": 86.0, "edge": 84.0},
    "fine": {"lighting": 95.0, "shadow": 93.0, "color": 94.0, "edge": 94.0},
}
_MISSING = 60.0

# 维度由哪个节点的质量决定
_DIM_ROLE = {
    "lighting": "relight",
    "shadow": "shadow_generate",
    "color": "harmonize",
    "edge": "matting",
}
# 评分依据的最终图：优先 export 产物，其次 harmonize / composited
_FINAL_ROLES = ("export", "harmonize", "shadow_generate")


class RuleCritic(BaseCritic):
    def __init__(self, settings: Settings):
        self.settings = settings

    def score(self, run: RunState) -> CriticResult:
        dag = run.dag
        scores: dict[str, float] = {}
        quality_note: dict[str, str] = {}
        for dim in DIMS:
            role = _DIM_ROLE[dim]
            node = dag.node_by_role(role)
            if node is None or node.status != NodeStatus.done:
                scores[dim] = _MISSING
                quality_note[dim] = "节点缺失或未完成"
            else:
                q = node.quality.value if isinstance(node.quality, Quality) else str(node.quality)
                scores[dim] = _BASE.get(q, _BASE["normal"])[dim]
                quality_note[dim] = f"{role}@{q}"

        # 客观图像分量：清晰度统计（确定性），夹在 edge 维度 ±3 分内
        delta = self._objective_delta(dag)
        scores["edge"] = round(min(100.0, max(0.0, scores["edge"] + delta)), 1)
        scores = {d: round(v, 1) for d, v in scores.items()}

        overall = round(sum(WEIGHTS[d] * scores[d] for d in DIMS), 1)
        lowest = min(DIMS, key=lambda d: scores[d])
        threshold = self.settings.critic_threshold

        if overall >= threshold:
            return CriticResult(scores=scores, overall=overall, threshold=threshold, passed=True,
                                lowest_dim=lowest, action="pass",
                                comment=f"PASS：overall {overall} ≥ 阈值 {threshold}；评分依据 {quality_note}")
        if run.replan_count >= self.settings.max_replans:
            return CriticResult(scores=scores, overall=overall, threshold=threshold, passed=False,
                                lowest_dim=lowest, action="force_pass",
                                comment=f"已重规划 {run.replan_count} 次，强制通过并提示用户")
        rerun_role = DIM_TO_TOOL[lowest]
        if dag.node_by_role(rerun_role) is None:
            return CriticResult(scores=scores, overall=overall, threshold=threshold, passed=False,
                                lowest_dim=lowest, action="force_pass",
                                comment=f"最低维 {lowest} 对应工具 {rerun_role} 不在当前计划中，无法重跑，强制通过")
        return CriticResult(scores=scores, overall=overall, threshold=threshold, passed=False,
                            lowest_dim=lowest, action="rerun", rerun_role=rerun_role,
                            comment=f"overall {overall} < {threshold}：{lowest} 最低，触发 {rerun_role} 升档重跑")

    def _objective_delta(self, dag) -> float:
        """对最终图测边缘强度（清晰度代理），归一到 [-2, +2]。mock 图确定性 -> 分数确定。"""
        for role in _FINAL_ROLES:
            node = dag.node_by_role(role)
            if node and node.status == NodeStatus.done and node.primary_artifact:
                try:
                    from agent.schema import resolve_uri
                    path: Path = resolve_uri(node.primary_artifact, self.settings.artifacts_dir)
                    img = Image.open(path).convert("L")
                    img.thumbnail((256, 256))
                    edges = img.filter(ImageFilter.FIND_EDGES)
                    std = ImageStat.Stat(edges).stddev[0]
                    return round(max(-2.0, min(2.0, (std - 18.0) / 9.0)), 2)
                except Exception as e:  # 客观分失败不影响主观分闭环
                    logger.warning("objective delta failed: %s", e)
                    return 0.0
        return 0.0
