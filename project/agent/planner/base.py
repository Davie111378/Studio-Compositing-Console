"""Planner 接口与数据结构（规范 A2：指令 -> JSON DAG）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent.dag.models import PlanDAG, Quality


@dataclass
class PlanningRequest:
    session_id: str
    text: str
    asset_uris: list[str] = field(default_factory=list)          # asset:// 上传图
    round_index: int = 1
    quality: Quality = Quality.normal
    spatial: Optional[dict] = None                               # {"click":{x,y}} / {"box":[x1,y1,x2,y2]}
    # 会话内可用冻结产物：role -> {"outputs": {...}, "artifacts": [...], "versions": [...]}
    available_roles: dict[str, dict] = field(default_factory=dict)


@dataclass
class PlanningResult:
    dag: Optional[PlanDAG] = None
    planner: str = "rule"                                        # rule | llm
    kind: str = "plan"                                           # plan | rollback
    rollback: Optional[dict] = None                              # {"role":..., "version":..., "preserve":[...]}


class Planner:
    name = "base"

    async def plan(self, req: PlanningRequest) -> PlanningResult:
        raise NotImplementedError
