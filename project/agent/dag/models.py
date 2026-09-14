"""执行计划 DAG 数据模型（规范 4.3 / 4.4）。

节点级版本化是条件回滚的基础：每个 node 保存 version + output_artifacts，
而非整图版本化（规范 4.4 实现要点）。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    """时间戳+短随机，形如 r20260910_3f2a：可读、无中文、可做文件名（规范 N7）。"""
    t = time.strftime("%Y%m%d", time.localtime())
    return f"{prefix}{t}_{uuid.uuid4().hex[:6]}"


class NodeStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"    # 上游失败导致未执行
    cancelled = "cancelled"


class Quality(str, Enum):
    draft = "draft"
    normal = "normal"
    fine = "fine"

    def lower(self) -> Optional["Quality"]:
        """L1 降级：fine->normal->draft->None。"""
        return {"fine": Quality.normal, "normal": Quality.draft, "draft": None}.get(self)

    def higher(self) -> Optional["Quality"]:
        """Critic 重跑升档：draft->normal->fine->None。"""
        return {"draft": Quality.normal, "normal": Quality.fine, "fine": None}.get(self)


class DAGNode(BaseModel):
    id: str
    tool: str                       # 工具名（8 个之一）
    role: str                       # 语义槽位，回滚/版本管理用它定位（默认=tool）
    label: str = ""                 # 前端 DAG 节点显示名
    inputs: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.pending
    quality: Quality = Quality.normal
    version: int = 1                # 节点输出版本（重跑/回滚后递增）
    attempts: int = 0
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error: Optional[dict[str, Any]] = None
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    latency_ms: Optional[int] = None

    @property
    def primary_artifact(self) -> Optional[str]:
        return self.artifacts[0] if self.artifacts else None


class CriticResult(BaseModel):
    scores: dict[str, float]        # lighting/shadow/color/edge
    overall: float
    threshold: int
    passed: bool
    lowest_dim: Optional[str] = None
    action: str = "pass"            # pass | rerun | force_pass
    rerun_role: Optional[str] = None
    comment: str = ""


class RunStatus(str, Enum):
    planning = "planning"
    running = "running"
    paused = "paused"
    critiquing = "critiquing"
    replanning = "replanning"       # Critic 触发的重跑轮
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class PlanDAG(BaseModel):
    plan_id: str = Field(default_factory=lambda: new_id("p"))
    run_id: str = ""
    revision: int = 1               # Critic 自动重跑的第几轮
    nodes: list[DAGNode] = Field(default_factory=list)
    planner: str = "rule"           # rule | llm
    created_at: int = Field(default_factory=now_ms)
    instruction: str = ""
    frozen: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="冻结输入：{role: {input_name: uri}}，节点构造输入时优先取用（多轮/回滚）",
    )
    quality: Quality = Quality.normal

    # ---- 查询 ----
    def node(self, node_id: str) -> DAGNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"节点不存在: {node_id}")

    def node_by_role(self, role: str) -> Optional[DAGNode]:
        for n in self.nodes:
            if n.role == role:
                return n
        return None

    def dependencies_of(self, node_id: str) -> list[str]:
        return list(self.node(node_id).depends_on)

    def dependents_of(self, node_id: str) -> list[str]:
        return [n.id for n in self.nodes if node_id in n.depends_on]

    def edges(self) -> list[list[str]]:
        return [[dep, n.id] for n in self.nodes for dep in n.depends_on]

    def topological_order(self) -> list[str]:
        """Kahn 拓扑排序；有环抛 ValueError。"""
        indeg = {n.id: 0 for n in self.nodes}
        for a, b in self.edges():
            indeg[b] += 1
        queue = sorted([nid for nid, d in indeg.items() if d == 0])
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for succ in sorted(self.dependents_of(nid)):
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.nodes):
            raise ValueError("DAG 存在环")
        return order

    def downstream_of(self, node_id: str, exclude_roles: set[str] | None = None) -> list[str]:
        """传递闭包的下游节点；exclude_roles 中的节点不进入结果也不继续传播（条件回滚用）。"""
        exclude_roles = exclude_roles or set()
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop(0)
            for succ in self.dependents_of(cur):
                if succ in seen:
                    continue
                role = self.node(succ).role
                if role in exclude_roles:
                    continue
                seen.add(succ)
                frontier.append(succ)
        return sorted(seen)


class RunState(BaseModel):
    """一次运行 = 初始 DAG + 若干 Critic 重跑轮 + 最终 Critic 结果。"""

    run_id: str = Field(default_factory=lambda: new_id("r"))
    session_id: str = ""
    instruction: str = ""
    status: RunStatus = RunStatus.planning
    dag: PlanDAG
    critic: Optional[CriticResult] = None
    critic_history: list[CriticResult] = Field(default_factory=list)
    replan_count: int = 0
    created_at: int = Field(default_factory=now_ms)
    finished_at: Optional[int] = None
    error: Optional[dict[str, Any]] = None

    def all_nodes(self) -> list[DAGNode]:
        return self.dag.nodes
