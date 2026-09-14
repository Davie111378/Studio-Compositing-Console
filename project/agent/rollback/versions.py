"""A5 条件回滚：节点级版本树 + 部分回滚语义。

语义（规范 4.4 / Demo 04）：
    「换回背景 A，但保留现在的光」
    = 目标节点恢复到指定版本（不重跑）
    + 保留集合（preserve）中的节点保持当前版本（不重跑）
    + 目标节点的下游节点（沿非保留路径）全部重跑
实现：选最近一个包含目标角色的计划作为拓扑基准，构造"全节点"新 DAG——
    重跑节点 pending（版本 = 会话内该角色最新版本 +1），
    恢复/保留节点 done 并写入版本历史中的产物，
    Executor 天然只执行 pending 节点，且 @ 引用自动取到 done 节点的产物。
"""

from __future__ import annotations

from typing import Any

from agent.dag.models import DAGNode, NodeStatus, PlanDAG, RunState
from agent.errors import E_ROLLBACK_INVALID, AgentError


def select_plan(plans: list[dict[str, Any]], role: str) -> PlanDAG:
    """从计划历史中选最近一个包含目标角色的计划（小步修改轮可能不含目标）。"""
    for plan in reversed(plans):
        if any(n.get("role") == role for n in plan.get("nodes", [])):
            return PlanDAG(**plan)
    raise AgentError(E_ROLLBACK_INVALID, f"计划历史中找不到包含 {role} 的计划，无法回滚")


def _latest(versions: dict[str, dict[str, Any]], role: str):
    hist = versions.get(role) or {}
    if not hist:
        return None, None
    v = max(hist.keys())
    return v, hist[v]


def build_rollback_dag(
    base_dag: PlanDAG,
    versions: dict[str, dict[str, Any]],
    target_role: str,
    target_version: int,
    preserve_roles: list[str],
) -> tuple[PlanDAG, dict[str, Any]]:
    """构造回滚 DAG。

    versions: role -> {version: vinfo}（来自 SessionManager.node_versions）。
    返回 (新 PlanDAG, meta)。meta 含 restored / rerun / frozen 三个清单。
    """
    target_node = base_dag.node_by_role(target_role)
    if target_node is None:
        raise AgentError(E_ROLLBACK_INVALID, f"基准计划中没有 {target_role} 节点，无法回滚")
    if target_role in preserve_roles:
        raise AgentError(E_ROLLBACK_INVALID, "目标节点不能同时出现在保留集合中")

    vhist = versions.get(target_role) or {}
    if target_version not in vhist:
        known = sorted(vhist.keys())
        raise AgentError(E_ROLLBACK_INVALID,
                         f"{target_role} 不存在版本 v{target_version}（已有: {known}）")
    restored_info = vhist[target_version]

    preserve = set(preserve_roles)
    rerun_ids = set(base_dag.downstream_of(target_node.id, exclude_roles=preserve | {target_role}))

    new_nodes: list[DAGNode] = []
    rerun_roles: list[str] = []
    frozen_roles: list[str] = []
    for n in base_dag.nodes:
        clone = n.model_copy(deep=True)
        clone.error = None
        clone.attempts = 0
        clone.started_at = None
        clone.finished_at = None
        clone.latency_ms = None
        latest_v, latest_info = _latest(versions, n.role)
        if n.role == target_role:
            # 恢复：直接落到历史版本的产物
            clone.status = NodeStatus.done
            clone.version = target_version
            clone.outputs = restored_info.get("outputs", {})
            clone.artifacts = restored_info.get("artifacts", [])
        elif n.id in rerun_ids:
            # 重跑：版本取会话最新 +1，清空产物
            clone.status = NodeStatus.pending
            clone.version = (latest_v + 1) if latest_v is not None else n.version + 1
            clone.outputs = {}
            clone.artifacts = []
            rerun_roles.append(n.role)
        else:
            # 保留/无关节点：冻结为会话内当前（最新）产物
            clone.status = NodeStatus.done
            if latest_info:
                clone.version = latest_v
                clone.outputs = latest_info.get("outputs", {})
                clone.artifacts = latest_info.get("artifacts", [])
            # 无历史记录的节点保持基准计划中已有产物（未跑完的链路兜底）
            frozen_roles.append(n.role)
        new_nodes.append(clone)

    dag = PlanDAG(
        nodes=new_nodes,
        planner="rollback",
        instruction=base_dag.instruction,
        quality=base_dag.quality,
        revision=base_dag.revision,
    )
    meta = {
        "target": {"role": target_role, "version": target_version,
                   "artifact": (restored_info.get("artifacts") or [None])[0]},
        "rerun": rerun_roles,
        "frozen": frozen_roles,
    }
    return dag, meta


def rollback_from_run(run: RunState, versions: dict[str, dict[str, Any]],
                      target_role: str, target_version: int,
                      preserve_roles: list[str]) -> tuple[PlanDAG, dict[str, Any]]:
    return build_rollback_dag(run.dag, versions, target_role, target_version, preserve_roles)
