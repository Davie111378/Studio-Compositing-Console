# -*- coding: utf-8 -*-
"""
dag.py — 执行计划 DAG 模型: 解析 / jsonschema 强校验 / 拓扑排序 / 环检测
依据冲刺计划 A3: topo + 幂等节点; Planner 输出经 _dag_meta.json 校验。
"""
from __future__ import annotations
import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
META = json.loads((SCHEMA_DIR / "_dag_meta.json").read_text(encoding="utf-8"))
TOOL_SCHEMAS = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                for p in SCHEMA_DIR.glob("T0*.json")}
VALID_TOOLS = set(TOOL_SCHEMAS)


class DagError(Exception):
    pass


def validate(dag: dict) -> list[str]:
    """jsonschema 校验 + 语义校验, 返回错误列表 (空 = 合法)。"""
    errs = []
    try:
        import jsonschema
        jsonschema.validate(dag, META)
    except Exception as e:
        return [f"schema: {str(e)[:300]}"]
    ids = [n["id"] for n in dag["nodes"]]
    if len(ids) != len(set(ids)):
        errs.append("节点 id 重复")
    idset = set(ids)
    for n in dag["nodes"]:
        if n["tool"] not in VALID_TOOLS:
            errs.append(f"{n['id']}: 未知工具 {n['tool']}")
        for d in n.get("depends_on", []):
            if d not in idset:
                errs.append(f"{n['id']}: 依赖不存在的节点 {d}")
    # 环检测 (Kahn)
    indeg = {i: 0 for i in ids}
    adj: dict[str, list[str]] = {i: [] for i in ids}
    for n in dag["nodes"]:
        for d in n.get("depends_on", []):
            adj[d].append(n["id"])
            indeg[n["id"]] += 1
    q = [i for i in ids if indeg[i] == 0]
    seen = 0
    while q:
        u = q.pop()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if seen != len(ids):
        errs.append("DAG 存在环")
    # outputs 必须存在
    for o in dag.get("outputs", []):
        if o not in idset:
            errs.append(f"outputs 引用不存在节点 {o}")
    return errs


def topo_order(dag: dict, key=None) -> list[dict]:
    """Kahn 拓扑排序 (依赖是硬约束; 就绪层按 key 升序, 默认按 id 保持确定性)。

    key: nid -> 可比较值 (如工具号)。2026-09-12: LLM 会交叉命名节点 id (n1=T02,
    n2=T01), 依赖图正确但纯 id 序执行会让 T02 先于 T01 (回归 I2 失败形态);
    engine 传工具号 key 后同层执行序与 planner._toposort_nodes 对齐。
    """
    import heapq
    nodes = {n["id"]: n for n in dag["nodes"]}
    indeg = {i: 0 for i in nodes}
    adj: dict[str, list[str]] = {i: [] for i in nodes}
    for n in dag["nodes"]:
        for d in n.get("depends_on", []):
            adj[d].append(n["id"])
            indeg[n["id"]] += 1
    k = key or (lambda i: (i,))
    heap = [(k(i), i) for i in nodes if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(nodes[u])
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (k(v), v))
    return order


def descendants(dag: dict, node_id: str) -> set[str]:
    """node_id 的全部下游 (含间接)。"""
    adj: dict[str, list[str]] = {n["id"]: [] for n in dag["nodes"]}
    for n in dag["nodes"]:
        for d in n.get("depends_on", []):
            adj[d].append(n["id"])
    seen, stack = set(), [node_id]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def summary(dag: dict) -> str:
    """人读摘要 (供日志/演示)。"""
    lines = [f"DAG: {dag.get('intent', '')}  ({len(dag['nodes'])} 节点)"]
    for n in topo_order(dag):
        dep = ",".join(n.get("depends_on", [])) or "-"
        lines.append(f"  {n['id']} {n['tool']} (依赖: {dep})")
    return "\n".join(lines)
