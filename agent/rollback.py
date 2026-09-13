# -*- coding: utf-8 -*-
"""
rollback.py — 条件回滚 (冲刺计划 A5 / PPT 创新三 Demo04)
- 节点级版本化: ArtifactTable.versions 已存每次执行产物
- "换回背景A保留光": swap_and_rerun(dag, "n_bg", {"semantic": "A"}, keep_tools={"T03","T04"})
  → 只重跑被改节点 + 真正依赖其产物的下游; keep_tools 节点复用缓存产物 (保留光照)
- 编辑历史树: VersionTree 记录每次 swap, 支持分支回溯
"""
from __future__ import annotations
import copy
from pathlib import Path

import dag as dagmod


def nodes_to_rerun(dag: dict, changed_id: str) -> set[str]:
    """被改节点 + 全部下游 (拓扑序执行时自动重算)。"""
    return {changed_id, *dagmod.descendants(dag, changed_id)}


def swap_param(dag: dict, node_id: str, patch: dict) -> dict:
    """深拷贝 DAG 并给指定节点打参数补丁。"""
    d2 = copy.deepcopy(dag)
    for n in d2["nodes"]:
        if n["id"] == node_id:
            n.setdefault("params", {}).update(patch)
            return d2
    raise KeyError(f"节点不存在: {node_id}")


def conditional_rerun(dag: dict, artifacts, node_id: str, patch: dict,
                      keep_tools: set[str] | None = None) -> tuple[dict, set[str], dict]:
    """条件回滚核心:
    返回 (new_dag, rerun_ids, keep_map)
      rerun_ids: 需要重新执行的节点
      keep_map:  {node_id: 旧产物} — 执行器跳过这些节点时直接回填缓存
    语义: 改 node_id 参数; 下游默认全部重跑, 但 keep_tools 中的工具节点
    (如 T03/T04 光照类) 复用旧产物 = "保留光照"。
    """
    keep_tools = keep_tools or set()
    new_dag = swap_param(dag, node_id, patch)
    rerun = nodes_to_rerun(new_dag, node_id)
    keep_map: dict[str, dict] = {}
    order = {n["id"]: i for i, n in enumerate(dagmod.topo_order(new_dag))}
    for rid in list(rerun):
        tool = next(n["tool"] for n in new_dag["nodes"] if n["id"] == rid)
        if tool in keep_tools and artifacts.get(rid):
            keep_map[rid] = artifacts.get(rid)
            rerun.discard(rid)
    return new_dag, rerun, keep_map


class VersionTree:
    """编辑历史树: 每个版本 {vid, parent, note, dag, artifacts_snapshot}。"""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.cur: str | None = None
        self.counter = 0

    def commit(self, dag: dict, artifacts_snapshot: dict, note: str = "") -> str:
        self.counter += 1
        vid = f"v{self.counter}"
        self.nodes[vid] = {"vid": vid, "parent": self.cur, "note": note,
                           "dag": copy.deepcopy(dag), "snap": artifacts_snapshot}
        self.cur = vid
        return vid

    def checkout(self, vid: str) -> dict:
        """分支回溯: 切到任意历史版本 (不删除兄弟分支)。"""
        if vid not in self.nodes:
            raise KeyError(vid)
        self.cur = vid
        return self.nodes[vid]

    def lineage(self) -> list[str]:
        out, v = [], self.cur
        while v:
            out.append(v)
            v = self.nodes[v]["parent"]
        return list(reversed(out))
