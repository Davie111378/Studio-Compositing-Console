# -*- coding: utf-8 -*-
"""
artifact.py — 产物表: 节点输出登记 + "$nN.key" 引用解析 + 节点级版本化 (供回滚)
"""
from __future__ import annotations
import re
from pathlib import Path

REF = re.compile(r"^\$([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$")


class ArtifactTable:
    """artifacts[node_id] = {key: value}; versions[node_id] = [data_v1, data_v2...]"""

    def __init__(self):
        self.artifacts: dict[str, dict] = {}
        self.versions: dict[str, list[dict]] = {}

    def store(self, node_id: str, data: dict):
        self.artifacts[node_id] = dict(data)
        self.versions.setdefault(node_id, []).append(dict(data))

    def get(self, node_id: str, key: str | None = None):
        d = self.artifacts.get(node_id)
        if d is None:
            return None
        return d if key is None else d.get(key)

    def latest_version(self, node_id: str) -> dict | None:
        vs = self.versions.get(node_id) or []
        return vs[-1] if vs else None

    def resolve(self, params: dict) -> dict:
        """把 params 里的 "$nN.key" 引用替换为实际产物值。"""
        out = {}
        for k, v in params.items():
            if isinstance(v, str):
                m = REF.match(v.strip())
                if m:
                    node, key = m.group(1), m.group(2)
                    val = self.get(node, key)
                    if val is None:
                        raise KeyError(f"引用悬空: {v} (节点 {node} 无产物 {key})")
                    out[k] = val
                    continue
            out[k] = v
        return out

    def snapshot(self) -> dict:
        return {"artifacts": {k: dict(v) for k, v in self.artifacts.items()},
                "versions": {k: list(v) for k, v in self.versions.items()}}

    def restore(self, snap: dict):
        self.artifacts = {k: dict(v) for k, v in snap["artifacts"].items()}
        self.versions = {k: list(v) for k, v in snap["versions"].items()}
