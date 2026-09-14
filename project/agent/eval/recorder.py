"""A8 评测与日志：每次运行落盘可回放记录。

目录：RUNS_DIR/runs/<run_id>/
    plan.json      初始计划（含 instruction / quality / planner）
    events.jsonl   全事件流（节点状态、Critic、控制动作）
    result.json    终态快照（DAG 各节点产物/耗时/Critic 历史）
回放 = 读 result.json + events.jsonl，可在 UI 逐节点复现（规范 A8 验收）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent.eval")


class RunRecorder:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _dir(self, run_id: str) -> Path:
        p = self.runs_dir / "runs" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_plan(self, run_id: str, dag_dict: dict) -> None:
        self._write(self._dir(run_id) / "plan.json", dag_dict)

    def log_event(self, run_id: str, event: dict[str, Any]) -> None:
        path = self._dir(run_id) / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def save_result(self, run_id: str, state_dict: dict) -> None:
        self._write(self._dir(run_id) / "result.json", state_dict)

    def load_result(self, run_id: str) -> dict | None:
        path = self._dir(run_id) / "result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[str]:
        root = self.runs_dir / "runs"
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    @staticmethod
    def _write(path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        tmp.replace(path)
