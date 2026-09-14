"""会话管理：上传资产、多轮指令、节点版本树、对话记录；JSON 落盘可恢复。"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.errors import E_ASSET_NOT_FOUND, E_SESSION_NOT_FOUND, AgentError
from agent.schema import resolve_uri

logger = logging.getLogger("agent.session")


def new_session_id() -> str:
    return "s" + uuid.uuid4().hex[:12]


class SessionManager:
    """内存索引 + 磁盘持久化（sessions/<sid>.json）。上传文件存 artifacts/uploads/<sid>/。"""

    def __init__(self, artifacts_dir: Path, runs_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.runs_dir = runs_dir
        self.sessions_dir = runs_dir / "sessions"
        self.uploads_root = artifacts_dir / "uploads"
        for d in (self.sessions_dir, self.uploads_root):
            d.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}

    # ---- 基础 ----

    def create(self, session_id: str | None = None) -> dict[str, Any]:
        sid = session_id or new_session_id()
        state = {
            "session_id": sid,
            "created_at": int(time.time() * 1000),
            "uploads": {},            # asset_id -> {asset_id, uri, filename, width, height, content_type}
            "runs": [],               # [run_id]
            "node_versions": {},      # role -> [ {version, run_id, artifacts, outputs, quality, ts} ]
            "current": {},            # role -> 当前生效版本号（回滚后会指向旧版本）
            "conversation": [],       # [{role, text, ts}]
            "plans": [],              # 计划历史（回滚选基准用，最多保留 30 份）
            "last_plan": None,        # 最近一次 PlanDAG dict
        }
        self._cache[sid] = state
        self._persist(state)
        return state

    def get(self, session_id: str) -> dict[str, Any]:
        if session_id in self._cache:
            return self._cache[session_id]
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise AgentError(E_SESSION_NOT_FOUND, f"会话不存在: {session_id}")
        state = json.loads(path.read_text(encoding="utf-8"))
        self._cache[session_id] = state
        return state

    def exists(self, session_id: str) -> bool:
        try:
            self.get(session_id)
            return True
        except AgentError:
            return False

    # ---- 上传 ----

    def add_upload(self, session_id: str, filename: str, data: bytes,
                   content_type: str = "image/png") -> dict[str, Any]:
        state = self.get(session_id)
        from PIL import Image
        import io
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:
            raise AgentError("E_INVALID_INPUT", f"无法解析的图片文件: {e}") from e
        asset_id = "a" + uuid.uuid4().hex[:8]
        safe_name = "".join(c for c in Path(filename or "upload.png").name if c.isascii() and c.isalnum() or c in "._-") or "upload.png"
        dest_dir = self.uploads_root / session_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{asset_id}_{safe_name}"
        img.convert("RGBA").save(dest, format="PNG")
        asset = {
            "asset_id": asset_id,
            "uri": "asset://" + dest.resolve().relative_to(self.artifacts_dir.resolve()).as_posix(),
            "filename": safe_name,
            "width": img.width,
            "height": img.height,
            "content_type": "image/png",
            "ts": int(time.time() * 1000),
        }
        state["uploads"][asset_id] = asset
        self._persist(state)
        return asset

    def get_asset(self, session_id: str, asset_id: str) -> dict[str, Any]:
        state = self.get(session_id)
        asset = state["uploads"].get(asset_id)
        if not asset:
            raise AgentError(E_ASSET_NOT_FOUND, f"资产不存在: {asset_id}")
        return asset

    # ---- 版本树 / 对话 / 运行 ----

    def record_run(self, session_id: str, run_id: str) -> None:
        state = self.get(session_id)
        if run_id not in state["runs"]:
            state["runs"].append(run_id)
        self._persist(state)

    def record_node_version(self, session_id: str, role: str, version: int, run_id: str,
                            artifacts: list[str], outputs: dict, quality: str) -> None:
        state = self.get(session_id)
        history = state["node_versions"].setdefault(role, [])
        history = [v for v in history if v["version"] != version]
        history.append({
            "version": version, "run_id": run_id, "artifacts": artifacts,
            "outputs": outputs, "quality": quality, "ts": int(time.time() * 1000),
        })
        history.sort(key=lambda v: v["version"])
        state["node_versions"][role] = history
        state.setdefault("current", {})[role] = version
        self._persist(state)

    def set_current(self, session_id: str, role: str, version: int) -> None:
        """回滚后把该 role 的"当前生效版本"指回旧版本（不产生新版本）。"""
        state = self.get(session_id)
        self.version_info(session_id, role, version)  # 校验存在性
        state.setdefault("current", {})[role] = version
        self._persist(state)

    def current_version(self, session_id: str, role: str) -> Optional[int]:
        state = self.get(session_id)
        return (state.get("current") or {}).get(role)

    def latest_version(self, session_id: str, role: str) -> Optional[dict[str, Any]]:
        cur = self.current_version(session_id, role)
        if cur is not None:
            try:
                return self.version_info(session_id, role, cur)
            except AgentError:
                pass
        state = self.get(session_id)
        history = state["node_versions"].get(role) or []
        return history[-1] if history else None

    def version_info(self, session_id: str, role: str, version: int) -> dict[str, Any]:
        state = self.get(session_id)
        for v in state["node_versions"].get(role) or []:
            if v["version"] == version:
                return v
        raise AgentError("E_ROLLBACK_INVALID", f"{role} 不存在版本 v{version}")

    def available_roles(self, session_id: str) -> dict[str, dict[str, Any]]:
        """Planner 视角：每个 role 当前生效版本的 outputs / artifacts / 版本号列表。"""
        state = self.get(session_id)
        out: dict[str, dict[str, Any]] = {}
        for role, history in (state["node_versions"] or {}).items():
            if not history:
                continue
            cur = self.current_version(session_id, role)
            last = next((v for v in history if v["version"] == cur), history[-1])
            out[role] = {
                "outputs": last["outputs"],
                "artifacts": last["artifacts"],
                "versions": [v["version"] for v in history],
            }
        return out

    def add_conversation(self, session_id: str, role: str, text: str) -> None:
        state = self.get(session_id)
        state["conversation"].append({"role": role, "text": text, "ts": int(time.time() * 1000)})
        self._persist(state)

    def set_last_plan(self, session_id: str, dag_dict: dict) -> None:
        state = self.get(session_id)
        state["last_plan"] = dag_dict
        plans = state.setdefault("plans", [])
        plans.append(dag_dict)
        if len(plans) > 30:
            state["plans"] = plans[-30:]
        self._persist(state)

    def plan_history(self, session_id: str) -> list[dict]:
        state = self.get(session_id)
        plans = state.get("plans") or []
        if not plans and state.get("last_plan"):
            plans = [state["last_plan"]]
        return plans

    def resolve_local(self, uri: str) -> Path:
        return resolve_uri(uri, self.artifacts_dir)

    # ---- 持久化 ----

    def _persist(self, state: dict[str, Any]) -> None:
        path = self.sessions_dir / f"{state['session_id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
