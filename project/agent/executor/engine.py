"""A3 DAG 执行引擎。

能力（验收口径）：
- 拓扑分层执行，无依赖节点并发（matting 与 background_generate 同层并行）；
- Run / Pause / Resume / Cancel / Retry（单节点）/ Skip / Re-run；
- 单节点失败可重试且不影响已完成节点（L1：重试 -> 降精度档位）；
- 节点级幂等：重跑 = 节点版本 +1 + 下游重跑，已完成节点产物不动；
- Critic 闭环：全部完成后打分，不达标按维度->工具映射升档重跑（最多 max_replans 次）；
- 事件总线发布节点级状态（前端实时可见），全事件流落盘可回放。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from agent.config import Settings
from agent.critic.base import BaseCritic
from agent.dag.models import (
    DAGNode,
    NodeStatus,
    PlanDAG,
    Quality,
    RunState,
    RunStatus,
    now_ms,
)
from agent.dag.validate import validate_dag
from agent.errors import (
    E_CANCELLED,
    E_DEP_FAILED,
    E_DAG_INVALID,
    E_RUN_NOT_FOUND,
    E_STATE_CONFLICT,
    E_TIMEOUT,
    AgentError,
)
from agent.eval.recorder import RunRecorder
from agent.executor.bus import EventBus
from agent.schema import get_tool_schema, public_artifact_url, validate_tool_inputs
from agent.session import SessionManager
from agent.tools.provider import ToolProvider

logger = logging.getLogger("agent.executor")


class RunRuntime:
    def __init__(self, state: RunState):
        self.state = state
        self.pause_gate = asyncio.Event()
        self.pause_gate.set()  # set = 运行中；clear = 暂停
        self.cancel_requested = False
        self.task: Optional[asyncio.Task] = None
        self.kind: str = "plan"
        self.session_id: str = ""

    @property
    def driving(self) -> bool:
        return self.task is not None and not self.task.done()


class Executor:
    def __init__(self, provider: ToolProvider, critic: BaseCritic | None, recorder: RunRecorder,
                 bus: EventBus, settings: Settings, sessions: SessionManager | None = None):
        self.provider = provider
        self.critic = critic
        self.recorder = recorder
        self.bus = bus
        self.settings = settings
        self.sessions = sessions
        self.runs: dict[str, RunRuntime] = {}

    # ================= 启动 =================

    async def start(self, session_id: str, dag: PlanDAG, instruction: str = "", kind: str = "plan") -> RunState:
        validate_dag(dag)
        state = RunState(session_id=session_id, dag=dag, instruction=instruction,
                         status=RunStatus.running)
        dag.run_id = state.run_id
        rt = RunRuntime(state)
        rt.kind = kind
        rt.session_id = session_id
        self.runs[state.run_id] = rt
        if self.sessions:
            self.sessions.record_run(session_id, state.run_id)
            self.sessions.set_last_plan(session_id, dag.model_dump(mode="json"))
        self.recorder.save_plan(state.run_id, dag.model_dump(mode="json"))
        self._log(rt, "run_started", {"instruction": instruction, "kind": kind,
                                      "planner": dag.planner, "nodes": len(dag.nodes)})
        rt.task = asyncio.create_task(self._drive(rt))
        return state

    def get_run(self, run_id: str) -> RunState:
        rt = self.runs.get(run_id)
        if rt is None:
            raise AgentError(E_RUN_NOT_FOUND, f"运行不存在: {run_id}")
        return rt.state

    # ================= 控制（A3）=================

    def pause(self, run_id: str) -> RunState:
        rt = self._runtime(run_id)
        self._check_active(rt)
        rt.pause_gate.clear()
        rt.state.status = RunStatus.paused
        self._log(rt, "run_paused", {})
        return rt.state

    def resume(self, run_id: str) -> RunState:
        rt = self._runtime(run_id)
        if rt.state.status != RunStatus.paused:
            raise AgentError(E_STATE_CONFLICT, f"运行处于 {rt.state.status}，无法恢复")
        rt.state.status = RunStatus.running
        rt.pause_gate.set()
        self._log(rt, "run_resumed", {})
        return rt.state

    async def cancel(self, run_id: str) -> RunState:
        rt = self._runtime(run_id)
        rt.cancel_requested = True
        rt.pause_gate.set()
        if rt.driving:
            try:
                await asyncio.wait_for(asyncio.shield(rt.task), timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                rt.task.cancel()
        return rt.state

    def retry_node(self, run_id: str, node_id: str) -> RunState:
        """失败节点重试：节点 + 下游重置 pending，已完成节点不动。"""
        return self._reset_chain(run_id, node_id, bump_version=False)

    def rerun_node(self, run_id: str, node_id: str, higher_quality: bool = False) -> RunState:
        """手动重跑：节点版本 +1，下游重跑。"""
        rt = self._runtime(run_id)
        node = self._node(rt, node_id)
        if higher_quality:
            node.quality = node.quality.higher() or node.quality
        state = self._reset_chain(run_id, node_id, bump_version=True)
        self._log(rt, "node_rerun", {"node_id": node_id, "quality": node.quality})
        return state

    def skip_node(self, run_id: str, node_id: str) -> RunState:
        rt = self._runtime(run_id)
        node = self._node(rt, node_id)
        if node.status != NodeStatus.pending:
            raise AgentError(E_STATE_CONFLICT, f"节点 {node_id} 状态为 {node.status}，只能跳过 pending 节点")
        node.status = NodeStatus.skipped
        node.error = {"code": "E_SKIPPED", "message": "用户跳过", "retryable": False}
        self._log(rt, "node_skipped", {"node_id": node_id})
        return rt.state

    # ================= 引擎主循环 =================

    async def _drive(self, rt: RunRuntime) -> None:
        state = rt.state
        try:
            state.status = RunStatus.running
            self._publish_run(rt, "run_status", {"status": state.status})
            while True:
                await rt.pause_gate.wait()
                if rt.cancel_requested:
                    self._mark_pending_cancelled(rt)
                    state.status = RunStatus.cancelled
                    state.error = {"code": E_CANCELLED, "message": "用户取消", "retryable": False}
                    break
                executed = await self._run_wave(rt)
                failed = [n for n in state.dag.nodes if n.status == NodeStatus.failed]
                if failed:
                    state.status = RunStatus.failed
                    state.error = failed[0].error
                    self._block_dependents(rt)
                    break
                if not executed:
                    blocked = [n for n in state.dag.nodes if n.status == NodeStatus.pending]
                    for n in blocked:
                        n.status = NodeStatus.skipped
                        n.error = {"code": E_DEP_FAILED, "message": "上游节点未产出，无法执行", "retryable": False}
                        self._publish_node(rt, n)
                    if any(n.status == NodeStatus.skipped for n in state.dag.nodes):
                        state.status = RunStatus.done
                        break
                    # 全部完成 -> Critic
                    action = await self._critique(rt)
                    if action == "rerun":
                        continue
                    state.status = RunStatus.done
                    break
        except asyncio.CancelledError:
            self._mark_pending_cancelled(rt)
            state.status = RunStatus.cancelled
        except Exception as e:  # 引擎级异常不得无声吞掉
            logger.exception("run %s engine error", state.run_id)
            state.status = RunStatus.failed
            state.error = {"code": "E_ENGINE", "message": str(e), "retryable": False}
        finally:
            state.finished_at = now_ms()
            self.recorder.save_result(state.run_id, state.model_dump(mode="json"))
            self._publish_run(rt, "run_finished", {
                "status": state.status, "error": state.error,
                "critic": state.critic.model_dump(mode="json") if state.critic else None,
            })

    async def _run_wave(self, rt: RunRuntime) -> bool:
        """执行当前所有就绪节点（并发）；返回是否执行了任何节点。"""
        dag = rt.state.dag
        ready = [n for n in dag.nodes
                 if n.status == NodeStatus.pending
                 and all(dag.node(d).status == NodeStatus.done for d in n.depends_on)]
        if not ready:
            return False
        self._log(rt, "wave", {"nodes": [n.id for n in ready]})
        await asyncio.gather(*(self._execute_node(rt, n) for n in ready))
        return True

    async def _execute_node(self, rt: RunRuntime, node: DAGNode) -> None:
        state = rt.state
        dag = state.dag
        quality = node.quality
        while True:
            node.attempts += 1
            node.status = NodeStatus.running
            node.started_at = now_ms()
            node.error = None
            self._publish_node(rt, node)
            await rt.pause_gate.wait()
            if rt.cancel_requested:
                node.status = NodeStatus.pending
                return
            try:
                inputs = self._resolve_inputs(dag, node)
                validate_tool_inputs(node.tool, inputs, node.options)
                options = {**node.options, "quality": quality.value}
                out_dir = f"runs/{state.run_id}/{node.id}/v{node.version}"
                schema = get_tool_schema(node.tool)
                envelope = await asyncio.wait_for(
                    self.provider.invoke(node.tool, inputs, options, out_dir,
                                         timeout_s=schema["latency"]["timeout_ms"] / 1000),
                    timeout=self.settings.node_timeout_s,
                )
                node.outputs = envelope.get("outputs", {})
                node.artifacts = envelope.get("artifacts") or [v for v in node.outputs.values()
                                                               if isinstance(v, str)]
                node.latency_ms = envelope.get("latency_ms")
                node.finished_at = now_ms()
                node.status = NodeStatus.done
                node.quality = quality
                self._publish_node(rt, node)
                if self.sessions and state.session_id:
                    self.sessions.record_node_version(state.session_id, node.role, node.version,
                                                      state.run_id, node.artifacts, node.outputs,
                                                      node.quality.value)
                return
            except AgentError as e:
                node.error = e.to_dict()
                self._log(rt, "node_error", {"node_id": node.id, "error": node.error,
                                             "attempt": node.attempts})
                if e.retryable and node.attempts <= self.settings.max_node_retries:
                    lowered = quality.lower()  # L1 降级：重试换低精度档位
                    if lowered is not None:
                        quality = lowered
                        self._log(rt, "node_degrade_retry", {"node_id": node.id, "quality": quality})
                    continue
                node.finished_at = now_ms()
                node.status = NodeStatus.failed
                self._publish_node(rt, node)
                return
            except asyncio.TimeoutError:
                node.error = {"code": E_TIMEOUT, "message": f"节点超时(>{self.settings.node_timeout_s}s)",
                              "retryable": True}
                if node.attempts <= self.settings.max_node_retries:
                    lowered = quality.lower()
                    if lowered is not None:
                        quality = lowered
                    continue
                node.finished_at = now_ms()
                node.status = NodeStatus.failed
                self._publish_node(rt, node)
                return

    # ================= Critic 闭环 =================

    async def _critique(self, rt: RunRuntime) -> str:
        """返回 'rerun'（进入下一轮重跑）或 'done'。"""
        state = rt.state
        if self.critic is None or not self.settings.critic_enabled:
            return "done"
        state.status = RunStatus.critiquing
        self._publish_run(rt, "run_status", {"status": state.status})
        result = self.critic.score(state)
        state.critic = result
        state.critic_history.append(result)
        self._log(rt, "critic", result.model_dump(mode="json"))
        self.bus.publish(state.session_id, {
            "type": "critic", "ts": now_ms(), "run_id": state.run_id,
            "data": result.model_dump(mode="json"),
        })
        if result.action != "rerun" or not result.rerun_role:
            return "done"
        state.replan_count += 1
        state.status = RunStatus.replanning
        self._apply_rerun(rt, result.rerun_role)
        return "rerun"

    def _apply_rerun(self, rt: RunRuntime, role: str) -> None:
        """Critic 触发：该节点升档重跑，下游连带重跑（版本 +1）。"""
        dag = rt.state.dag
        node = dag.node_by_role(role)
        assert node is not None
        node.quality = node.quality.higher() or node.quality
        self._reset(dag, node, bump_version=True)
        for nid in dag.downstream_of(node.id):
            self._reset(dag, dag.node(nid), bump_version=True)
        self._log(rt, "replan", {"role": role, "replan_count": rt.state.replan_count,
                                 "quality": node.quality})

    # ================= 内部工具 =================

    def _reset(self, dag: PlanDAG, node: DAGNode, bump_version: bool) -> None:
        if bump_version:
            node.version += 1
        node.status = NodeStatus.pending
        node.attempts = 0
        node.error = None
        node.outputs = {}
        node.artifacts = []
        node.latency_ms = None
        node.started_at = None
        node.finished_at = None

    def _reset_chain(self, run_id: str, node_id: str, bump_version: bool) -> RunState:
        rt = self._runtime(run_id)
        node = self._node(rt, node_id)
        if node.status == NodeStatus.running:
            raise AgentError(E_STATE_CONFLICT, "节点正在运行，无法重置")
        dag = rt.state.dag
        self._reset(dag, node, bump_version=bump_version)
        for nid in dag.downstream_of(node.id):
            n = dag.node(nid)
            if bump_version:
                n.version += 1
            self._reset(dag, n, bump_version=False)
        rt.cancel_requested = False
        if not rt.pause_gate.is_set():
            rt.pause_gate.set()
        if rt.state.status in (RunStatus.failed, RunStatus.cancelled, RunStatus.done,
                               RunStatus.paused) or not rt.driving:
            rt.state.status = RunStatus.running
            rt.task = asyncio.create_task(self._drive(rt))
        self._log(rt, "node_reset", {"node_id": node_id, "bump_version": bump_version})
        return rt.state

    def _resolve_inputs(self, dag: PlanDAG, node: DAGNode) -> dict[str, Any]:
        def resolve(val: Any) -> Any:
            if isinstance(val, str) and val.startswith("@"):
                ref = val[1:]
                if "." in ref:
                    nid, key = ref.split(".", 1)
                else:
                    nid, key = ref, None
                dep = dag.node(nid)
                if key is not None:
                    if key not in dep.outputs:
                        raise AgentError(E_DAG_INVALID,
                                         f"节点 {node.id} 引用 @{nid}.{key}，但上游无此输出")
                    return dep.outputs[key]
                if not dep.outputs:
                    raise AgentError(E_DAG_INVALID, f"节点 {node.id} 引用 @{nid}，但上游无输出")
                return next(iter(dep.outputs.values()))
            return val

        inputs = {k: resolve(v) for k, v in node.inputs.items()}
        frozen = dag.frozen.get(node.role)
        if frozen:
            inputs.update(frozen)
        return inputs

    def _block_dependents(self, rt: RunRuntime) -> None:
        dag = rt.state.dag
        for n in dag.nodes:
            if n.status == NodeStatus.pending:
                n.status = NodeStatus.skipped
                n.error = {"code": E_DEP_FAILED, "message": "上游失败，未执行", "retryable": False}
                self._publish_node(rt, n)

    def _mark_pending_cancelled(self, rt: RunRuntime) -> None:
        for n in rt.state.dag.nodes:
            if n.status in (NodeStatus.pending, NodeStatus.running):
                n.status = NodeStatus.cancelled
                n.error = {"code": E_CANCELLED, "message": "已取消", "retryable": False}
                self._publish_node(rt, n)

    def _runtime(self, run_id: str) -> RunRuntime:
        rt = self.runs.get(run_id)
        if rt is None:
            raise AgentError(E_RUN_NOT_FOUND, f"运行不存在: {run_id}")
        return rt

    @staticmethod
    def _node(rt: RunRuntime, node_id: str) -> DAGNode:
        try:
            return rt.state.dag.node(node_id)
        except KeyError as e:
            raise AgentError("E_NODE_NOT_FOUND", str(e)) from e

    @staticmethod
    def _check_active(rt: RunRuntime) -> None:
        if rt.state.status not in (RunStatus.running, RunStatus.paused, RunStatus.replanning,
                                   RunStatus.critiquing):
            raise AgentError(E_STATE_CONFLICT, f"运行处于终态 {rt.state.status}")

    # ================= 观测 =================

    def _node_snapshot(self, node: DAGNode) -> dict[str, Any]:
        data = node.model_dump(mode="json")
        data["artifact_urls"] = [public_artifact_url(a) for a in node.artifacts]
        return data

    def _publish_node(self, rt: RunRuntime, node: DAGNode) -> None:
        self.bus.publish(rt.state.session_id, {
            "type": "node_update", "ts": now_ms(), "run_id": rt.state.run_id,
            "data": self._node_snapshot(node),
        })
        self._log(rt, "node_update", {"node_id": node.id, "status": node.status,
                                      "version": node.version, "attempts": node.attempts})

    def _publish_run(self, rt: RunRuntime, type_: str, data: dict[str, Any]) -> None:
        self.bus.publish(rt.state.session_id, {"type": type_, "ts": now_ms(),
                                               "run_id": rt.state.run_id, "data": data})

    def _log(self, rt: RunRuntime, event: str, data: dict[str, Any]) -> None:
        self.recorder.log_event(rt.state.run_id, {"event": event, "ts": now_ms(), **data})
