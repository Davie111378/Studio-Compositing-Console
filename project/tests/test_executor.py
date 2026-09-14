"""A3 执行引擎：全链路成功、Critic 自动重规划、失败重试、暂停恢复、跳过、取消、手动重跑。"""

import asyncio

import pytest

from conftest import wait_done

from agent.critic import build_critic
from agent.dag.models import NodeStatus, Quality, RunStatus
from agent.eval.recorder import RunRecorder
from agent.executor import EventBus, Executor
from agent.planner import PlanningRequest
from agent.planner.rule_planner import RulePlanner
from agent.session import SessionManager
from agent.tools import EmbeddedProvider, ToolFailureError, ToolProvider


def make_executor(settings, sessions=None, provider=None, critic=None) -> Executor:
    provider = provider or EmbeddedProvider(settings.project_root / "ai-service", settings.artifacts_dir)
    critic = critic if critic is not None else build_critic(settings)
    recorder = RunRecorder(settings.runs_dir)
    return Executor(provider, critic, recorder, EventBus(), settings, sessions)


async def full_chain(settings, sessions: SessionManager, image_uri: str, text="把这个人物放进傍晚的咖啡馆，光从左边照过来"):
    planner = RulePlanner()
    req = PlanningRequest(session_id="s1", text=text, asset_uris=[image_uri],
                          available_roles=sessions.available_roles("s1"))
    result = await planner.plan(req)
    assert result.kind == "plan"
    return result.dag


async def _run_full_chain(tmp_env, sample_image_bytes):
    sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
    sessions.create("s1")
    asset = sessions.add_upload("s1", "person.png", sample_image_bytes)
    executor = make_executor(tmp_env, sessions)
    dag = await full_chain(tmp_env, sessions, asset["uri"])
    state = await executor.start("s1", dag, instruction="demo")
    state = await wait_done(executor, state.run_id)
    return executor, sessions, state


def test_full_chain_success_with_critic_replan(tmp_env, sample_image_bytes):
    executor, sessions, state = asyncio.run(_run_full_chain(tmp_env, sample_image_bytes))
    assert state.status == RunStatus.done
    nodes = state.dag.nodes
    assert all(n.status == NodeStatus.done for n in nodes)
    # Critic 闭环：normal 档阴影不达标 -> 触发 shadow 升档重跑 -> PASS
    assert state.replan_count == 1
    assert state.critic.passed is True
    shadow = state.dag.node_by_role("shadow_generate")
    assert shadow.quality == Quality.fine and shadow.version == 2
    # 产物真实落盘
    export = state.dag.node_by_role("export")
    assert export.outputs["meta"]["width"] > 0
    path = tmp_env.artifacts_dir / export.outputs["file_url"].split("://", 1)[1]
    assert path.exists() and path.stat().st_size > 0
    # 版本树已登记
    assert sessions.current_version("s1", "background_generate") == 1
    # 可回放（A8）
    result = RunRecorder(tmp_env.runs_dir).load_result(state.run_id)
    assert result["status"] == "done"


def test_retryable_failure_degrades_and_recovers(tmp_env, sample_image_bytes):
    class Flaky(ToolProvider):
        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        async def invoke(self, tool, inputs, options, out_dir, request_id=None, timeout_s=None):
            if tool == "matting":
                self.calls += 1
                if self.calls == 1:
                    raise ToolFailureError("E_MATTING_OOM", "模拟显存不足", retryable=True)
            return await self.inner.invoke(tool, inputs, options, out_dir, request_id, timeout_s)

        async def aclose(self):
            await self.inner.aclose()

    async def main():
        sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        provider = EmbeddedProvider(tmp_env.project_root / "ai-service", tmp_env.artifacts_dir)
        executor = make_executor(tmp_env, sessions, provider=Flaky(provider))
        dag = await full_chain(tmp_env, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        return executor, await wait_done(executor, state.run_id)

    executor, state = asyncio.run(main())
    assert state.status == RunStatus.done
    # L1 降级重试从事件流回放断言（Critic 事后会把 matting 升档重跑，终态不再是 draft）
    events = RunRecorder(tmp_env.runs_dir).load_result(state.run_id)
    assert events["status"] == "done"
    replay = _read_events(tmp_env, state.run_id)
    assert any(e["event"] == "node_degrade_retry" and e.get("node_id")
               for e in replay)


def _read_events(tmp_env, run_id):
    import json
    path = tmp_env.runs_dir / "runs" / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_non_retryable_failure_then_retry_node(tmp_env, sample_image_bytes):
    class HardFail(ToolProvider):
        def __init__(self, inner):
            self.inner, self.failed = inner, False

        async def invoke(self, tool, inputs, options, out_dir, request_id=None, timeout_s=None):
            if tool == "harmonize" and not self.failed:
                self.failed = True
                raise ToolFailureError("E_HARMONIZE_WEIGHT_MISSING", "权重缺失", retryable=False)
            return await self.inner.invoke(tool, inputs, options, out_dir, request_id, timeout_s)

        async def aclose(self):
            await self.inner.aclose()

    async def main():
        sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        provider = EmbeddedProvider(tmp_env.project_root / "ai-service", tmp_env.artifacts_dir)
        executor = make_executor(tmp_env, sessions, provider=HardFail(provider))
        dag = await full_chain(tmp_env, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        state = await wait_done(executor, state.run_id)
        assert state.status == RunStatus.failed
        # 失败不影响已完成节点
        assert state.dag.node_by_role("matting").status == NodeStatus.done
        assert state.dag.node_by_role("export").status == NodeStatus.skipped
        # 手动重试失败节点 -> 全链路完成
        state = executor.retry_node(state.run_id, state.dag.node_by_role("harmonize").id)
        state = await wait_done(executor, state.run_id)
        return state

    state = asyncio.run(main())
    assert state.status == RunStatus.done
    assert all(n.status == NodeStatus.done for n in state.dag.nodes)


def test_pause_and_resume(tmp_env, sample_image_bytes, monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_MS", "80")
    from agent.config import reset_settings
    settings = reset_settings()

    async def main():
        sessions = SessionManager(settings.artifacts_dir, settings.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        executor = make_executor(settings, sessions)
        dag = await full_chain(settings, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        await asyncio.sleep(0.02)
        executor.pause(state.run_id)
        await asyncio.sleep(0.3)  # 当前 wave 跑完，引擎在闸口等待
        state = executor.get_run(state.run_id)
        assert state.status == RunStatus.paused
        running_or_pending = {n.tool: n.status for n in state.dag.nodes}
        assert running_or_pending["lighting_estimate"] == NodeStatus.pending
        executor.resume(state.run_id)
        return await wait_done(executor, state.run_id)

    state = asyncio.run(main())
    assert state.status == RunStatus.done


def test_skip_node_user_intent_wins(tmp_env, sample_image_bytes, monkeypatch):
    """用户显式跳过：该节点及其下游保持 skipped，运行正常收尾（Critic 不覆盖用户意图）。"""
    monkeypatch.setenv("MOCK_DELAY_MS", "80")
    from agent.config import reset_settings
    settings = reset_settings()

    async def main():
        sessions = SessionManager(settings.artifacts_dir, settings.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        executor = make_executor(settings, sessions)
        dag = await full_chain(settings, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        await asyncio.sleep(0.05)  # 此时尚未轮到阴影节点
        shadow = state.dag.node_by_role("shadow_generate")
        state = executor.skip_node(state.run_id, shadow.id)
        return await wait_done(executor, state.run_id)

    state = asyncio.run(main())
    assert state.status == RunStatus.done
    assert state.dag.node_by_role("shadow_generate").status == NodeStatus.skipped
    assert state.dag.node_by_role("harmonize").status == NodeStatus.skipped
    assert state.dag.node_by_role("export").status == NodeStatus.skipped
    # 已完成节点不受影响
    assert state.dag.node_by_role("matting").status == NodeStatus.done
    assert state.critic is None  # 无完整产物，跳过评审


def test_cancel_run(tmp_env, sample_image_bytes, monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_MS", "300")
    from agent.config import reset_settings
    settings = reset_settings()

    async def main():
        sessions = SessionManager(settings.artifacts_dir, settings.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        executor = make_executor(settings, sessions)
        dag = await full_chain(settings, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        await asyncio.sleep(0.1)
        state = await executor.cancel(state.run_id)
        return state

    state = asyncio.run(main())
    assert state.status == RunStatus.cancelled
    assert any(n.status == NodeStatus.cancelled for n in state.dag.nodes)


def test_manual_rerun_bumps_version(tmp_env, sample_image_bytes):
    async def main():
        sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        executor = make_executor(tmp_env, sessions)
        dag = await full_chain(tmp_env, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        state = await wait_done(executor, state.run_id)
        harm = state.dag.node_by_role("harmonize")
        v_before = harm.version
        state = executor.rerun_node(state.run_id, harm.id, higher_quality=True)
        state = await wait_done(executor, state.run_id)
        return v_before, state

    v_before, state = asyncio.run(main())
    assert state.status == RunStatus.done
    harm = state.dag.node_by_role("harmonize")
    assert harm.version == v_before + 1
    assert harm.quality == Quality.fine


def test_run_wave_parallelizes_roots(tmp_env, sample_image_bytes):
    """matting 与 background_generate 无依赖，应在同一 wave 并发。"""
    events = []

    class SpyBus(EventBus):
        def publish(self, session_id, event):
            events.append(event)
            super().publish(session_id, event)

    async def main():
        sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        executor = make_executor(tmp_env, sessions)
        executor.bus = SpyBus()
        dag = await full_chain(tmp_env, sessions, asset["uri"])
        state = await executor.start("s1", dag)
        return await wait_done(executor, state.run_id)

    asyncio.run(main())
    wave_events = [e for e in events if e["type"] == "node_update"]
    starts = [e for e in wave_events if e["data"]["status"] == "running"]
    first_two = [e["data"]["tool"] for e in starts[:2]]
    assert set(first_two) == {"matting", "background_generate"}


from tests.conftest import wait_done  # noqa: E402
