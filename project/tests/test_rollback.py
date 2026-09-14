"""A5 条件回滚：版本树登记、回滚 DAG 构造、端到端回滚执行（Demo 04）。"""

import asyncio

from conftest import wait_done

from agent.dag.models import NodeStatus, RunStatus
from agent.planner import PlanningRequest
from agent.planner.rule_planner import RulePlanner
from agent.rollback import build_rollback_dag, select_plan
from agent.session import SessionManager

from test_executor import make_executor


def _plan_req(sessions, text, asset_uri):
    return PlanningRequest(session_id="s1", text=text, asset_uris=[asset_uri],
                           available_roles=sessions.available_roles("s1"))


async def _run(executor, sessions, asset_uri, text):
    result = await RulePlanner().plan(_plan_req(sessions, text, asset_uri))
    assert result.kind == "plan"
    state = await executor.start("s1", result.dag, instruction=text)
    return await wait_done(executor, state.run_id)


def test_rollback_keeps_light_rebuilds_downstream(tmp_env, sample_image_bytes):
    async def main():
        sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
        sessions.create("s1")
        asset = sessions.add_upload("s1", "p.png", sample_image_bytes)
        executor = make_executor(tmp_env, sessions)

        # 第 1 轮：完整链路（背景 v1：傍晚咖啡馆）
        r1 = await _run(executor, sessions, asset["uri"], "把这个人物放进傍晚的咖啡馆，光从左边照过来")
        assert r1.status == RunStatus.done
        bg_v1 = sessions.latest_version("s1", "background_generate")
        assert bg_v1["version"] == 1

        # 第 2 轮：换背景（背景 v2：海边）
        r2 = await _run(executor, sessions, asset["uri"], "把背景换成海边沙滩")
        assert r2.status == RunStatus.done
        hist = sessions.get("s1")["node_versions"]["background_generate"]
        assert [v["version"] for v in hist] == [1, 2]

        # Demo 04：换回背景 v1，保留现在的光
        # 第 4 类小步修改轮的计划可能不含目标角色 -> 从计划历史中选基准
        plans = sessions.plan_history("s1")
        base_dag = select_plan(plans, "background_generate")
        versions = {role: {v["version"]: v for v in hist2}
                    for role, hist2 in sessions.get("s1")["node_versions"].items()}
        dag, meta = build_rollback_dag(base_dag, versions, "background_generate", 1,
                                       ["lighting_estimate", "relight"])
        assert set(meta["rerun"]) == {"shadow_generate", "harmonize", "export"}
        assert "matting" in meta["frozen"] and "relight" in meta["frozen"]

        state = await executor.start("s1", dag, instruction="换回第一版背景，保留现在的光", kind="rollback")
        sessions.set_current("s1", "background_generate", 1)  # 服务层在回滚后设置当前指针
        state = await wait_done(executor, state.run_id)
        return sessions, state, meta

    sessions, state, meta = asyncio.run(main())
    assert state.status == RunStatus.done
    assert all(n.status == NodeStatus.done for n in state.dag.nodes)
    bg = state.dag.node_by_role("background_generate")
    # 背景节点直接落回 v1 产物，没有重跑
    assert bg.version == 1
    assert bg.artifacts == sessions.version_info("s1", "background_generate", 1)["artifacts"]
    # 重跑节点版本 +1
    assert state.dag.node_by_role("shadow_generate").version >= 2
    # 回滚后当前指针指向 v1
    assert sessions.current_version("s1", "background_generate") == 1
    # 最终导出产物更新（基于 v1 背景 + 保留光照的新合成）
    export = state.dag.node_by_role("export")
    path = tmp_env.artifacts_dir / export.outputs["file_url"].split("://", 1)[1]
    assert path.exists()


def test_rollback_invalid_version_rejected(tmp_env, sample_image_bytes):
    from agent.errors import AgentError
    versions = {"background_generate": {1: {"outputs": {}, "artifacts": ["artifact://x"]}}}
    sessions = SessionManager(tmp_env.artifacts_dir, tmp_env.runs_dir)
    sessions.create("s1")
    dag = asyncio.run(RulePlanner().plan(_plan_req(sessions, "把人物放进咖啡馆", "asset://y"))).dag
    try:
        build_rollback_dag(dag, versions, "background_generate", 9, [])
        raise AssertionError("应当拒绝不存在的版本")
    except AgentError as e:
        assert e.code == "E_ROLLBACK_INVALID"
