"""规则 Planner：覆盖 5 轮多轮剧本与 4 个 Demo 场景的路由；LLM 降级链路。"""

import asyncio

import pytest

from agent.dag.models import Quality
from agent.errors import AgentError
from agent.planner import FallbackPlanner, build_planner
from agent.planner.base import PlanningRequest
from agent.planner.llm_planner import LLMPlanner
from agent.planner.rule_planner import RulePlanner


def _req(text, assets=("asset://uploads/s/a.png",), roles=None, quality=Quality.normal, spatial=None):
    return PlanningRequest(session_id="s", text=text, asset_uris=list(assets),
                           available_roles=roles or {}, quality=quality, spatial=spatial)


def _roles_full():
    """模拟一次完整合成后的会话产物。"""
    def v(role, **outs):
        return {"outputs": outs, "artifacts": list(outs.values()), "versions": [1]}
    return {
        "matting": v("matting", rgba_png="artifact://m_rgba", alpha_png="artifact://m_a", mask_png="artifact://m_k"),
        "background_generate": v("bg", bg_png="artifact://bg", depth_png="artifact://d"),
        "lighting_estimate": v("light", light_dir={"azimuth": 180.0, "polar": 35.0}, color_temp=3600.0, intensity=1.0),
        "relight": v("relight", relit_png="artifact://r"),
        "shadow_generate": v("shadow", shadow_png="artifact://sh", composited_png="artifact://c"),
        "harmonize": v("harmonize", harmonized_png="artifact://h"),
        "export": v("export", file_url="artifact://f"),
    }


def test_round1_full_chain():
    dag = asyncio.run(RulePlanner().plan(_req("把这个人物放进傍晚的咖啡馆，光从左边照过来"))).dag
    tools = [n.tool for n in dag.nodes]
    assert tools == ["matting", "background_generate", "lighting_estimate", "relight",
                     "shadow_generate", "harmonize", "export"]
    validate = dag.topological_order()
    assert validate[0] in ("n1", "n2")  # matting 与背景并行
    bg = dag.node_by_role("background_generate")
    assert "咖啡馆" in bg.inputs["prompt"]


def test_matting_only():
    dag = asyncio.run(RulePlanner().plan(_req("帮我把人物抠出来，要透明背景"))).dag
    assert [n.tool for n in dag.nodes] == ["matting", "export"]


def test_no_asset_raises():
    with pytest.raises(AgentError):
        asyncio.run(RulePlanner().plan(_req("把人物放进咖啡馆", assets=())))


def test_round3_light_edit_uses_frozen_artifacts():
    dag = asyncio.run(RulePlanner().plan(_req("光线太冷了，暖一点", roles=_roles_full()))).dag
    tools = [n.tool for n in dag.nodes]
    assert tools == ["lighting_estimate", "relight", "shadow_generate", "harmonize", "export"]
    relight = dag.node_by_role("relight")
    assert relight.inputs["rgba_png"] == "artifact://m_rgba"  # 冻结引用
    assert relight.inputs["color_temp"] == 3600.0  # 语义"暖"解析为色温


def test_round3_explicit_direction_skips_estimate():
    dag = asyncio.run(RulePlanner().plan(_req("光从右边照过来", roles=_roles_full()))).dag
    tools = [n.tool for n in dag.nodes]
    assert "lighting_estimate" not in tools
    relight = dag.node_by_role("relight")
    assert relight.inputs["light_dir"]["azimuth"] == 0.0


def test_round4_shadow_edit_lighter():
    dag = asyncio.run(RulePlanner().plan(_req("阴影轻一点", roles=_roles_full()))).dag
    tools = [n.tool for n in dag.nodes]
    assert tools == ["shadow_generate", "harmonize", "export"]
    assert dag.nodes[0].inputs["shadow_strength"] == 0.3


def test_spatial_click_passes_point():
    dag = asyncio.run(RulePlanner().plan(
        _req("把这个人物抠出来", spatial={"click": {"x": 120, "y": 200}}))).dag
    matting = dag.node_by_role("matting")
    assert matting.inputs["point"] == {"x": 120, "y": 200}
    assert matting.inputs["mode"] == "trimap"


def test_round5_rollback_intent():
    roles = _roles_full()
    roles["background_generate"]["versions"] = [1, 2]
    result = asyncio.run(RulePlanner().plan(_req("背景换回上一版，但是保留现在的光线", roles=roles)))
    assert result.kind == "rollback"
    assert result.rollback["role"] == "background_generate"
    assert result.rollback["version"] == 1  # 上一版
    assert set(result.rollback["preserve"]) == {"lighting_estimate", "relight"}


def test_enhance_inserted_on_fine_quality():
    dag = asyncio.run(RulePlanner().plan(_req("把人物放进咖啡馆，要精修", quality=Quality.fine))).dag
    assert "enhance" in [n.tool for n in dag.nodes]


def test_fallback_planner_llm_down_to_rule():
    """LLM 未配置 -> 直接规则；LLM 抛错 -> 回落规则（L2）。"""
    planner = FallbackPlanner(primary=_BrokenPlanner(), rule=RulePlanner())
    result = asyncio.run(planner.plan(_req("把人物放进咖啡馆")))
    assert result.dag is not None and result.planner == "rule"

    with pytest.raises(AgentError):
        asyncio.run(FallbackPlanner(primary=_BrokenPlanner(), rule=RulePlanner(),
                                    strict_llm=True).plan(_req("x")))


class _BrokenPlanner(RulePlanner):
    name = "broken"

    async def plan(self, req):
        raise AgentError("E_PLANNER_LLM_UNAVAILABLE", "boom")


def test_build_planner_factory_modes(tmp_env):
    tmp_env.planner_mode = "auto"
    p = build_planner(tmp_env)
    assert isinstance(p, FallbackPlanner)
    tmp_env.planner_mode = "llm"
    with pytest.raises(AgentError):
        build_planner(tmp_env)  # llm 模式但未配置


def test_llm_planner_parse_contract():
    llm = LLMPlanner("http://x", "k", "m")
    content = ('{"nodes": [{"id": "n1", "tool": "matting", "inputs": {"image": "asset://UPLOAD_0"}, '
               '"depends_on": []}, {"id": "n2", "tool": "export", "inputs": {"image": "@n1.rgba_png"}, '
               '"depends_on": ["n1"]}]}')
    dag = llm._parse(content, _req("test"))
    assert dag.nodes[0].inputs["image"] == "asset://uploads/s/a.png"  # UPLOAD_0 已映射
    assert dag.planner == "llm"
