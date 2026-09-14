"""DAG 模型与静态校验。"""

import pytest

from agent.dag.models import DAGNode, PlanDAG, Quality
from agent.dag.validate import check_dag, validate_dag
from agent.errors import AgentError


def _node(nid, tool, depends=(), inputs=None):
    return DAGNode(id=nid, tool=tool, role=tool, inputs=inputs or {}, depends_on=list(depends))


def _full_chain_dag():
    return PlanDAG(nodes=[
        _node("n1", "matting", inputs={"image": "asset://x"}),
        _node("n2", "background_generate", inputs={"prompt": "咖啡馆"}),
        _node("n3", "lighting_estimate", depends=["n2"], inputs={"bg_png": "@n2.bg_png"}),
        _node("n4", "relight", depends=["n1", "n2", "n3"],
              inputs={"rgba_png": "@n1.rgba_png", "bg_hint": "@n2.bg_png",
                      "light_dir": "@n3.light_dir"}),
        _node("n5", "shadow_generate", depends=["n1", "n2", "n3", "n4"],
              inputs={"foreground_png": "@n4.relit_png", "background_png": "@n2.bg_png",
                      "mask_png": "@n1.mask_png", "light_dir": "@n3.light_dir"}),
        _node("n6", "harmonize", depends=["n1", "n5"],
              inputs={"composite_png": "@n5.composited_png", "mask_png": "@n1.mask_png"}),
    ])


def test_topological_order_and_edges():
    dag = _full_chain_dag()
    order = dag.topological_order()
    assert order.index("n1") < order.index("n4") and order.index("n2") < order.index("n3")
    assert len(dag.edges()) == 10


def test_cycle_detected():
    dag = PlanDAG(nodes=[
        _node("a", "relight", depends=["b"], inputs={"rgba_png": "@b.relit_png", "light_dir": {"azimuth": 0, "polar": 0}}),
        _node("b", "harmonize", depends=["a"], inputs={"composite_png": "@a.harmonized_png", "mask_png": "asset://m"}),
    ])
    errs = check_dag(dag)
    assert any("环" in e for e in errs)
    with pytest.raises(ValueError):
        dag.topological_order()


def test_validate_dag_rejects_bad_reference():
    dag = PlanDAG(nodes=[
        _node("n1", "matting", inputs={"image": "asset://x"}),
        _node("n2", "harmonize", depends=[], inputs={"composite_png": "@n9.composited_png", "mask_png": "@n1.mask_png"}),
    ])
    with pytest.raises(AgentError):
        validate_dag(dag)


def test_downstream_of_with_exclude_roles():
    dag = _full_chain_dag()
    down = dag.downstream_of("n2", exclude_roles={"lighting_estimate", "relight"})
    assert set(down) == {"n5", "n6"}  # 阴影/和谐化重跑，光照链被保留截断


def test_required_input_check_with_frozen():
    dag = _full_chain_dag()
    dag.frozen["matting"] = {"image": "artifact://frozen.png"}
    dag.nodes[0].inputs = {}
    validate_dag(dag)  # 冻结输入补足必填 -> 通过
