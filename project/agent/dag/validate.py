"""DAG 静态校验：Planner 产出必须全绿才允许进入 Executor（规范 N1）。"""

from __future__ import annotations

from agent.dag.models import PlanDAG, Quality
from agent.errors import E_DAG_INVALID, AgentError
from agent.schema import load_registry


def validate_dag(dag: PlanDAG) -> None:
    errors = check_dag(dag)
    if errors:
        raise AgentError(E_DAG_INVALID, "DAG 校验失败", detail=errors)


def check_dag(dag: PlanDAG) -> list[str]:
    registry = load_registry()
    errors: list[str] = []
    ids = [n.id for n in dag.nodes]
    if not ids:
        errors.append("DAG 至少需要一个节点")
    if len(ids) != len(set(ids)):
        errors.append(f"节点 id 重复: {[i for i in ids if ids.count(i) > 1]}")
    idset = set(ids)
    roles = [n.role for n in dag.nodes]
    if len(roles) != len(set(roles)):
        errors.append(f"同一次规划中 role 必须唯一（版本管理依赖）: {[r for r in roles if roles.count(r) > 1]}")

    node_map = {n.id: n for n in dag.nodes}
    for n in dag.nodes:
        if n.tool not in registry:
            errors.append(f"节点 {n.id}: 未知工具 {n.tool}")
            continue
        if n.quality not in Quality:
            errors.append(f"节点 {n.id}: 非法 quality {n.quality}")
        # 依赖存在性
        for dep in n.depends_on:
            if dep not in idset:
                errors.append(f"节点 {n.id}: 依赖 {dep} 不存在")
        # 输入引用：@node / @node.output_key / asset:// / artifact:// / 字面量
        for key, val in n.inputs.items():
            if not isinstance(val, str):
                continue
            if val.startswith("@"):
                ref = val[1:]
                ref_node = ref.split(".", 1)[0]
                if ref_node not in idset:
                    errors.append(f"节点 {n.id}.inputs.{key}: 引用 @{ref_node} 不存在")
                elif ref_node not in n.depends_on:
                    errors.append(f"节点 {n.id}.inputs.{key}: 引用 {ref_node} 必须先加入 depends_on")
            elif val.startswith(("asset://", "artifact://")):
                pass
    # 静态校验工具必填输入（@ 引用与冻结输入在运行期解析，此处按可解析处理）
    for n in dag.nodes:
        if n.tool in registry:
            required = registry[n.tool]["input"].get("required", [])
            frozen_inputs = set(dag.frozen.get(n.role, {}).keys())
            for req in required:
                if req not in n.inputs and req not in frozen_inputs:
                    errors.append(f"节点 {n.id}({n.tool}): 缺少必填输入 '{req}'")
    # 无环
    try:
        dag.topological_order()
    except ValueError as e:
        errors.append(str(e))
    return errors
