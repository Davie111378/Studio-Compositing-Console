"""LLM Planner：OpenAI 兼容 chat/completions + JSON 输出强校验（规范 5.4）。

失败语义：LLM 未配置 -> E_PLANNER_LLM_UNAVAILABLE；输出连续 2 次校验失败 -> E_PLANNER_EXHAUSTED。
调用方（工厂）捕获后回落 RulePlanner（L2 降级）。
"""

from __future__ import annotations

import json
import logging

import httpx

from agent.dag.models import DAGNode, PlanDAG, Quality
from agent.dag.validate import validate_dag
from agent.errors import E_PLANNER_EXHAUSTED, E_PLANNER_LLM_UNAVAILABLE, AgentError
from agent.planner.base import Planner, PlanningRequest, PlanningResult
from agent.schema import load_registry, validate_payload

logger = logging.getLogger("agent.planner.llm")

_SYSTEM_PROMPT = """你是图像合成 Agent 的 Planner。把用户指令解析为工具执行计划 DAG，不生成像素。

可用工具（JSON Schema 摘要）：
{tools}

输出格式（严格 JSON，无 markdown 围栏）：
{{
  "nodes": [
    {{"id": "n1", "tool": "matting", "inputs": {{"image": "asset://UPLOAD_0"}}, "depends_on": []}},
    {{"id": "n2", "tool": "background_generate", "inputs": {{"prompt": "..."}}, "depends_on": []}},
    ...
  ]
}}

硬性规则：
1. tool 只能取上面 8 个名字；role 省略（默认等于 tool）。
2. 引用上游输出写 "@节点id.输出键"，且该节点必须出现在 depends_on。
3. 用户上传图用 "asset://UPLOAD_0"（第 2 张为 asset://UPLOAD_1）。
4. 完整合成链 = matting → background_generate → lighting_estimate → relight → shadow_generate → harmonize → export；
   matting 与 background_generate 可并行（都无依赖）。局部修改只输出受影响的下游子链。
5. 引用型键（*_png、image、light_dir 等）只允许填引用 "@节点id.输出键" 或 "asset://UPLOAD_n"；
   严禁内联数值、数组或自造对象（如 light_dir 必须写 "@n?.light_dir"，禁止写 [x,y,z]）。
6. 用户要把前景放进"另一张上传的照片"时：跳过 background_generate，把那张照片的 asset://UPLOAD_n
   直接作为 lighting_estimate 的 bg_png 和 shadow_generate 的 background_png。
7. 不新增工具、不改工具输入输出键名、不输出解释文字。

示例（把人物放进傍晚的咖啡馆）：
{example}"""


def _tool_summary() -> str:
    parts = []
    for tool, s in load_registry().items():
        req = s["input"].get("required", [])
        props = list(s["input"].get("properties", {}).keys())
        out = list(s["output"].get("properties", {}).keys())
        parts.append(f"- {s['id']} {tool}: 输入必填={req} 可选={props} 输出={out}")
    return "\n".join(parts)


def _example_json() -> str:
    return json.dumps({
        "nodes": [
            {"id": "n1", "tool": "matting", "inputs": {"image": "asset://UPLOAD_0"}, "depends_on": []},
            {"id": "n2", "tool": "background_generate", "inputs": {"prompt": "傍晚的咖啡馆"}, "depends_on": []},
            {"id": "n3", "tool": "lighting_estimate", "inputs": {"bg_png": "@n2.bg_png"}, "depends_on": ["n2"]},
            {"id": "n4", "tool": "relight", "inputs": {"rgba_png": "@n1.rgba_png", "light_dir": "@n3.light_dir"},
             "depends_on": ["n1", "n3"]},
            {"id": "n5", "tool": "shadow_generate", "inputs": {"foreground_png": "@n4.relit_png",
                                                               "background_png": "@n2.bg_png",
                                                               "mask_png": "@n1.mask_png",
                                                               "light_dir": "@n3.light_dir"},
             "depends_on": ["n1", "n2", "n3", "n4"]},
            {"id": "n6", "tool": "harmonize", "inputs": {"composite_png": "@n5.composited_png",
                                                         "mask_png": "@n1.mask_png"},
             "depends_on": ["n1", "n5"]},
            {"id": "n7", "tool": "export", "inputs": {"image": "@n6.harmonized_png", "format": "png"},
             "depends_on": ["n6"]},
        ]
    }, ensure_ascii=False)


class LLMPlanner(Planner):
    name = "llm"

    def __init__(self, api_base: str, api_key: str, model: str, timeout_s: float = 20.0, max_retries: int = 2):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._system = _SYSTEM_PROMPT.format(tools=_tool_summary(), example=_example_json())

    @staticmethod
    def configured(api_base: str, model: str) -> bool:
        return bool(api_base and model)

    async def plan(self, req: PlanningRequest) -> PlanningResult:
        messages = [{"role": "system", "content": self._system}]
        if req.asset_uris:
            n = len(req.asset_uris)
            messages.append({"role": "system",
                             "content": f"本次会话可用上传图共 {n} 张：asset://UPLOAD_0 到 asset://UPLOAD_{n - 1}，"
                                        "指令中不存在的上传图编号不得使用。"})
        if req.spatial:
            messages.append({"role": "system",
                             "content": f"用户空间指代：{json.dumps(req.spatial, ensure_ascii=False)}"})
        messages.append({"role": "user", "content": req.text})
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                content = await self._call(messages)
                dag = self._parse(content, req)
                validate_dag(dag)
                self._validate_inputs(dag)
                return PlanningResult(dag=dag, planner=self.name)
            except (httpx.HTTPError, AgentError, KeyError, TypeError, ValueError) as e:
                last_err = e
                logger.warning("LLM planner attempt failed: %s", e)
                feedback = (f"你的输出未通过校验：{e}。请修正后重新输出完整 JSON 计划（只输出 JSON，"
                            "所有图像/遮罩/光照等键必须写上游引用 \"@节点id.输出键\"，禁止内联数值或数组）。")
                if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("你的输出未通过校验"):
                    messages[-1]["content"] = feedback
                else:
                    messages.append({"role": "user", "content": feedback})
        if isinstance(last_err, AgentError) and last_err.code == E_PLANNER_LLM_UNAVAILABLE:
            raise last_err
        raise AgentError(E_PLANNER_EXHAUSTED, f"LLM Planner 连续 {self.max_retries + 1} 次失败",
                         detail=str(last_err))

    @staticmethod
    def _validate_inputs(dag: PlanDAG) -> None:
        """计划期输入类型校验：引用 "@..." 是运行期才解析的合法占位，其余按 Schema 校验。

        执行期才校验会让坏计划在跑到一半时失败，代价远高于规划期重试，故提前至此。
        """
        registry = load_registry()
        for n in dag.nodes:
            props = registry[n.tool]["input"].get("properties", {})
            for key, value in (n.inputs or {}).items():
                if isinstance(value, str) and value.startswith(("@", "asset://", "artifact://")):
                    continue
                sch = props.get(key)
                if not sch:
                    continue  # 未声明键交由执行端兜底校验
                errs = validate_payload(sch, value, f"节点 {n.id}({n.tool}).inputs.{key}")
                if errs:
                    raise AgentError(E_PLANNER_EXHAUSTED, "计划输入类型不合法", detail=errs)

    async def _call(self, messages: list[dict]) -> str:
        if not self.configured(self.api_base, self.model):
            raise AgentError(E_PLANNER_LLM_UNAVAILABLE, "LLM 未配置（LLM_API_BASE/LLM_MODEL）")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={"model": self.model, "messages": messages, "temperature": 0,
                      "response_format": {"type": "json_object"}},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _parse(self, content: str, req: PlanningRequest) -> PlanDAG:
        data = json.loads(content)
        nodes_raw = data["nodes"]
        asset_map = {f"asset://UPLOAD_{i}": uri for i, uri in enumerate(req.asset_uris)}
        nodes: list[DAGNode] = []
        for raw in nodes_raw:
            inputs = {}
            for k, v in (raw.get("inputs") or {}).items():
                inputs[k] = asset_map.get(v, v) if isinstance(v, str) else v
            nodes.append(DAGNode(
                id=str(raw["id"]),
                tool=str(raw["tool"]),
                role=str(raw.get("role") or raw["tool"]),
                label=str(raw.get("label") or raw["tool"]),
                inputs=inputs,
                options=raw.get("options") or {},
                depends_on=[str(d) for d in (raw.get("depends_on") or [])],
                quality=req.quality,
            ))
        # 版本号承接会话历史
        for n in nodes:
            versions = (req.available_roles.get(n.role) or {}).get("versions") or []
            n.version = (max(versions) + 1) if versions else 1
        return PlanDAG(nodes=nodes, planner=self.name, instruction=req.text, quality=req.quality)
