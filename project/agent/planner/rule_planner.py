"""规则 Planner：关键词模板 -> 固定 DAG（L2 兜底，也是 LLM 未配置时的默认 Planner）。

设计目标（规范 3.4 / 3.5）：确定性覆盖 5 轮多轮剧本与 4 个 Demo 场景，离线可测。
模板自由度刻意压低：Plan 只做增量修改，防止多轮编辑退化（规范 6.2）。
"""

from __future__ import annotations

from agent.dag.models import DAGNode, PlanDAG, Quality
from agent.errors import E_INVALID_INPUT, AgentError
from agent.planner.base import Planner, PlanningRequest, PlanningResult

TOOL_LABELS = {
    "matting": "抠取",
    "background_generate": "背景生成",
    "lighting_estimate": "光照估计",
    "relight": "重打光",
    "shadow_generate": "接触阴影",
    "harmonize": "和谐化",
    "enhance": "细节增强",
    "export": "导出",
}

_BG_WORDS = ("放进", "放到", "合成", "换个场景", "置于", "背景换成", "换成", "改成", "换背景", "换个背景", "换场景")
_MATTING_WORDS = ("抠图", "抠像", "抠取", "抠出", "去背", "透明背景", "matting")
_LIGHT_WORDS = ("光", "亮", "暗", "冷", "暖", "明", "昏")
_SHADOW_WORDS = ("阴影", "影子")
_HARMONIZE_WORDS = ("和谐", "融合", "协调")
_ENHANCE_WORDS = ("细节", "景深", "虚化", "清晰", "锐化", "精修", "高清")
_ROLLBACK_WORDS = ("换回", "回退", "回到", "恢复到", "还原")
_VERSION_FIRST = ("第一版", "最开始", "最初")
_VERSION_PREV = ("上一版", "上一个", "刚才那版", "之前那版")
_PRESERVE_HINT = "保留"

# 光向关键词 -> {azimuth, polar}（约定见 docs/api-spec.md：0=右 90=上 180=左 270=下）
_LIGHT_DIR_WORDS = {
    ("左",): {"azimuth": 180.0, "polar": 35.0},
    ("右",): {"azimuth": 0.0, "polar": 35.0},
    ("头顶", "顶光", "正上方"): {"azimuth": 0.0, "polar": 85.0},
    ("逆光", "背后"): {"azimuth": 90.0, "polar": 30.0},
}
_WARM_WORDS = ("暖", "夕阳", "傍晚", "烛光", "黄昏")
_COLD_WORDS = ("冷", "清冷", "月", "雪")


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


def _extract_light_dir(text: str) -> dict | None:
    for words, d in _LIGHT_DIR_WORDS.items():
        if any(w in text for w in words):
            return dict(d)
    return None


def _extract_color_temp(text: str) -> float | None:
    if _has(text, _WARM_WORDS):
        return 3600.0
    if _has(text, _COLD_WORDS):
        return 7200.0
    return None


def _extract_intensity(text: str) -> float | None:
    if any(w in text for w in ("亮一点", "亮一些", "更亮")):
        return 1.3
    if any(w in text for w in ("暗一点", "暗一些", "更暗")):
        return 0.7
    return None


def _extract_shadow_strength(text: str) -> float | None:
    if any(w in text for w in ("轻", "淡", "少")):
        return 0.3
    if any(w in text for w in ("重", "深", "浓")):
        return 0.9
    return None


def _frozen(req: PlanningRequest, role: str, key: str) -> str | None:
    """取会话内某工具最近一次产物的指定输出；无则 None。"""
    info = req.available_roles.get(role)
    if not info:
        return None
    outputs = info.get("outputs") or {}
    if key in outputs:
        return outputs[key]
    arts = info.get("artifacts") or []
    return arts[0] if arts else None


class RulePlanner(Planner):
    name = "rule"

    async def plan(self, req: PlanningRequest) -> PlanningResult:
        text = req.text.strip()
        if not text:
            raise AgentError(E_INVALID_INPUT, "指令为空")

        # ---- 条件回滚意图（Demo 04 / 第 5 轮）----
        if _has(text, _ROLLBACK_WORDS):
            return self._plan_rollback(req, text)

        has_asset = bool(req.asset_uris)
        has_prior = any(r in req.available_roles for r in ("harmonize", "shadow_generate", "relight"))
        wants_matting_only = _has(text, _MATTING_WORDS) and not _has(text, _BG_WORDS)
        wants_bg = _has(text, _BG_WORDS)
        wants_light = _has(text, _LIGHT_WORDS)
        wants_shadow = _has(text, _SHADOW_WORDS)
        wants_enhance = _has(text, _ENHANCE_WORDS) or req.quality == Quality.fine

        if wants_matting_only and not has_prior:
            dag = self._matting_only(req, text)
        elif wants_bg or not has_prior:
            if not has_asset:
                raise AgentError(E_INVALID_INPUT, "没有可用图片：请先上传，再下指令")
            dag = self._full_chain(req, text, with_enhance=wants_enhance)
        elif wants_shadow:
            dag = self._shadow_edit(req, text)
        elif wants_light:
            dag = self._light_edit(req, text)
        elif wants_enhance:
            dag = self._enhance_edit(req, text)
        else:
            # 未识别的后续修改：默认按和谐化 + 导出的小步修正处理
            dag = self._harmonize_edit(req, text)
        return PlanningResult(dag=dag, planner=self.name)

    # ---- 模板 ----

    def _matting_only(self, req: PlanningRequest, text: str) -> PlanDAG:
        n1 = DAGNode(id="n1", tool="matting", role="matting", label=TOOL_LABELS["matting"],
                     inputs={"image": req.asset_uris[0]}, options={},
                     quality=req.quality)
        if req.spatial and "click" in req.spatial:
            n1.inputs["point"] = req.spatial["click"]
            n1.inputs["mode"] = "trimap"
        if req.spatial and "box" in req.spatial:
            n1.inputs["box"] = req.spatial["box"]
            n1.inputs["mode"] = "trimap"
        n2 = DAGNode(id="n2", tool="export", role="export", label=TOOL_LABELS["export"],
                     inputs={"image": "@n1.rgba_png", "format": "png"},
                     depends_on=["n1"], quality=req.quality)
        return self._assemble(req, [n1, n2], text)

    def _full_chain(self, req: PlanningRequest, text: str, with_enhance: bool) -> PlanDAG:
        n1 = DAGNode(id="n1", tool="matting", role="matting", label=TOOL_LABELS["matting"],
                     inputs={"image": req.asset_uris[0]}, quality=req.quality)
        if req.spatial and "click" in req.spatial:
            n1.inputs["point"] = req.spatial["click"]
            n1.inputs["mode"] = "trimap"
        n2 = DAGNode(id="n2", tool="background_generate", role="background_generate",
                     label=TOOL_LABELS["background_generate"],
                     inputs={"prompt": text}, quality=req.quality)
        n3 = DAGNode(id="n3", tool="lighting_estimate", role="lighting_estimate",
                     label=TOOL_LABELS["lighting_estimate"],
                     inputs={"bg_png": "@n2.bg_png"}, depends_on=["n2"], quality=req.quality)
        n4 = DAGNode(id="n4", tool="relight", role="relight", label=TOOL_LABELS["relight"],
                     inputs={"rgba_png": "@n1.rgba_png", "bg_hint": "@n2.bg_png",
                             "light_dir": "@n3.light_dir", "color_temp": "@n3.color_temp",
                             "intensity": "@n3.intensity"},
                     depends_on=["n1", "n2", "n3"], quality=req.quality)
        n5 = DAGNode(id="n5", tool="shadow_generate", role="shadow_generate",
                     label=TOOL_LABELS["shadow_generate"],
                     inputs={"foreground_png": "@n4.relit_png", "background_png": "@n2.bg_png",
                             "mask_png": "@n1.mask_png", "light_dir": "@n3.light_dir"},
                     depends_on=["n1", "n2", "n3", "n4"], quality=req.quality)
        n6 = DAGNode(id="n6", tool="harmonize", role="harmonize", label=TOOL_LABELS["harmonize"],
                     inputs={"composite_png": "@n5.composited_png", "mask_png": "@n1.mask_png"},
                     depends_on=["n1", "n5"], quality=req.quality)
        nodes = [n1, n2, n3, n4, n5, n6]
        last_id, last_out = "n6", "harmonized_png"
        if with_enhance:
            n7 = DAGNode(id="n7", tool="enhance", role="enhance", label=TOOL_LABELS["enhance"],
                         inputs={"image": f"@n6.{last_out}"}, depends_on=["n6"], quality=req.quality)
            nodes.append(n7)
            last_id, last_out = "n7", "enhanced_png"
        n8 = DAGNode(id=f"n{len(nodes) + 1}", tool="export", role="export", label=TOOL_LABELS["export"],
                     inputs={"image": f"@{last_id}.{last_out}", "format": "png",
                             "metadata": {"instruction": text}},
                     depends_on=[last_id], quality=req.quality)
        nodes.append(n8)
        return self._assemble(req, nodes, text)

    def _light_edit(self, req: PlanningRequest, text: str) -> PlanDAG:
        ld = _extract_light_dir(text)
        ct = _extract_color_temp(text)
        it = _extract_intensity(text)
        nodes: list[DAGNode] = []
        idx = 1
        if ld is None:
            bg_uri = _frozen(req, "background_generate", "bg_png")
            if not bg_uri:
                raise AgentError(E_INVALID_INPUT, "会话中没有背景产物，无法估计光照；请先完成一次合成")
            nl = DAGNode(id=f"n{idx}", tool="lighting_estimate", role="lighting_estimate",
                         label=TOOL_LABELS["lighting_estimate"], inputs={"bg_png": bg_uri}, quality=req.quality)
            idx += 1
            nodes.append(nl)
            light_ref = f"@{nl.id}.light_dir"
            ct_ref = f"@{nl.id}.color_temp"
            it_ref = f"@{nl.id}.intensity"
        else:
            light_ref = ld
            ct_ref = ct if ct is not None else 5500.0
            it_ref = it if it is not None else 1.0
        fg_uri = _frozen(req, "matting", "rgba_png")
        mask_uri = _frozen(req, "matting", "mask_png")
        bg_uri = _frozen(req, "background_generate", "bg_png")
        if not (fg_uri and mask_uri and bg_uri):
            raise AgentError(E_INVALID_INPUT, "会话缺少抠取/背景产物，请先完成一次完整合成")
        estimate_id = nodes[-1].id if ld is None else None  # 光照估计节点 id（可能为空）
        nr = DAGNode(id=f"n{idx}", tool="relight", role="relight", label=TOOL_LABELS["relight"],
                     inputs={"rgba_png": fg_uri, "light_dir": light_ref,
                             "color_temp": ct_ref, "intensity": it_ref, "bg_hint": bg_uri},
                     depends_on=[estimate_id] if estimate_id else [],
                     quality=req.quality)
        # 用户显式表达暖/冷/亮/暗时，覆盖估计值（方向未提则仍走估计）
        if ld is None:
            if ct is not None:
                nr.inputs["color_temp"] = ct
            if it is not None:
                nr.inputs["intensity"] = it
        idx += 1
        nodes.append(nr)
        shadow_deps = [nr.id] + ([estimate_id] if estimate_id else [])  # light_dir 来自估计节点
        ns = DAGNode(id=f"n{idx}", tool="shadow_generate", role="shadow_generate",
                     label=TOOL_LABELS["shadow_generate"],
                     inputs={"foreground_png": f"@{nr.id}.relit_png", "background_png": bg_uri,
                             "mask_png": mask_uri, "light_dir": light_ref},
                     depends_on=shadow_deps,
                     quality=req.quality)
        idx += 1
        nodes.append(ns)
        nh = DAGNode(id=f"n{idx}", tool="harmonize", role="harmonize", label=TOOL_LABELS["harmonize"],
                     inputs={"composite_png": f"@{ns.id}.composited_png", "mask_png": mask_uri},
                     depends_on=[ns.id], quality=req.quality)
        idx += 1
        nodes.append(nh)
        ne = DAGNode(id=f"n{idx}", tool="export", role="export", label=TOOL_LABELS["export"],
                     inputs={"image": f"@{nh.id}.harmonized_png", "format": "png",
                             "metadata": {"instruction": text}},
                     depends_on=[nh.id], quality=req.quality)
        nodes.append(ne)
        return self._assemble(req, nodes, text)

    def _shadow_edit(self, req: PlanningRequest, text: str) -> PlanDAG:
        fg_uri = _frozen(req, "relight", "relit_png") or _frozen(req, "matting", "rgba_png")
        bg_uri = _frozen(req, "background_generate", "bg_png")
        mask_uri = _frozen(req, "matting", "mask_png")
        ld = _extract_light_dir(text) or {"azimuth": 0.0, "polar": 35.0}
        if not (fg_uri and bg_uri and mask_uri):
            raise AgentError(E_INVALID_INPUT, "会话缺少合成所需产物，请先完成一次完整合成")
        n1 = DAGNode(id="n1", tool="shadow_generate", role="shadow_generate",
                     label=TOOL_LABELS["shadow_generate"],
                     inputs={"foreground_png": fg_uri, "background_png": bg_uri,
                             "mask_png": mask_uri, "light_dir": ld,
                             "shadow_strength": _extract_shadow_strength(text)},
                     quality=req.quality)
        n2 = DAGNode(id="n2", tool="harmonize", role="harmonize", label=TOOL_LABELS["harmonize"],
                     inputs={"composite_png": "@n1.composited_png", "mask_png": mask_uri},
                     depends_on=["n1"], quality=req.quality)
        n3 = DAGNode(id="n3", tool="export", role="export", label=TOOL_LABELS["export"],
                     inputs={"image": "@n2.harmonized_png", "format": "png",
                             "metadata": {"instruction": text}},
                     depends_on=["n2"], quality=req.quality)
        return self._assemble(req, [n1, n2, n3], text)

    def _harmonize_edit(self, req: PlanningRequest, text: str) -> PlanDAG:
        comp_uri = _frozen(req, "shadow_generate", "composited_png") or _frozen(req, "harmonize", "harmonized_png")
        mask_uri = _frozen(req, "matting", "mask_png")
        if not (comp_uri and mask_uri):
            raise AgentError(E_INVALID_INPUT, "会话缺少合成产物，请先完成一次完整合成")
        n1 = DAGNode(id="n1", tool="harmonize", role="harmonize", label=TOOL_LABELS["harmonize"],
                     inputs={"composite_png": comp_uri, "mask_png": mask_uri}, quality=req.quality)
        n2 = DAGNode(id="n2", tool="export", role="export", label=TOOL_LABELS["export"],
                     inputs={"image": "@n1.harmonized_png", "format": "png",
                             "metadata": {"instruction": text}},
                     depends_on=["n1"], quality=req.quality)
        return self._assemble(req, [n1, n2], text)

    def _enhance_edit(self, req: PlanningRequest, text: str) -> PlanDAG:
        src = (_frozen(req, "harmonize", "harmonized_png")
               or _frozen(req, "shadow_generate", "composited_png"))
        depth = _frozen(req, "background_generate", "depth_png")
        if not src:
            raise AgentError(E_INVALID_INPUT, "会话缺少合成产物，请先完成一次完整合成")
        inputs: dict = {"image": src, "strength": 0.6}
        if depth:
            inputs["depth_map"] = depth
        n1 = DAGNode(id="n1", tool="enhance", role="enhance", label=TOOL_LABELS["enhance"],
                     inputs=inputs, quality=req.quality)
        n2 = DAGNode(id="n2", tool="export", role="export", label=TOOL_LABELS["export"],
                     inputs={"image": "@n1.enhanced_png", "format": "png",
                             "metadata": {"instruction": text}},
                     depends_on=["n1"], quality=req.quality)
        return self._assemble(req, [n1, n2], text)

    def _plan_rollback(self, req: PlanningRequest, text: str) -> PlanningResult:
        """识别回滚目标与保留项；具体 DAG 由 rollback 模块构造。"""
        role_words = [
            ("background_generate", ("背景", "场景")),
            ("lighting_estimate", ("光照", "光线", "光源")),
            ("relight", ("光", "打光", "光影")),
            ("shadow_generate", ("阴影", "影子")),
            ("matting", ("抠图", "抠取")),
            ("harmonize", ("和谐", "融合")),
        ]
        target_role = None
        for role, words in role_words:
            if _has(text, words):
                target_role = role
                break
        if target_role is None:
            target_role = "background_generate"  # 默认语境：换回背景
        versions = (req.available_roles.get(target_role) or {}).get("versions") or []
        if _has(text, _VERSION_FIRST):
            version = versions[0] if versions else 1
        elif _has(text, _VERSION_PREV) and len(versions) >= 2:
            version = versions[-2]
        else:
            version = versions[-1] if versions else 1
        preserve: list[str] = []
        if _PRESERVE_HINT in text:
            for role, words in role_words:
                if role != target_role and _has(text, words):
                    preserve.append(role)
        return PlanningResult(planner=self.name, kind="rollback",
                              rollback={"role": target_role, "version": int(version), "preserve": preserve})

    def _assemble(self, req: PlanningRequest, nodes: list[DAGNode], text: str) -> PlanDAG:
        # 节点版本号承接会话历史（第 2 轮的 background_generate 是 v2 而不是覆盖 v1）
        for n in nodes:
            versions = (req.available_roles.get(n.role) or {}).get("versions") or []
            n.version = (max(versions) + 1) if versions else 1
        return PlanDAG(nodes=nodes, planner=self.name, instruction=text, quality=req.quality)
