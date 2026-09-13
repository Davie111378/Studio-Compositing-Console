# -*- coding: utf-8 -*-
"""
planner.py — Planner: 用户指令 → 合法执行计划 DAG
- agnes LLM 输出 JSON → 提取 → jsonschema 强校验 → 失败带错误重试 ≤2
- 全部失败 → 关键词模板兜底 (规范 L2 降级)
"""
from __future__ import annotations
import json, re, time
from pathlib import Path

import dag as dagmod
from prompt import build_messages


class AgnesLLM:
    """轻包装: 与 agent/agnes_client 解耦, 支持注入测试桩。

    provider: "agnes" | "qwen" | None(走 LLM_PROVIDER 环境变量)
    """

    def __init__(self, provider: str | None = None):
        sys_path = str(Path(__file__).resolve().parent.parent / "ai-agent")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        from agnes_client import AgnesClient
        self.provider = provider
        self.cli = AgnesClient(provider=provider)

    def complete(self, messages: list[dict], max_tokens: int = 1400) -> str:
        d = self.cli.chat(messages, tools=None, max_tokens=max_tokens, temperature=0.1)
        return d["choices"][0]["message"].get("content") or ""


def extract_json(text: str) -> dict | None:
    """从 LLM 文本提取 JSON 对象 (容忍 ```json 围栏/前后杂文)。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


# ---------------- 关键词模板兜底 (L2 降级) ----------------
BG_KEYS = [("新闻", "新闻LED"), ("播报", "新闻LED"), ("LED", "新闻LED"),
           ("访谈", "访谈"), ("沙发", "访谈"), ("书架", "访谈"),
           ("全景", "全景"), ("城市", "全景"), ("黄昏", "全景"), ("窗", "全景"),
           ("天气", "天气"), ("综艺", "综艺"), ("舞台", "综艺"),
           ("播客", "播客"), ("录音", "播客"), ("霓虹", "播客")]


def _bg_semantic(text: str) -> str | None:
    for k, v in BG_KEYS:
        if k in text:
            return v
    return None


def fallback_plan(instruction: str, cur_image: str | None = None,
                  bg_image: str | None = None,
                  bg_from_upload: bool = False) -> dict:
    t = instruction
    # ---- abstain: 域外请求检测 ----
    # 非图像编辑/合成/特效类指令 → 拒绝, 不调用任何工具
    NON_IMAGE_KEYWORDS = ("写诗", "写一首", "讲笑话", "聊天", "问天气", "翻译",
                          "写代码", "写文章", "算数", "计算", "讲故事",
                          "唱歌", "作曲", "写歌词", "百科", "知识问答")
    # 物体级/像素级编辑超出 8 工具能力
    BEYOND_ABILITY = ("换颜色", "变色", "改成红色", "改成蓝色", "换成红色", "换成蓝色",
                       "把眼睛", "把鼻子", "把嘴巴", "瘦脸", "丰胸",
                       "改文字", "把文字", "把图里的字", "修改文字")
    if any(k in t for k in NON_IMAGE_KEYWORDS) or any(k in t for k in BEYOND_ABILITY):
        return {"intent": "abstain", "nodes": [], "outputs": []}
    nodes: list[dict] = []
    # 删主体(保留背景) 优先判断: "扣去/去掉/移除/删除人物" —— 只 T01 单节点, keep=background
    remove_subject = any(k in t for k in ("扣去", "去掉", "移除", "删除", "P掉", "p掉", "抹掉", "擦除")) \
        and any(k in t for k in ("人物", "人", "主播", "主持人", "人物主体", "主体", "前景"))
    if remove_subject:
        img = cur_image or "$cur"
        nodes.append({"id": "n1", "tool": "T01_matting",
                      "params": {"image": img, "keep": "background"},
                      "depends_on": []})
        return {"intent": f"[fallback] 删除主体保留背景: {instruction[:40]}",
                "nodes": nodes, "outputs": ["n1"]}
    # 色度键直合成: "绿幕合成/键控/抠干净/绿边" 或 视频换背景 —— T02 mode=green_key 单节点
    green_key_intent = any(k in t for k in ("绿幕合成", "键控", "色度键", "绿幕键", "抠干净",
                                            "绿边", "chroma", "直接合成到背景")) \
        or (("视频" in t) and any(k in t for k in ("换背景", "换个", "背景合成", "合成背景",
                                                   "替换背景", "换到", "合成到"))) \
        or (("绿幕" in t) and ("背景" in t) and any(k in t for k in (
            "替换", "换上", "换成", "换掉", "用")))
    if green_key_intent:
        fg = cur_image or "$cur"
        params = {"mode": "green_key", "app_fg": fg}
        if bg_image:
            # 用户上传了背景图 → 必须用 file, 禁用 semantic (否则会去素材库/T2I 另造背景)
            params["file"] = bg_image
        else:
            sem = _bg_semantic(t)
            if sem:
                params["semantic"] = sem
        if "视频" in t:
            params["media_type"] = "video"
        # 前置色卡滤镜名(如 "5小纸条")→ media_filter
        import re as _re
        _fm = _re.search(r"([0-9]{1,2}[^0-9\s,，。]*?)(?:滤镜|色调|风格)", t)
        if _fm:
            params["media_filter"] = _fm.group(1) + ".png"
        nodes.append({"id": "n1", "tool": "T02_background_generate",
                      "params": params, "depends_on": []})
        return {"intent": f"[fallback] 绿幕色度键合成: {instruction[:40]}",
                "nodes": nodes, "outputs": ["n1"]}
    want_bg = any(k in t for k in ("换背景", "放到", "放进", "合成", "背景", "搬到", "替换背景"))
    green_studio = ("绿幕替换" in t) or (("保留" in t) and ("绿幕" in t))
    want_fx = any(k in t for k in ("滤镜", "特效", "贴纸", "水印", "虚化", "景深", "风格", "锐化", "颗粒"))
    need_matting = want_bg and not green_studio
    if need_matting:
        img = cur_image or "$cur"
        nodes.append({"id": "n1", "tool": "T01_matting",
                      "params": {"image": img}, "depends_on": []})
    bg_sem = _bg_semantic(t)
    bg_file = bg_image if (bg_image and (want_bg or "替换" in t or "背景" in t)) else None
    if bg_file:
        # 用户上传了背景图: T02 file 直通 (最优先)
        if need_matting:
            nodes.append({"id": "n2", "tool": "T02_background_generate",
                          "params": {"file": bg_file}, "depends_on": []})
            nodes.append({"id": "n3", "tool": "T03_lighting_estimate",
                          "params": {"bg_path": "$n2.bg_path"}, "depends_on": ["n2"]})
            nodes.append({"id": "n4", "tool": "T04_relight",
                          "params": {"fg_path": "$n1.fg_path", "bg_path": "$n2.bg_path",
                                     "light_dir": "$n3.light_dir", "color_temp": "$n3.color_temp"},
                          "depends_on": ["n1", "n2", "n3"]})
            nodes.append({"id": "n5", "tool": "T05_shadow_generate",
                          "params": {"alpha_path": "$n1.alpha_path", "bg_path": "$n2.bg_path",
                                     "light_dir": "$n3.light_dir"}, "depends_on": ["n1", "n2", "n3"]})
            nodes.append({"id": "n6", "tool": "T06_harmonize",
                          "params": {"fg_path": "$n4.relit_fg_path", "alpha_path": "$n1.alpha_path",
                                     "bg_path": "$n5.shadow_bg_path"}, "depends_on": ["n4", "n5"]})
            last = "n6"
        else:
            nodes.append({"id": "n2", "tool": "T02_background_generate",
                          "params": {"file": bg_file}, "depends_on": []})
            last = "n2"
        return {"intent": f"[fallback] {instruction[:40]}", "nodes": nodes,
                "outputs": [last]}
    if want_bg and bg_sem:
        nodes.append({"id": "n2", "tool": "T02_background_generate",
                      "params": {"semantic": bg_sem}, "depends_on": []})
    if need_matting and bg_sem:
        nodes.append({"id": "n3", "tool": "T03_lighting_estimate",
                      "params": {"bg_path": "$n2.bg_path"}, "depends_on": ["n2"]})
        nodes.append({"id": "n4", "tool": "T04_relight",
                      "params": {"fg_path": "$n1.fg_path", "bg_path": "$n2.bg_path",
                                 "light_dir": "$n3.light_dir", "color_temp": "$n3.color_temp"},
                      "depends_on": ["n1", "n2", "n3"]})
        nodes.append({"id": "n5", "tool": "T05_shadow_generate",
                      "params": {"alpha_path": "$n1.alpha_path", "bg_path": "$n2.bg_path",
                                 "light_dir": "$n3.light_dir"}, "depends_on": ["n1", "n2", "n3"]})
        nodes.append({"id": "n6", "tool": "T06_harmonize",
                      "params": {"fg_path": "$n4.relit_fg_path", "alpha_path": "$n1.alpha_path",
                                 "bg_path": "$n5.shadow_bg_path"}, "depends_on": ["n4", "n5"]})
        last = "n6"
    elif green_studio and bg_sem:
        src = cur_image or "$cur"
        nodes = [{"id": "n1", "tool": "T06_harmonize",
                  "params": {"fg_path": src, "alpha_path": src, "bg_path": "$n2.bg_path",
                             "mode": "greenscreen"},
                  "depends_on": ["n2"]}]
        nodes.insert(0, {"id": "n2", "tool": "T02_background_generate",
                         "params": {"semantic": bg_sem}, "depends_on": []})
        # 修正依赖 id 顺序 (n2 在前)
        nodes = [{"id": "n2", "tool": "T02_background_generate",
                  "params": {"semantic": bg_sem}, "depends_on": []},
                 {"id": "n1", "tool": "T06_harmonize",
                  "params": {"fg_path": src, "alpha_path": src, "bg_path": "$n2.bg_path",
                             "mode": "greenscreen"}, "depends_on": ["n2"]}]
        last = "n1"
    elif want_fx:
        mode = ("sharpen" if "锐化" in t else "grain" if "颗粒" in t else "depth_blur")
        nodes.append({"id": "n1", "tool": "T07_enhance",
                      "params": {"image_path": cur_image or "$cur", "mode": mode},
                      "depends_on": []})
        last = "n1"
    elif want_bg and not bg_sem:
        # 要换背景但没指明 → 默认访谈
        nodes = [{"id": "n2", "tool": "T02_background_generate",
                  "params": {"semantic": "访谈"}, "depends_on": []},
                 {"id": "n1", "tool": "T01_matting",
                  "params": {"image": cur_image or "$cur"}, "depends_on": []},
                 {"id": "n3", "tool": "T03_lighting_estimate",
                  "params": {"bg_path": "$n2.bg_path"}, "depends_on": ["n2"]},
                 {"id": "n4", "tool": "T04_relight",
                  "params": {"fg_path": "$n1.fg_path", "bg_path": "$n2.bg_path",
                             "light_dir": "$n3.light_dir", "color_temp": "$n3.color_temp"},
                  "depends_on": ["n1", "n2", "n3"]},
                 {"id": "n5", "tool": "T05_shadow_generate",
                  "params": {"alpha_path": "$n1.alpha_path", "bg_path": "$n2.bg_path",
                             "light_dir": "$n3.light_dir"}, "depends_on": ["n1", "n2", "n3"]},
                 {"id": "n6", "tool": "T06_harmonize",
                  "params": {"fg_path": "$n4.relit_fg_path", "alpha_path": "$n1.alpha_path",
                             "bg_path": "$n5.shadow_bg_path"}, "depends_on": ["n4", "n5"]}]
        last = "n6"
    else:
        # 兜底: 导出当前图
        nodes.append({"id": "n1", "tool": "T08_export",
                      "params": {"final_path": cur_image or "$cur", "quality": "normal"},
                      "depends_on": []})
        last = "n1"
    # 指令明确要求导出且链尾不是 T08 时补上 (否则丢导出步骤)
    want_export = any(k in t for k in ("导出", "保存", "输出成", "存成", "导出成"))
    if want_export and nodes and nodes[-1]["tool"] != "T08_export":
        prev = nodes[-1]["id"]
        out_key = {"T06_harmonize": "composite_path", "T07_enhance": "enhanced_path"}.get(
            nodes[-1]["tool"], "exported_path")
        nodes.append({"id": f"n{len(nodes)+1}", "tool": "T08_export",
                      "params": {"final_path": f"${prev}.{out_key}", "quality": "normal"},
                      "depends_on": [prev]})
        last = nodes[-1]["id"]
    return {"intent": f"[fallback] {instruction[:40]}", "nodes": nodes, "outputs": [last]}


# ---------------- 主入口 ----------------
def plan(instruction: str, cur_image: str | None = None,
         llm: AgnesLLM | None = None, fewshot_variant: str = "base",
         max_retry: int = 2, verbose: bool = False,
         bg_image: str | None = None,
         provider: str | None = None,
         bg_from_upload: bool = False) -> tuple[dict, str]:
    """返回 (dag, source)。source ∈ {llm, llm_retry, fallback}

    provider: "agnes" | "qwen" | None(走 LLM_PROVIDER)。仅当 llm 未注入时生效。
    bg_from_upload: bg_image 来自用户上传(而非语义检索) → 强制 T02 用 file, 禁用 semantic。
    """
    llm = llm or AgnesLLM(provider=provider)
    msgs = build_messages(instruction, cur_image, fewshot_variant,
                          bg_image=bg_image, bg_from_upload=bg_from_upload)
    last_err = ""
    for attempt in range(max_retry + 1):
        try:
            text = llm.complete(msgs)
        except Exception as e:
            last_err = f"LLM 调用失败: {e}"
            if verbose:
                print(f"[planner] attempt{attempt} {last_err}")
            time.sleep(1.0)
            continue
        dag = extract_json(text)
        if dag is None:
            last_err = "输出不是合法 JSON"
        elif dag.get("intent") == "abstain":
            # LLM 正确识别域外请求 → 直接返回 abstain, 跳过 schema 校验 (schema 要求 minItems=1)
            return dag, ("llm" if attempt == 0 else "llm_retry")
        elif not dag.get("nodes") and not dag.get("outputs"):
            # 2026-09-12: 空 dag 但 intent≠abstain —— 不能当拒识早退 (下游 run.py 取
            # run['total_ms'] 会对 engine 的 invalid 早退结构 KeyError→500, 回归 D5 实测),
            # 记错走重试/兜底
            last_err = "空计划 (无节点且非拒识)"
        else:
            dag = _enforce_upload_bg(dag, bg_image, bg_from_upload, cur_image=cur_image)
            dag = _prune_redundant_export(dag, instruction)
            dag = _toposort_nodes(dag)
            dag = _fix_dual_image_shortcut(dag, instruction, cur_image,
                                           bg_image, bg_from_upload)
            errs = dagmod.validate(dag)
            if not errs:
                return dag, ("llm" if attempt == 0 else "llm_retry")
            last_err = "; ".join(errs)[:300]
        if verbose:
            print(f"[planner] attempt{attempt} 失败: {last_err[:120]}")
        # 带错误反馈重试
        msgs = msgs + [{"role": "assistant", "content": text[:800]},
                       {"role": "user", "content":
                        f"上面的 DAG 有错误: {last_err}。请修正后重新只输出 JSON。"}]
    return (fallback_plan(instruction, cur_image, bg_image=bg_image,
                          bg_from_upload=bg_from_upload),
            "fallback")


def _fix_dual_image_shortcut(dag: dict, instruction: str, cur_image: str | None,
                             bg_image: str | None, bg_from_upload: bool) -> dict:
    """后校验: 双图换背景被 LLM 抄近路成"单节点"时, 确定性重建正规链。

    web 层 bg_from_upload=True 已判定"换背景"意图; 此时任何单节点计划都出不了
    "人物+新背景"成片 (T02 只供背景/键控直合成, 无抠图无 T06 合成)。
    实测形态 (I2 同型复发): 双图"图一背景图二换背景"被规划成 T02 mode=green_key 单节点。
    豁免 (合法单节点管线):
      - T02 green_key + media_type=video (视频键控 key_video 一步出片, T06 不支持视频)
      - T06 greenscreen 单节点 (绿幕范围合成自带键控+合成, 完整管线)
    其余 (含绿幕前景的 T02 green_key 图片直合成 — 规则15已禁止, 且回归 I1/I2 断言
    全链) → 委托 fallback_plan 重建 T01→T06 全链 (与重试耗尽后兜底同源, 不烧 LLM 重试)。
    """
    if not bg_from_upload or not bg_image or not cur_image:
        return dag
    nodes = dag.get("nodes") if isinstance(dag, dict) else None
    if not nodes or len(nodes) != 1:
        return dag
    n = nodes[0]
    tool = n.get("tool")
    p = n.get("params") or {}
    if tool == "T02_background_generate" and p.get("mode") == "green_key" \
            and p.get("media_type") == "video":
        return dag                       # 视频键控: 单节点即正规链
    if tool == "T06_harmonize" and p.get("mode") == "greenscreen":
        return dag                       # 绿幕范围合成单节点: 键控+合成完整
    return fallback_plan(instruction, cur_image, bg_image=bg_image,
                         bg_from_upload=bg_from_upload)


def _prune_redundant_export(dag: dict, instruction: str) -> dict:
    """后校验: 用户没要求导出时, 删除多余的尾部 T08_export (规则7)。

    LLM 常给"再加X"这类多轮指令自动补一个 T08 导出 (把增强结果存一份)。
    这会让 outputs 指向导出节点而非增强节点, 且违反"没要求导出就不加 T08"。
    仅当: 用户指令**没有**导出语义 且 T08 唯一依赖是链尾 (T07/T06) 时 → 剪掉 T08,
    并把 outputs 里对 T08 的引用改回其上游。
    """
    if not dag or not isinstance(dag.get("nodes"), list):
        return dag
    _EXPORT_WORDS = ("导出", "保存", "存下来", "存成", "下载", "另存", "输出文件",
                     "export", "save")
    if any(w in (instruction or "") for w in _EXPORT_WORDS):
        return dag                       # 用户确实要导出 → 不动
    nodes = dag["nodes"]
    t08 = [n for n in nodes if n.get("tool") == "T08_export"]
    if not t08:
        return dag
    # 2026-09-12: LLM 也会把 T08 直接挂在 T06 后 (无 T07 的全链), 同属"没要求导出却补导出"
    _tail_ids = {n["id"] for n in nodes if n.get("tool") in ("T07_enhance", "T06_harmonize")}
    drop_ids: set[str] = set()
    for n in t08:
        deps = n.get("depends_on") or []
        # 只剪"纯挂在链尾 (T07/T06) 后面"的导出节点
        if len(deps) == 1 and deps[0] in _tail_ids:
            drop_ids.add(n["id"])
    if not drop_ids:
        return dag
    # T08 被剪 → 其他节点对它的依赖重接到它的上游 T07
    _up_of = {n["id"]: (n.get("depends_on") or [None])[0] for n in t08
              if n["id"] in drop_ids}
    dag["nodes"] = [n for n in nodes if n["id"] not in drop_ids]
    for n in dag["nodes"]:
        deps = n.get("depends_on")
        if isinstance(deps, list):
            n["depends_on"] = [_up_of.get(d, d) for d in deps]
            n["depends_on"] = [d for d in n["depends_on"] if d]
    outs = dag.get("outputs")
    if isinstance(outs, list):
        new_outs = []
        for o in outs:
            o2 = _up_of.get(o, o)
            if o2 not in new_outs:
                new_outs.append(o2)
        dag["outputs"] = new_outs
    return dag


def _toposort_nodes(dag: dict) -> dict:
    """后校验: 节点确定性排序 (依赖优先 + 同层按工具号)。

    LLM 偶发输出 "T02 在 T01 前" 的顺序——executor 按依赖调度不影响正确性,
    但 steps 展示与回归用例的 tools 顺序断言都会乱。Kahn 拓扑 + 就绪层内
    按 (工具号, 原位置) 排序: 依赖关系是硬约束, 同层顺序是确定性约定。
    """
    if not dag or not isinstance(dag.get("nodes"), list):
        return dag
    nodes = dag["nodes"]
    ids = [n.get("id") for n in nodes]
    pos = {nid: i for i, nid in enumerate(ids)}
    by_id = {n.get("id"): n for n in nodes}
    deps_map = {n.get("id"): [d for d in (n.get("depends_on") or []) if d in pos]
                for n in nodes}

    def _rank(nid):
        m = re.match(r"T(\d+)", (by_id[nid].get("tool") or ""))
        return (int(m.group(1)) if m else 99, pos[nid])

    ordered: list[str] = []
    done: set[str] = set()
    pending = list(ids)
    while pending:
        ready = [nid for nid in pending if all(d in done for d in deps_map[nid])]
        if not ready:                    # 环依赖 (validate 会拦) → 放弃重排
            return dag
        ready.sort(key=_rank)
        for nid in ready:
            ordered.append(nid)
            done.add(nid)
        pending = [nid for nid in pending if nid not in done]
    if ordered != ids:
        dag["nodes"] = [by_id[nid] for nid in ordered]
    return dag


def _enforce_upload_bg(dag: dict, bg_image: str | None,
                       bg_from_upload: bool, cur_image: str | None = None) -> dict:
    """后校验: 用户上传了背景图时, 强制 T02 用 file=bg_image 并删除 semantic。

    防止 LLM 幻觉出 semantic → 走素材库/文生图 另造一张背景 (用户会看到"训练过的图")。
    2026-09-12: green_key 模式同时纠正 app_fg —— 双图绿幕替换时 LLM 偶发把 app_fg/file
    填反或两槽同图 (前端实测报"背景不存在或与前景同源")。web 层已确定性地解析出
    fg=当前输入图 / bg=上传背景, 此处按同一事实强制对齐; 仅在 app_fg 缺失或与背景
    同路径时纠正, 不碰视频等合法自定义 app_fg。
    """
    if not bg_image or not bg_from_upload:
        return dag

    def _same(a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        import os as _os
        return _os.path.normcase(_os.path.normpath(a)) == _os.path.normcase(_os.path.normpath(b))

    for n in dag.get("nodes", []):
        if n.get("tool") != "T02_background_generate":
            continue
        p = n.setdefault("params", {})
        p.pop("semantic", None)          # 禁用语义分支
        p.pop("prompt", None)            # 禁用文生图分支
        if p.get("mode") == "green_key" or p.get("app_fg"):
            p["mode"] = "green_key"
            p["file"] = bg_image         # green_key: 背景走 file
            if cur_image and (not p.get("app_fg") or _same(p.get("app_fg"), bg_image)):
                p["app_fg"] = cur_image  # 前景=当前输入图 (防填反/同源)
        elif not p.get("bg") and not p.get("file"):
            p["file"] = bg_image         # 普通合成: file 直通
    return dag

