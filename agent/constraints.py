# -*- coding: utf-8 -*-
"""
constraints.py — L0 需求形式化层 (《专用Agent系统架构指南》八元组之 C 约束集 + R 效用函数)
- 硬约束 HARD: 安全/合规类, 违反即拒 (工具白名单/节点上限/禁环/延迟预算)
- 软约束 SOFT: 风格/偏好/成本类, 尽量满足 (质量档位/文生图次数)
- 效用函数 U: 多目标加权 (质量/延迟/成本) — 供 Critic 与评估层 (L8) 使用
- 拒识 (Abstention): OOD 输入的快速判定 + 触发条件 (L2 异常检测的前置)
"""
from __future__ import annotations
import re
from pathlib import Path

import dag as dagmod

# ------------------------------------------------ L0 约束集 C
HARD = {
    "allowed_tools": sorted(dagmod.VALID_TOOLS),   # 动作空间 A 封闭集
    "max_nodes": 10,                                # DAG 规模上限
    "critic_max_replan": 2,                         # 反思循环上限
    "forbid_cycles": True,                          # 禁环 (Kahn 检测)
    "budget_ms": {"draft": 10000, "normal": 20000, "fine": 30000},   # ≤10s/≤30s 指标
    "require_final_artifact": True,                 # 必须产出可见成片
}
SOFT = {
    "prefer_quality": "draft",
    "max_t2i_per_run": 1,                           # 文生图成本控制
    "style": "photorealistic",
    "bg_semantic_whitelist": ["访谈", "新闻LED", "全景", "综艺", "播客", "天气"],
}

# ------------------------------------------------ 拒识 (Abstention)
# 图像任务意图词典: 命中任一 → 属于域内
_INTENT_WORDS = [
    "抠图", "抠出", "抠像", "换背景", "背景", "合成", "叠加", "放到", "放进", "搬到",
    "滤镜", "特效", "贴纸", "水印", "美颜", "风格", "黑白", "油画", "漫画", "素描",
    "水彩", "赛博", "霓虹", "暗角", "聚光", "虚化", "景深", "锐化", "颗粒", "调色",
    "光影", "打光", "重打光", "阴影", "影子", "和谐化", "演播室", "绿幕", "主播",
    "人像", "背景图", "导出", "保存", "成片", "增强", "去雾", "模糊",
]
# 明确域外意图
_OOD_WORDS = ["天气查询", "写代码", "编程", "翻译", "写论文", "股票", "导航", "订票", "放音乐"]


def abstain_reason(instruction: str) -> str | None:
    """返回拒识原因; None = 域内指令, 放行。
    触发条件 (L0 边界条件): 指令不含任何图像合成意图词, 或明确命中域外意图。"""
    t = instruction.strip()
    if not t:
        return "空指令"
    for w in _OOD_WORDS:
        if w in t:
            return f"域外意图 ({w}): 本系统仅处理演播室图像合成任务"
    if any(w in t for w in _INTENT_WORDS):
        return None
    # 含图片文件名/路径 → 视为域内 (用户直接丢图)
    if re.search(r"\.(png|jpe?g|webp|bmp|mp4|webm)", t, re.I) or "图" in t or "画" in t:
        return None
    return "未识别到图像合成意图 (抠图/换背景/滤镜/光影等), 已拒识 —— 请描述图像编辑需求"


# ------------------------------------------------ 效用函数 U (多目标加权)
def utility(overall_score: float | None, latency_ms: int, quality: str,
            n_t2i: int = 0) -> dict:
    """U = 0.6*质量 + 0.25*延迟达标 + 0.15*成本达标 → [0,1]
    overall_score: Critic Overall 维 (0-100), 无 Critic 时按档位期望值。"""
    if overall_score is None:
        overall_score = {"draft": 70, "normal": 80, "fine": 88}[quality]
    q = float(overall_score) / 100.0
    budget = HARD["budget_ms"].get(quality, 30000)
    lat = max(0.0, 1.0 - latency_ms / budget)              # 预算内=1, 超时线性衰减
    cost = max(0.0, 1.0 - n_t2i / max(SOFT["max_t2i_per_run"], 1))
    u = 0.6 * q + 0.25 * lat + 0.15 * cost
    return {"U": round(u, 4), "quality_term": round(q, 3),
            "latency_term": round(lat, 3), "cost_term": round(cost, 3),
            "latency_ms": latency_ms, "budget_ms": budget}


# ------------------------------------------------ 计划级约束校验 (L6 前置)
def check_plan(dag: dict, quality: str = "draft") -> list[str]:
    """对 Planner 产出做约束集 C 校验 (与 dag.validate 互补: 这里查预算/规模/软约束)。"""
    errs = []
    if len(dag["nodes"]) > HARD["max_nodes"]:
        errs.append(f"节点数 {len(dag['nodes'])} 超上限 {HARD['max_nodes']}")
    # 延迟预算: 用 Schema 的 latency_budget_ms 估算
    total = 0
    for n in dag["nodes"]:
        sch = dagmod.TOOL_SCHEMAS.get(n["tool"])
        q = n.get("params", {}).get("quality", quality)
        if sch:
            total += sch.get("latency_budget_ms", {}).get(q, 3000)
    budget = HARD["budget_ms"].get(quality, 30000)
    if total > budget:
        errs.append(f"预估延迟 {total}ms 超出 {quality} 档预算 {budget}ms "
                    f"(建议降档或拆分)")
    # 文生图次数
    n_t2i = sum(1 for n in dag["nodes"]
                if n["tool"] == "T02_background_generate"
                and n.get("params", {}).get("quality") == "fine")
    if n_t2i > SOFT["max_t2i_per_run"]:
        errs.append(f"文生图 {n_t2i} 次超软约束上限 {SOFT['max_t2i_per_run']}")
    return errs
