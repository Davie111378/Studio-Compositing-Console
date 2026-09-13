# -*- coding: utf-8 -*-
"""
critic.py — VLM Critic (冲刺计划 A4 压缩版)
- 5 维结构化解析: Lighting / Shadow / Color / Edge / Overall (0~100)
- 最低维 → 工具映射 (Edge→T01 精修, Lighting→T04 加强度, Shadow→T05, Color→T06)
- re-plan 闭环: 返回补丁参数, 由 executor 重跑该节点+下游 (≤2 次, 调用方控制)
"""
from __future__ import annotations
import re

DIMS = ["Lighting", "Shadow", "Color", "Edge", "Overall"]

PROMPT = """请审视这张演播室合成图, 严格按以下格式输出 5 维评分 (每维 0-100 整数) 与一句话诊断:
Lighting: <分数> - <一句话>
Shadow: <分数> - <一句话>
Color: <分数> - <一句话>
Edge: <分数> - <一句话>
Overall: <分数> - <一句话>
最后给一行建议, 格式: FIX: <可执行修正 (如 "重抠并开精修" / "增强主光强度" / "加深阴影" / "提高和谐化强度")>"""

# 维度 → 修复动作 (engine 侧执行: 找到对应节点, 打补丁重跑)
FIX_ACTIONS = {
    "Edge": {"tool": "T01_matting", "patch": {"quality": "fine"}},
    "Lighting": {"tool": "T04_relight", "patch": {"intensity": 1.0}},
    "Shadow": {"tool": "T05_shadow_generate", "patch": {"opacity": 0.65}},
    "Color": {"tool": "T06_harmonize", "patch": {"strength": 0.9}},
    "Overall": {"tool": "T06_harmonize", "patch": {"strength": 0.9}},
}


def parse(text: str) -> dict:
    """解析 VLM 输出 → {dim: {score, note}}, fix}"""
    scores, notes = {}, {}
    for line in text.splitlines():
        for dim in DIMS:
            m = re.match(rf"\s*{dim}\s*[:：]\s*(\d+)", line, re.I)
            if m:
                scores[dim] = int(m.group(1))
                notes[dim] = line.split("-", 1)[-1].strip()[:120]
    fm = re.search(r"FIX\s*[:：]\s*(.+)", text)
    fix = fm.group(1).strip()[:200] if fm else None
    return {"scores": scores, "notes": notes, "fix": fix}


def lowest_dim(scores: dict) -> str | None:
    scored = {k: v for k, v in scores.items() if k in DIMS}
    if not scored:
        return None
    return min(scored, key=scored.get)


def needs_replan(scores: dict, threshold: int = 80) -> bool:
    ov = scores.get("Overall")
    return bool(ov is not None and ov < threshold)


def replan_patch(scores: dict) -> dict | None:
    """最低维 → {tool, patch} 补丁 (供调用方定位 DAG 节点并重跑)。"""
    dim = lowest_dim(scores)
    if dim is None or dim not in FIX_ACTIONS:
        return None
    return {"dim": dim, **FIX_ACTIONS[dim]}


def review_image(client, image_path: str) -> dict:
    """调用 VLM 评图并解析。client: agnes_client.AgencestClient"""
    d = client.chat_with_image(PROMPT, [image_path], max_tokens=500)
    text = d["choices"][0]["message"].get("content") or ""
    out = parse(text)
    out["raw"] = text
    out["replan"] = replan_patch(out["scores"])
    out["needs_replan"] = needs_replan(out["scores"])
    return out
