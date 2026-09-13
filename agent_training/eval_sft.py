# -*- coding: utf-8 -*-
"""
eval_sft.py — Stage 4: 本地 SFT Planner 泛化评测
- 评测集: 与训练集不同表述的改写指令 (测泛化, 非背题)
- 指标: Tool Selection / DAG Order / Parameter 准确率 + 拒识准确率
- 对比: SFT 本地模型 vs agnes LLM 基线
运行: python agent_training/eval_sft.py [--engine local|agnes|both]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "ai-agent"))

import planner as planner_mod

FULL = ["T01_matting", "T02_background_generate", "T03_lighting_estimate",
        "T04_relight", "T05_shadow_generate", "T06_harmonize"]
GS = ["T02_background_generate", "T06_harmonize"]

# ---- 泛化评测集 (训练集外的新表述) ----
CASES = [
    # (id, instruction, expect)  expect: {"tools": [...], "bg":?, "fx"?, "sticker"?, "abstain"?}
    ("P1", "麻烦帮我把 fg_01_anchor_male.png 从绿幕里抠出来，放到综艺舞台上", {"tools": FULL, "bg": "综艺"}),
    ("P2", "我想让 fg_03_glasses.png 站在新闻演播室的大屏前", {"tools": FULL, "bg": "新闻LED"}),
    ("P3", "这位主播（fg_04_hair.png）需要出现在播客直播间", {"tools": FULL, "bg": "播客"}),
    ("P4", "帮我合成：fg_02_anchor_female.png + 天气预报演播室", {"tools": FULL, "bg": "天气"}),
    ("P5", "让照片里的虚化程度明显一点", {"tools": ["T07_enhance"], "fx": "depth_blur"}),
    ("P6", "画面整体锐化一些", {"tools": ["T07_enhance"], "fx": "sharpen"}),
    ("P7", "这张演播室照片，绿幕部分换成城市黄昏全景，桌子和设备别动", {"tools": GS, "bg": "全景"}),
    ("P8", "把图里的绿幕抠掉换成综艺大屏，其他东西都留着", {"tools": GS, "bg": "综艺"}),
    ("P9", "帮我查一下明天去上海的高铁票", {"abstain": True}),
    ("P10", "给我讲讲相对论", {"abstain": True}),
    ("P11", "帮我把这个Excel表格排序", {"abstain": True}),
    ("P12", "给我的人像加一个皇冠贴纸", {"tools": ["T07_enhance"], "fx": "sticker", "sticker": "crown"}),
    ("P13", "右上角加个星星", {"tools": ["T07_enhance"], "fx": "sticker", "sticker": "star"}),
    ("P14", "给这张图打上文字水印：演播室专用", {"tools": ["T07_enhance"], "fx": "watermark_text"}),
    # ---- G 域: 删除主体保留背景 (keep=background, 语义与"抠出人物"相反) ----
    ("G1", "帮我把图里的主持人P掉，只留背景", {"tools": ["T01_matting"], "keep": "background"}),
    ("G2", "移除画面中的人物，保留演播室背景", {"tools": ["T01_matting"], "keep": "background"}),
    ("G3", "把这位主播从画面里去掉", {"tools": ["T01_matting"], "keep": "background"}),
    ("G4", "只保留绿幕背景，人物抹掉", {"tools": ["T01_matting"], "keep": "background"}),
    ("G5", "抹掉前景里的两个人", {"tools": ["T01_matting"], "keep": "background"}),
]


def score(dag, expect):
    # 期望拒识的用例 (OOD): 模型也必须拒识
    if expect.get("abstain"):
        hit = bool(isinstance(dag, dict) and dag.get("abstain"))
        return {"tool": hit, "order": hit, "param": hit, "abstain_hit": hit}
    if dag is None:
        return {"tool": False, "order": False, "param": False, "abstain_hit": False}
    if isinstance(dag, dict) and dag.get("abstain"):
        return {"tool": bool(expect.get("abstain")), "order": bool(expect.get("abstain")),
                "param": bool(expect.get("abstain")), "abstain_hit": bool(expect.get("abstain"))}
    tools = [n["tool"] for n in dag.get("nodes", [])]
    exp = expect["tools"]
    tool = set(tools) == set(exp)
    order = tools == exp
    param = True
    if expect.get("bg"):
        n2 = next((n for n in dag["nodes"] if n["tool"] == "T02_background_generate"), {})
        sem = (n2.get("params") or {}).get("semantic", "")
        param = param and (sem == expect["bg"] or sem.isdigit() or (n2.get("params") or {}).get("file"))
    if expect.get("fx"):
        n7 = next((n for n in dag["nodes"] if n["tool"] == "T07_enhance"), {})
        pp = n7.get("params") or {}
        mode = pp.get("mode", "")
        param = param and (mode == expect["fx"] and
                           (expect.get("sticker") is None or pp.get("sticker") == expect["sticker"]))
    if expect.get("keep"):
        n1 = next((n for n in dag["nodes"] if n["tool"] == "T01_matting"), {})
        param = param and ((n1.get("params") or {}).get("keep", "subject") == expect["keep"])
    return {"tool": tool, "order": order, "param": param, "abstain_hit": None}


def run_engine(name, llm, cases):
    acc = {"tool": 0, "order": 0, "param": 0, "abstain_hit": 0}
    n_abstain_exp = 0
    rows = []
    for cid, instr, expect in cases:
        try:
            dag, src = planner_mod.plan(instr, llm=llm, fewshot_variant="best")
        except Exception as e:
            dag = {"abstain": False, "nodes": []}
            rows.append((cid, f"ERROR {str(e)[:60]}"))
            continue
        sc = score(dag, expect)
        for k in ("tool", "order", "param"):
            acc[k] += 1 if sc[k] else 0
        if sc["abstain_hit"] is not None:
            n_abstain_exp += 1
            acc["abstain_hit"] += 1 if sc["abstain_hit"] else 0
        rows.append((cid, "ok" if sc["tool"] else f"tool={tools_of(dag)}"))
    n = len(cases)
    print(f"\n[{name}] n={n}")
    print(f"  Tool  Selection: {acc['tool']}/{n} = {acc['tool']/n:.0%}")
    print(f"  DAG   Order     : {acc['order']}/{n} = {acc['order']/n:.0%}")
    print(f"  Parameter       : {acc['param']}/{n} = {acc['param']/n:.0%}")
    if n_abstain_exp:
        print(f"  拒识准确        : {acc['abstain_hit']}/{n_abstain_exp}")
    for cid, note in rows:
        if note != "ok":
            print(f"  ✗ {cid}: {note}")
    return {"n": n, **{k: round(v / n, 3) for k, v in acc.items()}}


def tools_of(dag):
    if isinstance(dag, dict) and dag.get("abstain"):
        return ["ABSTAIN"]
    return [n["tool"] for n in dag.get("nodes", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="both", choices=["local", "agnes", "both"])
    a = ap.parse_args()
    out = {}
    if a.engine in ("local", "both"):
        from local_planner import LocalPlannerLLM
        llm = LocalPlannerLLM()
        out["local_sft"] = run_engine("本地 SFT Planner (Qwen2.5-0.5B LoRA)", llm, CASES)
    if a.engine in ("agnes", "both"):
        out["agnes"] = run_engine("agnes 基线 (3.0-flash)", planner_mod.AgnesLLM(), CASES)
    (ROOT / "agent_training" / "eval_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告: agent_training/eval_report.json")


if __name__ == "__main__":
    main()
