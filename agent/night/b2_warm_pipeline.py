# -*- coding: utf-8 -*-
"""
b2_warm_pipeline.py — [夜间 B2] 真实 matting+composite 批量热身
- 用 engine.execute 跑完整 6 节点链 (draft 为主 + 2 条 fine)
- 逐节点真实 latency → agent/night/out/latency_table.json (供 ≤10s 预算审计)
- 产物落盘 agent/night/out/frames/ (喂给 C5 DAG 可视化)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # agent/
sys.path.insert(0, str(HERE.parent.parent))  # ROOT

import engine

ROOT = Path(__file__).resolve().parent.parent.parent
FGS = ["fg_01_anchor_male.png", "fg_02_anchor_female.png",
       "fg_03_glasses.png", "fg_04_hair.png"]
BGS = [("bg_02_interview.png", "访谈"), ("bg_01_news_led.png", "新闻LED"),
       ("bg_03_panorama.png", "全景")]


def chain(fg_path: str, bg_sem: str) -> dict:
    return {
        "intent": f"[b2 warm] {fg_path} -> {bg_sem}",
        "nodes": [
            {"id": "n1", "tool": "T01_matting", "params": {"image": f"data/ai_generated/green_fg/{fg_path}"}, "depends_on": []},
            {"id": "n2", "tool": "T02_background_generate", "params": {"semantic": bg_sem}, "depends_on": []},
            {"id": "n3", "tool": "T03_lighting_estimate", "params": {"bg_path": "$n2.bg_path"}, "depends_on": ["n2"]},
            {"id": "n4", "tool": "T04_relight", "params": {"fg_path": "$n1.fg_path", "bg_path": "$n2.bg_path", "light_dir": "$n3.light_dir", "color_temp": "$n3.color_temp"}, "depends_on": ["n1", "n2", "n3"]},
            {"id": "n5", "tool": "T05_shadow_generate", "params": {"alpha_path": "$n1.alpha_path", "bg_path": "$n2.bg_path", "light_dir": "$n3.light_dir"}, "depends_on": ["n1", "n2", "n3"]},
            {"id": "n6", "tool": "T06_harmonize", "params": {"fg_path": "$n4.relit_fg_path", "alpha_path": "$n1.alpha_path", "bg_path": "$n5.shadow_bg_path"}, "depends_on": ["n4", "n5"]},
            {"id": "n7", "tool": "T08_export", "params": {"final_path": "$n6.composite_path", "quality": "normal"}, "depends_on": ["n6"]},
        ],
        "outputs": ["n7"]}


def main():
    rows = []
    combos = [(fg, bg_file, bg_sem) for fg in FGS for bg_file, bg_sem in BGS]
    for i, (fg, _, bg_sem) in enumerate(combos):
        d = chain(fg, bg_sem)
        r = engine.execute(d, quality_default="draft")
        per = {nr["node_id"]: {"tool": nr["tool"], "ms": nr.get("latency_ms"),
                               "status": nr["status"]} for nr in r["node_results"]}
        rows.append({"combo": f"{fg}->{bg_sem}", "quality": "draft",
                     "total_ms": r["total_ms"], "status": r["status"], "nodes": per})
        print(f"[{i+1}/{len(combos)}] {fg}->{bg_sem} draft: {r['status']} {r['total_ms']}ms", flush=True)
    # fine 档 2 条 (T01 refine + T06 fine)
    for fg, (_, bg_sem) in [(FGS[0], BGS[0]), (FGS[2], BGS[2])]:
        d = chain(fg, bg_sem)
        for n in d["nodes"]:
            n["params"]["quality"] = "fine"
        r = engine.execute(d, quality_default="fine")
        per = {nr["node_id"]: {"tool": nr["tool"], "ms": nr.get("latency_ms"),
                               "status": nr["status"]} for nr in r["node_results"]}
        rows.append({"combo": f"{fg}->{bg_sem}", "quality": "fine",
                     "total_ms": r["total_ms"], "status": r["status"], "nodes": per})
        print(f"[fine] {fg}->{bg_sem}: {r['status']} {r['total_ms']}ms", flush=True)

    # 汇总: 每工具 draft/fine 中位延迟
    agg: dict[str, dict[str, list]] = {}
    for row in rows:
        for nid, info in row["nodes"].items():
            if info["ms"] is None:
                continue
            agg.setdefault(info["tool"], {}).setdefault(row["quality"], []).append(info["ms"])
    summary = {}
    for tool, q in agg.items():
        summary[tool] = {}
        for qual, ms_list in q.items():
            ms_list.sort()
            summary[tool][qual] = {"median_ms": ms_list[len(ms_list)//2],
                                   "max_ms": ms_list[-1], "n": len(ms_list)}
    out = {"time": time.strftime("%F %T"), "rows": rows, "summary": summary}
    (HERE / "out" / "latency_table.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 延迟汇总 (供 ≤10s 预算) ===")
    for tool, q in summary.items():
        line = " | ".join(f"{k}: med {v['median_ms']}ms / max {v['max_ms']}ms" for k, v in q.items())
        print(f"  {tool:26s} {line}")
    print("wrote latency_table.json")


if __name__ == "__main__":
    main()
