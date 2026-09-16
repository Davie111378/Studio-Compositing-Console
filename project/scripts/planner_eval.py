# -*- coding: utf-8 -*-
"""Planner 评测：>=100 条测试指令集（规范 §2.2A）。

指标：Tool Selection / DAG Order / Parameter / Spatial Reference 准确率 + 计划延迟。
用法：
  python scripts/planner_eval.py                  # RulePlanner（确定性，离线）
  python scripts/planner_eval.py --planner llm    # LLMPlanner（需 .env 配 LLM_API_BASE/KEY/MODEL）
  python scripts/planner_eval.py --planner auto   # FallbackPlanner（LLM 优先、规则兜底）
报告：data/eval/planner_eval_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "ai-service"))

from agent.config import Settings  # noqa: E402
from agent.dag.models import Quality  # noqa: E402
from agent.errors import AgentError  # noqa: E402
from agent.planner import build_planner  # noqa: E402
from agent.planner.base import PlanningRequest  # noqa: E402
from agent.planner.llm_planner import LLMPlanner  # noqa: E402

CASES = PROJECT / "agent" / "eval" / "planner_eval_cases.jsonl"
OUT = PROJECT / "data" / "eval" / "planner_eval_report.json"


def _roles_full():
    """模拟一次完整合成后的会话产物（含多版本，供回滚语义用）。"""
    def v(role, versions, **outs):
        return {"outputs": outs, "artifacts": list(outs.values()), "versions": versions}
    return {
        "matting": v("matting", [1], rgba_png="artifact://m_rgba", alpha_png="artifact://m_a",
                     mask_png="artifact://m_k"),
        "background_generate": v("background_generate", [1, 2, 3], bg_png="artifact://bg",
                                 depth_png="artifact://d"),
        "lighting_estimate": v("lighting_estimate", [1, 2],
                               light_dir={"azimuth": 180.0, "polar": 35.0},
                               color_temp=3600.0, intensity=1.0),
        "relight": v("relight", [1, 2], relit_png="artifact://r"),
        "shadow_generate": v("shadow_generate", [1], shadow_png="artifact://sh",
                             composited_png="artifact://c"),
        "harmonize": v("harmonize", [1], harmonized_png="artifact://h"),
        "export": v("export", [1], file_url="artifact://f"),
    }


def _dig(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def check_param(node_inputs: dict, spec: dict) -> bool:
    val, found = _dig(node_inputs, spec["key"])
    if not found or val is None:
        return False
    if "equals" in spec:
        try:
            return abs(float(val) - float(spec["equals"])) < 1e-6
        except (TypeError, ValueError):
            return val == spec["equals"]
    if "contains" in spec:
        return spec["contains"] in str(val)
    return True


async def run_case(planner, case: dict) -> dict:
    prior = bool(case.get("prior"))
    assets = [] if case["expect"].get("error") else ["asset://uploads/s/a.png"]
    req = PlanningRequest(
        session_id="eval", text=case["instruction"], asset_uris=assets,
        available_roles=_roles_full() if prior else {},
        quality=Quality.normal, spatial=case.get("spatial"),
    )
    out = {"id": case["id"], "group": case["group"], "tool": False, "order": False,
           "param": True, "spatial": True, "ms": 0, "kind": "plan"}
    t0 = time.perf_counter()
    try:
        result = await planner.plan(req)
    except AgentError as e:
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        want = case["expect"].get("error")
        ok = bool(want) and (e.code == want)
        out["tool"] = out["order"] = out["param"] = ok
        if not ok:
            out["got"] = f"AgentError {e.code}"
        return out
    except Exception as e:  # 非预期异常按失败计
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        out["got"] = f"unexpected {type(e).__name__}: {e}"
        return out
    out["ms"] = round((time.perf_counter() - t0) * 1000)

    exp = case["expect"]
    if exp.get("kind") == "rollback":
        out["kind"] = "rollback"
        got = result.rollback or {}
        out["tool"] = got.get("role") == exp["role"]
        out["order"] = got.get("version") == exp["version"]
        out["param"] = list(got.get("preserve") or []) == list(exp.get("preserve") or [])
        if not out["tool"] or not out["order"] or not out["param"]:
            out["got"] = json.dumps(got, ensure_ascii=False)
        return out

    if result.dag is None:
        out["got"] = "empty dag"
        return out
    tools = [n.tool for n in result.dag.nodes]
    exp_tools = exp["tools"]
    out["tool"] = set(tools) == set(exp_tools)
    out["order"] = tools == exp_tools
    # 参数：显式 param + spatial_param + 默认检查 bg prompt
    node_by_tool = {}
    for n in result.dag.nodes:
        node_by_tool.setdefault(n.tool, n)
    if exp.get("param"):
        spec = exp["param"]
        n = node_by_tool.get(spec["tool"])
        out["param"] = bool(n) and check_param(n.inputs, spec)
    if exp.get("spatial_param"):
        spec = exp["spatial_param"]
        n = node_by_tool.get(spec["tool"])
        out["spatial"] = bool(n) and check_param(n.inputs, spec)
    if not (out["tool"] and out["order"]):
        out["got"] = " -> ".join(tools)
    return out


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="rule", choices=["rule", "llm", "auto"])
    args = ap.parse_args()

    if args.planner == "rule":
        from agent.planner.rule_planner import RulePlanner
        planner = RulePlanner()
        label = "RulePlanner（确定性关键词路由）"
    else:
        settings = Settings()
        if args.planner == "llm":
            if not LLMPlanner.configured(settings.llm_api_base, settings.llm_model):
                print("LLM 未配置（LLM_API_BASE/LLM_API_KEY/LLM_MODEL），无法评测 LLM 档。")
                return 2
            planner = LLMPlanner(settings.llm_api_base, settings.llm_api_key,
                                 settings.llm_model, timeout_s=settings.llm_timeout_s)
            label = f"LLMPlanner（{settings.llm_model}）"
        else:
            planner = build_planner(settings)
            label = f"FallbackPlanner（mode={settings.planner_mode}）"

    cases = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for c in cases:
        rows.append(await run_case(planner, c))

    def acc(pred):
        n = len(rows)
        return round(sum(1 for r in rows if r[pred]) / max(n, 1), 4)

    groups = sorted({r["group"] for r in rows})
    per_group = {}
    for g in groups:
        sub = [r for r in rows if r["group"] == g]
        per_group[g] = {
            "n": len(sub),
            "tool": round(sum(1 for r in sub if r["tool"]) / len(sub), 4),
            "order": round(sum(1 for r in sub if r["order"]) / len(sub), 4),
            "param": round(sum(1 for r in sub if r["param"]) / len(sub), 4),
        }
    lat = [r["ms"] for r in rows]
    report = {
        "planner": label,
        "n_cases": len(rows),
        "tool_acc": acc("tool"),
        "order_acc": acc("order"),
        "param_acc": acc("param"),
        "spatial_acc": acc("spatial"),
        "plan_latency_ms": {"avg": round(sum(lat) / len(lat)), "max": max(lat)},
        "per_group": per_group,
        "fail_cases": [r for r in rows if not (r["tool"] and r["order"] and r["param"])],
        "generated_at": time.strftime("%F %T"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== Planner 评测 · {label} · n={len(rows)} ==")
    print(f"  Tool Selection : {report['tool_acc']:.2%}   (目标 >=95%)")
    print(f"  DAG Order      : {report['order_acc']:.2%}   (目标 >=95%)")
    print(f"  Parameter      : {report['param_acc']:.2%}   (目标 >=90%)")
    print(f"  Spatial Ref    : {report['spatial_acc']:.2%}   (目标 >=80%)")
    print(f"  计划延迟 avg/max: {report['plan_latency_ms']['avg']}ms / {report['plan_latency_ms']['max']}ms (目标 <=3000ms)")
    for g, s in per_group.items():
        print(f"    {g:4s} n={s['n']:3d} tool={s['tool']:.0%} order={s['order']:.0%} param={s['param']:.0%}")
    fails = report["fail_cases"]
    if fails:
        print(f"  失败 {len(fails)} 例:")
        for r in fails[:10]:
            print(f"    {r['id']} [{r['group']}] got={r.get('got', '')}")
    print(f"  报告 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
