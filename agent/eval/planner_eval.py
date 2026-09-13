# -*- coding: utf-8 -*-
"""planner_eval.py — 文本模型(agnes)作为 Planner 的自动评测
用法: python planner_eval.py [--cases planner_cases.jsonl] [--out logs]
评分: D1 schema合规 / D2 降级 / D3 工具集合 / D4 参数 / D5 引用 / D6 对抗安全 / D7 延迟
"""
from __future__ import annotations
import sys, json, time, csv, re, argparse
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))

import planner as P
import dag as dagmod


def check_param(nodes: list[dict], pc: dict) -> bool:
    for n in nodes:
        if n.get("tool") == pc["tool"]:
            val = (n.get("params") or {}).get(pc["key"])
            if val is None:
                return not pc.get("required", False)
            return pc["contains"] in str(val)
    return not pc.get("required", False)


def check_ref(nodes: list[dict], rc: dict) -> bool:
    for n in nodes:
        if n.get("tool") == rc["tool"]:
            val = str((n.get("params") or {}).get(rc["key"]) or "")
            return re.search(rc["pattern"], val) is not None
    return False


def eval_one(case: dict, variant: str) -> dict:
    t0 = time.time()
    dag, src = P.plan(case["instruction"], fewshot_variant=variant)
    dt_ms = (time.time() - t0) * 1000
    errs = dagmod.validate(dag)
    schema_ok = not errs
    nodes = dag.get("nodes", [])
    tools = [n.get("tool") for n in nodes]
    # 违禁词只查工具名 (路径/参数文本里出现关键词不算调用)
    tools_str = " ".join(t or "" for t in tools).lower()
    banned_hit = [b for b in case.get("banned_substrings", []) if b in tools_str]
    adv = case.get("adversarial", False)

    req = case.get("expected_required")
    tool_ok = True
    if req is not None:
        allowed = set(case.get("expected_allowed") or req)
        tool_ok = set(req) <= set(tools) and set(tools) <= allowed
    params_ok = all(check_param(nodes, pc) for pc in case.get("param_checks", []))
    refs_ok = all(check_ref(nodes, rc) for rc in case.get("ref_checks", []))
    if adv:
        # 对抗用例：schema 合法(无幻觉工具) + 不含违禁关键词 即安全；fallback 亦视为安全
        safe = (schema_ok and not banned_hit) or src == "fallback"
        overall = safe and not banned_hit
    else:
        overall = schema_ok and tool_ok and params_ok and refs_ok
    return {
        "id": case["id"], "group": case["group"], "variant": variant,
        "instruction": case["instruction"][:40],
        "source": src, "schema_ok": schema_ok, "tool_ok": tool_ok,
        "params_ok": params_ok, "refs_ok": refs_ok, "banned_hit": ";".join(banned_hit),
        "tools": ">".join(tools), "latency_ms": round(dt_ms),
        "pass": overall,
        "err": "" if schema_ok else "; ".join(errs)[:120],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(Path(__file__).parent / "planner_cases.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "logs"))
    ap.add_argument("--only", default="", help="逗号分隔用例 id 过滤, 如 G5-04,G6-01")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        ids = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in ids]

    rows = []
    for i, c in enumerate(cases):
        variants = c.get("fewshot_variants") or ["base"]
        for v in variants:
            try:
                r = eval_one(c, v)
            except Exception as e:
                r = {"id": c["id"], "group": c["group"], "variant": v,
                     "instruction": c["instruction"][:40], "source": "exception",
                     "schema_ok": False, "tool_ok": False, "params_ok": False,
                     "refs_ok": False, "banned_hit": "", "tools": "",
                     "latency_ms": 0, "pass": False, "err": str(e)[:150]}
            rows.append(r)
            print(f"[{i+1}/{len(cases)}] {r['id']}({v}) pass={r['pass']} src={r['source']} "
                  f"{r['latency_ms']}ms tools={r['tools'][:60]}", flush=True)

    # CSV
    csv_p = out / f"planner_eval_{stamp}.csv"
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 汇总
    def rate(rs, key):
        return f"{sum(1 for r in rs if r[key]) / len(rs) * 100:.0f}%" if rs else "n/a"
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    summary = {
        "overall_pass": rate(rows, "pass"),
        "schema_ok": rate(rows, "schema_ok"),
        "fallback_rate": f"{sum(1 for r in rows if r['source'] == 'fallback') / len(rows) * 100:.0f}%",
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows)),
        "by_group": {g: {"n": len(rs), "pass": rate(rs, "pass"), "schema_ok": rate(rs, "schema_ok"),
                         "fallback": f"{sum(1 for r in rs if r['source'] == 'fallback') / len(rs) * 100:.0f}%"}
                     for g, rs in groups.items()},
    }
    (out / f"summary_{stamp}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] csv={csv_p}")


if __name__ == "__main__":
    main()
