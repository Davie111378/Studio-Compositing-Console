# -*- coding: utf-8 -*-
"""
eval_corpus.py — 语料内 held-out 评测: SFT Planner 在提示词库 val 子集上的表现

- 数据: agent_training/data/sft_val.jsonl 中来自语料库 (新) 的条目
- 分层抽样: FULL / GS / T01 / T07 / abstain 各抽 N 条
- 指标: Tool Selection / DAG Order / Parameter (bg semantic, fx mode, sticker, watermark text) / 拒识
- 输出: data/prompt_corpus/eval_corpus_report.json

运行: python agent_training/eval_corpus.py [--n_per 15]
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent_training"))
sys.path.insert(0, str(ROOT / "agent"))

DATA = ROOT / "agent_training" / "data" / "sft_val.jsonl"
OUT = ROOT / "data" / "prompt_corpus" / "eval_corpus_report.json"

FULL_TOOLS = {"T01_matting", "T02_background_generate", "T03_lighting_estimate",
              "T04_relight", "T05_shadow_generate", "T06_harmonize"}


def bucket_of(dag_json: str) -> str:
    try:
        d = json.loads(dag_json)
    except Exception:
        return "bad"
    if d.get("abstain"):
        return "abstain"
    tools = [n["tool"] for n in d.get("nodes", [])]
    ts = set(tools)
    if ts == FULL_TOOLS:
        return "FULL" if "T07_enhance" not in ts else "FULL+T07"
    if "T02_background_generate" in ts and "T06_harmonize" in ts:
        return "GS" if "T07_enhance" not in ts else "GS+T07"
    if tools and tools[0] == "T01_matting" and len(tools) == 1:
        return "T01"
    if "T07_enhance" in ts:
        return "T07"
    return "other"


def tools_of(dag_json: str):
    try:
        d = json.loads(dag_json)
    except Exception:
        return []
    return [n["tool"] for n in d.get("nodes", [])]


def score(pred_json: str, gold_json: str) -> dict:
    """Tool/Order/Param 三维。"""
    try:
        g = json.loads(gold_json)
    except Exception:
        return {"tool": None}
    # 拒识
    if g.get("abstain"):
        try:
            p = json.loads(pred_json)
            hit = bool(p.get("abstain"))
        except Exception:
            hit = False
        return {"tool": hit, "order": hit, "param": hit, "abstain": hit}
    try:
        p = json.loads(pred_json)
        if p.get("abstain"):
            return {"tool": False, "order": False, "param": False, "abstain": False}
    except Exception:
        return {"tool": False, "order": False, "param": False}
    pt, gt = tools_of(pred_json), tools_of(gold_json)
    tool_ok = set(pt) == set(gt)
    order_ok = pt == gt
    # 参数: bg semantic / fx mode / sticker / text
    gp = {n["tool"]: n.get("params", {}) for n in g.get("nodes", [])}
    pp = {n["tool"]: n.get("params", {}) for n in p.get("nodes", [])}
    param_ok = True
    g_t02 = gp.get("T02_background_generate", {})
    p_t02 = pp.get("T02_background_generate", {})
    if g_t02.get("semantic") and g_t02["semantic"] != p_t02.get("semantic"):
        param_ok = False
    g_t07 = gp.get("T07_enhance", {})
    p_t07 = pp.get("T07_enhance", {})
    for k in ("mode", "sticker", "text"):
        if g_t07.get(k) and g_t07[k] != p_t07.get(k):
            param_ok = False
    return {"tool": tool_ok, "order": order_ok, "param": param_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]
    # 只评测语料库新增条目 (assistant 是 JSON DAG; 旧 F 域 QA 是 answer 文本, 跳过)
    corpus_rows = []
    for r in rows:
        asst = r["messages"][2]["content"]
        if asst.startswith("{") and ("nodes" in asst or "abstain" in asst):
            corpus_rows.append((r["messages"][1]["content"], asst))
    by_bucket: dict[str, list] = {}
    for text, gold in corpus_rows:
        b = bucket_of(gold)
        by_bucket.setdefault(b, []).append((text, gold))
    print("val 语料桶分布:", {k: len(v) for k, v in by_bucket.items()})

    sample = []
    for b, items in by_bucket.items():
        rng.shuffle(items)
        sample += items[:a.n_per]
    print(f"评测条数: {len(sample)}")

    from local_planner import LocalPlannerLLM, SFT_SYSTEM
    llm = LocalPlannerLLM()

    agg = Counter()
    per_bucket = {}
    detail = []
    for i, (text, gold) in enumerate(sample):
        try:
            pred = llm.complete([
                {"role": "system", "content": SFT_SYSTEM},
                {"role": "user", "content": text}])
        except Exception as e:
            pred = f'{{"error": "{e}"}}'
        s = score(pred, gold)
        if s.get("tool") is None:
            continue
        b = bucket_of(gold)
        for k in ("tool", "order", "param"):
            if s.get(k):
                agg[k] += 1
        per_bucket.setdefault(b, Counter())
        for k in ("tool", "order", "param"):
            if s.get(k):
                per_bucket[b][k] += 1
        detail.append({"text": text[:60], "bucket": b, **s,
                       "pred_ok": bool(s.get("tool"))})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    n = max(len(detail), 1)
    report = {
        "n_eval": len(detail),
        "tool_acc": round(agg["tool"] / n, 4),
        "order_acc": round(agg["order"] / n, 4),
        "param_acc": round(agg["param"] / n, 4),
        "per_bucket": {b: {"n": c_total, **{k: round(v / max(c_total, 1), 3)
                                            for k, v in c.items()}}
                       for b, c_total in
                       ((bb, sum(1 for d in detail if d["bucket"] == bb)) for bb in per_bucket)
                       for c in [per_bucket[b]]},
        "fail_cases": [d for d in detail if not d["pred_ok"]][:15],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "fail_cases"},
                     ensure_ascii=False, indent=2))
    print(f"报告 -> {OUT}")


if __name__ == "__main__":
    main()
