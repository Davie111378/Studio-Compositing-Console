# -*- coding: utf-8 -*-
"""
b3_planner_sweep.py — [夜间 B3] Planner 全量扫描: 3 组 few-shot × 100 条指令
- 评分: Tool Selection(集合全等) / DAG Order(序列全等) / Parameter(bg语义、fx模式)
- 限速 (每调用间 sleep), 断点续扫 (jsonl checkpoint, 重跑只补失败/缺失项)
- 产出: sweep_results.csv + best_fewshot.json (次日早选优写回 agent/planner/fewshot_best.json)
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # agent/
sys.path.insert(0, str(HERE.parent.parent))

import dag as dagmod
import planner as planner_mod

OUT = HERE / "out"
CKPT = OUT / "sweep_checkpoint.jsonl"
SLEEP_S = 2.0


def make_fewshot_variants():
    """生成 3 组 few-shot: base / strict(补绿幕实景+更强参数约束) / lite(只留 1 例, 测鲁棒)。"""
    import prompt as P
    pdir = Path(__file__).resolve().parent.parent / "planner"
    pdir.mkdir(exist_ok=True)
    # a) strict: base + greenscreen 示例 + fx 示例
    strict = list(P.FEWSHOT_BASE) + [
        {"role": "user", "content": "这张演播室照片只把绿幕换成全景背景, 桌子话筒保留"},
        {"role": "assistant", "content": json.dumps({
            "intent": "绿幕实景区域替换(保留实物)",
            "nodes": [
                {"id": "n1", "tool": "T02_background_generate", "params": {"semantic": "全景", "quality": "draft"}, "depends_on": []},
                {"id": "n2", "tool": "T06_harmonize", "params": {"fg_path": "$cur", "alpha_path": "$cur", "bg_path": "$n1.bg_path", "mode": "greenscreen"}, "depends_on": ["n1"]}],
            "outputs": ["n2"]}, ensure_ascii=False)},
        {"role": "user", "content": "给这张图加景深虚化再导出"},
        {"role": "assistant", "content": json.dumps({
            "intent": "特效+导出",
            "nodes": [
                {"id": "n1", "tool": "T07_enhance", "params": {"image_path": "$cur", "mode": "depth_blur"}, "depends_on": []},
                {"id": "n2", "tool": "T08_export", "params": {"final_path": "$n1.enhanced_path", "quality": "normal"}, "depends_on": ["n1"]}],
            "outputs": ["n2"]}, ensure_ascii=False)},
    ]
    (pdir / "fewshot_strict.json").write_text(json.dumps(strict, ensure_ascii=False), encoding="utf-8")
    # b) lite: 只留第 1 例 (测无示例鲁棒性)
    (pdir / "fewshot_lite.json").write_text(json.dumps(P.FEWSHOT_BASE[:2], ensure_ascii=False), encoding="utf-8")


def score(dag: dict, expect: dict) -> dict:
    tools = [n["tool"] for n in dag["nodes"]]
    exp_tools = expect["tools"]
    tool_ok = set(tools) == set(exp_tools)
    # 顺序: 只比较期望序列是否为实际序列的子序列且相对顺序一致 (LLM 可能并行无关节点)
    order_ok = tools == exp_tools
    # 参数: bg 语义 / fx 模式
    param_ok = True
    if expect.get("bg"):
        bg_node = next((n for n in dag["nodes"] if n["tool"] == "T02_background_generate"), None)
        sem = (bg_node or {}).get("params", {}).get("semantic", "")
        # 允许中文语义或数字序号: 数字序号视为正确(映射表等价)
        param_ok = param_ok and (sem == expect["bg"] or sem.isdigit())
    if expect.get("fx"):
        fx_node = next((n for n in dag["nodes"] if n["tool"] == "T07_enhance"), None)
        mode = (fx_node or {}).get("params", {}).get("mode", "")
        param_ok = param_ok and (mode == expect["fx"])
    return {"tool_ok": tool_ok, "order_ok": order_ok, "param_ok": param_ok,
            "n_nodes": len(tools), "extra_tools": sorted(set(tools) - set(exp_tools)),
            "missing_tools": sorted(set(exp_tools) - set(tools))}


def main():
    make_fewshot_variants()
    items = json.loads((OUT / "instructions_100.json").read_text(encoding="utf-8"))
    # 断点续扫: 读取已有 checkpoint
    done: dict[str, dict] = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[f"{r['variant']}|{r['id']}"] = r
            except Exception:
                pass
    llm = planner_mod.AgnesLLM()
    variants = ["base", "strict", "lite"]
    n_new = 0
    for variant in variants:
        for it in items:
            key = f"{variant}|{it['id']}"
            if key in done:
                continue
            t0 = time.time()
            try:
                dag, src = planner_mod.plan(it["text"], cur_image=None,
                                            llm=llm, fewshot_variant=variant)
                sc = score(dag, it["expect"]) if src != "fallback" else \
                     {"tool_ok": False, "order_ok": False, "param_ok": False,
                      "n_nodes": len(dag["nodes"]), "extra_tools": [], "missing_tools": ["FALLBACK"]}
                rec = {"variant": variant, "id": it["id"], "text": it["text"][:60],
                       "source": src, "sec": round(time.time() - t0, 2), **sc}
            except Exception as e:
                rec = {"variant": variant, "id": it["id"], "text": it["text"][:60],
                       "source": "error", "error": str(e)[:150], "sec": round(time.time() - t0, 2),
                       "tool_ok": False, "order_ok": False, "param_ok": False,
                       "n_nodes": 0, "extra_tools": [], "missing_tools": []}
            with CKPT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[f"{variant}|{it['id']}"] = rec
            n_new += 1
            time.sleep(SLEEP_S)   # 限速防 429
        print(f"[{variant}] done ({n_new} new so far)", flush=True)

    # 汇总
    rows = list(done.values())
    variants_stat = {}
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        n = len(sub)
        variants_stat[v] = {
            "n": n,
            "tool_acc": round(sum(r["tool_ok"] for r in sub) / n, 3),
            "order_acc": round(sum(r["order_ok"] for r in sub) / n, 3),
            "param_acc": round(sum(r["param_ok"] for r in sub) / n, 3),
            "fallback_rate": round(sum(r["source"] == "fallback" for r in sub) / n, 3),
            "avg_sec": round(sum(r["sec"] for r in sub) / n, 2),
        }
    best = max(variants_stat, key=lambda v: (variants_stat[v]["tool_acc"],
                                             variants_stat[v]["order_acc"])) if variants_stat else "base"
    (OUT / "best_fewshot.json").write_text(
        json.dumps({"best": best, "stats": variants_stat}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (OUT / "sweep_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "id", "text", "source", "sec",
                                          "tool_ok", "order_ok", "param_ok", "n_nodes",
                                          "extra_tools", "missing_tools"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print("\n=== Planner 扫描汇总 ===")
    for v, st in variants_stat.items():
        print(f"  {v:8s} tool {st['tool_acc']:.0%} | order {st['order_acc']:.0%} | "
              f"param {st['param_acc']:.0%} | fallback {st['fallback_rate']:.0%} | {st['avg_sec']}s/条")
    print(f"BEST: {best}  (写回 fewshot_{best}.json → fewshot_best.json)")
    src = Path(__file__).resolve().parent.parent / "planner" / f"fewshot_{best}.json"
    if src.exists():
        (src.parent / "fewshot_best.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
