# -*- coding: utf-8 -*-
"""e2e_image_test.py — 文本指令端到端驱动图片处理 效果测试
文本提示词(agent/eval/e2e_prompts.jsonl) → AgentSession(Planner+Executor+Critic) → 成片
评价: 执行状态 / critic VLM 5维分 / 耗时 / 成片清单 → 汇总 JSON + 联络表
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
ROOT = AGENT.parent
sys.path.insert(0, str(AGENT))

from run import AgentSession

OUT = ROOT / "experiments" / "studio" / "e2e_text2img"
OUT.mkdir(parents=True, exist_ok=True)
CASES = [json.loads(l) for l in (AGENT / "eval" / "e2e_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def run_all():
    rows = []
    s = AgentSession(quality="draft", enable_critic=True, verbose=True)
    for c in CASES:
        cid = c["id"]
        if cid == "P7b":   # 多轮第二跳: 条件回滚
            r = s.rollback_background("综艺")
            rows.append({"id": cid, "instruction": c["instruction"], **{k: r.get(k) for k in ("status", "final", "rerun", "kept")}})
            print(f"[{cid}] rollback -> {r.get('final')}", flush=True)
            continue
        s.cur_image = str(ROOT / c["input_image"]) if c.get("input_image") else s.cur_image
        r = s.process(c["instruction"])
        critic = (r.get("critic") or {}).get("scores", {})
        rows.append({
            "id": cid, "instruction": c["instruction"],
            "plan_source": r.get("plan_source"), "sec": r.get("sec"),
            "final": r.get("final"), "critic": critic,
            "utility": (r.get("utility") or {}).get("U"),
            "exec_ms": (r.get("run") or {}).get("total_ms"),
            "abstain": r.get("abstain"),
        })
        print(f"[{cid}] final={r.get('final')} critic={critic}", flush=True)
        # 成片复制到 OUT 便于归档
        if r.get("final") and Path(r["final"]).exists():
            import shutil
            ext = Path(r["final"]).suffix or ".png"
            shutil.copy(r["final"], OUT / f"{cid}_final{ext}")
    (OUT / "e2e_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[e2e] saved ->", OUT / "e2e_results.json")
    return rows


def contact_sheet(rows):
    from PIL import Image, ImageDraw
    finals = [(r["id"], r.get("final")) for r in rows if r.get("final") and Path(r["final"]).exists()]
    if not finals:
        print("[sheet] 无成片"); return
    W = 420
    tiles = []
    for cid, p in finals:
        im = Image.open(p).convert("RGB")
        tiles.append((cid, im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)))
    cols = 3
    th = max(t.height for _, t in tiles)
    grid = Image.new("RGB", (W * cols + 24, (th + 34) * ((len(tiles) + cols - 1) // cols) + 10), "white")
    d = ImageDraw.Draw(grid)
    for i, (cid, t) in enumerate(tiles):
        x = (i % cols) * (W + 8) + 4
        y = (i // cols) * (th + 34) + 26
        grid.paste(t, (x, y))
        d.text((x + 2, y - 18), cid, fill="black")
    grid.save(OUT / "e2e_contact_sheet.png")
    print("[e2e] sheet ->", OUT / "e2e_contact_sheet.png")


if __name__ == "__main__":
    rows = run_all()
    contact_sheet(rows)
