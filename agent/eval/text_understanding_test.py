# -*- coding: utf-8 -*-
"""text_understanding_test.py — 文本理解能力测试 (Planner-only, 不出图)
目标: 验证 Agent 是否能把用户的中文指令正确映射为 8 工具 DAG (工具选择 + 关键参数)。
覆盖: 抠图 / 绿幕背景 / 滤镜 / 特效 四类 + 混淆与边界指令。
输出: eval/logs/text_mu_report.json + CSV; 只调 LLM 规划, 不执行图像。
"""
from __future__ import annotations
import sys, json, time, csv
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT))
import planner, dag as dagmod


def tools_of(d: dict) -> list[str]:
    return [n["tool"] for n in d.get("nodes", [])]


def find_param(d: dict, tool: str) -> dict:
    for n in d.get("nodes", []):
        if n["tool"] == tool:
            return n.get("params", {})
    return {}


# ---------- 用例: (id, group, instruction, judge) ----------
# judge(d, src) -> (ok: bool, desc: str)
CASES: list[tuple[str, str, str, object]] = []


def C(cid, group, inst, judge):
    CASES.append((cid, group, inst, judge))


# ============ 1. 抠图语义 ============
C("K1", "抠图", "把当前图里的主播台抠出来", lambda d, s: "T01_matting" in tools_of(d) or d.get("intent") == "abstain")
C("K2", "抠图", "把人物抠出来", lambda d, s: tools_of(d) == ["T01_matting"] and find_param(d, "T01_matting").get("keep", "subject") == "subject")
C("K3", "抠图", "扣掉图中的人物, 只留背景", lambda d, s: tools_of(d) == ["T01_matting"] and find_param(d, "T01_matting").get("keep", "background") == "background")
C("K4", "抠图", "把图里的人P掉,铺平背景", lambda d, s: tools_of(d) == ["T01_matting"] and find_param(d, "T01_matting").get("keep", "subject") == "background")
C("K5", "抠图", "去掉主播只保留演播室", lambda d, s: tools_of(d) == ["T01_matting"] and find_param(d, "T01_matting").get("keep", "subject") == "background")
C("K6", "抠图", "用MODNet把人像抠出来", lambda d, s: "T01_matting" in tools_of(d) and str(find_param(d, "T01_matting").get("engine", "")).startswith("humanmatting"))
# ============ 2. 绿幕背景 ============
C("G1", "绿幕", "绿幕合成到城市背景图", lambda d, s: "T02_background_generate" in tools_of(d) and find_param(d, "T02_background_generate").get("mode") == "green_key")
C("G2", "绿幕", "用绿幕键控把这卷视频换到沙滩背景", lambda d, s: "T02_background_generate" in tools_of(d) and find_param(d, "T02_background_generate").get("mode") == "green_key" and find_param(d, "T02_background_generate").get("media_type") == "video")
C("G3", "绿幕", "这张绿幕演播室照片只换掉绿幕部分, 桌子和话筒都要保留", lambda d, s: set(tools_of(d)) == {"T02_background_generate", "T06_harmonize"} and find_param(d, "T06_harmonize").get("mode") == "greenscreen")
C("G4", "绿幕", "绿幕区域替换成综艺背景, 保留所有实物", lambda d, s: "T06_harmonize" in tools_of(d) and find_param(d, "T06_harmonize").get("mode") == "greenscreen")
C("G5", "绿幕", "把绿幕人物直接合成到新闻LED演播室背景", lambda d, s: "T02_background_generate" in tools_of(d) and find_param(d, "T02_background_generate").get("mode") == "green_key")

# ============ 3. 滤镜 ============
C("F1", "滤镜", "给这张图加个胶片颗粒质感", lambda d, s: tools_of(d) == ["T07_enhance"] and find_param(d, "T07_enhance").get("mode") in ("grain", "film_grain"))
C("F2", "滤镜", "调成冷色调", lambda d, s: tools_of(d) == ["T07_enhance"] and find_param(d, "T07_enhance").get("mode") == "cool")
C("F3", "滤镜", "弄成暖色调的风格", lambda d, s: tools_of(d) == ["T07_enhance"] and find_param(d, "T07_enhance").get("mode") == "warm")
C("F4", "滤镜", "加一个「5小纸条」滤镜", lambda d, s: ("T02_background_generate" in tools_of(d) and "5小纸条" in str(find_param(d, "T02_background_generate").get("media_filter", ""))) or (tools_of(d) == ["T07_enhance"]))
C("F5", "滤镜", "用色卡做甜美风", lambda d, s: "media_filter" in find_param(d, "T02_background_generate") if "T02_background_generate" in tools_of(d) else tools_of(d) == ["T07_enhance"])

# ============ 4. 特效 ============
C("E1", "特效", "给这张图加景深虚化", lambda d, s: tools_of(d) == ["T07_enhance"] and find_param(d, "T07_enhance").get("mode") == "depth_blur")
C("E2", "特效", "画面有点糊, 帮我锐化一下", lambda d, s: tools_of(d) == ["T07_enhance"] and find_param(d, "T07_enhance").get("mode") == "sharpen")
C("E3", "特效", "加一个爱心贴纸", lambda d, s: tools_of(d) == ["T07_enhance"] and find_param(d, "T07_enhance").get("mode") == "sticker" and "heart" in str(find_param(d, "T07_enhance").get("sticker", "heart")))
C("E4", "特效", "在右下角加文字水印'样片'", lambda d, s: "T07_enhance" in tools_of(d) and find_param(d, "T07_enhance").get("mode") == "watermark_text")

# ============ 5. 混淆/边界 (易错) ============
C("M1", "边界", "抠出人物换到访谈背景", lambda d, s: tools_of(d) == ["T01_matting", "T02_background_generate", "T03_lighting_estimate", "T04_relight", "T05_shadow_generate", "T06_harmonize"])
C("M2", "边界", "把人物放到播客背景再加暗角", lambda d, s: set(tools_of(d)) >= {"T07_enhance"})
C("M3", "边界", "把当前图保存到本地", lambda d, s: "T08_export" in tools_of(d) or (d.get("nodes") or []) and tools_of(d)[-1] == "T08_export")
C("M4", "边界", "帮我写一首诗", lambda d, s: d.get("intent") == "abstain" or (not d.get("nodes") and not d.get("outputs")))
C("M5", "边界", "这把椅子换颜色", lambda d, s: d.get("intent") == "abstain" or (not d.get("nodes") and not d.get("outputs")))


def main():
    out_dir = AGENT / "eval" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = planner.AgnesLLM()
    rows = []
    for cid, group, inst, judge in CASES:
        t0 = time.time()
        try:
            dag, src = planner.plan(inst, cur_image=None, llm=llm, verbose=False)
        except Exception as e:
            dag, src = {}, f"error:{str(e)[:80]}"
        sec = round(time.time() - t0, 2)
        try:
            ok, desc = judge(dag, src), ""
        except Exception as e:
            ok, desc = False, f"judge err: {e}"
        rows.append({"id": cid, "group": group, "instruction": inst,
                     "ok": ok, "src": src, "sec": sec,
                     "tools": tools_of(dag), "desc": desc,
                     "intent": (dag or {}).get("intent", "")})
        print(f"[{cid}|{group}] {'PASS' if ok else 'FAIL'} [{src}] {tools_of(dag)} | {inst[:36]}",
              flush=True)
    n_pass = sum(1 for r in rows if r["ok"])
    print(f"\n===== 文本理解: {n_pass}/{len(rows)} PASS =====")
    for r in rows:
        if not r["ok"]:
            print(f"  ✗ {r['id']}: tools={r['tools']} src={r['src']} | {r['instruction'][:40]}")
    (out_dir / "text_mu_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / "text_mu_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group", "instruction", "ok",
                                          "src", "sec", "tools", "desc"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print("报告:", out_dir / "text_mu_report.json")


if __name__ == "__main__":
    main()