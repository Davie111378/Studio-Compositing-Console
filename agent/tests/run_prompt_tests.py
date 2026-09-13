# -*- coding: utf-8 -*-
"""
run_prompt_tests.py — 《Agent功能测试提示词全集》自动化逐条测试
- 全部走 HTTP /api/agent/dag (含上传/回滚/边界), 与用户网页操作同路径
- 逐条判定 PASS/FAIL, 输出报告表 + 明细 json
- 运行: python agent/tests/run_prompt_tests.py [--base http://127.0.0.1:8765]
"""
from __future__ import annotations
import argparse, base64, json, socket, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def post(base, url, obj, timeout=600):
    req = urllib.request.Request(base + url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http": e.code, "error": e.read().decode()[:300]}
    except (TimeoutError, socket.timeout, OSError) as e:
        # 超时/断连不崩套件: 记为伪响应, 让该用例判 FAIL 后继续
        return {"_timeout": True, "error": f"net: {e}"}


def tools_of(r):
    return [x["tool"] for x in r.get("steps", []) if x.get("tool")]


FULL = ["T01_matting", "T02_background_generate", "T03_lighting_estimate",
        "T04_relight", "T05_shadow_generate", "T06_harmonize"]


def build_cases():
    C = []
    # A 拒识
    C.append(("A1", {"instruction": "今天天气怎么样", "critic": False},
              lambda r, s: bool(r.get("abstain"))))
    C.append(("A2", {"instruction": "帮我写一个python爬虫", "critic": False},
              lambda r, s: bool(r.get("abstain"))))
    C.append(("A3", {"instruction": "讲个笑话", "critic": False},
              lambda r, s: bool(r.get("abstain"))))
    # B 单工具
    C.append(("B1", {"instruction": "给这张图加景深虚化", "critic": False},
              lambda r, s: tools_of(r) == ["T07_enhance"]))
    C.append(("B2", {"instruction": "锐化这张图", "critic": False},
              lambda r, s: tools_of(r) == ["T07_enhance"]))
    C.append(("B3", {"instruction": "给当前画面加胶片颗粒", "critic": False},
              lambda r, s: tools_of(r) == ["T07_enhance"]))
    C.append(("B4", {"instruction": "把这张图导出保存", "critic": False},
              lambda r, s: tools_of(r) == ["T08_export"]))
    # C 绿幕实景
    C.append(("C1", {"instruction": "这张演播室照片只把绿幕换成全景背景, 桌子和话筒都要保留", "critic": False},
              lambda r, s: set(tools_of(r)) == {"T02_background_generate", "T06_harmonize"}))
    C.append(("C2", {"instruction": "绿幕区域替换成综艺背景, 保留所有实物", "critic": False},
              lambda r, s: set(tools_of(r)) == {"T02_background_generate", "T06_harmonize"}))
    # D 完整链
    C.append(("D1", {"instruction": "把 fg_02_anchor_female.png 抠图放到访谈背景", "critic": False},
              lambda r, s: tools_of(r) == FULL))
    C.append(("D2", {"instruction": "将 fg_01_anchor_male.png 合成到新闻LED演播室", "critic": False},
              lambda r, s: tools_of(r) == FULL))
    C.append(("D3", {"instruction": "fg_04_hair 换背景到全景", "critic": False},
              lambda r, s: tools_of(r) == FULL))
    C.append(("D4", {"instruction": "把 fg_03_glasses.png 放进播客场景, 再加暗角", "critic": False},
              lambda r, s: tools_of(r) == FULL + ["T07_enhance"]))
    C.append(("D5", {"instruction": "再加胶片颗粒", "critic": False},
              lambda r, s: tools_of(r) == ["T07_enhance"] and
              _uses_cur(r)))   # 多轮: 作用于上一轮成片
    # E 模式
    C.append(("E1", {"instruction": "把 fg_02_anchor_female.png 抠图放到访谈背景", "critic": False},
              lambda r, s: tools_of(r) == FULL and "refiner" not in _t01_engine(r)))
    C.append(("E2", {"instruction": "把 fg_02_anchor_female.png 抠图放到全景背景", "critic": True},
              lambda r, s: tools_of(r) == FULL and bool(r.get("critic"))))
    C.append(("E3", {"instruction": "把 fg_03_glasses.png 抠图放到播客背景", "critic": False,
                     "quality": "fine"},
              lambda r, s: tools_of(r) == FULL and "refiner" in _t01_engine(r)))
    # F 回滚 (依赖前一轮 DAG)
    C.append(("F1", {"rollback": "综艺"},
              lambda r, s: "kept" in r and any(x["status"] == "kept" for x in r.get("steps", []))
              and bool(r.get("cur_url"))))
    # G 上传图 + 指代
    C.append(("G1", {"instruction": "把这张图里的人物抠出来换到访谈背景", "image": "UPLOAD", "critic": False},
              lambda r, s: tools_of(r) == FULL))
    C.append(("G2", {"instruction": "再加爱心贴纸", "critic": False},
              lambda r, s: tools_of(r) == ["T07_enhance"] and _uses_cur(r)))
    # H 边界
    C.append(("H1", {"instruction": "", "critic": False},
              lambda r, s: r.get("_http") == 400))
    C.append(("H2", {"instruction": "处理这张图", "image": "../../etc/passwd", "critic": False},
              lambda r, s: ("error" in r) or bool(r.get("abstain")) or r.get("final") is None))
    # I 双图合成 (2026-09-11): 两张上传图 + 指令指认谁是背景 (角色解析优先于前端 chip 角色)
    C.append(("I1", {"instruction": "将A图片作为B图片的背景", "critic": False,
                     "images": [{"path": "UP_BG", "role": "input"},    # 故意标反角色
                                {"path": "UP_FG", "role": "bg"}]},
              lambda r, s: tools_of(r) == FULL and _resolved_bg(r) == _bn(s["up_bg"])))
    C.append(("I2", {"instruction": "第二张的背景用第一张", "critic": False,
                     "images": [{"path": "UP_BG", "role": "input"},
                                {"path": "UP_FG", "role": "input"}]},
              lambda r, s: tools_of(r) == FULL and _resolved_bg(r) == _bn(s["up_bg"])))
    return C


def _bn(p):
    import os
    return os.path.basename(p or "")


def _resolved_bg(r):
    """web 层实际解析出的背景图路径 (调试字段 _resolved.bg, 绝对路径 → 取文件名比)。"""
    return _bn((r.get("_resolved") or {}).get("bg") or "")


def _t01_engine(r):
    for x in r.get("steps", []):
        if x.get("tool") == "T01_matting":
            return x.get("engine", "")
    return ""


def _uses_cur(r):
    """多轮判定: 请求作用于上一轮成片 (T01 未出现 = 未重新抠图 = 用了 cur)。"""
    return "T01_matting" not in tools_of(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--only", default="", help="只跑指定用例号, 逗号分隔 (如 A1,D1)")
    a = ap.parse_args()
    base = a.base
    only = {x.strip().upper() for x in a.only.split(",") if x.strip()}

    # 预置: 上传测试图 (供 G1/I1/I2) — 人像前景 + 全景背景各一张
    up = post(base, "/api/upload", {"data": "data:image/png;base64," + base64.b64encode(
        (ROOT / "data/ai_generated/green_fg/fg_03_glasses.png").read_bytes()).decode()})
    upload_path = up.get("path")
    up2 = post(base, "/api/upload", {"data": "data:image/png;base64," + base64.b64encode(
        (ROOT / "data/ai_generated/studio_bg/bg_03_panorama.png").read_bytes()).decode()})
    upload_bg_path = up2.get("path")
    print(f"[setup] 上传前景测试图 -> {upload_path}")
    print(f"[setup] 上传背景测试图 -> {upload_bg_path}")

    post(base, "/api/agent/dag", {"reset": True})
    results = []
    for cid, payload, check in build_cases():
        if only and cid not in only:
            continue
        p = dict(payload)
        if p.get("image") == "UPLOAD":
            p["image"] = upload_path
        for _it in (p.get("images") or []):
            if _it.get("path") == "UP_BG":
                _it["path"] = upload_bg_path
            elif _it.get("path") == "UP_FG":
                _it["path"] = upload_path
        p["up_bg"] = upload_bg_path          # 供 check 闭包断言
        t0 = time.time()
        r = post(base, "/api/agent/dag", p)
        sec = time.time() - t0
        try:
            ok = bool(check(r, p))
        except Exception as e:
            ok = False
            r = {"_checker_error": str(e)[:150], **r}
        status = "PASS" if ok else "FAIL"
        results.append({"id": cid, "ok": ok, "sec": round(sec, 1),
                        "tools": tools_of(r), "abstain": bool(r.get("abstain")),
                        "final": bool(r.get("cur_url")), "resp": r})
        print(f"[{cid}] {status}  {sec:5.1f}s  tools={tools_of(r) or ('拒识' if r.get('abstain') else '-')}",
              flush=True)

    n_pass = sum(1 for x in results if x["ok"])
    print(f"\n===== 汇总: {n_pass}/{len(results)} PASS =====")
    for x in results:
        if not x["ok"]:
            print(f"  ✗ {x['id']}: tools={x['tools']} abstain={x['abstain']} final={x['final']}")
    (OUT / "prompt_test_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("报告:", OUT / "prompt_test_report.json")


if __name__ == "__main__":
    main()
