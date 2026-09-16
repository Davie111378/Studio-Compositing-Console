# -*- coding: utf-8 -*-
"""后端全链路冒烟：会话 → 上传 → 指令 → 轮询 run 至完成。

用法：python tools/backend_flow_test.py [图片路径]
前置：agent 服务已在 127.0.0.1:8000 运行（PLANNER_MODE=rule python scripts/run_agent.py）。
"""
import io
import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
IMG = sys.argv[1] if len(sys.argv) > 1 else \
    str(Path(__file__).resolve().parent.parent / "media" / "source_original.jpg")

s = requests.Session()
s.trust_env = False  # 本机直连，绕过系统代理


def main() -> int:
    r = s.get(BASE + "/healthz", timeout=5)
    r.raise_for_status()
    print("healthz:", r.json()["status"], "planner:", r.json()["planner"])

    sid = s.post(BASE + "/api/v1/sessions", timeout=10).json()["session_id"]
    print("session:", sid)

    with open(IMG, "rb") as f:
        up = s.post(
            BASE + f"/api/v1/sessions/{sid}/uploads",
            params={},  # multipart
            files={"file": ("portrait.jpg", f, "image/jpeg")},
            timeout=60,
        )
    up.raise_for_status()
    asset = up.json()
    print("uploaded:", asset["asset_id"], asset.get("width"), "x", asset.get("height"))

    instr = {"text": "把这个人物放到夕阳湖边，光线要暖。", "quality": "draft"}
    run = s.post(
        BASE + f"/api/v1/sessions/{sid}/instructions",
        data=json.dumps(instr, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=60,
    ).json()
    rid = run.get("run_id")
    if not rid:
        print("instruction failed:", run)
        return 1
    print("run:", rid, "planner:", run.get("planner"))

    deadline = time.time() + 180
    last = None
    while time.time() < deadline:
        state = s.get(BASE + f"/api/v1/runs/{rid}", timeout=15).json()
        nodes = state.get("dag", {}).get("nodes", [])
        sig = " ".join(f"{n['role']}:{n['status']}" for n in nodes)
        if sig != last:
            print("  ", sig)
            last = sig
        done = nodes and all(n["status"] in ("done", "skipped", "failed") for n in nodes)
        if done:
            break
        time.sleep(1.5)

    ok = all(n["status"] in ("done", "skipped") for n in nodes)
    arts = {n["role"]: n["artifact_urls"] for n in nodes if n.get("artifact_urls")}
    print("final status:", state.get("status"), "all_ok:", ok)
    for role, urls in arts.items():
        print("  artifact", role, "->", urls[0])
    return 0 if ok and "export" in arts else 1


if __name__ == "__main__":
    sys.exit(main())
