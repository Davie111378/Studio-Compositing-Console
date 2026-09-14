"""端到端冒烟验证：对一个运行中的 Agent 服务走完 Demo 01 + Demo 04 全流程。

用法：
    python scripts/smoke_test.py                          # 对 http://127.0.0.1:8000
    python scripts/smoke_test.py --url http://127.0.0.1:9000
    python scripts/smoke_test.py --image path/to/person.png

验证点：healthz -> tools(8) -> 建会话 -> 上传 -> Demo01 一句话合成（轮询至 done，
Critic 通过）-> 产物可下载 -> Demo04 条件回滚（保留光照重跑下游）-> 回放记录完整。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_sample_png() -> bytes:
    img = Image.new("RGB", (256, 320), (244, 240, 235))
    d = ImageDraw.Draw(img)
    d.ellipse([98, 40, 158, 100], fill=(96, 66, 50))
    d.rounded_rectangle([78, 110, 178, 290], radius=24, fill=(40, 60, 110))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Smoke:
    def __init__(self, base: str, image: bytes):
        self.base = base.rstrip("/")
        self.client = httpx.Client(timeout=180)
        self.image = image
        self.passed: list[str] = []

    def ok(self, name: str, cond: bool, extra: str = "") -> None:
        mark = "PASS" if cond else "FAIL"
        print(f"[{mark}] {name}{(' - ' + extra) if extra else ''}")
        if not cond:
            raise SystemExit(f"冒烟失败于: {name}")
        self.passed.append(name)

    def run(self) -> None:
        r = self.client.get(f"{self.base}/healthz")
        self.ok("healthz", r.status_code == 200, json.dumps(r.json(), ensure_ascii=False))

        tools = self.client.get(f"{self.base}/api/v1/tools").json()["tools"]
        self.ok("8 个工具 Schema", len(tools) == 8, ", ".join(sorted(tools)))

        sid = self.client.post(f"{self.base}/api/v1/sessions").json()["session_id"]
        asset = self.client.post(f"{self.base}/api/v1/sessions/{sid}/uploads",
                                 files={"file": ("person.png", self.image, "image/png")}).json()
        png = self.client.get(self.base + asset["url"])
        self.ok("上传与产物静态服务", png.status_code == 200 and png.content[:4] == b"\x89PNG")

        # ---- Demo 01：一句话合成 ----
        r = self.client.post(f"{self.base}/api/v1/sessions/{sid}/instructions",
                             json={"text": "把这个人物放进傍晚的咖啡馆，光从左边照过来"})
        self.ok("Demo01 计划生成", r.status_code == 200,
                f"planner={r.json()['planner']} nodes={len(r.json()['plan']['nodes'])}")
        run_id = r.json()["run_id"]
        state = self.wait_run(run_id)
        self.ok("Demo01 执行完成", state["status"] == "done",
                f"critic={state['critic']['overall'] if state.get('critic') else '-'} "
                f"replan={state['replan_count']}")
        export = next(n for n in state["dag"]["nodes"] if n["tool"] == "export")
        final_url = export["artifacts"][0].replace("artifact://", "/artifacts/")
        final = self.client.get(self.base + final_url)
        self.ok("最终产物下载", final.status_code == 200, f"{len(final.content)} bytes -> {final_url}")

        # ---- Demo 04：条件回滚（换回背景 v1，保留现在的光）----
        r = self.client.post(f"{self.base}/api/v1/sessions/{sid}/rollback",
                             json={"role": "background_generate", "version": 1,
                                   "preserve": ["lighting_estimate", "relight"]})
        self.ok("Demo04 回滚计划", r.status_code == 200,
                f"rerun={r.json()['rollback']['rerun']}")
        rb_state = self.wait_run(r.json()["run_id"])
        self.ok("Demo04 回滚执行", rb_state["status"] == "done")

        replay = self.client.get(f"{self.base}/api/v1/runs/{run_id}/replay").json()
        events = replay["events"]
        self.ok("A8 运行可回放", replay["result"]["status"] == "done"
                and any(e["event"] == "critic" for e in events),
                f"{len(events)} 条事件")

        print(f"\n冒烟全部通过（{len(self.passed)} 项）。会话 {sid} 可在前端继续多轮测试。")

    def wait_run(self, run_id: str, timeout_s: float = 180) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            state = self.client.get(f"{self.base}/api/v1/runs/{run_id}").json()
            if state["status"] in ("done", "failed", "cancelled"):
                return state
            time.sleep(0.3)
        raise SystemExit(f"运行 {run_id} 超时未完成")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default="", help="可选：用于上传的图片路径（默认程序合成）")
    args = ap.parse_args()
    image = Path(args.image).read_bytes() if args.image else make_sample_png()
    Smoke(args.url, image).run()
    sys.exit(0)
