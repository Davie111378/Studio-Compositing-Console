"""B 组引擎合并验收：内嵌 ai-service 跑 T01→T08 全链，验证引擎选择与产物。

用法:
    python scripts/bsrc_engine_check.py [--image 路径] [--bg 路径]

默认样本: 工作区 imagecompose-site/media/source_original.png + scene_lakeside.jpg。
产物与中间帧写入 data/artifacts/bsrc_check/，可直接肉眼比对效果。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "ai-service"))

DEFAULT_IMAGE = PROJECT.parent / "imagecompose-site" / "media" / "source_original.png"
DEFAULT_BG = PROJECT.parent / "imagecompose-site" / "media" / "scene_lakeside.jpg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--bg", default=str(DEFAULT_BG))
    args = ap.parse_args()

    artifacts = PROJECT / "data" / "artifacts"
    check_dir = artifacts / "bsrc_check"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    assets = artifacts / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for stale in assets.glob("input.png"):
        stale.unlink()
    shutil.copy(args.image, assets / "input.png")
    shutil.copy(args.bg, assets / "bg.png")

    os.environ["ARTIFACTS_DIR"] = str(PROJECT / "data" / "artifacts")
    os.environ["MOCK_DELAY_MS"] = "0"

    from starlette.testclient import TestClient

    from aiservice.app import create_app
    from aiservice import bsrc_loader

    print(f"engine mode={bsrc_loader.engine_mode()} available={bsrc_loader.available()} "
          f"use_bsrc={bsrc_loader.use_bsrc()} upstream={bsrc_loader.upstream_rev()[:12]}")

    client = TestClient(create_app())

    def invoke(tool: str, inputs: dict, out_dir: str, options: dict | None = None) -> dict:
        payload = {"tool": tool, "version": "1.0", "request_id": "check",
                   "inputs": inputs, "options": options or {}, "out_dir": out_dir}
        t0 = time.perf_counter()
        env = client.post(f"/invoke/{tool}", json=payload).json()
        ms = int((time.perf_counter() - t0) * 1000)
        assert env["status"] == "success", f"{tool} 失败: {env}"
        extra = {k: v for k, v in env["outputs"].items()
                 if k in ("engine", "light_dir", "color_temp", "intensity",
                          "effects_applied", "shadow_meta", "harmony_meta")}
        print(f"[{tool:>20}] {ms:5d}ms  {extra}")
        return env["outputs"]

    t01 = invoke("matting", {"image": "asset://assets/input.png"}, "runs/bsrc_check/t01")
    t03 = invoke("lighting_estimate", {"bg_png": "asset://assets/bg.png"}, "runs/bsrc_check/t03")
    ld = t03["light_dir"]
    t04 = invoke("relight", {"rgba_png": t01["rgba_png"], "light_dir": ld,
                             "color_temp": t03["color_temp"], "intensity": t03["intensity"],
                             "bg_hint": "asset://assets/bg.png"}, "runs/bsrc_check/t04")
    t05 = invoke("shadow_generate", {"foreground_png": t04["relit_png"],
                                     "background_png": "asset://assets/bg.png",
                                     "mask_png": t01["mask_png"], "light_dir": ld,
                                     "shadow_strength": 0.6}, "runs/bsrc_check/t05")
    t06 = invoke("harmonize", {"composite_png": t05["composited_png"],
                               "mask_png": t01["mask_png"], "strength": 0.7}, "runs/bsrc_check/t06")
    t07 = invoke("enhance", {"image": t06["harmonized_png"],
                             "effects": [
                                 {"effect": "spotlight", "params": {"intensity": 0.5}},
                                 {"effect": "vignette", "params": {"intensity": 0.45}},
                             ]}, "runs/bsrc_check/t07")
    t08 = invoke("export", {"image": t07["enhanced_png"], "format": "png",
                            "metadata": {"check": "bsrc-merge"}}, "runs/bsrc_check/t08")

    engines_ok = (t03.get("engine") == "bsrc" and t04.get("engine") == "bsrc"
                  and t05.get("engine") == "bsrc" and t06.get("engine") == "bsrc"
                  and t07.get("engine") == "bsrc")
    print("\n结果:")
    print("  B 组引擎生效 (T03-T07):", "OK" if engines_ok else "FAIL -> 回落了 mock，检查 cv2/IMC_ENGINE")
    final_uri = t08["file_url"].replace("artifact://", "")
    print("  成片:", artifacts / final_uri)
    return 0 if engines_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
