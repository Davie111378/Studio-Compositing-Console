"""测试公共夹具：临时产物目录、样本图、上下文构建。"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "ai-service")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    """隔离的 artifacts/runs 目录 + 零延迟 mock。"""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOCK_DELAY_MS", "0")
    monkeypatch.delenv("AI_SERVICE_URL", raising=False)
    # 千问相关 key 全部剥离：单测永远走规则 Planner / 规则 Critic，不触网
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.setenv("PLANNER_MODE", "auto")
    from agent.config import reset_settings
    return reset_settings()


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """合成一张'人物'测试图：浅背景 + 深色人形，mock 抠取能明确分离。"""
    img = Image.new("RGB", (256, 320), (244, 240, 235))
    d = ImageDraw.Draw(img)
    d.ellipse([98, 40, 158, 100], fill=(96, 66, 50))          # 头
    d.rounded_rectangle([78, 110, 178, 290], radius=24, fill=(40, 60, 110))  # 身体
    d.ellipse([108, 55, 122, 70], fill=(230, 230, 230))
    d.ellipse([134, 55, 148, 70], fill=(230, 230, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def context(tmp_env):
    """完整应用上下文（内嵌 provider + 规则 planner + 规则 critic）。"""
    from agent.server.state import build_context
    ctx = build_context(tmp_env)
    yield ctx
    import asyncio
    asyncio.run(ctx.aclose())


async def wait_done(executor, run_id: str, timeout_s: float = 90.0):
    """轮询到终态。"""
    import asyncio
    from agent.dag.models import RunStatus
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    state = executor.get_run(run_id)
    while state.status in (RunStatus.planning, RunStatus.running, RunStatus.paused,
                           RunStatus.critiquing, RunStatus.replanning):
        if loop.time() > deadline:
            raise TimeoutError(f"run {run_id} 未在 {timeout_s}s 内完成: {state.status}")
        await asyncio.sleep(0.03)
        state = executor.get_run(run_id)
    return state
