"""Agent 服务启动器：python scripts/run_agent.py（默认 :8000）。

环境变量见 agent/config.py；常用：
    AI_SERVICE_URL=http://127.0.0.1:8100   # 分离部署 ai-service（B 组替换真模型时）
    MOCK_DELAY_MS=0                         # 关闭模拟延迟（测试）
    PLANNER_MODE=rule|auto|llm
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    os.environ.setdefault("ARTIFACTS_DIR", str(PROJECT_ROOT / "data" / "artifacts"))
    os.environ.setdefault("RUNS_DIR", str(PROJECT_ROOT / "data" / "runs"))
    uvicorn.run("agent.server.app:app",
                host=os.environ.get("AGENT_HOST", "127.0.0.1"),
                port=int(os.environ.get("AGENT_PORT", "8000")),
                reload=False)


if __name__ == "__main__":
    main()
