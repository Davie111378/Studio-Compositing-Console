"""ai-service 启动器：python ai-service/run.py（默认 :8100）。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    os.environ.setdefault("ARTIFACTS_DIR", str(Path(__file__).resolve().parents[1] / "data" / "artifacts"))
    uvicorn.run("aiservice.app:app", host=os.environ.get("AI_SERVICE_HOST", "127.0.0.1"),
                port=int(os.environ.get("AI_SERVICE_PORT", "8100")), reload=False)
