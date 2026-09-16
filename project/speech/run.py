# -*- coding: utf-8 -*-
"""speech 服务启动器：python speech/run.py（默认 127.0.0.1:8200）。

与 agent 一致：启动时加载项目根 .env（真实环境变量优先，不覆盖已存在键）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

import uvicorn  # noqa: E402

from speech.service import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("SPEECH_HOST", "127.0.0.1"),
                port=int(os.environ.get("SPEECH_PORT", "8200")))

