"""ai-service 启动脚本（与 ai-service/run.py 等价的入口，便于记忆）。"""

import runpy
import sys
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "ai-service" / "run.py"), run_name="__main__")
