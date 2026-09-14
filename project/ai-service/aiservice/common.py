"""工具实现公共层：信封错误、URI 解析、图像 IO、确定性随机。

Mock 实现约定（B 组替换时保持）：
    impl.run(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict(outputs)
    失败时抛 ToolFailure(code, message, retryable)。
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw


class ToolFailure(Exception):
    """工具执行失败（对应信封 error 字段）。"""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def fail(code: str, message: str, retryable: bool = False) -> ToolFailure:
    return ToolFailure(code, message, retryable)


# ---- URI 与路径 ----

def resolve_uri(uri: str, root: Path) -> Path:
    """asset:// / artifact:// / 绝对路径 -> 本地路径；不存在抛不可重试错误。"""
    s = uri
    for prefix in ("asset://", "artifact://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    else:
        if not (s.endswith(".png") or s.endswith(".jpg") or s.endswith(".jpeg") or "/" in s or "\\" in s):
            raise fail("E_INVALID_INPUT", f"无法识别的 URI: {uri}")
    p = (root / s).resolve() if not Path(s).is_absolute() else Path(s)
    try:
        p.relative_to(root.resolve())
    except ValueError:
        raise fail("E_INVALID_INPUT", f"URI 越界（禁止访问 artifacts 之外）: {uri}")
    if not p.exists():
        raise fail("E_INVALID_INPUT", f"输入文件不存在: {uri}", retryable=False)
    return p


def ensure_out_dir(out_dir: Path, root: Path) -> Path:
    p = (root / out_dir).resolve() if not out_dir.is_absolute() else out_dir
    try:
        p.relative_to(root.resolve())
    except ValueError:
        raise fail("E_INVALID_INPUT", f"输出目录越界: {out_dir}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_png(img: Image.Image, out_dir: Path, root: Path, name: str) -> str:
    path = out_dir / name
    img.save(path, format="PNG")
    return "artifact://" + path.resolve().relative_to(root.resolve()).as_posix()


def load_rgb(uri: str, root: Path) -> Image.Image:
    return Image.open(resolve_uri(uri, root)).convert("RGB")


def load_rgba(uri: str, root: Path) -> Image.Image:
    img = Image.open(resolve_uri(uri, root))
    return img.convert("RGBA")


# ---- 确定性随机（同输入同输出，保证测试与回放稳定）----
# 注意：这是 mock 产物可复现用的伪随机，刻意非加密安全；任何安全场景请用 secrets 模块。

def seeded_random(*parts: object) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


# ---- 常用小工具 ----

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def new_canvas(w: int, h: int, color=(0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (int(w), int(h)), color)


def draw_crosshair(img: Image.Image, xy: tuple[float, float], r: int = 6, color=(255, 60, 60)) -> None:
    """调试标记（mock 产物可视化用）。"""
    d = ImageDraw.Draw(img)
    x, y = xy
    d.line([(x - r, y), (x + r, y)], fill=color, width=2)
    d.line([(x, y - r), (x, y + r)], fill=color, width=2)
