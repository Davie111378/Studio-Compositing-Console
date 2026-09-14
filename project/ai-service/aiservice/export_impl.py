"""T08 export —— mock 实现（纯 IO，无模型，A 组自实现）。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from aiservice.common import ensure_out_dir, fail, load_rgb, resolve_uri, save_png

_FORMATS = {"png": "PNG", "jpg": "JPEG"}


def run(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    out_dir = ensure_out_dir(out_dir, root)
    img = load_rgb(inputs["image"], root)
    fmt = inputs.get("format", "png")
    if fmt not in _FORMATS:
        raise fail("E_INVALID_INPUT", f"不支持的导出格式: {fmt}")
    size = inputs.get("size")
    if size:
        img = img.resize((int(size["width"]), int(size["height"])))

    ext = "png" if fmt == "png" else "jpg"
    path = out_dir / f"final.{ext}"
    if fmt == "jpg":
        img.save(path, format="JPEG", quality=92)
    else:
        img.save(path, format="PNG")
    uri = "artifact://" + path.resolve().relative_to(root.resolve()).as_posix()
    return {
        "file_url": uri,
        "meta": {
            "width": img.width,
            "height": img.height,
            "format": fmt,
            "bytes": path.stat().st_size,
        },
    }
