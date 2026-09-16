# -*- coding: utf-8 -*-
"""迁移记录（2026-09-14 已执行完毕，保留作审计痕迹）：
把旧前端目录 imagecompose-site/ 中剩余文件（app.js、INTEGRATION.md）
逐字节校验后搬入 project/frontend/，并删除搬空后的旧目录。
本脚本为该次迁移的留痕，不再需要重复执行。
"""
import hashlib
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "imagecompose-site"
DST = Path(__file__).resolve().parent.parent / "frontend"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not SRC.exists():
        print("src missing:", SRC)
        return 1
    moved = []
    for item in sorted(SRC.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(SRC)
        target = DST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        if sha(item) != sha(target):
            print("HASH MISMATCH:", rel)
            return 2
        moved.append(str(rel))
        item.unlink()
    # 清空后的空目录
    for d in sorted((p for p in SRC.rglob("*") if p.is_dir()), reverse=True):
        d.rmdir()
    SRC.rmdir()
    print("moved & verified:", len(moved))
    for m in moved:
        print("  -", m)
    print("removed empty dir:", SRC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
