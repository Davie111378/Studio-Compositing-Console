# -*- coding: utf-8 -*-
"""两张上传图的指令角色解析自测: _parse_two_image_roles / _ref_index。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "app"))
from studio_web import _parse_two_image_roles as P  # noqa: E402

CASES = [
    # (指令, 期望 (bg_idx, fg_idx) 或 None)
    ("将A图片作为B图片的背景", (0, 1)),
    ("用A图作为B图的背景", (0, 1)),
    ("把第一张作为第二张的背景", (0, 1)),
    ("用第一张做第二张的背景", (0, 1)),
    ("A图是B图的背景", (0, 1)),
    ("将第一张图片充当第二张图片的背景", (0, 1)),
    ("B图的背景换成A图", (0, 1)),
    ("第二张的背景用第一张", (0, 1)),
    ("把第二张的背景替换为第一张", (0, 1)),
    ("把A图P到B图的背景上", (1, 0)),
    ("把第一张放到第二张的背景上", (1, 0)),
    ("用A图当背景", (0, 1)),
    ("背景用第二张", (1, 0)),
    ("图1作为图2的背景", (0, 1)),
    ("第一张的背景换成第二张", (1, 0)),
    # 解析不出 → None (走前端角色兜底)
    ("给这张图加个滤镜", None),
    ("把两张图都美化一下", None),
    ("给A图加个景深虚化", None),      # 单引用但动词不是"作为背景"类
]

ok = True
for t, want in CASES:
    got = P(t)
    mark = "PASS" if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"[{mark}] {t!r} -> {got} (want {want})")

# _ref_index 边界
from studio_web import _ref_index as R  # noqa: E402
for tok, want in [("A图", 0), ("图B", 1), ("第 2 张", 1), ("第1张", 0), ("1号图", 0),
                  ("前者", 0), ("后者", 1), ("A 图片", 0)]:
    got = R(tok)
    if got != want:
        ok = False
        print(f"[FAIL] _ref_index({tok!r}) = {got} (want {want})")
    else:
        print(f"[PASS] _ref_index({tok!r}) = {got}")

print("ALL_PASS" if ok else "HAS_FAILURES")
sys.exit(0 if ok else 1)
