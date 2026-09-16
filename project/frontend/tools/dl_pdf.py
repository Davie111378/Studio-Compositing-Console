# -*- coding: utf-8 -*-
import os
import requests
from pathlib import Path

OUT = r"G:\myself\作业\生产实习\imagecompose-site\ImageCompose-首页纵向叙事重构-设计稿.pdf"


def _safe_out(base: str, name: str = "") -> Path:
    """输出路径校验：解析后必须仍位于基准目录内（防路径穿越）。"""
    base_p = Path(base).resolve()
    p = (base_p / name).resolve() if name else base_p
    if p != base_p and base_p not in p.parents:
        raise SystemExit(f"路径越界: {p}")
    return p


url = ("https://ai.d.gtimg.com/mcp/exports/724742044579898/"
       "ImageCompose%20%E9%A6%96%E9%A1%B5%20%C2%B7%20%E7%BA%B5%E5%90%91%E5%8F%99%E4%BA%8B%E9%87%8D%E6%9E%84-20260911_215609788.pdf"
       "?sign=68ed5b7d265687c3ab8bdd0a1161c3bd&t=1791726969")
s = requests.Session()
s.trust_env = False
r = s.get(url, timeout=300)
out = _safe_out(OUT)
out.write_bytes(r.content)
print(r.status_code, round(len(r.content) / 1024), "KB", out)
