# -*- coding: utf-8 -*-
import requests
from pathlib import Path

OUT = r"G:\myself\作业\生产实习\_build\_shot"
OUT_PATH = Path(OUT).resolve()
OUT_PATH.mkdir(parents=True, exist_ok=True)


def _safe_out(base: Path, name: str) -> Path:
    """输出路径校验：解析后必须仍位于基准目录内（防路径穿越，禁止 ../ 逃逸）。"""
    p = (base / name).resolve()
    if p != base and base not in p.parents:
        raise SystemExit(f"路径越界: {p}")
    return p


B = "https://ai.d.gtimg.com/mcp/screenshots/724742044579898/"
urls = {
    "01b": B + "screenshot-2_60-20260911_215439900.png?sign=8858a10a14353293dfba48931f23edab&t=1791726880",
    "02b": B + "screenshot-2_84-20260911_215439901.png?sign=fe1e94110baf4b483673126cc9bf01a2&t=1791726879",
    "03b": B + "screenshot-2_112-20260911_215439902.png?sign=47f14d1f54e1ff5118fdf60622aad056&t=1791726880",
    "05b": B + "screenshot-2_169-20260911_215439902.png?sign=9e5a783e392ddd83d01ef4bf44a5ab36&t=1791726880",
    "06b": B + "screenshot-2_189-20260911_215439903.png?sign=725cd45e926ed86f88d53426711d86da&t=1791726880",
    "08b": B + "screenshot-2_238-20260911_215439904.png?sign=31defc33eeb064236e6f9b955503fe32&t=1791726880",
    "10b": B + "screenshot-2_314-20260911_215439904.png?sign=0e2fa78be6a70cc04841b0ee2091431c&t=1791726880",
}
s = requests.Session()
s.trust_env = False
for k, u in urls.items():
    r = s.get(u, timeout=120)
    p = _safe_out(OUT_PATH, k + ".png")
    p.write_bytes(r.content)
    print(k, r.status_code, len(r.content))
