# -*- coding: utf-8 -*-
import requests
from pathlib import Path

OUT = r"G:\myself\作业\生产实习\_build\_shot2"
OUT_PATH = Path(OUT).resolve()
OUT_PATH.mkdir(parents=True, exist_ok=True)


def _safe_out(base: Path, name: str) -> Path:
    """输出路径校验：解析后必须仍位于基准目录内（防路径穿越，禁止 ../ 逃逸）。"""
    p = (base / name).resolve()
    if p != base and base not in p.parents:
        raise SystemExit(f"路径越界: {p}")
    return p


B = "https://ai.d.gtimg.com/mcp/screenshots/724747602162280/"
urls = {
    "hero":  B + "screenshot-3_17-20260911_222643249.png?sign=308dae23045b36c39eda979774319df3&t=1791728803",
    "relight": B + "screenshot-3_155-20260911_222643251.png?sign=c39a654a273b0a3afe63d351734b4d4a&t=1791728803",
    "enhance": B + "screenshot-3_201-20260911_222643251.png?sign=62c7e17c1e02653b563ca794ff086050&t=1791728803",
    "final10": B + "screenshot-3_248-20260911_222643252.png?sign=bcf336e3f829acdf627484e9e84e471f&t=1791728803",
    "critic":  B + "screenshot-3_225-20260911_222643252.png?sign=98610baa2b414dccbaed2b1d99f7a5d1&t=1791728803",
    "upload":  B + "screenshot-3_307-20260911_222643253.png?sign=d043958c6eed10a4d1ca16ca4043b372&t=1791728803",
    "spec2":   B + "screenshot-3_378-20260911_222643253.png?sign=dddc15064accee5938adcc1e2189f435&t=1791728803",
}
s = requests.Session()
s.trust_env = False
for k, u in urls.items():
    r = s.get(u, timeout=180)
    p = _safe_out(OUT_PATH, k + ".png")
    p.write_bytes(r.content)
    print(k, r.status_code, len(r.content))
