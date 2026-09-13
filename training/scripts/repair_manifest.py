# -*- coding: utf-8 -*-
"""
repair_manifest.py — 从磁盘补齐 manifest.csv / *_names.txt

背景: ingest_p3m_hm.py 重跑会整表重写, 把 synth_real_bg.py / synth_v3.py
      追加进来的合成样本条目冲掉 (文件还在 train/ 里, 只是没进 manifest)。
      本脚本只做"增量补齐": 已有条目原样保留, 仅追加磁盘上存在但 manifest
      缺失的样本, 并按文件名前缀推断 case。

用法: python repair_manifest.py
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "matting_real"
MF = DATA / "manifest.csv"

PREFIX_CASE = [
    ("sy3_", "syn"), ("sy2_", "syn"), ("syn_", "syn"),
    ("st_", "studio"), ("p3m_", "p3m"), ("hm_", "hm"),
    ("np_", "p3m_np"), ("pp_", "p3m_p"),
    ("vnp_", "p3m_np"), ("vpp_", "p3m_p"),
]


def case_of(name: str) -> str:
    for pre, c in PREFIX_CASE:
        if name.startswith(pre):
            return c
    return "unknown"


def main():
    with open(MF, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    header, body = rows[0], rows[1:]
    have = {r[0] for r in body}

    added = {}
    for split in ("train", "val", "test"):
        d = DATA / split
        if not d.exists():
            continue
        for ip in sorted(d.iterdir()):
            if not ip.is_file() or ip.stem.endswith("_alpha"):
                continue
            if not (d / f"{ip.stem}_alpha.png").exists():
                continue
            if ip.name in have:
                continue
            added.setdefault(split, []).append([ip.name, split, case_of(ip.name)])

    total = sum(len(v) for v in added.values())
    if not total:
        print("[repair] manifest 已完整, 无需补齐")
        return

    with open(MF, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for split in ("train", "val", "test"):
            w.writerows(added.get(split, []))

    for split in ("train", "val", "test"):
        rows_s = added.get(split, [])
        if not rows_s:
            continue
        nf = DATA / f"{split}_names.txt"
        cur = nf.read_text(encoding="utf-8").splitlines() if nf.exists() else []
        cur = [c.strip() for c in cur if c.strip()]
        cur += [r[0] for r in rows_s]
        nf.write_text("\n".join(cur) + "\n", encoding="utf-8")

    for split in ("train", "val", "test"):
        if added.get(split):
            from collections import Counter
            cc = Counter(r[2] for r in added[split])
            print(f"[repair] {split}: +{len(added[split])}  {dict(cc)}")
    print(f"[repair] 合计 +{total}")


if __name__ == "__main__":
    main()
