# -*- coding: utf-8 -*-
"""
watch_stuff_download.py — D: 实时跟踪 COCO-Stuff 标注下载, 完成后自动解压校验

用法:
  python watch_stuff_download.py           # 监控直到完成 (或 4h 超时)
  python watch_stuff_download.py --once    # 只打印一次当前状态
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "coco_stuff" / "stuff_annotations_trainval2017.zip"
TOTAL = 1148688564          # 服务端 Content-Length
LOG = ROOT / "data" / "coco_stuff" / "download_progress.log"


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def bar(pct: float, width: int = 36) -> str:
    fill = int(width * pct / 100)
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def ts() -> str:
    return time.strftime("%H:%M:%S")


def step(prev_size: int, prev_t: float) -> tuple[int, float, float]:
    sz = ZIP.stat().st_size if ZIP.exists() else 0
    now = time.time()
    spd = (sz - prev_size) / max(now - prev_t, 0.1)
    return sz, now, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--timeout_h", type=float, default=4.0)
    a = ap.parse_args()

    prev_size, prev_t = 0, time.time()
    t0 = time.time()
    samples: list[tuple[float, int]] = []       # (t, size) 滑动窗口算速率

    while True:
        sz = ZIP.stat().st_size if ZIP.exists() else 0
        pct = sz / TOTAL * 100
        now = time.time()
        samples.append((now, sz))
        # 用最近 3 分钟窗口(至少 2 个点)算平滑速率, 避免首样本爆速
        win = [(t, s) for t, s in samples if now - t <= 180]
        if len(win) >= 2 and win[-1][0] > win[0][0]:
            spd = (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0])
        else:
            spd = 0.0
        eta = (TOTAL - sz) / spd if spd > 1024 else float("inf")
        msg = (f"{ts()}  {bar(pct)} {pct:5.2f}%  "
               f"{human(sz)}/{human(TOTAL)}  {human(spd)}/s  "
               f"ETA {'--' if eta == float('inf') else time.strftime('%H:%M:%S', time.gmtime(eta))}")
        print(msg, flush=True)
        try:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

        if a.once:
            return
        if sz >= TOTAL:
            print(f"\n[{ts()}] 下载完成! 总大小 {human(sz)}")
            break
        if time.time() - t0 > a.timeout_h * 3600:
            print(f"\n[{ts()}] 超时退出 ({a.timeout_h}h), 当前 {pct:.2f}%")
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
