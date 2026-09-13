# -*- coding: utf-8 -*-
"""
mt_download.py — 多线程分段下载 (利用服务端 Accept-Ranges 断点续传)

COCO 官方源实测单连接仅 ~35KB/s, 但支持 Range: bytes。分段并行可显著提速。

特性:
  - 自动探测 Content-Length 与 Accept-Ranges
  - 保留已下载部分, 只补缺失的段 (断点续传, 中断可重复运行)
  - 分段落盘到 .parts/, 完成后拼接为最终文件
  - 实时进度条 + ETA

用法:
  python mt_download.py --url <URL> --out <path> --threads 8
  python mt_download.py ... --probe    # 只探测, 不下载
"""
from __future__ import annotations
import argparse, os, sys, threading, time
from pathlib import Path

import urllib.request

CHUNK = 1 << 20          # 每个 Range 请求 1MB
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def probe(url: str) -> tuple[int, bool]:
    """返回 (total_size, supports_range)。"""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers.get("Content-Length", 0))
        ar = (r.headers.get("Accept-Ranges", "") or "").lower()
    ok = "bytes" in ar
    if not ok:
        # 用一次 Range 请求二次确认
        try:
            req2 = urllib.request.Request(url, headers={"User-Agent": UA,
                                                        "Range": "bytes=0-1"})
            with urllib.request.urlopen(req2, timeout=30) as r2:
                ok = r2.status == 206
        except Exception:
            ok = False
    return total, ok


def fetch_segment(url: str, start: int, end: int, retries: int = 6) -> bytes:
    """下载 [start, end] 闭区间, 带重试。"""
    for k in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
            if data:
                return data
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(1.5 * (k + 1))
    return b""


class Tracker:
    def __init__(self, total: int):
        self.total = total
        self.lock = threading.Lock()
        self.done = 0
        self.t0 = time.time()
        self.last = 0.0
        self.last_t = time.time()

    def add(self, n: int):
        with self.lock:
            self.done += n

    def line(self) -> str:
        with self.lock:
            d = self.done
        now = time.time()
        recent = (d - self.last) / max(now - self.last_t, 0.1)
        if now - self.last_t > 3:
            self.last, self.last_t = d, now
        pct = d / self.total * 100 if self.total else 0
        fill = int(36 * pct / 100)
        bar = "[" + "#" * fill + "-" * (36 - fill) + "]"
        eta = (self.total - d) / recent if recent > 1024 else float("inf")
        eta_s = "--" if eta == float("inf") else \
            f"{int(eta//3600):02d}:{int(eta%3600//60):02d}:{int(eta%60):02d}"
        return (f"{time.strftime('%H:%M:%S')} {bar} {pct:5.2f}%  "
                f"{d/1048576:7.1f}/{self.total/1048576:.1f}MB  "
                f"{recent/1048576:5.2f}MB/s  ETA {eta_s}")


def download(url: str, out: Path, threads: int = 8, seed: Path | None = None,
             progress_log: Path | None = None):
    total, ok = probe(url)
    if not total:
        print("无法获取文件大小, 退回单线程")
        raise SystemExit(1)
    print(f"文件大小 {total/1048576:.1f} MB | 支持断点续传: {ok}")
    if not ok:
        print("服务端不支持 Range, 请用 curl -C - 续传")
        raise SystemExit(1)

    part_dir = out.with_suffix(out.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    segs = [(i, min(i + CHUNK - 1, total - 1)) for i in range(0, total, CHUNK)]
    print(f"分段 {len(segs)} 个 × {CHUNK//1024}KB, 线程 {threads}")

    # ---- 种子文件继承: 把已有 partial 按段切分存入 .parts/ (省重复下载) ----
    if seed and seed.exists():
        seed_size = seed.stat().st_size
        if 0 < seed_size <= total:
            n_full = seed_size // CHUNK
            if n_full > 0:
                data = seed.read_bytes()
                wrote = 0
                for k in range(n_full):
                    pf = part_dir / f"{k*CHUNK:012d}.part"
                    if pf.exists() and pf.stat().st_size == CHUNK:
                        continue
                    pf.write_bytes(data[k*CHUNK:(k+1)*CHUNK])
                    wrote += 1
                print(f"从 {seed.name} 继承 {wrote} 个完整段 "
                      f"({n_full*CHUNK/1048576:.1f} MB, 跳过重复下载)")
            # 最后不完整段丢弃 (留给下载器)
        else:
            print(f"种子文件大小异常 ({seed_size}), 忽略")

    tr = Tracker(total)
    # 已存在的段先计入
    todo = []
    for s, e in segs:
        pf = part_dir / f"{s:012d}.part"
        want = e - s + 1
        if pf.exists() and pf.stat().st_size == want:
            tr.add(want)
            continue
        todo.append((s, e, pf))
    print(f"待下载 {len(todo)} 段 (已完成 {len(segs)-len(todo)} 段, "
          f"{tr.done/1048576:.1f} MB)")

    if not todo:
        pass
    else:
        idx = [0]
        idx_lock = threading.Lock()
        err = []
        log_lock = threading.Lock()

        def worker():
            while True:
                with idx_lock:
                    if idx[0] >= len(todo):
                        return
                    s, e, pf = todo[idx[0]]
                    idx[0] += 1
                try:
                    data = fetch_segment(url, s, e)
                    if data:
                        pf.write_bytes(data)
                        tr.add(len(data))
                except Exception as ex:
                    err.append(f"{s}: {ex}")

        stop = threading.Event()

        def printer():
            while not stop.is_set():
                line = tr.line()
                print("\r" + line, end="", flush=True)
                if progress_log:
                    try:
                        with log_lock:
                            with open(progress_log, "a", encoding="utf-8") as f:
                                f.write(line + "\n")
                    except Exception:
                        pass
                time.sleep(5)

        pt = threading.Thread(target=printer, daemon=True)
        pt.start()
        ths = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        stop.set()
        time.sleep(5.2)
        print()
        if err:
            print(f"失败 {len(err)} 段 (可重跑本脚本续传): {err[:3]}")

    # 拼接
    missing = [f"{s}" for s, e in segs
               if not (part_dir / f"{s:012d}.part").exists()
               or (part_dir / f"{s:012d}.part").stat().st_size != e - s + 1]
    if missing:
        print(f"仍有 {len(missing)} 段缺失, 请重跑续传")
        return False
    print("拼接中...")
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as w:
        for s, e in segs:
            w.write((part_dir / f"{s:012d}.part").read_bytes())
    size = tmp.stat().st_size
    if size != total:
        print(f"拼接后大小不符 {size} != {total}")
        return False
    tmp.replace(out)
    print(f"完成 -> {out} ({size/1048576:.1f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", default="", help="已有部分文件, 其完整段会被继承")
    ap.add_argument("--log", default="", help="进度日志文件 (追加)")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        t, ok = probe(a.url)
        print(f"size={t} ({t/1048576:.1f}MB) supports_range={ok}")
        return
    download(a.url, Path(a.out), a.threads,
             seed=Path(a.seed) if a.seed else None,
             progress_log=Path(a.log) if a.log else None)


if __name__ == "__main__":
    main()
