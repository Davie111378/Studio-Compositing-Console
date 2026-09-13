# -*- coding: utf-8 -*-
"""
video_io.py — 视频读写与逐帧处理基础模块 (视频管线 Step1)
- VideoReader : cv2 VideoCapture 封装, 遍历帧 (BGR->RGB), 帧数/尺寸/fps 统计
- VideoWriter : 自动按原视频参数写回; 无 audio (纯视频管道), 编码选 x264 兼容 vp9/mp4v
- process_video: 通用"帧->帧"处理函数 (接收 callable), 带进度回调

API:
  probe(path) -> {width,height,fps,frames,duration}
  read_frames(path, fn, progress_cb=None, keep_size=True)
      fn(frame_rgb: (H,W,3) uint8) -> frame_rgb' (同一尺寸), 逐帧调用
"""
from __future__ import annotations
import os, time
from pathlib import Path

import cv2
import numpy as np


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = frames / fps if fps else 0
    cap.release()
    return {"width": w, "height": h, "fps": float(fps),
            "frames": frames, "duration": round(dur, 2)}


def pick_fourcc(codec="VP90"):
    """可用编码: vp09(VP9, .webm, 浏览器原生支持, 推荐) / mp4v(mp4, 部分播放器不可播)。
    返回 (fourcc, suffix)。"""
    c = {"vp09": "VP90", "vp9": "VP90", "vp90": "VP90",
         "vp8": "VP80", "vp80": "VP80",
         "mp4v": "mp4v", "avc1": "avc1", "x264": "avc1"}.get(str(codec).lower(), "VP90")
    suffix = ".webm" if c in ("VP90", "VP80") else ".mp4"
    return cv2.VideoWriter_fourcc(*c), suffix


def process_video(in_path, out_path, fn, progress_cb=None, codec="VP90",
                  every_n=1, max_frames=None):
    """逐帧处理视频。
    in_path : 源视频
    out_path: 输出视频 (推荐 .webm / VP9, 浏览器可直接播放)
    fn      : callable(frame_rgb_uint8_3ch) -> frame_rgb_uint8_3ch (同一 HxWx3)
    progress_cb(frame_idx, total) : 进度回调
    返回 probe 摘要 dict
    """
    info = probe(in_path)
    W, H, FPS = info["width"], info["height"], info["fps"]
    if FPS <= 0 or FPS != FPS: FPS = 25.0
    cap = cv2.VideoCapture(str(in_path))
    fourcc, suffix = pick_fourcc(codec)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # 扩展名必须与编码容器一致 (VP9 -> webm, mp4v -> mp4), 否则容器不匹配无法播放
    if suffix == ".webm" and out_p.suffix.lower() != ".webm":
        out_p = out_p.with_suffix(".webm")
    elif suffix == ".mp4" and out_p.suffix.lower() != ".mp4":
        out_p = out_p.with_suffix(".mp4")
    writer = cv2.VideoWriter(str(out_p), fourcc, FPS, (W, H))
    if not writer.isOpened():
        # fallback 到 mp4v
        writer.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if suffix == ".webm":
            out_p = out_p.with_suffix(".mp4")
            suffix = ".mp4"
        writer = cv2.VideoWriter(str(out_p), fourcc, FPS, (W, H))
    total = info["frames"]
    idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()          # BGR
        if not ok:
            break
        if idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_rgb = fn(rgb, idx) if every_n == 1 else fn(rgb, idx)
            if out_rgb is None:
                out_rgb = rgb
            # 保证尺寸一致
            if out_rgb.shape[:2] != (H, W):
                out_rgb = cv2.resize(out_rgb, (W, H))
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)
        idx += 1
        if progress_cb and idx % max(1, total // 20) == 0:
            progress_cb(idx, total)
        if max_frames and idx >= max_frames:
            break
    cap.release(); writer.release()
    if progress_cb:
        progress_cb(total, total)
    out_p = Path(str(out_p))
    return {"input": str(Path(in_path)), "output": str(out_p),
            "frames_written": idx, "size": [W, H], "fps": FPS,
            "codec": codec, "sec": round(time.time() - t0, 2)}


# ---- 便捷: 回调格式 (每帧给 RGB numpy)
def sample_first_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, f = cap.read(); cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/video/test_green.mp4"
    print("probe:", probe(src))
    # 冒烟: 逐帧原样回写 (验证 IO)
    out = "outputs/video_io_test.mp4"
    def noop(frame, idx): return frame
    r = process_video(src, out, noop, progress_cb=lambda i, t: print(f"  {i}/{t}", flush=True))
    print("写回 OK:", r)
