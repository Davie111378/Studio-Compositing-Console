# -*- coding: utf-8 -*-
"""视频管线 Step2 demo: 单帧引擎处理视频 (换背景 + 滤镜/贴纸/水印)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_io, frame_engine

def progress(i, t):
    if i % 15 == 0: print(f"  [{i}/{t}]", flush=True)

if __name__ == "__main__":
    SRC = "data/video/test_green.mp4"
    # ---- demo 1: 绿幕换背景 ----
    out1 = "outputs/video_demo1_bg.mp4"
    print("== Demo1 视频换背景 ==")
    cfg1 = {"green": {"enabled": True, "bg": "data/ai_generated/studio_bg/bg_02_interview.png"}}
    r1 = frame_engine.process(cfg1, SRC, out1, progress_cb=progress)
    print("  完成:", r1["output"], r1["sec"], "s")
    # ---- demo 2: 滤镜 + 贴纸 + 水印 ----
    out2 = "outputs/video_demo2_fx.mp4"
    print("== Demo2 滤镜+贴纸+水印 ==")
    cfg2 = {"filter": "teal_orange",
            "fx": [{"effect": "heart", "params": {"frac": 0.12}}],
            "watermark": {"text": "© 演播室视频合成", "opacity": 0.5, "size": 0.045}}
    r2 = frame_engine.process(cfg2, SRC, out2, progress_cb=progress)
    print("  完成:", r2["output"], r2["sec"], "s")
