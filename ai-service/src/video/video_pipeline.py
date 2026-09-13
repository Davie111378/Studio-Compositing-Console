# -*- coding: utf-8 -*-
"""
video_pipeline.py — 视频处理完整管线 (视频 Step4)
一行调用: pipeline.process(cfg, in_video, out_video)

预设:
  preset_replace_background(bg_path)        视频换背景 (绿幕键控)
  preset_apply_filter(effect_key)           全视频统一滤镜/风格
  preset_mosaic_track(bbox=None)           视频局部打码 (按区域)
  preset_watermark(text, ...)              全视频水印 (斜向/角落)
  preset_remove_object(bbox, every=1)       圈选删除+绿幕填补
  preset_pretty(tone="warm", heart=...)     一键美化+贴纸

可组合: pipelines.combine(preset_a, preset_b, ...) -> cfg
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fx"))

import video_io, frame_engine, temporal


def preset_replace_background(bg_path: str, enabled=True) -> Dict[str, Any]:
    return {"green": {"enabled": enabled, "bg": bg_path}}


def preset_apply_filter(effect_key: str) -> Dict[str, Any]:
    return {"filter": effect_key}


def preset_mosaic_track(bbox=None, cell=18, enabled=True) -> Dict[str, Any]:
    """局部打码, bbox=(x1,y1,x2,y2) 原图像素坐标, None=全图中心。"""
    return {"mosaic": {"enabled": enabled, "box": bbox, "cell": cell}}


def preset_watermark(text: str = "© 演播室合成", style: str = "diag",
                     opacity: float = 0.25, size: float = 0.06) -> Dict[str, Any]:
    return {"watermark": {"text": text, "style": style,
                          "opacity": opacity, "size": size}}


def preset_remove_object(bbox, every=1) -> Dict[str, Any]:
    """圈选删除+绿幕/原背景填补 (按 bbox 区域删除并填补)。"""
    return {"remove": {"enabled": True, "box": bbox, "every": every}}


def preset_pretty(filter_key: str = "warm", fx_list: list = None) -> Dict[str, Any]:
    """一键美化: 暖色温 + 贴纸。"""
    cfg = {"filter": filter_key, "fx": fx_list or [{"effect": "heart", "params": {"frac": 0.12}}]}
    return cfg


def combine(*cfgs: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个 preset (后置字段覆盖前者)。"""
    out = {}
    for c in cfgs:
        for k, v in c.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


# ---------- 管线预设 -> 实际处理
def _make_apply_fn(cfg: Dict[str, Any], stable: bool = True, **stab_kwargs):
    """根据 cfg 生成 (frame_rgb, idx) -> frame_rgb 处理器。
    stable=True: 启用时序一致性 (键控参考色 EMA + 亮度时间低通), 默认开。"""
    use_stable = stable and (cfg.get("green", {}).get("enabled") or
                              cfg.get("filter") in (None,) and False)
    # 只要含绿幕或滤镜就启用时序稳定 (低成本)
    if stable and (cfg.get("green", {}).get("enabled") or cfg.get("filter")):
        state = temporal.build_stabilized_processor(cfg, **stab_kwargs)
        def fn(rgb, idx):
            return temporal.stabilized_frame_apply(rgb, idx, cfg, state)
        return fn
    return frame_engine.make_frame_fn(cfg)


def process(cfg: Dict[str, Any], in_video: str, out_video: str,
            stable: bool = True, stabilizer_lr: float = 0.06,
            smoother_strength: float = 0.15, progress_cb=None,
            codec: str = "VP90"):
    """完整管线入口: 逐帧处理视频, 写回。
    stable: 启用时序一致性 (默认开)。
    """
    fn = _make_apply_fn(cfg, stable=stable, stabilizer_lr=stabilizer_lr,
                          smoother_strength=smoother_strength)
    return video_io.process_video(in_video, out_video, fn,
                                   progress_cb=progress_cb, codec=codec)


# ---------- 快捷 demo
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="视频管线 v1")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--bg", default="data/ai_generated/studio_bg/bg_02_interview.png")
    ap.add_argument("--filter", default=None, help="全视频滤镜 (如 teal_orange)")
    ap.add_argument("--fx", default=None, help="特效 (heart/mosaic/bokeh...)")
    ap.add_argument("--wm", default=None, help="水印文字 (空=无)")
    args = ap.parse_args()

    cfg = {}
    if args.bg: cfg = combine(cfg, preset_replace_background(args.bg))
    if args.filter: cfg = combine(cfg, preset_apply_filter(args.filter))
    if args.fx: cfg = combine(cfg, **{"fx":[{"effect":args.fx,"params":{"frac":0.15}}]})
    if args.wm: cfg = combine(cfg, preset_watermark(args.wm))
    def pc(i,t): print(f"  [{i}/{t}]",flush=True) if i%10==0 or i==t else None
    r = process(cfg, args.input, args.output, progress_cb=pc)
    print("done:", r)
