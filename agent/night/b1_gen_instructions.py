# -*- coding: utf-8 -*-
"""
b1_gen_instructions.py — [夜间 B1] 生成 100 条指令测试集 (含期望工具序列标注)
模板组合覆盖: 8 工具 / 完整链 / 单特效 / 绿幕实景 / 多轮指代。确定性生成(可复现, seed 固定)。
产出: agent/night/out/instructions_100.json
"""
from __future__ import annotations
import json, random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

FULL_CHAIN = ["T01_matting", "T02_background_generate", "T03_lighting_estimate",
              "T04_relight", "T05_shadow_generate", "T06_harmonize"]

BGS = [("访谈", "访谈"), ("新闻LED", "新闻"), ("全景", "全景"), ("综艺", "综艺"),
       ("播客", "播客"), ("天气", "天气")]
FGS = ["fg_01_anchor_male.png", "fg_02_anchor_female.png", "fg_03_glasses.png", "fg_04_hair.png"]
FULL_TPL = [
    "把 {fg} 抠图放到{bg_cn}背景里",
    "将 {fg} 合成到{bg_cn}演播室",
    "{fg} 换背景到{bg_cn}",
    "把 {fg} 放进{bg_cn}场景, 要求光影一致",
]
FXS = [("depth_blur", "景深虚化"), ("sharpen", "锐化"), ("grain", "胶片颗粒")]
FX_TPL = [
    "给这张图加{fx_cn}",
    "对当前画面做{fx_cn}处理",
    "增强这张图: {fx_cn}",
]
GS_TPL = [
    "这张演播室照片只要把绿幕换成{bg_cn}背景, 桌子和话筒都要保留",
    "绿幕区域替换成{bg_cn}, 保留所有实物",
]


def gen(n_target: int = 100, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []

    def add(text, tools, **kw):
        items.append({"id": f"i{len(items)+1:03d}", "text": text,
                      "expect": {"tools": tools, **kw}})

    # 1) 完整链 (36 条): 6 背景 × 4 人像 × 模板轮换
    for i in range(36):
        bg_key, bg_cn = BGS[i % 6]
        fg = FGS[i % 4]
        text = FULL_TPL[(i // 6) % len(FULL_TPL)].format(fg=fg, bg_cn=bg_cn)
        add(text, FULL_CHAIN, chain="full", bg=bg_key, fg=fg)

    # 2) 绿幕实景替换 (12 条)
    for i in range(12):
        bg_key, bg_cn = BGS[i % 6]
        add(GS_TPL[i % 2].format(bg_cn=bg_cn),
            ["T02_background_generate", "T06_harmonize"],
            chain="greenscreen", bg=bg_key)

    # 3) 单特效 (24 条): 3 模式 × 8 模板
    for i in range(24):
        fx_key, fx_cn = FXS[i % 3]
        add(FX_TPL[i % 3].format(fx_cn=fx_cn), ["T07_enhance"],
            chain="fx", fx=fx_key)

    # 4) 完整链 + 特效尾巴 (16 条)
    for i in range(16):
        bg_key, bg_cn = BGS[i % 6]
        fx_key, fx_cn = FXS[i % 3]
        text = FULL_TPL[i % len(FULL_TPL)].format(fg=FGS[i % 4], bg_cn=bg_cn) + f", 再加{fx_cn}"
        add(text, FULL_CHAIN + ["T07_enhance"], chain="full_fx", bg=bg_key, fx=fx_key)

    # 5) 多轮/指代 (12 条): 用"这张图/当前图"指代
    for i in range(12):
        bg_key, bg_cn = BGS[i % 6]
        ref = ["这张图", "当前图", "刚才那张"][i % 3]
        add(f"把{ref}里的人物抠出来换到{bg_cn}背景",
            FULL_CHAIN, chain="full", bg=bg_key, ref=True)

    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"i{i+1:03d}"
    return items[:n_target]


if __name__ == "__main__":
    items = gen(100)
    p = OUT / "instructions_100.json"
    p.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print("total:", len(items), "| chains:", dict(Counter(x["expect"]["chain"] for x in items)))
    print("wrote", p)
