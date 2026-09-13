# -*- coding: utf-8 -*-
"""
assets.py — Agent 可用的本地素材库索引。
每个素材含 path + 语义标签(cn描述) + 类别, 供规划器按用户语义挑选。
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 用户口语 → 素材库标签用词 (长词在前, 逐条 replace)
BG_ALIAS = [
    ("城市天际线", "全景"), ("城市夜景", "全景"), ("城市风光", "全景"), ("城市", "全景"),
    ("落地窗", "全景"), ("窗景", "全景"), ("天际线", "全景"), ("黄昏", "全景"), ("日落", "全景"),
    ("演播厅", "演播室"), ("新闻播报", "新闻"), ("播报", "新闻"), ("气象", "天气"),
    ("沙发", "访谈"), ("书架", "访谈"), ("对话", "访谈"),
    ("娱乐", "综艺"), ("舞台", "综艺"), ("录音", "播客"), ("电台", "播客"),
]

# 演播厅 / 背景
STUDIO_BG = [
    {"file": "data/ai_generated/studio_bg/bg_01_news_led.png",
     "cn": "新闻演播室: 弧形LED大屏播放蓝色科技画面, 深色主播台, 专业灯光", "cat": "news"},
    {"file": "data/ai_generated/studio_bg/bg_02_interview.png",
     "cn": "访谈/对话节目棚: 两张暖色扶手沙发+茶几+书架, 暖光温馨, 柔和背景虚化", "cat": "interview"},
    {"file": "data/ai_generated/studio_bg/bg_03_panorama.png",
     "cn": "全景落地窗演播室: 黄昏城市天际线, 紫橙日落光, 现代极简", "cat": "panorama"},
    {"file": "data/ai_generated/studio_bg/bg_04_weather.png",
     "cn": "天气预报演播室: 数字天气雷达大屏, 蓝白配色, 冷色调", "cat": "weather"},
    {"file": "data/ai_generated/studio_bg/bg_05_variety.png",
     "cn": "综艺综艺舞台: 多彩几何LED屏, 舞台桁架+烟雾+红金灯光, 娱乐氛围", "cat": "variety"},
    {"file": "data/ai_generated/studio_bg/bg_06_podcast.png",
     "cn": "播客直播间: 声学吸音板, 霓虹灯, 麦克风悬臂, 冷暖对比", "cat": "podcast"},
]

# 绿幕人像前景 (供合成/抠图测试)
GREEN_FG = [
    {"file": "data/ai_generated/green_fg/fg_01_anchor_male.png",
     "cn": "男新闻主播: 藏青西装领带, 半身照, 绿幕背景", "cat": "person", "gender": "male"},
    {"file": "data/ai_generated/green_fg/fg_02_anchor_female.png",
     "cn": "女主播: 酒红西装, 深发及肩有些散发, 半身坐姿(含桌面), 绿幕背景", "cat": "person", "gender": "female"},
    {"file": "data/ai_generated/green_fg/fg_03_glasses.png",
     "cn": "戴金属细框眼镜的年轻女性, 卷发, 卫衣, 绿幕背景", "cat": "person", "gender": "female"},
    {"file": "data/ai_generated/green_fg/fg_04_hair.png",
     "cn": "男长发飞扬(风吹散发丝), 深色夹克, 绿幕背景", "cat": "person", "gender": "male"},
]

# 难例 (透明/半透明/薄纱)
HARD_FG = [
    {"file": "data/ai_generated/green_fg_hard/hfg_01_frosted_glass.png",
     "cn": "磨砂玻璃后的男人(半透明透过玻璃), 绿幕背景", "cat": "semi"},
    {"file": "data/ai_generated/green_fg_hard/hfg_02_glass_pitcher.png",
     "cn": "商务女手持透明玻璃水壶(装水), 手透过玻璃可见, 绿幕背景", "cat": "transparent"},
    {"file": "data/ai_generated/green_fg_hard/hfg_03_sheer_shawl.png",
     "cn": "舞者长巾飞舞(半透明纱巾), 绿幕背景", "cat": "veil"},
]

ALL = {"bg": STUDIO_BG, "fg": GREEN_FG, "hard": HARD_FG}


# ---------------------------------------------------------------- COCO 素材 (B)
def _load_coco_assets() -> tuple[list, list]:
    """读 data/coco_assets/assets.json (由 training/scripts/coco_assets.py 生成)。"""
    import json
    p = ROOT / "data" / "coco_assets" / "assets.json"
    if not p.exists():
        return [], []
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    bg, fg = [], []
    for it in idx:
        rec = {"file": f"data/coco_assets/{it['file']}", "cn": it["cn"],
               "cat": it.get("cat", "")}
        (bg if it["kind"] == "bg" else fg).append(rec)
    return bg, fg


COCO_BG, COCO_FG = _load_coco_assets()



def list_backgrounds() -> str:
    """返回背景素材清单的文本(供 system prompt 注入)。"""
    lines = ["可用背景(供 background 参数):"]
    for i, b in enumerate(STUDIO_BG):
        lines.append(f"  {i}. {b['file']}  —— {b['cn']}")
    if COCO_BG:
        lines.append(f"  [实景背景库 {len(COCO_BG)} 张, 可用中文语义检索, 如 '实景/街道/室内/自然']:")
        for b in COCO_BG[:12]:
            lines.append(f"     - {b['file']}  —— {b['cn']}")
        if len(COCO_BG) > 12:
            lines.append(f"     ... 共 {len(COCO_BG)} 张 (list_assets kind=bg 查看全部)")
    return "\n".join(lines)


def list_persons() -> str:
    lines = ["可用绿幕人像前景(可作 source 输入):"]
    for b in GREEN_FG + HARD_FG:
        lines.append(f"  - {b['file']}  —— {b['cn']}")
    if COCO_FG:
        lines.append(f"  [实景前景库 {len(COCO_FG)} 个透明底物体, 按中文语义选, 如 '椅子/沙发/杯子/盆栽']")
    return "\n".join(lines)


def find_background(keyword: str) -> str | None:
    """按中文关键词/编号匹配背景 (计分制, 兼容组合词如"新闻LED")。
    整串命中=直接返回; 否则按 bigram 命中数计分, **且要求命中占比达标**才采纳,
    避免"城市背景"这种弱匹配误召回访谈棚 (幻觉背景的根因之一)。
    优先演播室素材, 其次 COCO 实景库 (带"实景/真实/照片"等词时优先实景)。"""
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return None
    if keyword.isdigit() and 0 <= int(keyword) < len(STUDIO_BG):
        return STUDIO_BG[int(keyword)]["file"]
    # 中文同义词归一: 用户常用词 → 素材库标签用词
    for src, dst in BG_ALIAS:
        keyword = keyword.replace(src, dst)
    want_real = any(w in keyword for w in ("实景", "真实", "照片", "现实", "街", "室内", "自然"))

    def _hays(b):
        return (b["cn"] + " " + b["cat"] + " " + Path(b["file"]).stem).lower()

    for b in STUDIO_BG:
        if keyword in _hays(b):
            return b["file"]
    pool = (COCO_BG if want_real else []) + STUDIO_BG + (COCO_BG if not want_real else [])
    best, best_score = None, 0.0
    n = max(1, len(keyword) - 1)
    for b in pool:
        hay = _hays(b)
        if keyword in hay:
            return b["file"]
        hits = sum(1 for i in range(len(keyword) - 1) if keyword[i:i + 2] in hay)
        ratio = hits / n
        if ratio > best_score:
            best, best_score = b["file"], ratio
    # 命中占比 >= 50% 才认为语义相关 (否则宁可不匹配 → 交给文生图/让 LLM 重规划)
    return best if best_score >= 0.5 else None


def find_coco_fg(keyword: str, k: int = 5) -> list[dict]:
    """按中文语义找 COCO 透明底前景素材。"""
    keyword = (keyword or "").strip()
    if not keyword:
        return COCO_FG[:k]
    scored = []
    for b in COCO_FG:
        hay = b["cn"] + " " + b["cat"]
        if keyword in hay:
            scored.append((100, b))
        else:
            hits = sum(1 for i in range(len(keyword) - 1) if keyword[i:i + 2] in hay)
            if hits:
                scored.append((hits, b))
    scored.sort(key=lambda x: -x[0])
    return [b for _, b in scored[:k]]


def all_background_paths() -> list[str]:
    return [b["file"] for b in STUDIO_BG] + [b["file"] for b in COCO_BG]


def describe_background(path: str) -> str:
    for b in STUDIO_BG + COCO_BG:
        if path.endswith(b["file"]):
            return b["cn"]
    return ""


if __name__ == "__main__":
    print(list_backgrounds())
    print(find_background("访谈"))
    print(find_background("3"))
