# -*- coding: utf-8 -*-
"""生成 planner 评测指令集 planner_eval_cases.jsonl（规范 §2.2A：>=100 条）。

分组：
  G1  抠图单步            G2  完整换背景链       G2E 换背景 + 细节增强
  G3  阴影局部修改        G3L 光线修改·无方向词   G3D 光线修改·显式方向
  G4  细节增强            G4H 和谐化小修正       G5  条件回滚（角色/版本/保留）
  G6  空间指代（点击/框选） G7  无图拒绝（E_INVALID_INPUT）
  G8  已知局限（诚实失败例：光影歧义目标 / 仅导出）

用法：python agent/eval/build_eval_set.py   （幂等，重写同内容）
"""
import json
from collections import Counter
from pathlib import Path

cases = []


def add(group, text, expect, prior=False, spatial=None):
    cases.append({"id": f"{group}-{len(cases) + 1:03d}", "group": group, "instruction": text,
                  "expect": expect, "prior": prior, "spatial": spatial})


FULL = ["matting", "background_generate", "lighting_estimate", "relight",
        "shadow_generate", "harmonize", "export"]
FULL_E = FULL[:-1] + ["enhance", "export"]

# ---- G1 抠图单步（无 prior）15 条 ----
for t in ["帮我把这张图抠出来", "人像抠图，要透明背景", "把主体抠出来", "抠像处理一下",
          "把照片里的人物抠取出来", "去背，输出透明底", "这张图需要抠图", "抠出画面中的模特",
          "帮我去背", "透明背景版本", "把图里的宠物抠出来", "对这张商品图抠图",
          "抠出人物并导出PNG", "抠取前景主体", "图片抠图处理"]:
    add("G1", t, {"tools": ["matting", "export"]})

# ---- G2 完整换背景链（无 prior）25 条 ----
scenes = ["傍晚的咖啡馆", "夕阳下的海边", "雪原", "夜晚的霓虹街头", "森林晨雾",
          "现代办公室", "赛博都市天台", "星空下的草原"]
verbs = ["把这个人放进", "把人物放到", "和", "合成到", "背景换成", "置于"]
for n, s in enumerate(scenes):
    tail = "，光从左边照过来" if n % 3 == 0 else ""
    add("G2", verbs[n % len(verbs)] + s + tail,
        {"tools": FULL, "param": {"tool": "background_generate", "key": "prompt", "contains": s[:4]}})
for t in ["换个场景，要黄昏的湖边，暖一点", "把人放到雪山之巅，光从右边来",
          "合成一张咖啡店内的画面，色调温暖", "背景换成海滩，夕阳感觉",
          "换成访谈场景背景，保持自然", "换个背景：江南水乡，冷色调",
          "把模特置于樱花林中，光从左侧打", "换背景为太空站内，冷光",
          "改成城市黄昏，暖光从左边", "放进图书馆内景，要有安静的氛围",
          "背景换成篝火营地，暖光", "放到清晨的稻田，光线清冷",
          "合成到古堡庭院，光从右上方来", "换个场景到瀑布边，湿润通透",
          "置于极地冰原，冷色", "把人物放进洞穴，光从背后逆光"]:
    add("G2", t, {"tools": FULL})

# ---- G2E 换背景 + 细节增强 5 条 ----
for t in ["把人物放进咖啡馆并加景深虚化，画面更高级", "背景换成雪原，整体精修一下",
          "合成到霓虹街头，最后锐化细节", "放到海边黄昏，做出景深效果",
          "换成森林背景，细节增强，高清导出"]:
    add("G2E", t, {"tools": FULL_E})

# ---- G3 阴影局部修改（prior）8 条 ----
for t in ["阴影轻一点", "影子太重了，淡一些", "阴影再浓一点", "把影子弄淡",
          "接触阴影深一点", "阴影少一点，自然些", "影子轻一些", "阴影重量适中偏重"]:
    exp = {"tools": ["shadow_generate", "harmonize", "export"]}
    if any(k in t for k in ("轻", "淡", "少")):
        exp["param"] = {"tool": "shadow_generate", "key": "shadow_strength", "equals": 0.3}
    elif any(k in t for k in ("重", "深", "浓")):
        exp["param"] = {"tool": "shadow_generate", "key": "shadow_strength", "equals": 0.9}
    add("G3", t, exp, prior=True)

# ---- G3L 光线局部修改·无方向词（prior）8 条 ----
for t in ["光线太冷了，暖一点", "画面偏暗，亮一点", "色调太暖，冷一些", "更亮一些",
          "暗一点，氛围感", "暖色调处理", "冷光感觉", "亮度提高一点"]:
    add("G3L", t, {"tools": ["lighting_estimate", "relight", "shadow_generate",
                             "harmonize", "export"]}, prior=True)

# ---- G3D 光线局部修改·显式方向（prior）8 条 ----
for t, az in [("光从左边来", 180.0), ("光从右边打", 0.0), ("头顶打光", 0.0), ("逆光效果", 90.0),
              ("左侧来光，暖一点", 180.0), ("右侧光，冷一些", 0.0), ("正上方光源", 0.0),
              ("背后打光", 90.0)]:
    add("G3D", t, {"tools": ["relight", "shadow_generate", "harmonize", "export"],
                   "param": {"tool": "relight", "key": "light_dir.azimuth", "equals": az}},
        prior=True)

# ---- G4 细节增强（prior）6 条 ----
for t in ["背景虚化一些", "画面锐化一下", "更清晰一点", "加点景深效果", "细节增强处理", "高清精修"]:
    add("G4", t, {"tools": ["enhance", "export"]}, prior=True)

# ---- G4H 和谐化小修正（prior）6 条 ----
for t in ["色彩再和谐一点", "前景背景融合得更自然", "色调协调一下", "更和谐统一",
          "融合感加强", "画面协调些"]:
    add("G4H", t, {"tools": ["harmonize", "export"]}, prior=True)

# ---- G5 条件回滚（prior）12 条：role / version（相对 versions）/ preserve ----
RB = [
    ("背景换回上一版，但是保留现在的光线", "background_generate", 2, ["lighting_estimate", "relight"]),
    ("背景换回第一版", "background_generate", 1, []),
    ("换回最初背景", "background_generate", 1, []),
    ("背景恢复到之前那版，保留现在的光影", "background_generate", 2, ["relight"]),
    ("回到上一版背景", "background_generate", 2, []),
    ("光照换回上一版", "lighting_estimate", 1, []),
    ("光线恢复到第一版", "lighting_estimate", 1, []),
    ("打光回退到上一版，保留背景", "relight", 1, ["background_generate"]),
    ("光影还原到最初，背景别动", "relight", 1, ["background_generate"]),
    ("抠图换回第一版", "matting", 1, []),
    ("和谐化回到上一版", "harmonize", 1, []),
    ("阴影恢复到之前那版", "shadow_generate", 1, []),
]
for t, role, ver, keep in RB:
    add("G5", t, {"kind": "rollback", "role": role, "version": ver, "preserve": keep}, prior=True)

# ---- G6 空间指代（无 prior）6 条 ----
G6 = [
    ("点击这里，把这个主体抠出来", {"click": {"x": 120, "y": 260}}),
    ("框选这个区域抠图", {"box": [10, 10, 200, 300]}),
    ("点击人物头部，抠取主体", {"click": {"x": 300, "y": 88}}),
    ("按框选范围去背", {"box": [0, 0, 500, 500]}),
    ("点一下这里抠像", {"click": {"x": 88, "y": 220}}),
    ("框出人物并抠图", {"box": [5, 5, 120, 220]}),
]
for t, sp in G6:
    add("G6", t, {"tools": ["matting", "export"],
                  "spatial_param": {"tool": "matting", "key": "mode", "equals": "trimap"}},
        spatial=sp)

# ---- G7 无图拒绝（无 asset，非抠图指令）6 条：期望 E_INVALID_INPUT ----
for t in ["把人物放进咖啡馆", "合成到雪山背景", "放到海边", "背景换成森林",
          "置于星空之下", "换个场景到草原"]:
    add("G7", t, {"error": "E_INVALID_INPUT"})

# ---- G8 已知局限（诚实失败例）：光/影歧义目标 / 仅导出 3 条 ----
add("G8", "把阴影回退到上一版，保留现在的光",
    {"kind": "rollback", "role": "shadow_generate", "version": 1,
     "preserve": ["relight"], "known_limit": True}, prior=True)
add("G8", "影子换回第一版，保留光照",
    {"kind": "rollback", "role": "shadow_generate", "version": 1,
     "preserve": ["lighting_estimate"], "known_limit": True}, prior=True)
add("G8", "只导出成 PNG，不要再改画面",
    {"tools": ["export"], "known_limit": True}, prior=True)


def main():
    out = Path(__file__).resolve().parent / "planner_eval_cases.jsonl"
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
    print("total:", len(cases), dict(Counter(c["group"] for c in cases)))


if __name__ == "__main__":
    main()
