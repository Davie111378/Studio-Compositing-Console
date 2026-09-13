# -*- coding: utf-8 -*-
"""
build_domain_corpus.py — Stage 2 领域适应: 演播室合成 Agent 领域语料构建
来源: ①T01-T08 工具契约  ②领域术语表  ③规划规则/约束集  ④背景语义体系(COCO-Stuff)
      ⑤拒识/降级规则  ⑥领域问答对 (QA, 供 SFT 混入)
输出: agent_training/data/domain_corpus.jsonl  {"type": "...", "text": "..."}
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
records = []


def add(type_: str, text: str):
    records.append({"type": type_, "text": text.strip()})


# ---------------- ① 工具契约 (来自 schema) ----------------
SCHEMA_DIR = ROOT / "agent" / "schema"
for p in sorted(SCHEMA_DIR.glob("T0*.json")):
    sc = json.loads(p.read_text(encoding="utf-8"))
    props = sc["input"].get("properties", {})
    outp = sc["output"].get("properties", {})
    lines = [f"{sc['id']} {sc['name']}（{sc.get('desc','')}）",
             "  输入参数: " + "; ".join(f"{k}({v.get('description', v.get('type',''))})"
                                      for k, v in props.items()),
             "  输出: " + ", ".join(outp.keys()),
             "  质量档位: " + json.dumps(sc.get("quality_tiers", {}), ensure_ascii=False)]
    errs = sc.get("errors", [])
    if errs:
        lines.append("  错误码: " + "; ".join(f"{e['code']}({e.get('retryable','') and '可重试' or '不可重试'})"
                                          for e in errs))
    add("tool_contract", "\n".join(lines))

# ---------------- ② 领域术语表 ----------------
TERMS = {
    "抠图 (Matting)": "把前景从背景中分离，输出 alpha 透明通道与前景图。本项目用 BiRefNet（自动）+ GrabCut（圈选）+ AlphaRefiner（残差精修）三档。",
    "绿幕 / 色度键 (Chroma Key)": "纯色（绿色）背景拍摄，通过颜色距离场把绿幕像素判为背景。本项目的 chroma_key_scene 从全图最绿颜色簇估计参考色，容忍光照渐变与褶皱。",
    "Despill 去溢色": "绿幕反射会让主体边缘泛绿，despill 把绿色通道压回 (R+B)/2，是丝滑边缘的关键。",
    "和谐化 (Harmonization)": "让前景与背景色调统一。v2 方法：Lab 空间 a/b 色度对齐 + 低频亮度迁移 + 高频细节保留 + FDR 防压黑 + 肤色保护（皮肤区保留 80% 原色度）。",
    "重打光 (Relighting)": "按背景光照方向与色温给前景重新打光。directional 方法：方向光渐变 + 色温（warm/cool）。",
    "接触阴影 (Contact Shadow)": "物体落地处的柔和阴影，由 alpha 高斯模糊 + 偏移 + 衰减程序化生成，是消除“悬浮感/贴纸感”的关键。",
    "FDR (前景细节保持率)": "sigma(输出前景区)/sigma(输入前景区)，∈[0.7,1.3] 合格；越界即“压黑/过增强”，一票否决。",
    "绿幕实景键控": "演播室实景照片（含桌台/话筒等实物）只替换绿幕像素、保留非绿实物，区别于人像抠图换整背景。",
    "执行计划 DAG": "Agent 规划器把指令编译为工具调用有向无环图，节点间用 $nN.key 引用产物，支持拓扑执行、失败重试与条件回滚。",
    "条件回滚": "“换回背景A保留光”：只重跑背景节点及其下游，光照类节点（T03/T04）复用缓存产物。",
    "拒识 (Abstention)": "指令超出领域（非图像合成）时礼貌拒绝，不硬套工具。",
    "质量档位": "draft（草稿, ≤10s）/ normal / fine（精修, ≤30s），映射到推理尺寸、精修开关与强度参数。",
    "混合域训练": "通用域+目标域数据混合微调 Refiner，避免灾难性遗忘；纯目标域微调会遗忘原域能力。",
}
for k, v in TERMS.items():
    add("term", f"术语【{k}】：{v}")

# ---------------- ③ 规划规则 / 约束 ----------------
RULES_TEXT = """规划规则：用户指令编译为执行计划 DAG（纯 JSON）。8 原子工具：
T01_matting 抠图（auto/interactive，fine 档带 AlphaRefiner 精修）；
T02_background_generate 背景（file 用户上传直通 / semantic 素材库语义检索 / prompt 文生图）；
T03_lighting_estimate 光照估计（输出 light_dir/色温/强度/SH9）；
T04_relight 重打光（light_dir 可引用 $nT03.light_dir）；
T05_shadow_generate 接触阴影（alpha+背景+方向）；
T06_harmonize 和谐化合成（fg+alpha+bg；mode=greenscreen 时只换绿幕保留实物）；
T07_enhance 增强/特效（景深/锐化/颗粒/暗角/色温/贴纸/水印）；
T08_export 导出。
默认完整链：T01→T02→T03→T04→T05→T06→(T07)→(T08)。上游产物用 $节点id.输出key 引用。"""
add("rule", RULES_TEXT)
add("rule", "约束集：单计划节点数 ≤10；禁环；draft 预算 10s / normal 20s / fine 30s；文生图每次 ≤1 张；动作空间仅限 8 工具白名单；OOD 指令（非图像合成）必须拒识输出 {\"abstain\": true, \"reason\": ...}。")
add("rule", "背景语义表：新闻/播报→新闻LED；访谈/沙发/书架→访谈；全景/城市/黄昏/落地窗→全景；天气→天气；综艺/舞台→综艺；播客/录音/霓虹→播客。")

# ---------------- ④ 背景语义体系 (COCO-Stuff stuff 类) ----------------
STUFF_CLASSES = ("sky 天空, trees 树, grass,草 地板, road 道路, wall 墙, building 建筑, window 窗, "
                 "ceiling 天花板, floor 地板, mountain 山, sea 海, river 河, sand 沙地, "
                 "clouds 云, snow 雪, water 水, sky-other 其他天空, wood 木头, rock 岩石, "
                 "field 田野, platform 站台, runway 跑道, bridge 桥, desk 桌, table 桌子, "
                 "floor-other 其他地面, bush 灌木, plant 植物, curtain 窗帘, screen 幕布")
add("taxonomy", f"COCO-Stuff 背景类别体系（可作背景语义）: {STUFF_CLASSES}。这些是场景级（stuff）类别，"
                "用于把自然语言里的场景描述映射到背景素材检索与生成。")
add("taxonomy", "COCO-Stuff 数据集: COCO 2017 图像（val2017 共 5000 张，带 5 条/图的说明文字）+ 171 类"
                "密集标注（80 things + 91 stuff）。用途：①真实背景素材库 ②文本→场景理解的训练/评测对。")

# ---------------- ⑤ 领域问答对 (QA, SFT 混入) ----------------
QA = [
    ("什么是 FDR？", "FDR 是前景细节保持率：sigma(输出前景)/sigma(输入前景)，合格区间 [0.7,1.3]。低于 0.7 是压黑，高于 1.3 是过增强，都会被一票否决并自动降 strength 重算。"),
    ("绿幕实景照片怎么换背景？", "用 T06_harmonize 且 mode=greenscreen：场景色度键只把绿幕像素判为背景并替换，桌台、话筒等非绿实物全部保留，边缘做 despill 去溢色与羽化。"),
    ("纯绿幕人像怎么换背景？", "完整链 T01 抠图（alpha+前景）→ T02 背景检索/生成 → T03 光照估计 → T04 重打光 → T05 接触阴影 → T06 和谐化合成，输出尺寸恒等于输入图。"),
    ("什么是贴纸感？怎么消除？", "“贴纸感”指前景像贴纸一样浮在背景上：光照方向矛盾、无接触阴影、色调不统一。消除手段：T03 估计背景光照 → T04 按其重打光 → T05 生成接触阴影 → T06 色彩和谐化。"),
    ("为什么需要混合域训练？", "纯目标域微调会让 Refiner 遗忘通用域能力（灾难性遗忘，实测原域 SAD 恶化 3.9 倍）；通用+目标混合训练既保留原域又能适配新域，实测原域还提升 28%。"),
    ("一个合成任务最长允许多久？", "草稿档 ≤10 秒、精修档 ≤30 秒（端到端）。超预算时规划器应降档（fine→draft）或拆分节点。"),
]
for q, a in QA:
    add("qa", f"问：{q}\n答：{a}")

# ---------------- 写出 ----------------
out_p = OUT / "domain_corpus.jsonl"
with out_p.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print("领域语料:", len(records), "条 |", dict(Counter(r["type"] for r in records)))
print("写出:", out_p)
