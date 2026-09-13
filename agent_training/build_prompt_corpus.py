# -*- coding: utf-8 -*-
"""
build_prompt_corpus.py — 四域大规模提示词库 (≥100,000 字) + SFT 数据扩充

四域: A 绿幕背景(整人换景 full-chain / 绿幕区域键控 GS) | B 抠图 | C 特效 | D 滤镜
加: E 多步组合链 | F 拒识负样本

产物:
  data/prompt_corpus/prompt_corpus.jsonl   全量语料 {id, domain, text, expect}
  data/prompt_corpus/corpus_stats.json     字数/域分布/句式覆盖统计
  agent_training/data/sft_train.jsonl      旧数据(备份) + 语料分层抽样 → 新训练集
  agent_training/data/sft_val.jsonl        同上

语料每条带 expect (工具序列/参数/背景语义), 可直接转 DAG → SFT messages。

运行: python agent_training/build_prompt_corpus.py [--target_chars 100000]
"""
from __future__ import annotations
import argparse, json, random, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent_training"))
from build_sft_data import SFT_SYSTEM, dag_of, full_chain_dag, sample  # noqa: E402

OUT_DIR = ROOT / "data" / "prompt_corpus"
DATA_DIR = ROOT / "agent_training" / "data"

rng = random.Random(20260910)

# ================================================================ 词汇表
BG_SEMANTIC = [
    # 6 演播室 (assets 语义)
    "新闻LED", "访谈", "全景", "天气", "综艺", "播客",
    # 24 语义实景 (T02 semantic 支持生成)
    "城市黄昏天际线", "星空夜景", "海边日落", "森林晨雾", "雪原极光", "沙漠戈壁",
    "古堡庭院", "赛博都市", "水下珊瑚", "月球表面", "牧场草原", "樱花林",
    "江南水乡", "雪山之巅", "热带雨林", "极地冰原", "复古咖啡馆", "现代图书馆",
    "工业风loft", "顶楼天台", "樱花铁道", "稻田黄昏", "瀑布山涧", "薰衣草花田",
]
BG_SOFT = ["尽量自然", "光照要一致", "要有接触阴影", "人物光影和背景匹配", "看起来真实一点",
           "要有电影感", "有景深效果", "色彩协调", "构图自然", "灯光氛围统一", "质感高级",
           "整体和谐不突兀"]
FG_IMAGES = ["fg_01_anchor_male.png", "fg_02_anchor_female.png",
             "fg_03_glasses.png", "fg_04_hair.png",
             "data/ai_generated/green_fg/fg_02_anchor_female.png",
             "data/ai_generated/green_fg/fg_01_anchor_male.png"]
FG_DESC = ["这张绿幕图", "当前这张图", "女主播的图", "男主播那张", "我这张人像",
           "这张主播照片", "图里的主持人", "这张素材图"]

# 域 A1: 整人换景句式 (full chain)
A1_PATTERNS = [
    "把{img}的背景换成{bg}",
    "帮我把{img}从绿幕换到{bg}",
    "{img}换个{bg}背景",
    "请把{img}的绿幕背景替换为{bg}",
    "我想让{img}出现在{bg}",
    "{img}合成到{bg}, {soft}",
    "把{img}放到{bg}里, {soft}",
    "{img}换背景, 目标场景是{bg}",
    "麻烦将{img}的绿幕抠掉, 换成{bg}",
    "{img}→{bg}, {soft}",
    "帮{img}换个场景: {bg}",
    "{img}这张绿幕人像, 背景改成{bg}",
    "给{img}换个{bg}当背景, {soft}",
    "把{img}合成到{bg}场景中, {soft}",
    "{img}需要出现在{bg}, 帮我合成一下",
    "将{img}抠出并放置到{bg}",
    "{img}的背景替换成{bg}, 要自然",
    "把{img}与{bg}合成, 注意光照一致",
    "帮我把{img}置身于{bg}",
    "{img}转场到{bg}",
    "{img}挪到{bg}里去",
    "帮我做一个{bg}版本的{img}",
]
# 域 A2: 绿幕区域键控句式 (保留实物, GS 两步)
A2_PATTERNS = [
    "{img}只把绿幕换成{bg}, 桌台话筒都保留",
    "{img}的绿幕区域替换为{bg}, 实物别动",
    "保留{img}里的桌子和设备, 只把绿幕部分换成{bg}",
    "{img}做虚拟演播室键控: 绿幕→{bg}",
    "{img}只替换绿幕部分为{bg}",
    "{img}的绿幕抠掉换成{bg}, 其他全保留",
    "对{img}做绿幕键控, 背景换成{bg}, 台子留着",
    "{img}: 绿幕区域→{bg}, 实物保留",
    "只把{img}的绿幕像素换成{bg}",
    "{img}键控换景: {bg}, 保留一切实物",
    "{img}的绿幕改成{bg}, 不要抠掉实物",
    "虚拟背景: {img}的绿幕换成{bg}",
    "{img}局部换景, 绿幕→{bg}, 东西都留着",
    "把{img}绿幕部分替换为{bg}, 保持其余不变",
    "{img}中的绿幕区域用{bg}替换",
    "{img}换成{bg}背景, 但桌台话筒灯架都保留",
    "对{img}执行绿幕区域替换, 目标{bg}",
    "{img}绿幕像素替换为{bg}, 其他像素不动",
]
A2_IMGS = ["这张演播室照片", "当前绿幕实景图", "这张含绿幕的演播室图", "那张带桌台的照片",
           "这张演播室素材"]

# 域 B: 抠图
B_ENGINES = ["", "用色度键", "用精修引擎", "用最强模型", "自动选择最佳引擎", "用混合模式"]
B_PATTERNS = [
    "把{img}抠出来",
    "帮{img}抠图, {eng}",
    "{img}抠成透明底",
    "将{img}中的人像抠出, {eng}",
    "{img}做抠图处理",
    "把{img}的前景分离出来",
    "{img}抠图, 只要人",
    "对{img}执行抠图, {eng}",
    "{img}抠出带alpha的版本",
    "帮我把{img}里的人物抠下来",
    "{img}去背景, 输出透明PNG",
    "把{img}抠成无背景版本",
    "{img}需要抠图, {eng}",
    "分离{img}的前景与背景",
    "{img}抠像处理, {eng}",
    "抠出{img}的人像部分",
    "把{img}中的人物完整抠出, 保留发丝细节",
    "{img}抠图输出fg和alpha",
    "对{img}做前景提取",
    "{img}去掉绿幕, 给我透明底结果",
]
B_OBJECTS = ["这张人像图", "fg_03_glasses.png", "当前这张主播图", "这张绿幕照片",
             "我上传的图", "那张男主播", "图片里的女士", "画面中的人物"]

# 域 C: 特效 (T07)
C_FX = [
    ("depth_blur", ["加一点景深虚化", "背景虚化一些", "做出景深效果", "虚化程度明显一点",
                    "加个背景模糊", "把背景做虚", "景深感强一点", "像单反一样的虚化"]),
    ("sharpen", ["画面锐化一下", "把图变锐利", "提高清晰度", "锐化处理",
                 "画面不够清晰, 锐化", "边缘更锐一点", "整体锐一点", "细节加强锐化"]),
    ("grain", ["加点胶片颗粒", "做出老电影颗粒感", "加一点颗粒噪点", "胶片质感处理",
               "颗粒感重一点", "复古颗粒", "加上胶片噪点", "做出胶片味道"]),
    ("vignette", ["四周加暗角", "加个暗角氛围", "四角压暗一些", "边缘暗化处理",
                  "做点暗角效果", "加暗角, 电影感", "周围压暗", "边角淡出效果"]),
    ("sticker:heart", ["加个爱心贴纸", "贴一颗爱心", "加上爱心", "来个心形贴纸",
                       "点缀几颗爱心", "爱心贴纸安排上", "放个红心", "加个love贴纸"]),
    ("sticker:crown", ["加个皇冠贴纸", "头上来个皇冠", "贴个王冠", "加皇冠",
                       "给她戴个皇冠贴纸", "皇冠安排", "加个国王皇冠", "放上王冠贴纸"]),
    ("sticker:star", ["加个星星贴纸", "贴几颗星星", "加上星光", "右上角加星星",
                      "来点星星点缀", "星空贴纸", "加闪星", "放几颗五角星"]),
    ("sticker:flower", ["加朵花贴纸", "贴一朵小花", "加上花朵", "樱花贴纸来一个",
                        "放几朵花", "花贴纸点缀", "加个花朵装饰", "贴花"]),
]
C_SOFT = ["", "轻一点", "明显一些", "效果强一点", "适度就好", "夸张一点", "自然一些", "力度适中"]
C_WM_TEXTS = ["内部资料", "演播室专用", "样片预览", "版权所有", "仅供内部使用",
              "PROTOTYPE", "演示版", "请勿外传", "制作中", "CHROMA STUDIO"]
C_WM = ["给{img}打上文字水印：{t}", "{img}加水印, 文字是{t}", "在{img}上写水印{t}",
        "帮{img}加个「{t}」水印", "{img}右下角水印: {t}", "水印文字{t}, 加到{img}上",
        "{img}需要水印保护, 内容{t}", "给{img}盖上{t}水印"]
C_IMGS = ["这张图", "当前图", "刚才的成片", "这张照片", "fg_02_anchor_female.png",
          "上面那张", "处理后的图", "这张素材"]

# 域 D: 滤镜 (warm/cool 口语化全覆盖)
D_WARM = ["调成暖色调", "画面偏暖一点", "调暖色温", "做成日落暖光感觉", "暖色滤镜",
          "加一点暖调, 黄昏感", "让画面暖起来", "温馨暖光滤镜", "橙色暖调处理",
          "夕阳色温", "暖黄滤镜", "给画面加点温度", "暖化处理", "暖暖的氛围感",
          "做成金色时刻的感觉", "暖色调, 咖啡馆感觉", "来点暖阳滤镜", "整体调暖",
          "光线暖一些", "冬季壁炉暖光感", "加暖色, 像傍晚", "暖色氛围灯感觉",
          "暖光滤镜安排", "调出午后阳光的暖", "偏暖处理"]
D_COOL = ["调成冷色调", "画面偏冷一点", "调冷色温", "冬季清冷感觉", "冷色滤镜",
          "加一点冷调, 极地感", "让画面冷静下来", "冰蓝滤镜", "青色调处理",
          "清晨冷光", "冷白滤镜", "给画面降温", "冷淡风处理", "高冷氛围感",
          "做成北极光的冷感", "冷色调, 科技感", "来点深海蓝滤镜", "整体调冷",
          "色调冷一些", "月夜清辉感", "加冷色, 像凌晨", "冷色科技蓝感觉",
          "冷光滤镜安排", "调出冰岛的冷", "偏冷处理"]
D_DEG = ["", "轻微", "适度", "明显", "强烈", "一点点", "大幅", "稍微"]

# 域 E: 多步链
E_FULL_FX = [
    "{img}换到{bg}, 然后加{fx}",
    "{img}合成{bg}, 再来点{fx}",
    "把{img}放到{bg}, 最后{fx}",
    "{img}: 换{bg} + {fx}",
    "{img}先换{bg}背景, 再{fx}",
    "{img}→{bg}, 完成后{fx}",
    "给{img}换{bg}并{fx}",
    "{img}换景{bg}, 收尾加{fx}",
]
E_GS_FX = [
    "{img}绿幕换成{bg}(保留实物), 然后{fx}",
    "{img}键控到{bg}, 再{fx}",
    "{img}只换绿幕为{bg}, 完事后{fx}",
]
E_DOUBLE_FX = [
    "{img}先{fx1}, 再{fx2}",
    "{img}: {fx1} + {fx2}",
    "给{img}加{fx1}, 然后{fx2}",
    "{img}先做{fx1}处理, 接着{fx2}",
]
FX_CN = {"depth_blur": "景深虚化", "sharpen": "锐化", "grain": "胶片颗粒",
         "vignette": "暗角", "sticker:heart": "爱心贴纸", "sticker:crown": "皇冠贴纸",
         "sticker:star": "星星贴纸", "sticker:flower": "花朵贴纸"}

# 域 F: 拒识主题 × 句式
F_TOPICS = ["今天天气怎么样", "帮我写一个python爬虫", "讲个笑话", "推荐一部电影",
            "1+1等于几", "翻译成英文：你好", "明天股票会涨吗", "帮我订个闹钟",
            "写一首关于秋天的诗", "北京有哪些景点", "怎么减肥最快", "帮我回一封邮件",
            "世界杯什么时候开赛", "解释一下量子力学", "附近有什么好吃的", "帮我算一下房贷",
            "推荐一本好书", "如何学英语", "孙悟空是谁", "帮我生成随机密码",
            "现在几点了", "红烧肉怎么做", "帮我修电脑", "下一个假期是什么时候",
            "帮我起个名字", "这段代码有bug", "地球到月球多远", "帮我写辞职信",
            "帮我规划旅游路线", "防脱发的办法", "帮我看看合同", "做一道数学题",
            "帮我做个PPT", "今天限行尾号", "苹果醋有什么好处", "帮我买机票",
            "写个周报", "帮我查快递", "帮我记账", "讲讲历史故事",
            "怎么注册公司", "帮我练口语", "帮我挑礼物", "股票怎么开户",
            "明天穿什么", "帮我做PPT动画", "帮我下载音乐", "教我开车",
            "帮我算汇率", "帮我写情书"]
F_PATTERNS = ["{t}", "麻烦{t}", "请问{t}", "帮我{t}", "你能{t}吗", "我想{t}",
              "帮个忙: {t}", "顺便{t}", "{t}?", "{t}！", "对了,{t}", "另外{t}"]

# 域 G: 删主体·保留背景 (T01 keep=background 单节点) —— 与"抠出人物"语义相反, 高频易错
G_SUBJ = ["图中人物", "画面里的人物", "照片里的人物", "图中的主播", "图里的主持人",
          "人物主体", "画面中的人", "图片里的人", "前景人物", "这位主播", "两个主持人",
          "人物", "人像", "主体人物"]
G_PATTERNS = [
    "扣去{s}", "把{s}去掉", "移除{s}", "删除{s}", "帮我把{s}P掉", "把{s}抹掉",
    "{s}不要了", "去掉{s}, 只留背景", "扣掉{s}, 保留演播室", "把{s}擦掉",
    "处理掉{s}, 只要背景", "{s}删掉, 背景留着", "去掉{s}, 让背景露出来",
    "帮我把{s}抠掉, 只要绿幕背景", "把{s}移除, 保留原背景", "抹掉{s}, 不要动背景",
    "{s}太碍事了, 去掉", "去掉{s}并保留演播室场景", "把{s}从画面里拿掉",
]
G_TAIL = ["", ", 只留背景", ", 保留演播室场景", ", 背景要原样保留",
          ", 只保留绿幕部分", ", 其他都不要动", ", 只要背景"]

ALL_DOMAINS = ["A_green_bg", "A2_gs_keying", "B_matting", "C_fx", "D_filter",
               "E_multi", "F_abstain", "G_remove_subject", "H_green_key"]


# ================================================================ 生成器
_dom_n = Counter()          # 每域已收条数 (cap 按域计, 不用全局)


def _dedup_add(out: list, seen: set, domain: str, text: str, expect: dict, cap: int) -> bool:
    if _dom_n[domain] >= cap:
        return False
    t = " ".join(text.split())
    if t in seen:
        return False
    seen.add(t)
    out.append({"id": f"{domain[:2]}{len(out):06d}", "domain": domain,
                "text": t, "expect": expect})
    _dom_n[domain] += 1
    return True


def gen_A1(out, seen, cap):
    """绿幕背景·整人换景 (full chain)。"""
    n = 0
    for bg in BG_SEMANTIC:
        for pat in A1_PATTERNS:
            for soft in rng.sample(BG_SOFT, 4):
                img = rng.choice(FG_IMAGES)
                text = pat.format(img=img, bg=bg, soft=soft)
                _dedup_add(out, seen, "A_green_bg", text,
                           {"tools": "FULL", "bg": bg}, cap)
                n += 1
                if n >= cap * 3:
                    return


def gen_A2(out, seen, cap):
    """绿幕区域键控 (GS 两步, 保留实物)。"""
    bgs = ["新闻LED", "访谈", "全景", "天气", "综艺", "播客",
           "城市夜景", "科技大屏", "复古剧场", "晨间新闻台"]
    for bg in bgs:
        for pat in A2_PATTERNS:
            for img in A2_IMGS:
                for soft in rng.sample(BG_SOFT, 2):
                    text = pat.format(img=img, bg=bg) + (f", {soft}" if rng.random() < 0.5 else "")
                    _dedup_add(out, seen, "A2_gs_keying", text,
                               {"tools": "GS", "bg": bg}, cap)


def gen_B(out, seen, cap):
    """抠图单步。"""
    objs = sorted(set(B_OBJECTS + FG_DESC))
    for eng in B_ENGINES:
        for pat in B_PATTERNS:
            for obj in objs:
                _dedup_add(out, seen, "B_matting", pat.format(img=obj, eng=eng),
                           {"tools": ["T01_matting"]}, cap)


def _fx_key(k: str):
    return k.split(":")[0] if ":" not in k else "sticker"


def gen_C(out, seen, cap):
    """特效单步 (贴纸/水印/虚化/锐化/颗粒/暗角)。"""
    for fx, pats in C_FX:
        key = fx.split(":")[1] if ":" in fx else fx
        for pat in pats:
            for soft in C_SOFT:
                for img in C_IMGS:
                    text = f"{pat}({soft}) — {img}" if soft else f"{pat} — {img}"
                    if fx.startswith("sticker:"):
                        exp = {"tools": ["T07_enhance"], "fx": "sticker", "sticker": key}
                    else:
                        exp = {"tools": ["T07_enhance"], "fx": key}
                    _dedup_add(out, seen, "C_fx", text, exp, cap)
    for t in C_WM_TEXTS:
        for pat in C_WM:
            for img in C_IMGS:
                _dedup_add(out, seen, "C_fx", pat.format(img=img, t=t),
                           {"tools": ["T07_enhance"], "fx": "watermark_text", "text": t}, cap)


def gen_D(out, seen, cap):
    """整体滤镜 (warm/cool)。"""
    for sents, temp in ((D_WARM, "warm"), (D_COOL, "cool")):
        for s in sents:
            for deg in D_DEG:
                for img in C_IMGS:
                    text = f"{s}{deg} — {img}" if deg else f"{s} — {img}"
                    _dedup_add(out, seen, "D_filter", text,
                               {"tools": ["T07_enhance"], "fx": temp}, cap)


def gen_E(out, seen, cap):
    """多步链 (换景+特效 / 键控+特效 / 双特效)。"""
    fx_keys = [k for k, _ in C_FX]
    for bg in rng.sample(BG_SEMANTIC, 14):
        for pat in E_FULL_FX:
            fx = rng.choice(fx_keys)
            img = rng.choice(FG_IMAGES)
            exp = {"tools": "FULL+T07", "bg": bg, "fx": _fx_key(fx)}
            if fx.startswith("sticker:"):
                exp.update({"fx": "sticker", "sticker": fx.split(":")[1]})
            _dedup_add(out, seen, "E_multi", pat.format(img=img, bg=bg, fx=FX_CN[fx]), exp, cap)
    for bg in ["新闻LED", "综艺", "全景", "播客"]:
        for pat in E_GS_FX:
            fx = rng.choice(fx_keys)
            exp = {"tools": "GS+T07", "bg": bg, "fx": _fx_key(fx)}
            if fx.startswith("sticker:"):
                exp.update({"fx": "sticker", "sticker": fx.split(":")[1]})
            _dedup_add(out, seen, "E_multi", pat.format(img=rng.choice(A2_IMGS), bg=bg, fx=FX_CN[fx]), exp, cap)
    for pat in E_DOUBLE_FX:
        for _ in range(60):
            f1, f2 = rng.sample(fx_keys, 2)
            exp = {"tools": ["T07_enhance"], "fx": _fx_key(f1), "fx2": _fx_key(f2)}
            if f1.startswith("sticker:"):
                exp.update({"fx": "sticker", "sticker": f1.split(":")[1]})
            if f2.startswith("sticker:"):
                exp.update({"fx2": "sticker", "sticker2": f2.split(":")[1]})
            _dedup_add(out, seen, "E_multi",
                       pat.format(img=rng.choice(C_IMGS), fx1=FX_CN[f1], fx2=FX_CN[f2]), exp, cap)


def gen_F(out, seen, cap):
    """拒识负样本。"""
    for t in F_TOPICS:
        for pat in F_PATTERNS:
            _dedup_add(out, seen, "F_abstain", pat.format(t=t),
                       {"abstain": True}, cap)


def gen_G(out, seen, cap):
    """删主体·保留背景 (T01 keep=background 单节点)。"""
    for subj in G_SUBJ:
        for pat in G_PATTERNS:
            for tail in rng.sample(G_TAIL, 3):
                text = pat.format(s=subj) + tail
                _dedup_add(out, seen, "G_remove_subject", text,
                           {"tools": ["T01_matting"], "keep": "background"}, cap)


# ---- H 域: 色度键直合成 (T02 mode=green_key 单节点) ----
H_PATTERNS = [
    "用绿幕键控把{img}合成到{bg}背景上",
    "把{img}色度键抠干净后放到{bg}演播室",
    "{img}绿边太重了，用键控重新合成到{bg}背景",
    "帮我做绿幕合成：{img} + {bg}背景",
    "用 chroma key 把{img}抠出来叠到{bg}上",
    "{img}直接键控合成到{bg}，人物位置我指定",
    "把{img}精细抠图后放到{bg}背景，绿幕要去干净",
    "用色度键处理{img}，背景换成{bg}",
    "{img}抠图合成到{bg}，要求边缘不留绿边",
    "把{img}从绿幕里键控出来，合成到{bg}背景里",
    "{img}用绿幕合成方式换{bg}背景，一步到位",
    "帮我把{img}的绿幕键掉，放到{bg}场景中",
]
H_TAIL = ["", "，人物居中", "，缩小到0.5居中", "，人物放在左下角",
          "，要精确控制人物位置", "，顺便加个滤镜", "，人物贴底居中"]


def gen_H(out, seen, cap):
    """色度键直合成 (T02 mode=green_key 单节点)。"""
    for bg in BG_SEMANTIC:
        for pat in H_PATTERNS:
            for img in rng.sample(FG_IMAGES, 2):
                tail = rng.choice(H_TAIL)
                text = pat.format(img=img, bg=bg) + tail
                exp = {"tools": ["T02_background_generate"], "green_key": True, "bg": bg}
                _dedup_add(out, seen, "H_green_key", text, exp, cap)


# ================================================================ expect → DAG
def expect_to_dag(text: str, expect: dict):
    if expect.get("abstain"):
        return {"abstain": True, "reason": "未识别到图像合成意图 (抠图/换背景/滤镜/光影等), 已拒识"}
    tools = expect["tools"]
    bg = expect.get("bg")
    fx = expect.get("fx")
    img = None
    for cand in FG_IMAGES:
        if cand in text:
            img = f"data/ai_generated/green_fg/{Path(cand).name}"
            break
    img = img or rng.choice([
        "data/ai_generated/green_fg/fg_01_anchor_male.png",
        "data/ai_generated/green_fg/fg_02_anchor_female.png"])

    def t07_node(nid, dep, mode=None, sticker=None, text_=None):
        params = {"image_path": "$cur"}
        if mode:
            params["mode"] = mode
        if sticker:
            params.update({"mode": "sticker", "sticker": sticker})
        if text_:
            params.update({"mode": "watermark_text", "text": text_})
        return {"id": nid, "tool": "T07_enhance", "params": params, "depends_on": dep}

    if tools == "FULL":
        return full_chain_dag(text, img, bg)
    if isinstance(tools, list) and tools == ["T01_matting"] and expect.get("keep") == "background":
        # 删主体·保留背景: 单节点 T01, keep=background
        return dag_of(text, [{"id": "n1", "tool": "T01_matting",
                              "params": {"image": img, "keep": "background", "quality": "draft"},
                              "depends_on": []}], ["n1"])
    if expect.get("green_key"):
        # 色度键直合成: 单节点 T02, mode=green_key
        params = {"mode": "green_key", "app_fg": img}
        if bg:
            params["semantic"] = bg
        if any(k in text for k in ("视频", "mp4", "片子")):
            params["media_type"] = "video"
        return dag_of(text, [{"id": "n1", "tool": "T02_background_generate",
                              "params": params, "depends_on": []}], ["n1"])
    if tools == "FULL+T07":
        if fx == "sticker":
            return full_chain_dag(text, img, bg, fx_mode="sticker", fx_is_sticker=True,
                                  sticker=expect.get("sticker", "heart"))
        return full_chain_dag(text, img, bg, fx_mode=fx or "depth_blur")
    if tools == "GS":
        nodes = [
            {"id": "n1", "tool": "T02_background_generate",
             "params": {"semantic": bg, "quality": "draft"}, "depends_on": []},
            {"id": "n2", "tool": "T06_harmonize",
             "params": {"fg_path": "$cur", "alpha_path": "$cur", "bg_path": "$n1.bg_path",
                        "mode": "greenscreen"}, "depends_on": ["n1"]}]
        return dag_of(text, nodes, ["n2"])
    if tools == "GS+T07":
        nodes = [
            {"id": "n1", "tool": "T02_background_generate",
             "params": {"semantic": bg, "quality": "draft"}, "depends_on": []},
            {"id": "n2", "tool": "T06_harmonize",
             "params": {"fg_path": "$cur", "alpha_path": "$cur", "bg_path": "$n1.bg_path",
                        "mode": "greenscreen"}, "depends_on": ["n1"]},
            t07_node("n3", ["n2"], mode=fx if fx != "sticker" else None,
                     sticker=expect.get("sticker") if fx == "sticker" else None,
                     text_=expect.get("text") if fx == "watermark_text" else None)]
        if fx == "sticker":
            nodes[-1]["params"] = {"image_path": "$n2.composite_path",
                                   "mode": "sticker", "sticker": expect.get("sticker", "heart")}
        elif fx == "watermark_text":
            nodes[-1]["params"] = {"image_path": "$n2.composite_path",
                                   "mode": "watermark_text", "text": expect.get("text", "水印")}
        else:
            nodes[-1]["params"] = {"image_path": "$n2.composite_path", "mode": fx}
        return dag_of(text, nodes, ["n3"])
    if tools == ["T01_matting"]:
        return dag_of(text, [{"id": "n1", "tool": "T01_matting",
                              "params": {"image": img, "quality": "draft"}, "depends_on": []}], ["n1"])
    if tools == ["T07_enhance"]:
        if fx == "sticker":
            node = t07_node("n1", [], sticker=expect.get("sticker", "heart"))
        elif fx == "watermark_text":
            node = t07_node("n1", [], text_=expect.get("text", "水印"))
        else:
            node = t07_node("n1", [], mode=fx or "depth_blur")
        if "$cur" in node["params"].get("image_path", ""):
            node["params"]["image_path"] = "$cur"
        return dag_of(text, [node], ["n1"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_chars", type=int, default=100000)
    ap.add_argument("--sft_sample", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    global rng
    rng = random.Random(a.seed)

    caps = {"A_green_bg": 2600, "A2_gs_keying": 1700, "B_matting": 1400,
            "C_fx": 2100, "D_filter": 2000, "E_multi": 680, "F_abstain": 600,
            "G_remove_subject": 800, "H_green_key": 700}
    gens = {"A_green_bg": gen_A1, "A2_gs_keying": gen_A2, "B_matting": gen_B,
            "C_fx": gen_C, "D_filter": gen_D, "E_multi": gen_E, "F_abstain": gen_F,
            "G_remove_subject": gen_G, "H_green_key": gen_H}

    out, seen = [], set()
    for d in ALL_DOMAINS:
        gens[d](out, seen, caps[d])
        print(f"[corpus] {d}: 累计 {len(out)}", flush=True)

    # 字数达标检查 (不足则对 A/C/D 域再扩量)
    total_chars = sum(len(x["text"]) for x in out)
    if total_chars < a.target_chars:
        for d in ("A_green_bg", "C_fx", "D_filter"):
            gens[d](out, seen, caps[d] + 1200)
        total_chars = sum(len(x["text"]) for x in out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompt_corpus.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out), encoding="utf-8")

    # ---- 统计 ----
    dom_cnt = Counter(x["domain"] for x in out)
    stats = {
        "total_prompts": len(out),
        "total_chars": total_chars,
        "avg_chars": round(total_chars / max(len(out), 1), 1),
        "unique_texts": len(seen),
        "domain_dist": dict(dom_cnt),
        "target_chars": a.target_chars,
        "target_met": total_chars >= a.target_chars,
        "tool_dist": dict(Counter(
            ("FULL" if x["expect"].get("tools") == "FULL" else
             "FULL+T07" if x["expect"].get("tools") == "FULL+T07" else
             "GS" if x["expect"].get("tools") == "GS" else
             "GS+T07" if x["expect"].get("tools") == "GS+T07" else
             "abstain" if x["expect"].get("abstain") else
             "+".join(x["expect"].get("tools", [])))
            for x in out)),
    }
    (OUT_DIR / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ---- SFT 转换 + 分层抽样 ----
    sft = []
    for x in out:
        dag = expect_to_dag(x["text"], x["expect"])
        if dag is None:
            continue
        sft.append(sample(x["text"], dag))
    rng.shuffle(sft)
    n_val = max(100, int(len(sft) * 0.08))
    n_val = min(n_val, len(sft) // 5)
    new_val, new_train = sft[:n_val], sft[n_val:n_val + a.sft_sample]
    # 旧数据备份 (仅首次; 防重复运行膨胀) + 合并
    for name in ("sft_train.jsonl", "sft_val.jsonl"):
        p = DATA_DIR / name
        bak = DATA_DIR / f"{name}.v1.bak"
        if p.exists() and not bak.exists():
            shutil.copy(p, bak)
    old_train = [json.loads(x) for x in
                 (DATA_DIR / "sft_train.jsonl.v1.bak").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (DATA_DIR / "sft_train.jsonl.v1.bak").exists() else []
    old_val = [json.loads(x) for x in
               (DATA_DIR / "sft_val.jsonl.v1.bak").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (DATA_DIR / "sft_val.jsonl.v1.bak").exists() else []
    # 旧数据 messages 兼容 (F 域 QA 是 answer 型, 保留)
    merged_train = old_train + new_train
    merged_val = old_val + new_val
    rng.shuffle(merged_train)
    (DATA_DIR / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in merged_train), encoding="utf-8")
    (DATA_DIR / "sft_val.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in merged_val), encoding="utf-8")
    print(f"[sft] train {len(merged_train)} (旧{len(old_train)}+新{len(new_train)}) "
          f"| val {len(merged_val)} (旧{len(old_val)}+新{len(new_val)})")


if __name__ == "__main__":
    main()
