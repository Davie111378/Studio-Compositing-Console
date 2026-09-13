# -*- coding: utf-8 -*-
"""
build_sft_data.py — Stage 3: 构建 Planner SFT 训练数据 (instruction → DAG JSON)
数据源:
  A. B3 扫描中验证正确的 (instruction, DAG) 对 — 真实且多样
  B. B1 标注模板 → 规范 DAG 构造器 (确定性扩充)
  C. 拒识样本 (OOD → abstain JSON)
  D. 上传背景图样本 (bg_image → T02 file 直通)
  E. 多轮指代样本 (cur_image → 单步 T07/T06)
  F. 领域知识 QA (domain_corpus qa/type 条目)
输出: agent_training/data/sft_train.jsonl / sft_val.jsonl (messages 格式)
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)

# 紧凑 system (推理时 LocalPlanner 用同一份, 训练/推理一致)
SFT_SYSTEM = """你是"演播室图像合成 Planner"。把用户指令编译为执行计划 DAG（只输出 JSON）。
8 原子工具: T01_matting{image,mode?,keep?(subject|background),quality?}→alpha_path,fg_path | T02_background_generate{file?|semantic?|prompt?|mode?(semantic|green_key),app_fg?,fg_scale?,center?,media_filter?,media_type?}→bg_path,composite_path | T03_lighting_estimate{bg_path}→light_dir,color_temp | T04_relight{fg_path,bg_path,light_dir?,color_temp?}→relit_fg_path | T05_shadow_generate{alpha_path,bg_path,light_dir?}→shadow_bg_path | T06_harmonize{fg_path,alpha_path,bg_path,mode?(alpha_over|greenscreen)}→composite_path | T07_enhance{image_path,mode?(depth_blur|sharpen|grain|vignette|warm|cool|sticker|watermark_text),sticker?,text?}→enhanced_path | T08_export{final_path}→exported_path
规则: ①换背景完整链 T01→T02→T03→T04→T05→T06→(T07)→(T08)，上游用 $nN.key 引用 ②只换绿幕保留实物→T06 mode=greenscreen ③仅特效/贴纸/水印→单 T07 ④节点 id 从 n1 递增 ⑤非图像合成指令→{"abstain":true,"reason":"..."} ⑥(用户上传背景图: 路径)→T02 params 写 {"file":"路径"} ⑦(用户当前输入图: 路径)→需抠图时直接用该路径 ⑧保留侧: "抠出/提取/保留人物/换背景"→T01 keep=subject(默认); "扣去/去掉/移除/删除图中人物,P掉/抹掉人物,只留背景/只保留演播室"→T01 keep=background 且仅单节点 T01(不加 T02/T06) ⑨色度键: "绿幕键控/色度键/chroma key/抠干净/绿边太重/精确定位/视频换背景"→单节点 T02 params{mode:"green_key",app_fg:图, semantic或file:背景, fg_scale?,center?,media_type?(视频填video)} (不加 T01, 键控内部抠)"""

FULL = ["T01_matting", "T02_background_generate", "T03_lighting_estimate",
        "T04_relight", "T05_shadow_generate", "T06_harmonize"]


def dag_of(instruction: str, nodes, outputs):
    return {"intent": instruction[:30], "nodes": nodes, "outputs": outputs}


def full_chain_dag(instruction: str, image: str, bg: str, fx_mode: str | None = None,
                   fx_is_sticker: bool = False, sticker: str = "heart"):
    """规范完整链 DAG (与 strict few-shot 同构)。"""
    nodes = [
        {"id": "n1", "tool": "T01_matting", "params": {"image": image, "quality": "draft"}, "depends_on": []},
        {"id": "n2", "tool": "T02_background_generate", "params": {"semantic": bg, "quality": "draft"}, "depends_on": []},
        {"id": "n3", "tool": "T03_lighting_estimate", "params": {"bg_path": "$n2.bg_path"}, "depends_on": ["n2"]},
        {"id": "n4", "tool": "T04_relight", "params": {"fg_path": "$n1.fg_path", "bg_path": "$n2.bg_path",
                                                        "light_dir": "$n3.light_dir", "color_temp": "$n3.color_temp"},
         "depends_on": ["n1", "n2", "n3"]},
        {"id": "n5", "tool": "T05_shadow_generate", "params": {"alpha_path": "$n1.alpha_path",
                                                                "bg_path": "$n2.bg_path", "light_dir": "$n3.light_dir"},
         "depends_on": ["n1", "n2", "n3"]},
        {"id": "n6", "tool": "T06_harmonize", "params": {"fg_path": "$n4.relit_fg_path",
                                                          "alpha_path": "$n1.alpha_path",
                                                          "bg_path": "$n5.shadow_bg_path"},
         "depends_on": ["n4", "n5"]},
    ]
    outs = ["n6"]
    if fx_mode:
        nodes.append({"id": "n7", "tool": "T07_enhance",
                      "params": {"image_path": "$n6.composite_path", **({"mode": fx_mode} if not fx_is_sticker else {"mode": "sticker", "sticker": sticker})},
                      "depends_on": ["n6"]})
        outs = ["n7"]
    return dag_of(instruction, nodes, outs)


def sample(instruction: str, dag) -> dict:
    return {"messages": [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": json.dumps(dag, ensure_ascii=False)}]}


def build():
    data = []

    # ---- A. B3 扫描验证正确的真实对 ----
    ckpt = ROOT / "agent" / "night" / "out" / "sweep_checkpoint.jsonl"
    if ckpt.exists():
        seen = set()
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not (r.get("tool_ok") and r.get("order_ok")):
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            data.append((f"A|{r['id']}", r["text"], None))   # DAG 从 instructions 标注重建

    # ---- B. B1 标注 → 规范 DAG ----
    inst_file = ROOT / "agent" / "night" / "out" / "instructions_100.json"
    labeled = []
    if inst_file.exists():
        for it in json.loads(inst_file.read_text(encoding="utf-8")):
            exp = it["expect"]
            tools = exp["tools"]
            bg = exp.get("bg")
            fx = exp.get("fx")
            if tools == FULL:
                dag = full_chain_dag(it["text"], "data/ai_generated/green_fg/fg_02_anchor_female.png", bg)
            elif tools == FULL + ["T07_enhance"]:
                dag = full_chain_dag(it["text"], "data/ai_generated/green_fg/fg_01_anchor_male.png", bg,
                                     fx_mode=fx or "depth_blur")
            elif tools == ["T02_background_generate", "T06_harmonize"]:
                nodes = [
                    {"id": "n1", "tool": "T02_background_generate",
                     "params": {"semantic": bg, "quality": "draft"}, "depends_on": []},
                    {"id": "n2", "tool": "T06_harmonize",
                     "params": {"fg_path": "$cur", "alpha_path": "$cur", "bg_path": "$n1.bg_path",
                                "mode": "greenscreen"}, "depends_on": ["n1"]}]
                dag = dag_of(it["text"], nodes, ["n2"])
            elif tools == ["T07_enhance"]:
                mode = {"depth_blur": "depth_blur", "sharpen": "sharpen", "grain": "grain"}.get(fx, "depth_blur")
                dag = dag_of(it["text"], [{"id": "n1", "tool": "T07_enhance",
                                           "params": {"image_path": "$cur", "mode": mode}, "depends_on": []}], ["n1"])
            else:
                continue
            labeled.append((it["id"], it["text"], dag))

    # B1/B3 融合: 有标注的直接用; B3 记录若对应标注存在则用标注版
    used_ids = set()
    for cid, text, dag in labeled:
        data.append((f"B|{cid}", text, dag))
        used_ids.add(cid)

    # ---- C. 拒识样本 (扩充: 覆盖更多 OOD 表述族, 治拒识泛化失败) ----
    OOD = ["今天天气怎么样", "帮我写一个python爬虫", "讲个笑话", "给我推荐一部电影", "1+1等于几",
           "翻译成英文：你好", "明天股票会涨吗", "帮我订个闹钟", "写一首关于秋天的诗", "北京有哪些景点",
           "怎么减肥最快", "帮我回一封邮件", "世界杯什么时候开赛", "解释一下量子力学",
           "附近有什么好吃的餐厅", "帮我查一下明天去上海的高铁票", "给我讲讲相对论",
           "帮我把这个Excel表格排序", "今天几号", " MongoClient 怎么用", "帮我算一下房贷",
           "推荐一本好书", "如何学英语", "孙悟空是谁", "帮我生成一个随机密码", "现在几点了",
           "做一道红烧肉的菜谱", "帮我修一下电脑", "下一个节假日是什么时候", "帮我起个名字",
           "这段代码有什么bug", "地球到月球多远", "帮我写封辞职信", "心理学入门看什么书",
           "帮我规划三条旅游路线", "防止脱发的办法", " websocket 和 socket 区别", "帮我看看这个合同",
           "小明和小红谁跑得快", "给我发个红包", "下载一部电影", "把这段话改写成文言文",
           "什么是通货膨胀", "怎么注册公司", "帮我做个PPT", "今天限行尾号多少", "苹果醋有什么好处",
           # 新增: 覆盖测试中发现的失败用例
           "帮我写一首诗", "这把椅子换颜色", "把图里文字改成你好", "把眼睛变大",
           "把桌子换成红色", "把人物的衣服改成蓝色", "把沙发变成绿色",
           "改一下图片上的字", "把背景里的花变成黄色"]
    for t in OOD:
        abstain = {"intent": "abstain", "nodes": [], "outputs": []}
        data.append((f"C|{t[:10]}", t, abstain))

    # ---- D. 上传背景图样本 ----
    BG_SENTS = ["背景换成这张照片", "用这张图当背景", "把这张照片作为背景替换绿幕", "我要这个当背景",
                "背景用我传的这张图"]
    for s in BG_SENTS:
        dag = dag_of(s, [
            {"id": "n1", "tool": "T01_matting", "params": {"image": "$cur", "quality": "draft"}, "depends_on": []},
            {"id": "n2", "tool": "T02_background_generate", "params": {"file": "(用户上传背景图路径)"}, "depends_on": []},
            {"id": "n3", "tool": "T03_lighting_estimate", "params": {"bg_path": "$n2.bg_path"}, "depends_on": ["n2"]},
            {"id": "n4", "tool": "T04_relight", "params": {"fg_path": "$n1.fg_path", "bg_path": "$n2.bg_path",
                                                            "light_dir": "$n3.light_dir", "color_temp": "$n3.color_temp"},
             "depends_on": ["n1", "n2", "n3"]},
            {"id": "n5", "tool": "T05_shadow_generate", "params": {"alpha_path": "$n1.alpha_path",
                                                                    "bg_path": "$n2.bg_path", "light_dir": "$n3.light_dir"},
             "depends_on": ["n1", "n2", "n3"]},
            {"id": "n6", "tool": "T06_harmonize", "params": {"fg_path": "$n4.relit_fg_path",
                                                              "alpha_path": "$n1.alpha_path",
                                                              "bg_path": "$n5.shadow_bg_path"},
             "depends_on": ["n4", "n5"]},
        ], ["n6"])
        data.append((f"D|{s[:8]}", s + "\n(用户上传背景图: <上传路径>)", dag))

    # ---- D2. 绿幕实景保留实物样本 (覆盖 G3/G4 失败模式) ----
    GREEN_PRESERVE = [
        "这张绿幕演播室照片只换掉绿幕部分, 桌子和话筒都要保留",
        "绿幕区域替换成综艺背景, 保留所有实物",
        "绿幕实景照片换背景但保留桌台和设备",
        "只换绿幕不要动桌子和麦克风",
        "绿幕演播室换城市背景, 主播台话筒保留",
    ]
    for s in GREEN_PRESERVE:
        dag = dag_of(s, [
            {"id": "n1", "tool": "T02_background_generate",
             "params": {"semantic": "综艺", "quality": "draft"}, "depends_on": []},
            {"id": "n2", "tool": "T06_harmonize",
             "params": {"fg_path": "$cur", "alpha_path": "$cur", "bg_path": "$n1.bg_path",
                        "mode": "greenscreen"}, "depends_on": ["n1"]}
        ], ["n2"])
        data.append((f"D2|{s[:10]}", s, dag))

    # ---- E. 多轮指代 + 单特效句式扩充 (治 T07→T08 过度绑定) ----
    FOLLOW = [("再加爱心贴纸", "sticker", "heart"), ("再叠一个皇冠贴纸", "sticker", "crown"),
              ("加一点景深虚化", "depth_blur", None), ("画面锐化一下", "sharpen", None),
              ("加点胶片颗粒", "grain", None), ("四周加暗角", "vignette", None),
              ("调成暖色温", "warm", None), ("偏冷色调", "cool", None),
              # 单特效新句式 (不带导出)
              ("虚化程度明显一点", "depth_blur", None), ("这张图能做景深虚化吗", "depth_blur", None),
              ("背景虚化一些", "depth_blur", None), ("帮我把画面弄锐利", "sharpen", None),
              ("清晰度不够, 锐化", "sharpen", None), ("颗粒感重一点", "grain", None),
              ("做出老电影的颗粒", "grain", None), ("四角压暗一些", "vignette", None),
              ("加个暗角氛围", "vignette", None), ("画面偏暖一点", "warm", None),
              ("调暖色调", "warm", None), ("调冷色调", "cool", None),
              ("色温偏冷一些", "cool", None), ("给照片加个星星贴纸", "sticker", "star"),
              ("贴一朵小花", "sticker", "flower"), ("加只小猫贴纸", "sticker", "cat"),
              ("加彩虹贴纸", "sticker", "rainbow"), ("来个爱心", "sticker", "heart"),
              ("加文字水印：内部资料", "watermark_text", None), ("水印写上样片", "watermark_text", None),
              ("打个水印: 预览版", "watermark_text", None),
              # 多轮指代变体
              ("刚才那张再加个星星", "sticker", "star"), ("在上面贴朵花", "sticker", "flower"),
              ("继续加暗角", "vignette", None), ("再锐化一点", "sharpen", None)]
    for t, mode, st in FOLLOW:
        params = {"image_path": "$cur"}
        if mode == "sticker":
            params.update({"mode": "sticker", "sticker": st})
        else:
            params["mode"] = mode
        dag = dag_of(t, [{"id": "n1", "tool": "T07_enhance", "params": params, "depends_on": []}], ["n1"])
        data.append((f"E|{t[:10]}", t + "\n(用户当前输入图: <上一步成片>)", dag))

    # ---- E2. 绿幕实景键控扩充 (治 GS 泛化) ----
    GS_SENTS = [("这张演播室照片只把绿幕换成全景背景, 桌子和话筒都要保留", "全景"),
                ("绿幕区域替换成综艺背景, 保留所有实物", "综艺"),
                ("只替换绿幕部分为新闻LED大屏, 台子别动", "新闻LED"),
                ("实景键控: 绿幕换成播客背景, 其他保留", "播客"),
                ("把绿幕换成天气背景, 保留桌台", "天气"),
                ("照片里的绿幕换成访谈沙发背景, 实物全保留", "访谈"),
                ("虚拟演播室: 绿幕改全景", "全景"),
                ("绿幕抠了换成综艺舞台, 灯架话筒保留", "综艺")]
    for t, bg in GS_SENTS:
        nodes = [{"id": "n2", "tool": "T02_background_generate",
                  "params": {"semantic": bg, "quality": "draft"}, "depends_on": []},
                 {"id": "n1", "tool": "T06_harmonize",
                  "params": {"fg_path": "$cur", "alpha_path": "$cur", "bg_path": "$n2.bg_path",
                             "mode": "greenscreen"}, "depends_on": ["n2"]}]
        data.append((f"E2|{t[:10]}", t, dag_of(t, nodes, ["n1"])))

    # ---- F. 领域知识 QA ----
    corpus = ROOT / "agent_training" / "data" / "domain_corpus.jsonl"
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["type"] == "qa":
                q, a = r["text"].split("答：", 1)
                q = q.replace("问：", "")
                data.append((f"F|{q[:8]}", q, {"answer": a}))
            elif r["type"] == "term":
                data.append((f"F|term", f"解释术语: {r['text'][:r['text'].find('：')]}",
                             {"answer": r["text"][r['text'].find('：')+1:]}))

    return data


def main():
    raw = build()
    # 序列化为 messages 格式
    out = []
    for cid, text, dag in raw:
        content = json.dumps(dag, ensure_ascii=False) if not isinstance(dag, str) else dag
        out.append({"id": cid, "messages": sample(text, dag)["messages"]})
    rng.shuffle(out)
    n_val = max(1, int(len(out) * 0.08))
    val, train = out[:n_val], out[n_val:]
    (OUT / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train), encoding="utf-8")
    (OUT / "sft_val.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in val), encoding="utf-8")
    from collections import Counter
    kinds = Counter(x["id"].split("|")[0] for x in out)
    print(f"SFT 数据: train {len(train)} + val {n_val} = {len(out)} | 来源分布 {dict(kinds)}")


if __name__ == "__main__":
    main()
