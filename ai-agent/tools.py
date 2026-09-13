# -*- coding: utf-8 -*-
"""
tools.py — Agent 工具注册表: 把本项目的图像/视频处理能力封装成
LLM function-calling 工具 (JSON Schema + Python 执行器)。

设计要点:
  - 工具以"文件路径"交互, 但保留一个会话态 ctx["cur"] = 当前工作图;
    LLM 若不给 image, 执行器自动沿用 cur, 便于多步串联。
  - 工具名 snake_case, 与 openai tools 参数命名一致。
  - 每个执行函数捕获异常返回 {"error": ...} 而非抛错, 供 LLM 自行修正。

模块: 抠图 / 背景叠加 / 绿幕替换 / 特效滤镜 / 圈选删除 / 全流程 /
      素材查询 / 文生图 / 视频处理
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sp in [str(ROOT), str(ROOT / "app"),
           str(ROOT / "ai-service" / "src"),
           str(ROOT / "ai-service" / "src" / "fx"),
           str(ROOT / "ai-service" / "src" / "video"),
           str(ROOT / "training" / "scripts"),
           str(ROOT / "ai-agent")]:
    if sp not in sys.path:
        sys.path.insert(0, sp)
import numpy as np
from PIL import Image
import studio_cli  # noqa
from agnes_client import AgnesClient
import habits_gptimage as HGI  # my-gptimage-habits 技能接入

OUTROOT = ROOT / "outputs" / "agent"
OUTROOT.mkdir(parents=True, exist_ok=True)
_AGENT_CLIENT = None


def _client(provider: str | None = None) -> AgnesClient:
    """取 LLM 客户端。provider 为 None 时走 LLM_PROVIDER 默认后端 (带缓存)。"""
    global _AGENT_CLIENT
    if provider:
        return AgnesClient(provider=provider)
    if _AGENT_CLIENT is None:
        _AGENT_CLIENT = AgnesClient()
    return _AGENT_CLIENT


def _ensure_img(ctx: dict, image: str | None, default_desc: str) -> str:
    """解析 image 参数: 显式路径 | 素材名 | 空则沿用 ctx['cur']。"""
    if image:
        p = Path(image)
        # 素材相对根
        cand = ROOT / image
        if cand.exists():
            return str(cand)
        if p.exists():
            return str(p)
        # 尝试已知目录
        for d in ["data/ai_generated/green_fg", "data/ai_generated/studio_bg",
                  "data/ai_generated/green_fg_hard"]:
            c = ROOT / d / image
            if c.exists():
                return str(c)
        raise ValueError(f"找不到图片: {image}")
    if ctx.get("cur"):
        return ctx["cur"]
    raise ValueError(f"没有可用的当前图, 请先给出 image 路径(或先用素材/上传一张图)")


# ================= 工具执行函数 =================
def _norm_result(r: dict, ctx: dict) -> dict:
    # 统一记录并返回; 探测结果图路径放入 ctx['cur']
    out = dict(r)
    # 常见结果图字段 (habit 技能返回 path)
    for key in ("final", "fg", "rgba", "out_path", "output", "path",
                "greenscreen_composited", "saved", "composite"):
        v = out.get(key)
        if isinstance(v, str) and Path(v).exists():
            ctx["cur"] = v
            break
    return out


def t_matting(ctx, image: str = "", engine: str = "hybrid") -> dict:
    try:
        img = _ensure_img(ctx, image, "抠图")
        r = studio_cli.task_matting(img, str(OUTROOT / "matting"), engine=engine)
        r["note"] = "已抠出前景; 结果含 fg(去底) / alpha / rgba。若要换背景请接着用 composite 或 greenscreen。"
        return _norm_result(r, ctx)
    except Exception as e:
        return {"error": f"matting失败: {e}"}


def t_composite(ctx, image: str = "", bg: str = "", harmonize: bool = True,
                relight: bool = True, shadow: bool = True) -> dict:
    try:
        img = _ensure_img(ctx, image, "合成")
        bgp = _resolve_bg(bg)
        r = studio_cli.task_composite(img, bgp, str(OUTROOT / "composite"),
                                      harmonize=harmonize, relight=relight, shadow=shadow)
        r["note"] = "已把前景合成到背景上并做了光影一致融合; final 为成片。"
        return _norm_result(r, ctx)
    except Exception as e:
        return {"error": f"composite失败: {e}"}


def t_greenscreen(ctx, image: str = "", bg: str = "") -> dict:
    try:
        img = _ensure_img(ctx, image, "绿幕替换")
        bgp = _resolve_bg(bg)
        r = studio_cli.task_greenscreen(img, bgp, str(OUTROOT / "greenscreen"))
        return _norm_result(r, ctx)
    except Exception as e:
        return {"error": f"greenscreen失败: {e}"}


def _resolve_bg(bg: str) -> str:
    import assets
    if not bg:
        # 默认访谈背景
        return str(ROOT / "data/ai_generated/studio_bg/bg_02_interview.png")
    # assets 语义匹配
    hit = assets.find_background(bg)
    if hit:
        return str(ROOT / hit)
    p = ROOT / bg
    if p.exists():
        return str(p)
    if Path(bg).exists():
        return bg
    return str(ROOT / "data/ai_generated/studio_bg/bg_02_interview.png")


# 特效名 -> 中文
_FX_CN = {
    "vintage_1970s": "70年代胶片", "sepia": "怀旧棕褐", "teal_orange": "青橙电影",
    "bw_noir": "黑白暗影", "cyberpunk": "赛博朋克", "film_golden": "黄金时刻",
    "duotone": "双色调", "oil_painting": "油画", "cartoon_comic": "漫画",
    "watercolor": "水彩", "sketch_pencil": "铅笔素描", "halftone": "网点",
    "glitch": "故障", "vignette": "暗角", "bokeh": "光斑散景", "spotlight": "聚光灯",
    "bloom": "柔光", "film_grain": "胶片颗粒", "heart": "爱心贴纸", "star": "星星",
    "crown": "皇冠", "flower": "花朵", "watermark_text": "文字水印",
    "date_stamp": "日期戳", "vhs": "复古录影带", "thermal": "热成像",
}


def t_apply_fx(ctx, image: str = "", effect: str = "", intensity: float = 0.5,
               category: str = "", keyword: str = "") -> dict:
    """给当前图加特效/滤镜/贴纸/水印/光影。effect 可给英文 key 或中文描述;
    不给 effect 时用 category+keyword 语义匹配。"""
    try:
        img = _ensure_img(ctx, image, "特效")
        arr = np.array(Image.open(img).convert("RGB"))
        import fx_library as F
        # 解析 effect
        key = None
        if effect:
            key = effect
            # 中文反查
            for k, cn in _FX_CN.items():
                if cn in effect or effect in cn:
                    key = k
                    break
        if not key or key not in _fx_all_keys():
            # 语义找: 优先贴纸/水印等装饰
            if not key:
                key = _semantic_fx(category, keyword)
            if not key:
                raise ValueError(f"无法解析特效。可用中文: {list(_FX_CN.values())[:20]}... 或英文 key")
        out_arr, meta = F.fx_auto.place(arr, key, params={"frac": float(intensity)})
        op = OUTROOT / "fx" / f"{key}_{int(time.time())}.png"
        op.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_arr).save(op)
        r = {"effect": key, "final": str(op), "box": meta.get("box"),
             "note": f"已应用特效[{key}] (中文={_FX_CN.get(key,key)})"}
        return _norm_result(r, ctx)
    except Exception as e:
        return {"error": f"apply_fx失败: {e}"}


_fx_cache = {}


def _fx_all_keys():
    if not _fx_cache:
        import fx_library as F
        _fx_cache["k"] = list(F.FILTERS) + list(F.STYLES) + list(F.DECOR) + \
                         ["vignette", "bokeh", "spotlight", "bloom", "god_rays",
                          "film_grain", "tilt_shift", "warm", "cool", "light_leak",
                          "mosaic", "blur", "beautify", "whiten"]
    return _fx_cache["k"]


def _semantic_fx(category: str, keyword: str) -> str | None:
    import fx_library as F
    s = (category + " " + keyword).lower()
    # 类别映射
    if "黑白" in s or "bw" in s or "黑白" in category:
        return "bw_noir"
    if "胶片" in s or "复古" in s: return "vintage_1970s"
    if "青橙" in s or "电影" in s: return "teal_orange"
    if "赛博" in s or "霓虹" in s: return "cyberpunk"
    if "油画" in s: return "oil_painting"
    if "漫画" in s or "卡通" in s: return "cartoon_comic"
    if "素描" in s: return "sketch_pencil"
    if "水彩" in s: return "watercolor"
    if "暗角" in s: return "vignette"
    if "聚光" in s or "灯光" in s: return "spotlight"
    if "光斑" in s or "散景" in s or "梦幻" in s: return "bokeh"
    if "暖" in s or "色温" in s: return "warm"
    if "冷" in s: return "cool"
    if "颗粒" in s or "胶片颗粒" in s: return "film_grain"
    if "心" in s: return "heart"
    if "星" in s: return "star"
    if "花" in s: return "flower"
    if "水印" in s: return "watermark_text"
    if "日期" in s or "时间戳" in s: return "date_stamp"
    if "马赛克" in s: return "mosaic"
    if "美颜" in s: return "beautify"
    return None


def t_list_effects(ctx, category: str = "") -> dict:
    import fx_library as F
    cats = {
        "滤镜": sorted(F.FILTERS), "风格": sorted(F.STYLES),
        "贴纸": sorted([k for k in F.DECOR]), "光影": ["vignette", "bokeh", "spotlight", "bloom", "film_grain"],
        "美颜": ["beautify", "whiten"],
    }
    lines = []
    for c, keys in cats.items():
        if category and category not in c:
            continue
        lines.append(f"[{c}] " + ", ".join(f"{k}({_FX_CN.get(k,k)})" for k in keys))
    return {"effects": "\n".join(lines), "note": "effect 参数填括号前的英文 key"}


def t_remove_region(ctx, image: str = "", region: str = "center") -> dict:
    """圈选删除图像中央或指定区域的内容(用背景填补)。region 可为
    'center'(中央矩形30%) / 百分比矩形 'x1,y1,x2,y2'(像素) 。"""
    try:
        img = _ensure_img(ctx, image, "删除")
        arr = np.array(Image.open(img).convert("RGB"))
        h, w = arr.shape[:2]
        if region == "center":
            x1, y1, x2, y2 = int(w * 0.25), int(h * 0.2), int(w * 0.75), int(h * 0.8)
            box = [x1, y1, x2, y2]
        else:
            parts = [int(x) for x in str(region).replace(" ", "").split(",")]
            if len(parts) != 4:
                raise ValueError("region 需 'center' 或 'x1,y1,x2,y2'")
            box = parts
        import cv2
        mask = np.zeros(arr.shape[:2], np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        # 绿幕优先填绿, 否则 Telea
        f32 = arr.astype(np.float32)
        gr = float((f32[..., 1] - np.maximum(f32[..., 0], f32[..., 2]) > 25).mean())
        if gr > 0.15:
            gmask = f32[..., 1] - np.maximum(f32[..., 0], f32[..., 2]) > 25
            ref = np.median(arr[gmask], axis=0)
            mask_d = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            soft = cv2.GaussianBlur((mask_d.astype(np.float32) / 255)[..., None], (0, 0), 15)
            fill = np.stack([np.full(arr.shape[:2], ref[c]) for c in range(3)], -1)
            out = np.clip(arr * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
        else:
            mask_d = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            out = cv2.inpaint(arr, mask_d, 7, cv2.INPAINT_TELEA)
        op = OUTROOT / "remove" / f"removed_{int(time.time())}.png"
        op.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(op)
        return _norm_result({"final": str(op), "box": box,
                             "note": f"已删除区域 {box} 并自动填补背景"}, ctx)
    except Exception as e:
        return {"error": f"remove失败: {e}"}


def t_gen_image(ctx, prompt: str = "", role: str = "background") -> dict:
    """用 AI 文生图生成素材(背景/前景/贴纸)。prompt 描述画面, role 说明用途。"""
    try:
        if not prompt:
            raise ValueError("prompt 不能为空")
        op = OUTROOT / "gen" / f"{role}_{int(time.time())}.png"
        _client().generate_image(prompt, str(op), size="1024x1024")
        return _norm_result({"path": str(op), "role": role,
                             "note": f"已生成{'背景' if role=='background' else '前景'}素材: {prompt}"}, ctx)
    except Exception as e:
        return {"error": f"gen_image失败: {e}"}


def t_list_assets(ctx, kind: str = "bg") -> dict:
    import assets
    if kind == "bg":
        return {"assets": assets.list_backgrounds()}
    if kind == "person":
        return {"assets": assets.list_persons()}
    return {"assets": assets.list_backgrounds() + "\n" + assets.list_persons()}


def t_video_process(ctx, video: str = "", task: str = "replace_bg",
                    bg: str = "") -> dict:
    """视频处理。task: replace_bg(换背景)/filter(滤镜)/mosaic(打码)/combo(组合)。"""
    try:
        import video_pipeline as VP
        v = video or ctx.get("video")
        if not v:
            # 找预置测试视频
            tv = ROOT / "data/video/test_green.mp4"
            if tv.exists():
                v = str(tv)
            else:
                raise ValueError("无视频, 请给 video 路径或用预置")
        cfg = {"replace_bg": {"green": {"enabled": True, "bg": _resolve_bg(bg)}}}.get(task)
        if task == "filter":
            cfg = {"filter": "teal_orange"}
        elif task == "combo":
            cfg = {"green": {"enabled": True, "bg": _resolve_bg(bg)},
                   "filter": "teal_orange", "fx": [{"effect": "heart", "params": {"frac": 0.12}}]}
        elif task == "mosaic":
            cfg = {"mosaic": {"enabled": True, "box": [100, 100, 540, 320], "cell": 24}}
        op = OUTROOT / "video" / f"out_{int(time.time())}.webm"
        r = VP.process(cfg, v, str(op))
        ctx["video"] = r["output"]
        return {"output": r["output"], "frames": r["frames_written"], "sec": r["sec"],
                "note": "视频处理完成, output 为结果(.webm, 浏览器可播)"}
    except Exception as e:
        return {"error": f"video_process失败: {e}"}


def t_region_fx(ctx, image: str = "", effect: str = "spotlight_on",
                region: str = "auto", intensity: float = 0.7,
                filter: str = "", invert: bool = False) -> dict:
    """mask 驱动的区域特效: 聚光灯/局部滤镜/区域调色/背景虚化/辉光/描边。
    region: auto(自动抠出主体当 mask) | 像素框 'x1,y1,x2,y2' | mask 文件路径。"""
    try:
        img_path = _ensure_img(ctx, image, "区域特效")
        arr = np.array(Image.open(img_path).convert("RGB"))
        H, W = arr.shape[:2]
        # ---- 构造 mask ----
        if region == "auto" or not region:
            r = studio_cli.task_matting(img_path, str(OUTROOT / "regionfx"), engine="hybrid")
            ap = r.get("alpha") or r.get("rgba")
            if not ap or not Path(ap).exists():
                raise ValueError("无法自动抠出主体, 请显式给 region 框")
            if ap.endswith(".png") and "rgba" in Path(ap).stem:
                mask = np.array(Image.open(ap).convert("RGBA"))[..., 3]
            else:
                mask = np.array(Image.open(ap).convert("L"))
        elif Path(region).exists():
            mask = np.array(Image.open(region).convert("L"))
        elif "," in region:
            x1, y1, x2, y2 = [int(v) for v in region.replace(" ", "").split(",")]
            mask = np.zeros((H, W), np.uint8)
            mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 255
        else:
            raise ValueError("region 需 auto / 'x1,y1,x2,y2' / mask 路径")
        import region_fx as RF
        out, meta = RF.apply_region_fx(arr, mask, effect,
                                       {"intensity": float(intensity), "filter": filter,
                                        "invert": bool(invert)})
        op = OUTROOT / "regionfx" / f"{effect}_{int(time.time())}.png"
        op.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(op)
        return _norm_result({"final": str(op), "region_area": meta["area_ratio"],
                             "note": f"已应用区域特效[{meta['cn']}] 覆盖 {meta['area_ratio']*100:.1f}% 画面"},
                            ctx)
    except Exception as e:
        return {"error": f"region_fx失败: {e}"}


def t_semantic_fx(ctx, image: str = "", group: str = "天空",
                  effect: str = "region_color", intensity: float = 0.7,
                  warm: float = 0.0, cool: float = 0.0, invert: bool = False) -> dict:
    """语义级区域特效: 按画面语义(天空/建筑/树木植物/地面/人物/车辆...)精准作用。
    依赖 COCO-Stuff 语义分割, 只对原图为 COCO val2017 图 (image_id 命名如 000000001532) 有效。"""
    try:
        img_path = _ensure_img(ctx, image, "语义特效")
        stem = Path(img_path).stem
        if not (ROOT / "data" / "coco_stuff" / "annotations" / "stuff_val2017_pixelmaps"
                / f"{stem}.png").exists():
            return {"error": f"语义分割仅支持 COCO val2017 图 (需要 {stem}.png 像素图), "
                             f"当前图不支持。可改用 region_fx 的 auto 模式"}
        sys.path.insert(0, str(ROOT / "training" / "scripts"))
        import stuff_semantic as SS
        params = {"intensity": float(intensity)}
        if warm:
            params["warm"] = float(warm)
        if cool:
            params["cool"] = float(cool)
        op = OUTROOT / "semantic_fx" / f"{group}_{effect}_{int(time.time())}.png"
        out, meta = SS.apply_semantic_fx(stem, group, effect, params, op, invert=bool(invert))
        return _norm_result({"final": str(op), "group": group,
                             "group_ratio": meta["area_ratio"],
                             "note": f"已按语义[{group}]应用{meta['cn']}, 覆盖 {meta['area_ratio']*100:.1f}% 画面"},
                            ctx)
    except Exception as e:
        return {"error": f"semantic_fx失败: {e}"}


def t_analyze_scene(ctx, image: str = "") -> dict:
    """分析图片的场景语义构成 (各类别占比), 依赖 COCO-Stuff 语义分割。"""
    try:
        img_path = _ensure_img(ctx, image, "场景分析")
        stem = Path(img_path).stem
        sys.path.insert(0, str(ROOT / "training" / "scripts"))
        import stuff_semantic as SS
        r = SS.analyze(stem)
        if r.get("error"):
            return r
        top = ", ".join(f"{c['name']}({c['ratio']*100:.0f}%)" for c in r["classes"][:8])
        grp = ", ".join(f"{k}({v*100:.0f}%)" for k, v in list(r["groups"].items())[:6])
        return {"scene": top, "groups": grp, "note": "场景语义构成分析完成"}
    except Exception as e:
        return {"error": f"analyze_scene失败: {e}"}


# ================= my-gptimage-habits 技能工具 =================
def t_gen_subtitle_bar(ctx, name: str = "姓名", role: str = "头衔", theme: str = "演播室",
                       transparent: bool = True) -> dict:
    """生成视频字幕条(人名+头衔/职称), 透明底可直接叠加到视频上。文字逐字准确。"""
    try:
        r = HGI.generate_habit("W2", {"name": name, "role": role, "theme": theme,
                                    "transparent": transparent})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已生成字幕条: {name} | {role}"}, ctx)
    except Exception as e:
        return {"error": f"gen_subtitle_bar失败: {e}"}


def t_gen_title_card(ctx, title: str = "标题", subtitle: str = "", theme: str = "科技",
                     transparent: bool = True) -> dict:
    """生成视频片头字幕/标题卡(大字+副标题), 透明底, 文字逐字准确。"""
    try:
        r = HGI.generate_habit("W9", {"title": title, "subtitle": subtitle, "theme": theme,
                                    "transparent": transparent})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已生成标题卡: {title}"}, ctx)
    except Exception as e:
        return {"error": f"gen_title_card失败: {e}"}


def t_gen_ppt_cover(ctx, title: str = "标题", subtitle: str = "", style: str = "高级感设计感") -> dict:
    """生成 PPT 封面/演讲扉页。注: agnes 文生图可能偶有中文错字, 关键标题请后续用 add_cover_text 校对叠加。"""
    try:
        r = HGI.generate_habit("W6", {"title": title, "subtitle": subtitle, "style": style})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已生成 PPT 封面: {title}; 若中文显示异常请用 add_cover_text 重叠加"}, ctx)
    except Exception as e:
        return {"error": f"gen_ppt_cover失败: {e}"}


def t_gen_storyboard(ctx, title: str = "标题", subtitle: str = "", style: str = "高级感设计感") -> dict:
    """生成视频分镜/故事板缩略图。"""
    try:
        r = HGI.generate_habit("W7", {"title": title, "subtitle": subtitle, "style": style})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已生成分镜图: {title}"}, ctx)
    except Exception as e:
        return {"error": f"gen_storyboard失败: {e}"}


def t_add_cover_text(ctx, image: str = "", title: str = "", subtitle: str = "") -> dict:
    """给图片/截图加封面级标题文字(底部渐变遮罩+居中大字), 文字 100% 准确。"""
    try:
        img = _ensure_img(ctx, image, "加字")
        r = HGI.generate_habit("W1", {"image": img, "title": title, "subtitle": subtitle})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已加封面文字: {title}"}, ctx)
    except Exception as e:
        return {"error": f"add_cover_text失败: {e}"}


def t_extract_element(ctx, image: str = "", element: str = "图标", keep_person: bool = False) -> dict:
    """从图中抠出某个元素(如图标/物体)并转成透明底 PNG。非绿幕图会回退到本地抠图。"""
    try:
        img = _ensure_img(ctx, image, "抠元素")
        r = HGI.generate_habit("W3", {"image": img, "element": element, "keep_person": keep_person})
        return _norm_result({"path": r["path"], "rgba_path": r.get("rgba_path"),
                             "provider": r.get("provider"),
                             "note": f"已提取 {element} 为透明底"}, ctx)
    except Exception as e:
        return {"error": f"extract_element失败: {e}"}


def t_merge_images(ctx, image: str = "", image2: str = "", position: str = "画面右侧") -> dict:
    """把第二张图的人物/前景合成到第一张背景图上。image=背景, image2=人物/前景。"""
    try:
        bg = _ensure_img(ctx, image, "背景")
        fg = _ensure_img(ctx, image2, "人物") if image2 else None
        if not fg:
            raise ValueError("需要 image2 作为人物/前景")
        r = HGI.generate_habit("W4", {"image": bg, "image2": fg, "position": position})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已把人物合成到背景({position})"}, ctx)
    except Exception as e:
        return {"error": f"merge_images失败: {e}"}


def t_replace_text(ctx, image: str = "", new_text: str = "") -> dict:
    """把图片底部/字幕区域的文字改成新文字, 其他内容保持不变。"""
    try:
        img = _ensure_img(ctx, image, "改字")
        r = HGI.generate_habit("W5", {"image": img, "new_text": new_text})
        return _norm_result({"path": r["path"], "provider": r.get("provider"),
                             "note": f"已将文字改为: {new_text}"}, ctx)
    except Exception as e:
        return {"error": f"replace_text失败: {e}"}


# ================= 注册表 =================
def build_tool_schemas() -> list[dict]:
    """返回 openai 风格 tools 数组。"""
    def F(name, desc, props, req=None):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": req or []}}}

    S = {"type": "string"}
    return [
        F("matting", "抠图: 分离前景与背景(绿幕图给 alpha/fg)。engine: hybrid(默认,绿幕人像最佳)/chroma/refiner/birefnet。image 可为素材名或路径, 空则沿用当前图。",
          {"image": S, "engine": S}, ["engine"]),
        F("composite", "背景叠加合成: 把当前人像/前景放到背景上并做光影一致融合(可选 harmonize/relight/shadow)。bg 给背景文件或中文语义(如'访谈','新闻LED','全景','综艺')。输出尺寸=输入尺寸。",
          {"image": S, "bg": S, "harmonize": {"type": "boolean"}, "relight": {"type": "boolean"}, "shadow": {"type": "boolean"}}),
        F("greenscreen", "绿幕区域替换(虚拟演播室键控): 保留整张原图, 仅把绿幕像素替换为新背景(桌/灯/话筒保留)。适合同时含绿幕+实物的演播室照片。bg 同上语义。",
          {"image": S, "bg": S}),
        F("apply_fx", "给当前图添加特效/滤镜/贴纸/水印/光影。effect 用英文 key 或中文(如'黑白','青橙','赛博朋克','油画','漫画','暗角','聚光灯','爱心','水印','美颜','复古胶片')。",
          {"image": S, "effect": S, "intensity": {"type": "number", "description": "0~1强度/大小"},
           "category": S, "keyword": S}),
        F("remove_region", "圈选删除图像中某块区域的内容并用背景丝滑填补。region: 'center'(默认中央)或像素 'x1,y1,x2,y2'。",
          {"image": S, "region": S}),
        F("gen_image", "AI 文生图生成素材(可用作背景/前景)。prompt 详细描述想要的画面(如'黄昏城市天际线演播室背景'), role=background/fg。生成的图会保存并可作为后续 bg 使用。",
          {"prompt": S, "role": S}, ["prompt"]),
        F("list_effects", "列出可用的滤镜/风格/贴纸/光影/美颜特效清单(含英文key和中文名)。category: 滤镜/风格/贴纸/光影/美颜 可选。",
          {"category": S}),
        F("list_assets", "列出可用素材(背景/绿幕人像)及语义描述。kind: bg/person/all。",
          {"kind": S}),
        F("gen_subtitle_bar", "生成视频字幕条: 人名+头衔, 透明底 PNG, 可直接叠加到视频/画面上。文字逐字准确。",
          {"name": S, "role": S, "theme": S, "transparent": {"type": "boolean"}},
          ["name", "role"]),
        F("gen_title_card", "生成视频片头/标题卡: 大标题+副标题, 透明底 PNG。文字逐字准确。",
          {"title": S, "subtitle": S, "theme": S, "transparent": {"type": "boolean"}},
          ["title"]),
        F("gen_ppt_cover", "生成 PPT 封面/演讲扉页。基于 agnes 文生图, 视觉效果好但中文标题可能偶有错字, 关键文字建议再用 add_cover_text 校对。",
          {"title": S, "subtitle": S, "style": S}, ["title"]),
        F("gen_storyboard", "生成视频分镜/故事板缩略图。",
          {"title": S, "subtitle": S, "style": S}, ["title"]),
        F("add_cover_text", "给已有图片(截图/PPT/封面)加底部/居中封面文字, 文字 100% 准确, 带渐变遮罩。",
          {"image": S, "title": S, "subtitle": S}, ["title"]),
        F("extract_element", "从图中提取某个元素(图标/物体/人物)并输出透明底 PNG。",
          {"image": S, "element": S, "keep_person": {"type": "boolean"}}, ["image", "element"]),
        F("merge_images", "把第二张图的人物/前景合成到第一张背景图上(多图合成)。image=背景, image2=人物/前景。",
          {"image": S, "image2": S, "position": S}, ["image", "image2"]),
        F("replace_text", "把图片底部/字幕区域文字改成新文字, 其他保持不变。",
          {"image": S, "new_text": S}, ["image", "new_text"]),
        F("region_fx", "区域特效(mask 驱动, 比 apply_fx 更精准): 只对画面某个主体/区域生效。"
          "effect: spotlight_on(主体聚光灯,打亮人压暗背景) / spotlight_off(主体压暗模糊,隐私保护) / "
          "region_filter(仅主体套滤镜, 配 filter 如'青橙') / region_color(区域调暖/冷) / "
          "bg_blur(背景虚化景深) / region_glow(主体辉光) / edge_highlight(边缘描边发光)。"
          "region: auto(自动抠出主体) 或框 'x1,y1,x2,y2'。invert=true 则作用于背景而非主体。",
          {"image": S, "effect": S, "region": S,
           "intensity": {"type": "number", "description": "0~1 强度"},
           "filter": S, "invert": {"type": "boolean"}}, ["effect"]),
        F("semantic_fx", "语义级区域特效(最精准): 按画面语义精准作用, 如'只把天空调暖''只虚化地面建筑'。"
          "group: 天空/建筑/树木植物/地面/山石/水/雪/人物/车辆/家具/电子设备/食物/文字标识/室内空间/户外环境。"
          "effect 同 region_fx。仅对 COCO val2017 图有效。",
          {"image": S, "group": S, "effect": S,
           "intensity": {"type": "number"}, "warm": {"type": "number"}, "cool": {"type": "number"},
           "invert": {"type": "boolean"}}, ["group"]),
        F("analyze_scene", "分析图片场景语义构成(各类别占比), 如'天空30% 建筑20% 地面15%'。用于理解画面或决定对哪个语义区域做特效。",
          {"image": S}),
    ]


_DISPATCH = {
    "matting": t_matting, "composite": t_composite, "greenscreen": t_greenscreen,
    "apply_fx": t_apply_fx, "remove_region": t_remove_region, "gen_image": t_gen_image,
    "list_effects": t_list_effects, "list_assets": t_list_assets,
    "gen_subtitle_bar": t_gen_subtitle_bar, "gen_title_card": t_gen_title_card,
    "gen_ppt_cover": t_gen_ppt_cover, "gen_storyboard": t_gen_storyboard,
    "add_cover_text": t_add_cover_text, "extract_element": t_extract_element,
    "merge_images": t_merge_images, "replace_text": t_replace_text,
    "region_fx": t_region_fx,
    "semantic_fx": t_semantic_fx, "analyze_scene": t_analyze_scene,
}


def call_tool(name: str, arguments: dict, ctx: dict) -> dict:
    fn = _DISPATCH.get(name)
    if not fn:
        return {"error": f"未知工具 {name}"}
    try:
        return fn(ctx, **arguments)
    except TypeError as e:
        return {"error": f"工具 {name} 参数错误: {e}"}
    except Exception:
        return {"error": f"工具 {name} 异常:\n{traceback.format_exc()[:300]}"}


def _init_fx_auto_dir():
    import fx_library as F  # noqa 触发注册


if __name__ == "__main__":
    ctx = {}
    print("== tools 冒烟测试 ==")
    # matting 绿幕人像
    r = call_tool("matting", {"image": "fg_02_anchor_female.png", "engine": "hybrid"}, ctx)
    print("matting:", r.get("task"), "fg_ratio", r.get("fg_ratio"), "cur:", ctx.get("cur"))
    # composite
    r2 = call_tool("composite", {"bg": "访谈"}, ctx)
    print("composite:", r2.get("task"), r2.get("final"))
    # fx
    r3 = call_tool("apply_fx", {"effect": "青橙"}, ctx)
    print("fx:", r3)
    # list
    print(call_tool("list_effects", {"category": "光影"}, ctx))
