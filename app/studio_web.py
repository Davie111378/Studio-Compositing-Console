# -*- coding: utf-8 -*-
"""
演播室图像合成 · Web 交互工作台 (零新增依赖: stdlib http.server + 浏览器)
功能: 选择绿幕图片 -> 圈选 -> 抠图 / 背景叠加 / 添加特效 (区域圈选)

运行: python app/studio_web.py     (自动打开浏览器 http://127.0.0.1:8765)
后端复用 app/studio_cli 的任务函数, 与 CLI/GUI 同一引擎。
"""
from __future__ import annotations
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # project root (app/ -> root)
sys.path.insert(0, str(ROOT / "app"))
import studio_cli  # noqa: E402

OUT = ROOT / "outputs" / "web"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8765
BG_DIR = ROOT / "data" / "ai_generated" / "studio_bg"
FG_DIR = ROOT / "data" / "ai_generated" / "green_fg"


def safe_path(p: str) -> Path | None:
    """路径必须落在项目根内 (防目录穿越)。"""
    q = (ROOT / p).resolve()
    if str(q).startswith(str(ROOT.resolve())) and q.exists():
        return q
    return None


def to_png_b64(arr) -> str:
    import base64, io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# "把绿幕图用 XX 背景图替换" 类意图: 用户上传的那张图是**背景**, 前景(绿幕图)在会话上下文里
_BG_REPLACE_KEYS = ("背景图替换", "背景图换", "用.*背景图", "换上.*背景", "替换绿幕", "替换掉绿幕",
                    "绿幕图片用", "绿幕用", "把绿幕")
_FG_PRONOUNS = ("这张图", "这张", "该图", "当前图", "此图")


def _is_bg_replace_intent(instr: str) -> bool:
    """判断「单张上传图其实是背景」的意图 (保守判定, 宁可漏判不可误判)。

    仅当指令明确把绿幕说成**待处理的前景**、并把另一物说成**背景**时触发:
      例: "绿幕图片用城市背景图替换" / "把绿幕换成这张背景图"
    若指令用"这张图"指代上传图(说明它就是前景), 一律不触发。
    """
    import re as _re
    t = instr or ""
    # "这张图"指代上传图 → 它当前景, 不重定向。但"这张背景图"指的是背景本体, 不算指代上传图。
    for k in _FG_PRONOUNS:
        for _m in _re.finditer(_re.escape(k), t):
            _tail = t[_m.end():_m.end() + 4]
            if "背景" not in _tail and "背" not in _tail:
                return False
    # 必须同时出现"绿幕"(待换的前景) 与 "背景"(目标)
    if ("绿幕" not in t) or ("背景" not in t):
        return False
    return any(_re.search(p, t) for p in _BG_REPLACE_KEYS)


# ---- 两张上传图: 从指令解析「哪张是背景」(A图/图1/第一张/前者 → 上传序号) ----
# 语义约定 (2026-09-12 与用户确认): "图1作为图2的背景" = 图1是背景、图2是人物(前景)。
# 引用词正则: 图1(A)/图2(B)/第一张/第1张/1号图/前者/后者 …
_REF_PAT = (r"[ABab]\s*图(?:片)?|图\s*[ABab1-2]|第\s*[一二两1-2]\s*张(?:图片|照片|图)?"
            r"|[12]\s*号图|前者|后者")

# 角色模式 (按优先级依次尝试): 命名组 bg=背景引用词, fg=前景(主体)引用词
_ROLE_PATTERNS = [
    # X 作为/当/做/为/是/充当 Y (的)背景 → X=背景, Y=主体      "图1作为图2的背景"
    r"(?P<bg>_R_)\s*(?:作为|当|做|为|是|充当)\s*(?P<fg>_R_)\s*(?:的)?(?:图)?\s*背景",
    # Y (的)背景 换成/用/设为 X → X=背景, Y=主体               "图2的背景换成图1" / "图2绿幕的背景换成图1"
    r"(?P<fg>_R_)\s*的?\s*[^，。;；]{0,6}?背景\s*(?:换成|换为|替换成|替换为|改为|设为|设置为|用|是|为)\s*(?P<bg>_R_)",
    # X 替换/换掉 Y (的)背景 → X=背景, Y=主体                 "图1替换图2的背景" / "图1替换图2绿幕部分的背景"
    r"(?P<bg>_R_)\s*(?:替换|换掉|换掉|换成|顶替|代替)\s*(?P<fg>_R_)[^，。;；]{0,8}?(?:的)?(?:图)?\s*背景",
    # 用 X (来)?替换/换 Y → X=背景, Y=主体                    "用图1替换图2"
    r"用\s*(?P<bg>_R_)\s*(?:来)?(?:替换|换|换掉|换成|当|做|作为)\s*(?P<fg>_R_)\s*$",
    # X 是背景(，)Y 是(人物/前景/主体) → X=背景, Y=主体       "图1是背景，图2是人物"
    r"(?P<bg>_R_)\s*(?:是|为)\s*(?:背景|背景图)[，,;；\s]*(?P<fg>_R_)\s*(?:是|为|做)?\s*(?:人物|人像|前景|主体|主体图)?",
    # 把 X 放到/贴到/拼到/合成到 Y (的)背景(上/里) → X=主体, Y=背景  "把图2抠出来放到图1"
    r"(?:把|将)?\s*(?P<fg>_R_)[^，。;；]{0,8}?(?:放到|放入|放进|贴到|贴进|拼到|合到|合成到|P到|p到|移到|并入|抠出来放|抠出放)"
    r"\s*(?P<bg>_R_)\s*(?:的)?\s*(?:图)?\s*(?:背景|上|里|中)?",
    # 单引用: X 作为/当/做/充当 背景 → X=背景, 另一张为主体     "用图1当背景"
    r"(?P<bg>_R_)\s*(?:作为|当|做|为|充当)\s*(?:这)?(?:张)?\s*背景",
    # 单引用: 背景 用/换成/是/为 X → X=背景                     "背景用第二张"
    r"背景\s*(?:用|换成|换为|改为|设为|是|为)\s*(?P<bg>_R_)",
]


def _ref_index(tok: str) -> int:
    """引用词 → 上传序号 (0=图1/A/第一张, 1=图2/B/第二张)。与前端 chip 标签约定一致。"""
    import re as _re
    t = _re.sub(r"\s+", "", tok)
    if t == "前者":
        return 0
    if t == "后者":
        return 1
    if t.startswith(("图B", "图b")) or _re.fullmatch(r"[Bb]图(?:片)?", t):
        return 1
    if t.startswith(("图A", "图a")) or _re.fullmatch(r"[Aa]图(?:片)?", t):
        return 0
    m = _re.search(r"[一二两1-2]", t)
    if m:
        return {"一": 0, "1": 0, "两": 1, "二": 1, "2": 1}.get(m.group(0), 0)
    return 0


def _parse_two_image_roles(instr: str):
    """两张上传图时, 从指令解析哪张做背景。返回 (bg_idx, fg_idx) 或 None。

    例: "将A图片作为B图片的背景" → (0, 1)  "第二张的背景换成第一张" → (0, 1)
    """
    import re as _re
    t = instr or ""
    try:
        for pat in _ROLE_PATTERNS:
            m = _re.search(pat.replace("_R_", _REF_PAT), t)
            if not m:
                continue
            bg_i = _ref_index(m.group("bg"))
            fg_i = _ref_index(m.group("fg")) if m.groupdict().get("fg") else None
            if fg_i is None:                     # 单引用模式 → 另一张为主体
                fg_i = 1 - bg_i if bg_i in (0, 1) else None
            if bg_i in (0, 1) and fg_i in (0, 1) and bg_i != fg_i:
                return bg_i, fg_i
    except Exception:
        pass
    return None


def _green_roles(imgs: list) -> tuple | None:
    """双图物理判据 (2026-09-12 去角色化兜底): 绿幕色度占比差距显著时,
    绿的=待替换场景(前景), 另一张=背景。返回 (bg_idx, fg_idx) 或 None。

    色度绿幕场景图绿占比通常 >15%, 实景背景 ≈0; 差距 <=0.10 视为不可判,
    交回上层"背景"词/默认顺序逻辑, 避免误判两张普通图。
    """
    def _gr(p) -> float:
        try:
            sp = safe_path(str(p))
            if sp is None:
                return -1.0
            a = cv2.imdecode(np.fromfile(str(sp), dtype=np.uint8), 1)
            if a is None:
                return -1.0
            f = a.astype(np.float32)
            return float((f[..., 1] - np.maximum(f[..., 0], f[..., 2]) > 25).mean())
        except Exception:
            return -1.0
    if len(imgs) != 2:
        return None
    g = [_gr(x.get("path")) for x in imgs]
    if min(g) < 0 or abs(g[0] - g[1]) <= 0.10:
        return None
    fg_i = 0 if g[0] > g[1] else 1
    return (1 - fg_i, fg_i)


def _session_result(ag, path: str, tool: str, note: str, sec: float) -> dict:
    """会话级操作 (undo/reset/matte_edit) 的标准返回结构。"""
    return {"steps": [{"node_id": "n1", "tool": tool, "status": "done",
                       "latency_ms": int(sec * 1000), "engine": "", "mode": "",
                       "error": ""}],
            "total_ms": int(sec * 1000), "final": path,
            "cur_url": "/file?path=" + quote(str(path)),
            "final_text": note, "utility": None}


def _run_preset(ag, preset: dict) -> dict:
    """面板直通执行 (2026-09-12): 滤镜/抠图/特效的机械通道, 不经 LLM。

    作用对象 = 会话当前图 (ag.cur_image); 产物回写 ag.cur_image 支持链式操作。
    2026-09-12 晚补: undo(撤销上一步)/reset(返回原图)/matte_edit(圈选区域擦除恢复)
    /matting engine=grabcut+rect(圈选抠图)。特效可按操作顺序叠加, 每步入撤销栈。
    返回与 /api/agent/dag 消息流一致的结构 (steps/cur_url/final_text)。
    """
    t0 = time.time()
    kind = preset.get("kind")

    def _fail(msg):
        return {"steps": [{"node_id": "n1", "tool": kind or "preset", "status": "failed",
                           "latency_ms": 0, "engine": "", "mode": "", "error": msg}],
                "total_ms": 0, "final": None, "cur_url": None,
                "final_text": "⚠️ " + msg, "utility": None}

    # ---- 会话级操作: 撤销 / 返回原图 (不需要 src) ----
    if kind == "undo":
        hist = getattr(ag, "history", None) or []
        if not hist:
            return _fail("没有可撤销的操作")
        prev = hist.pop()
        ag.cur_image = prev
        return _session_result(ag, prev, "undo", f"已撤销上一步，恢复到上一个图像状态 (还可撤销 {len(hist)} 步)",
                               round(time.time() - t0, 2))
    if kind == "reset":
        orig = preset.get("image") or getattr(ag, "orig_image", None)
        orig = safe_path(str(orig)) if orig else None
        if orig is None or not Path(orig).exists():
            return _fail("本会话没有可返回的原图记录")
        if getattr(ag, "cur_image", None) and ag.cur_image != str(orig):
            ag.history.append(ag.cur_image)
        ag.cur_image = str(orig)
        return _session_result(ag, str(orig), "reset", "已返回原图 (操作历史已保留，仍可逐步撤销)",
                               round(time.time() - t0, 2))

    # 取图顺序: 面板显式指定 (前端最后产物/上传图) > 会话当前图
    _src_raw = str(preset.get("image") or getattr(ag, "cur_image", "") or "")
    src = safe_path(_src_raw) if _src_raw else None
    if src is None or not Path(src).exists():
        return _fail("没有可处理的图——请先上传一张图或在对话中生成一张")
    if getattr(ag, "orig_image", None) is None:      # 会话首张源图 → "返回原图"目标
        ag.orig_image = str(src)
    out_dir = OUT / ("preset_" + str(kind))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "filter":
            sys.path.insert(0, str(ROOT / ".workbuddy" / "skills" /
                                   "green-screen-composite" / "scripts"))
            import green_key as GK
            import filter_presets as FP
            pname = str(preset.get("preset") or "")
            fname = str(preset.get("filter") or "")
            arr = GK.imread(str(src))
            # 优先: 参数化自然调色引擎 (float32 连续曲线, 无 LUT 量化色带)
            if pname in FP.PHOTO_PRESETS:
                strength = float(preset.get("strength", 1.0))
                out = FP.apply_photo_filter(arr, FP.PHOTO_PRESETS[pname], strength)
                out_p = out_dir / f"{Path(src).stem}_{int(time.time())}.png"
                GK.imwrite(str(out_p), out)
                tool = f"T07_filter[{pname}]"
                note = f"已为当前图应用自然调色「{pname}」(强度 {strength:.2f})"
            elif fname:                                   # 兼容: 旧 LUT 色卡
                lut = GK.resolve_filter(fname)
                if lut is None:
                    return _fail(f"找不到色卡: {fname}")
                # 自然风: 强度线性混合 + 饱和度限幅 (色卡原风格浓烈, 默认 0.45/175)
                strength = float(preset.get("strength", 0.45))
                sat_max = float(preset.get("sat_max", 175))
                out = GK.apply_filter(arr, lut, strength=strength, sat_max=sat_max)
                out_p = out_dir / f"{Path(src).stem}_{int(time.time())}.png"
                GK.imwrite(str(out_p), out)
                tool = f"T02_filter[{fname}]"
                note = f"已为当前图叠加色卡「{fname.replace('.png', '')}」(自然风 强度{strength:.2f})"
            else:
                return _fail("未指定滤镜 (preset=自然调色名 或 filter=色卡名)")
        elif kind == "matting":
            engine = str(preset.get("engine") or "hybrid")
            rect = preset.get("rect")
            pts = preset.get("points")                 # 滑动圈选 (套索多边形)
            if engine == "grabcut" or rect or pts:
                # 圈选抠图 (2026-09-12): 矩形/套索 → GrabCut + 绿幕二次清理, 白底预乘输出
                # 松手即执行 (前端无确认弹窗); 套索失败时回退多边形直接填充 (涂抹语义)
                from matting.interactive_matting import grabcut_matte
                from ai_material_eval import chroma_key_alpha
                import numpy as _np
                from PIL import Image as _PIL
                arr = _np.array(_PIL.open(src).convert("RGB"))
                H, W = arr.shape[:2]
                if pts:
                    try:
                        xy = [float(v) for v in pts]
                    except (TypeError, ValueError):
                        return _fail("圈选坐标无效")
                    if len(xy) < 6:
                        return _fail("滑动圈选太短，请圈出更大的范围")
                    poly = _np.array([[min(max(x, 0), W - 1), min(max(y, 0), H - 1)]
                                      for x, y in zip(xy[0::2], xy[1::2])], _np.int32)
                    prompt = {"type": "polygon", "points": poly.tolist()}
                elif rect:
                    try:
                        x, y, w, h = [max(0, int(v)) for v in rect]
                    except (TypeError, ValueError):
                        return _fail("矩形圈选需要先在画布上拖拽框选主体区域")
                    x2, y2 = min(x + w, W), min(y + h, H)
                    if x2 - x < 8 or y2 - y < 8:
                        return _fail("圈选区域太小，请在画布上框出主体所在范围")
                    prompt = {"type": "box", "xyxy": [x, y, x2, y2]}
                else:
                    return _fail("圈选抠图需要先在画布上拖拽/涂抹圈选区域")
                alpha = grabcut_matte(arr, prompt)
                ck = chroma_key_alpha(arr)               # 绿幕残留二次清理 (同旧工作台)
                ck_ratio = float((ck > 0.5).mean())
                if 0.05 < ck_ratio < 0.95:
                    alpha = alpha * ck
                if (alpha > 0.5).mean() < 0.01:
                    if pts:                               # 套索回退: 圈内直接填充 (涂抹)
                        m = _np.zeros((H, W), _np.uint8)
                        import cv2 as _cv2
                        _cv2.fillPoly(m, [poly], 255)
                        alpha = m.astype("float32") / 255.0
                    else:
                        return _fail("圈选区域未找到前景 (GrabCut 失败)，请调整圈选范围重试")
                out = (arr * alpha[..., None] + 255 * (1 - alpha[..., None])).astype("uint8")
                out_p = out_dir / f"{Path(src).stem}_grabcut_{int(time.time())}.png"
                _PIL.fromarray(out).save(out_p)
                tool = "T01_grabcut[圈选]"
                note = "已完成圈选抠图 (GrabCut)，白底前景已生成；可继续滑动圈选修改或用区域擦除/恢复调整"
            else:
                r = studio_cli.task_matting(str(src), str(out_dir), engine=engine)
                if r.get("error"):
                    return _fail(r["error"])
                out_p = Path(r["fg"])                     # 白底预乘前景
                tool, note = f"T01_matting[{engine}]", f"已用 {engine} 引擎完成抠图，前景已生成（可继续换背景/加滤镜/加特效）"
        elif kind == "matte_edit":
            # 圈选区域修改: op=erase(擦除→白底) | restore(恢复原图) | inpaint(删去+丝滑融合)
            op = str(preset.get("op") or "erase")
            box = preset.get("box")
            pts = preset.get("points")
            import numpy as _np
            from PIL import Image as _PIL
            if op == "inpaint":
                # 删除圈选内容并用 LaMa 重建 (手机"AI 消除路人"效果; 背景纯色时主色自适应填充)
                sys.path.insert(0, str(ROOT / ".workbuddy" / "skills" /
                                       "green-screen-composite" / "scripts"))
                import inpaint_engine as IE
                arr = _np.array(_PIL.open(src).convert("RGB"))
                info = {}
                try:
                    out = IE.remove_object(arr[:, :, ::-1], box=box, points=pts, info=info)
                except ValueError as _e:
                    return _fail(str(_e))
                out_p = out_dir / f"{Path(src).stem}_inpaint_{int(time.time())}.png"
                _PIL.fromarray(out[:, :, ::-1]).save(out_p)
                _eng = {"flat": "背景同色自适应融合", "lama": "LaMa 智能重建",
                        "biharmonic": "双调和修复"}.get(info.get("engine"), info.get("engine"))
                tool = f"T01_inpaint[{info.get('engine')}]"
                note = f"已删去圈选内容并丝滑融合背景（{_eng}），看不出修改痕迹"
            else:
                arr = _np.array(_PIL.open(src).convert("RGB"))
                H, W = arr.shape[:2]
                try:
                    x, y, w, h = [int(v) for v in box]
                except (TypeError, ValueError):
                    return _fail("区域修改需要先在画布上拖拽圈选范围")
                x, y = max(0, x), max(0, y)
                w, h = min(w, W - x), min(h, H - y)
                if w < 4 or h < 4:
                    return _fail("圈选区域太小，请重新框选")
                if op == "erase":
                    arr[y:y + h, x:x + w] = 255              # 白底预乘语义: 擦除=置白
                    note = "已擦除圈选区域 (置白底)"
                elif op == "restore":
                    s_path = preset.get("source") or getattr(ag, "orig_image", None)
                    s_path = safe_path(str(s_path)) if s_path else None
                    if s_path is None or not Path(s_path).exists():
                        return _fail("找不到原图，无法恢复该区域")
                    s_arr = _np.array(_PIL.open(s_path).convert("RGB"))
                    if s_arr.shape[:2] != (H, W):
                        s_arr = _np.array(_PIL.fromarray(s_arr).resize((W, H), _PIL.BILINEAR))
                    arr[y:y + h, x:x + w] = s_arr[y:y + h, x:x + w]
                    note = "已将圈选区域恢复为原图像素"
                else:
                    return _fail(f"未知区域操作: {op}")
                out_p = out_dir / f"{Path(src).stem}_{op}_{int(time.time())}.png"
                _PIL.fromarray(arr).save(out_p)
                tool = f"T01_matte_edit[{op}]"
        elif kind == "fx":
            import numpy as _np
            from PIL import Image as _PIL
            arr = _np.array(_PIL.open(src).convert("RGB"))
            effect = str(preset.get("effect") or "spotlight")
            inten = float(preset.get("intensity", 0.6))
            r = studio_cli.task_fx_pil(arr, str(out_dir), effect, inten, None)
            if r.get("error"):
                return _fail(r["error"])
            out_p = Path(r["out_path"])
            tool, note = f"T07_fx[{effect}]", f"已添加特效「{effect}」(强度 {inten})"
        else:
            return _fail(f"未知面板类型: {kind}")
        cur = str(out_p)
        # 链式: 面板产物成为新的当前图; 旧图压入撤销栈 (特效可按顺序叠加+逐步撤销)
        if getattr(ag, "cur_image", None) and ag.cur_image != cur:
            ag.history.append(ag.cur_image)
        ag.cur_image = cur
        sec = round(time.time() - t0, 2)
        return {"steps": [{"node_id": "n1", "tool": tool, "status": "done",
                           "latency_ms": int(sec * 1000), "engine": "", "mode": "",
                           "error": ""}],
                "total_ms": int(sec * 1000), "final": cur,
                "cur_url": "/file?path=" + quote(cur),
                "final_text": f"{note}。耗时 {sec}s。",
                "utility": None}
    except Exception as e:
        import traceback as _tb
        sys.stderr.write("[preset] ERR: " + _tb.format_exc() + "\n")
        return _fail(f"{type(e).__name__}: {e}")


FX_CN_MAP = {
    "vintage_1970s": "70年代胶片", "vintage_1980s": "80年代霓虹", "polaroid": "拍立得",
    "sepia": "怀旧棕褐", "kodachrome": "柯达反转", "portra": "人像胶片", "cinestill": "电影暖调",
    "bleach_bypass": "漂白电影", "bw_classic": "经典黑白", "bw_highcontrast": "高反差黑白",
    "bw_noir": "黑白暗影", "infrared": "红外摄影", "clean_bright": "清新明亮", "moody_dark": "暗黑氛围",
    "pastel": "奶油粉彩", "vibrant_pop": "高饱和糖果", "teal_orange": "青橙电影", "film_golden": "黄金时刻",
    "cyberpunk": "赛博朋克", "vaporwave": "蒸汽波", "nightvision": "夜视仪", "thermal": "热成像", "duotone": "双色调",
    "mosaic_cell": "马赛克色块", "pixelate": "像素化", "oil_painting": "油画", "cartoon_comic": "漫画",
    "toon": "卡通", "sketch_pencil": "铅笔素描", "halftone": "网点印刷", "crystallize": "结晶玻璃",
    "stained_glass": "彩色玻璃", "pointillism": "点彩", "watercolor": "水彩", "vhs": "复古录影带",
    "glitch": "故障特效", "comic_halftone": "漫画网点",
    "vignette": "暗角", "bokeh": "光斑散景", "spotlight": "聚光灯", "bloom": "柔光发光",
    "god_rays": "体积光/神光", "film_grain": "胶片颗粒", "tilt_shift": "移轴微缩",
    "warm": "暖色温", "cool": "冷色温", "light_leak": "漏光",
    "beautify": "一键美颜", "whiten": "美白提亮",
    "mosaic": "马赛克(局部)", "blur_face": "脸部打码", "blur": "高斯模糊",
    "heart": "爱心贴纸", "star": "星星贴纸", "crown": "皇冠贴纸", "bubble": "气泡贴纸",
    "emoji_smile": "笑脸贴纸", "cat": "猫咪贴纸", "flower": "花朵贴纸", "music": "音符贴纸",
    "rainbow": "彩虹贴纸", "watermark_text": "文字水印", "brand_logo": "品牌Logo水印",
    "date_stamp": "日期戳", "qr_code": "二维码水印", "polaroid_frame": "拍立得相框",
    "film_border": "电影黑边", "photo_white": "白色相框", "circle_avatar": "圆形头像角标",
    "rect_badge": "圆角徽章", "camera_lens": "镜头光圈道具",
    # v2 扩充滤镜
    "noir": "黑白暗调", "soft_bw": "柔和黑白", "s_curve": "S曲线", "faded_film": "褪色胶片",
    "lomo": "LOMO", "teal_shadow": "青影", "orange_highlight": "橙光", "fresh_green": "清新绿",
    "royal_purple": "皇家紫", "rose_gold": "玫瑰金", "dusk_mood": "黄昏氛围", "dawn_warm": "晨曦暖",
    "neon_city": "霓虹都市", "retro_synth": "复古合成", "solarize": "反转负冲", "cross_process": "交叉冲印",
    "xpro": "极致冲印", "acid_pop": "酸性流行", "invert_color": "颜色反相", "emboss_light": "浅浮雕",
    "edge_glow": "边缘辉光", "sharp_photo": "锐化清晰", "grainy_film": "粗颗粒胶片", "desat_mood": "低饱和情绪",
    "cream_cream": "奶油白", "lavender": "薰衣草", "baby_blue": "婴儿蓝", "mint": "薄荷绿",
    "peach": "蜜桃粉", "warm_honey": "蜂蜜暖", "cool_steel": "冷钢蓝", "vintage_lomography": "复古LOMO", "night_walk": "夜行",
    # v2 风格
    "dot_screen": "波点网点", "comic_popart": "波普漫画", "psychedelic": "迷幻", "scanlines": "扫描线",
    "twirl": "漩涡扭曲", "lens_distort": "镜头畸变", "liquid": "液态变形", "mirror_kaleido": "镜面万花筒",
    "emboss_art": "浮雕艺术", "neon_edges": "霓虹描边", "smoke_bw": "烟雾素描", "pencil_color": "彩色铅笔", "water_dream": "水彩梦幻",
    # v2 贴纸/水印/边框
    "balloon": "气球贴纸", "moon": "月亮贴纸", "cloud": "云朵贴纸", "lightning": "闪电贴纸",
    "camera_icon": "相机贴纸", "gift": "礼物盒", "snowman": "雪人贴纸", "sun": "太阳贴纸",
    "fish": "小鱼贴纸", "icecream": "雪糕贴纸", "seal_red": "红色印章水印", "diag_watermark": "斜向平铺水印",
    "year_stamp": "年份戳", "frame_double": "白边相框", "corner_marks": "四角取景标记",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静默访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, p: Path, mime: str):
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file_range(self, p: Path, mime: str):
        """文件服务, 支持 HTTP Range / 206 分段(浏览器视频拖动必需), 及 Accept-Ranges 通告。"""
        size = p.stat().st_size
        rng = self.headers.get("Range")
        if rng:
            # 解析 bytes=start-end
            try:
                rng = rng.strip().replace("bytes=", "")
                start_s, _, end_s = rng.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with open(p, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            except Exception:
                pass
        # 无 Range: 全量
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._file(ROOT / "web" / "studio_web.html", "text/html; charset=utf-8")
        elif u.path == "/agent":
            self._file(ROOT / "web" / "studio_agent.html", "text/html; charset=utf-8")
        elif u.path == "/file":
            p = safe_path(q["path"][0])
            if p is None:
                self._json({"error": "invalid path"}, 403)
                return
            suf = p.suffix.lower()
            if suf == ".png": mime = "image/png"
            elif suf in (".jpg", ".jpeg"): mime = "image/jpeg"
            elif suf == ".mp4": mime = "video/mp4"
            elif suf == ".webm": mime = "video/webm"
            elif suf == ".gif": mime = "image/gif"
            else: mime = "application/octet-stream"
            # ?download=1 → 附件下载 (Content-Disposition), 否则内联展示
            if q.get("download"):
                import urllib.parse as _up
                _fname = _up.quote(p.name)
                size = p.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition",
                                 f"attachment; filename=\"{p.name}\"; filename*=UTF-8''{_fname}")
                self.end_headers()
                with open(p, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return
            # 支持 Range (浏览器 <video> 拖动进度条需要 206 Partial Content)
            self._file_range(p, mime)
        elif u.path == "/api/agent/providers":
            # LLM 后端清单 (供 /agent 页面下拉 + 切换)
            sys.path.insert(0, str(ROOT / "ai-agent"))
            import agnes_client as AC
            self._json({"providers": AC.list_providers(),
                        "default": AC.DEFAULT_PROVIDER,
                        "current": globals().get("_AGENT_PROVIDER") or AC.DEFAULT_PROVIDER})
        elif u.path == "/api/agent/provider":
            # 测试连接: ?name=qwen
            sys.path.insert(0, str(ROOT / "ai-agent"))
            import agnes_client as AC
            name = (q.get("name") or [None])[0]
            self._json(AC.AgnesClient(provider=name).ping())
        elif u.path == "/api/effects/catalog":
            sys.path.insert(0, str(ROOT / "ai-service" / "src" / "fx"))
            import fx_library as F
            cat = []
            for name in sorted(F.FILTERS):
                cat.append({"key": name, "cat": "滤镜", "cn": FX_CN_MAP.get(name, name)})
            for name in sorted(F.STYLES):
                cat.append({"key": name, "cat": "风格", "cn": FX_CN_MAP.get(name, name)})
            for name in sorted(F.DECOR):
                cat.append({"key": name, "cat": F.DECOR[name]["cat"], "cn": FX_CN_MAP.get(name, name)})
            for name in ["vignette", "bokeh", "spotlight", "bloom", "god_rays", "film_grain",
                         "tilt_shift", "warm", "cool", "light_leak"]:
                cat.append({"key": name, "cat": "光影", "cn": FX_CN_MAP.get(name, name)})
            self._json(cat)
        elif u.path == "/api/video/probe":
            vp_path = safe_path(q["path"][0])
            if vp_path is None or not vp_path.exists():
                self._json({"error": "video not found"}, 404); return
            try:
                sys.path.insert(0, str(ROOT / "ai-service" / "src" / "video"))
                import video_io as VI
                self._json(VI.probe(str(vp_path)))
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
        elif u.path == "/api/list":
            bgs = sorted(p.name for p in BG_DIR.glob("*.png"))
            fgs = sorted(p.name for p in FG_DIR.glob("*.png"))
            self._json({"bgs": bgs, "fgs": fgs})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n).decode())
        except Exception as e:
            self._json({"error": f"bad json: {e}"}, 400)
            return
        t0 = time.time()
        try:
            if self.path == "/api/upload":
                import base64, io
                raw = base64.b64decode(req["data"].split(",")[-1])
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if img.height > 2400:
                    s = 2400 / img.height
                    img = img.resize((int(img.width * s), 2400), Image.LANCZOS)
                kind = req.get("kind", "fg")
                import uuid
                p = OUT / f"uploaded_{kind}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                img.save(p)
                self._json({"path": str(p.relative_to(ROOT)), "w": img.width, "h": img.height,
                            "b64": to_png_b64(np.array(img))})
            elif self.path == "/api/matting":
                r = studio_cli.task_matting(req["image"], str(OUT / "matting"),
                                            engine=req.get("engine", "hybrid"))
                arr = np.array(Image.open(req if False else safe_path(req["image"])).convert("RGB"))
                alpha = np.array(Image.open(r["alpha"]).convert("L")).astype(np.float32) / 255.0
                alpha = np.array(Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                    (arr.shape[1], arr.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0
                yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
                c = (((yy // 16) + (xx // 16)) % 2).astype(np.uint8)
                board = np.stack([c * 40 + 160] * 3, -1).astype(np.float32)
                vis = (arr * alpha[..., None] + board * (1 - alpha[..., None])).astype(np.uint8)
                vp = OUT / "matting" / "vis.png"
                vp.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(vis).save(vp)
                r["vis"] = str(vp.relative_to(ROOT))
                r["sec"] = round(time.time() - t0, 2)
                self._json(r)
            elif self.path == "/api/grabcut":
                from matting.interactive_matting import grabcut_matte
                img_p = safe_path(req["image"])
                arr = np.array(Image.open(img_p).convert("RGB"))
                sel = req["sel"]
                if sel["type"] == "rect":
                    prompt = {"type": "box", "xyxy": sel["xyxy"]}
                else:
                    prompt = {"type": "polygon", "points": sel["points"]}
                alpha = grabcut_matte(arr, prompt)
                # 绿幕图专用: GrabCut 圈内残留的绿幕用色度键再清一次
                from ai_material_eval import chroma_key_alpha
                ck = chroma_key_alpha(arr)
                ck_ratio = float((ck > 0.5).mean())
                if 0.05 < ck_ratio < 0.95:        # 图中确实存在绿幕背景时才启用
                    alpha = alpha * ck
                if (alpha > 0.5).mean() < 0.01:
                    self._json({"error": "圈选区域无效 (GrabCut 未找到前景), 请重画"}, 400)
                    return
                (OUT / "grabcut").mkdir(parents=True, exist_ok=True)
                ap = OUT / "grabcut" / "alpha.png"
                fp = OUT / "grabcut" / "fg.png"
                Image.fromarray((alpha * 255).astype(np.uint8)).save(ap)
                Image.fromarray(arr).save(fp)
                # 棋盘预览: 让用户看到抠除效果 (而非原图)
                yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
                c = (((yy // 16) + (xx // 16)) % 2).astype(np.uint8)
                board = np.stack([c * 40 + 160] * 3, -1).astype(np.float32)
                vis = (arr * alpha[..., None] + board * (1 - alpha[..., None])).astype(np.uint8)
                vp = OUT / "grabcut" / "vis.png"
                Image.fromarray(vis).save(vp)
                self._json({"alpha": str(ap.relative_to(ROOT)), "fg": str(fp.relative_to(ROOT)),
                            "vis": str(vp.relative_to(ROOT)),
                            "fg_ratio": round(float((alpha > 0.5).mean()), 3)})
            elif self.path == "/api/video/upload":
                import base64 as _b64
                raw = _b64.b64decode(req["data"].split(",")[-1])
                vdir = OUT / "videos"; vdir.mkdir(parents=True, exist_ok=True)
                vp = vdir / req.get("name", f"upload_{int(time.time())}.mp4")
                vp.write_bytes(raw)
                self._json({"path": str(vp.relative_to(ROOT)), "size": vp.stat().st_size})
            elif self.path == "/api/video/process":
                sys.path.insert(0, str(ROOT / "ai-service" / "src" / "video"))
                import video_pipeline as VP
                cfg = req.get("cfg", {})
                inp = safe_path(req["input"])
                out = OUT / "videos" / f"proc_{int(time.time())}.mp4"
                r = VP.process(cfg, str(inp), str(out))
                self._json({"output": str(Path(r["output"]).relative_to(ROOT)),
                            "sec": r["sec"], "frames": r["frames_written"],
                            "size": r["size"]})
            elif self.path == "/api/agent/dag":
                # Agent v2 (Planner-DAG 八层架构版): 拒识/约束/验证器/Critic/回滚
                sys.path.insert(0, str(ROOT / "agent"))
                import traceback as _tb2
                from run import AgentSession
                global _AGENT_DAG, _AGENT_PROVIDER
                if "_AGENT_DAG" not in globals():
                    _AGENT_DAG = AgentSession(quality="draft", enable_critic=True, verbose=False)
                ag = _AGENT_DAG
                # LLM 后端切换 (agnes | qwen): 请求带 provider 即热切换
                _req_provider = (req.get("provider") or "").strip() or None
                if _req_provider and _req_provider != globals().get("_AGENT_PROVIDER"):
                    ag.set_provider(_req_provider)
                    globals()["_AGENT_PROVIDER"] = ag.provider
                ag.enable_critic = bool(req.get("critic", True))
                ag.verbose = False
                if req.get("reset"):
                    _AGENT_DAG = AgentSession(quality="draft", enable_critic=ag.enable_critic,
                                              verbose=False, provider=ag.provider)
                    self._json({"reset": True}); return
                # ---- 面板直通 (2026-09-12): 前端滤镜/抠图/特效按钮的机械执行通道 ----
                # 不经 LLM 规划, 零提示词风险; 作用于会话当前图 (ag.cur_image)
                _preset = req.get("preset")
                if isinstance(_preset, dict) and _preset.get("kind"):
                    _pr = _run_preset(ag, _preset)
                    self._json(_pr); return
                try:
                    _up_img = _up_bg = None          # 回滚分支也要能安全引用 (_resolved 字段)
                    _bg_from_upload = False
                    if req.get("rollback"):
                        r = ag.rollback_background(str(req["rollback"]))
                        r["action"] = "rollback"
                    else:
                        if req.get("quality"):
                            ag.quality = str(req["quality"])   # fast/std/fine 档位
                        else:
                            ag.quality = "draft"   # 未指定档位时复位, 防全局单例 quality 泄漏 (上个请求的 fine 粘住后续所有请求)
                        instr = (req.get("instruction") or "").strip()
                        if not instr:
                            self._json({"error": "instruction 不能为空"}, 400); return
                        _up_img = req.get("image")
                        _up_bg = req.get("bg_image")
                        # 多图上传: images=[{path, role}] (前端可多选, 按上传顺序 = 图1/图2…)
                        _imgs = []
                        for _it in (req.get("images") or []):
                            if isinstance(_it, dict) and _it.get("path"):
                                if safe_path(_it["path"]) is None:
                                    self._json({"error": "非法 images 路径 (目录穿越或不存在)"}, 400)
                                    return
                                _imgs.append({"path": _it["path"],
                                              "role": _it.get("role", "input")})
                        # 校验上传图路径 (防目录穿越)
                        for _k, _v in (("image", _up_img), ("bg_image", _up_bg)):
                            if _v and safe_path(_v) is None:
                                self._json({"error": f"非法 {_k} 路径 (目录穿越或不存在)"}, 400)
                                return
                        _bg_from_upload = False
                        if len(_imgs) >= 2:
                            # 两张图: 指令指代解析 ("图1作为图2的背景") 优先 →
                            # 绿幕占比物理判据 (绿的=待替换场景) → chip角色(兼容) → "背景"词默认
                            _parsed = _parse_two_image_roles(instr)
                            _green = None if _parsed else _green_roles(_imgs)
                            _role_bg = next((x["path"] for x in _imgs if x["role"] == "bg"), None)
                            if _parsed:
                                _bi, _fi = _parsed
                                _up_bg = str(safe_path(_imgs[_bi]["path"]))
                                _up_img = str(safe_path(_imgs[_fi]["path"]))
                                _bg_from_upload = True
                            elif _green:
                                _bi, _fi = _green
                                _up_bg = str(safe_path(_imgs[_bi]["path"]))
                                _up_img = str(safe_path(_imgs[_fi]["path"]))
                                _bg_from_upload = True
                            elif _role_bg:
                                _up_bg = str(safe_path(_role_bg))
                                _up_img = str(safe_path(next(
                                    (x["path"] for x in _imgs if x["path"] != _role_bg),
                                    _imgs[0]["path"])))
                                _bg_from_upload = True
                            elif "背景" in instr:
                                # 有"背景"意图但没解析出指代 → 约定第一张为背景
                                # (与"放入A,B两张, 将A作为B的背景"的上传顺序一致)
                                _up_bg = str(safe_path(_imgs[0]["path"]))
                                _up_img = str(safe_path(_imgs[1]["path"]))
                                _bg_from_upload = True
                            else:
                                # 非换背景类指令 → 只把第一张输入图当前景, 不进双图合成
                                _up_img = str(safe_path(next(
                                    (x["path"] for x in _imgs if x["role"] == "input"),
                                    _imgs[0]["path"])))
                        elif len(_imgs) == 1:
                            if _imgs[0]["role"] == "bg" and not _up_img:
                                _up_bg, _bg_from_upload = str(safe_path(_imgs[0]["path"])), True
                            elif not _up_img:
                                _up_img = str(safe_path(_imgs[0]["path"]))
                        # 单图上传 + "绿幕图用XX背景替换" 类意图 → 该上传图其实是**背景**。
                        # 判定见 _is_bg_replace_intent (保守: 仅当明确提到 绿幕=待换前景 且 背景=目标)
                        if _up_img and not _up_bg and not _bg_from_upload \
                                and _is_bg_replace_intent(instr):
                            _up_bg, _up_img = _up_img, None
                            _bg_from_upload = True
                        # 只有确定为前景的上传图才覆盖 cur_image (背景图不得污染前景上下文)
                        if _up_img and hasattr(ag, "set_cur"):
                            ag.set_cur(str(safe_path(_up_img)))   # 旧图入撤销栈, 首张记为原图
                        elif _up_img:
                            ag.cur_image = str(safe_path(_up_img))
                        r = ag.process(instr, critic_max=1,           # Web 端限 1 轮自评, 控时
                                       bg_image=_up_bg,
                                       bg_from_upload=_bg_from_upload)
                        r["action"] = "process"
                except Exception as e:
                    sys.stderr.write("[agent-dag] ERR: " + _tb2.format_exc() + "\n")
                    sys.stderr.flush()
                    self._json({"error": str(e)[:500]}, 500); return
                # LLM 生成对用户的自然语言汇报 (走当前选中的后端)
                final_text = ""
                try:
                    from agnes_client import AgnesClient
                    brief = {
                        "instruction": req.get("instruction") or f"回滚背景到{req.get('rollback')}",
                        "provider": ag.provider or "default",
                        "abstain": r.get("abstain"),
                        "plan_source": r.get("plan_source"),
                        "steps": [{"tool": x["tool"], "status": x["status"],
                                   "latency_ms": x.get("latency_ms"),
                                   "keep": (x.get("data") or {}).get("keep"),
                                   "source": (x.get("data") or {}).get("source"),
                                   "fx": (x.get("data") or {}).get("fx")} for x in (r.get("run") or {}).get("node_results", [])],
                        "critic": (r.get("critic") or {}).get("scores") if r.get("critic") else None,
                        "utility": r.get("utility"),
                        "rollback": {k: r.get(k) for k in ("rerun", "kept")} if r.get("action") == "rollback" else None,
                    }
                    d = AgnesClient(provider=ag.provider).chat([
                        {"role": "system", "content": "你是演播室合成助手的汇报模块。根据执行记录用不超过3句中文向用户汇报结果(做了什么/成片在哪/质量如何)。**铁律: 若任何 step 的 status 不是 done/kept (failed/skipped), 必须如实汇报失败与原因, 严禁编造'已生成/质量良好'**。注意 keep=background 表示\"删除主体、保留背景\"(不要汇报成抠出人物); keep=subject 或未标表示抠出主体。若 source=green_key 表示\"用色度键把绿幕抠净后直接合成到背景\"(适合绿边残留/精确定位/视频换背景场景); 若 fx 有值表示叠加了色卡滤镜(滤镜名如 5小纸条/8clean 等都是风格化色调)。拒识时礼貌说明只懂图像合成。"},
                        {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}],
                        max_tokens=900, temperature=0.3)  # 推理模型: reasoning 先耗 token
                    final_text = d["choices"][0]["message"].get("content") or ""
                except Exception as _e:
                    sys.stderr.write(f"[agent-dag] summary ERR: {_e}\n")
                    sys.stderr.flush()
                    final_text = ""
                final_path = r.get("final")
                _steps = [{"node_id": x.get("node_id"), "tool": x.get("tool"),
                           "status": x.get("status"), "latency_ms": x.get("latency_ms"),
                           "engine": (x.get("data") or {}).get("engine", ""),
                           "mode": (x.get("data") or {}).get("mode", ""),
                           "error": (x.get("error") or {}).get("message", "") if isinstance(x.get("error"), dict) else ""}
                          for x in (r.get("run") or {}).get("node_results", [])] if r.get("run") else []
                if not _steps and r.get("action") == "rollback":
                    # rollback_background 返回结构与 process 不同: 手工拼节点行
                    rerun = set(r.get("rerun") or []); kept = set(r.get("kept") or [])
                    _steps = [{"node_id": nid, "tool": "", "status": "done", "latency_ms": None, "error": ""}
                              for nid in sorted(rerun)]
                    _steps += [{"node_id": nid, "tool": "", "status": "kept", "latency_ms": None, "error": ""}
                               for nid in sorted(kept)]
                # 程序兜底: 有失败/跳过节点时强制如实汇报 (汇报 LLM 可能幻觉"成功")
                _failed = [x for x in _steps if x.get("status") in ("failed", "skipped")]
                if _failed:
                    _names = "、".join(filter(None, [x.get("tool") for x in _failed])) or "执行节点"
                    _err0 = next((x["error"] for x in _failed if x.get("error")), "")
                    _head = "⚠️ 部分步骤未完成" if final_path else "⚠️ 执行失败"
                    final_text = (f"{_head}: {_names}。{( _err0 or '')[:160]}"
                                  + ("请调整指令或素材后重试。" if not final_path else "已生成的部分结果仅供参考。"))
                self._json({
                    "action": r.get("action"),
                    "abstain": r.get("abstain"),
                    "plan_source": r.get("plan_source"),
                    "provider": ag.provider or "default",
                    "_resolved": {"fg": _up_img, "bg": _up_bg,
                                  "bg_from_upload": _bg_from_upload},
                    "vid": r.get("vid"),
                    "rerun": r.get("rerun"), "kept": r.get("kept"),
                    "steps": _steps,
                    "total_ms": (r.get("run") or {}).get("total_ms"),
                    "critic": (r.get("critic") or {}).get("scores") if r.get("critic") else None,
                    "critic_fix": (r.get("critic") or {}).get("fix") if r.get("critic") else None,
                    "utility": r.get("utility"),
                    "final": final_path,
                    "cur_url": ("/file?path=" + quote(final_path)) if final_path else None,
                    "final_text": final_text,
                    "sec": r.get("sec"),
                })
            elif self.path == "/api/agent/chat":
                sys.path.insert(0, str(ROOT / "ai-agent"))
                import traceback as _tb
                from agent import StudioAgent
                # 简单全局单例 (单用户本地使用足够, 多用户并发可换会话池)
                global _AGENT_SINGLETON
                if "_AGENT_SINGLETON" not in globals():
                    _AGENT_SINGLETON = StudioAgent(enable_critic=True, max_iter=12)
                ag = _AGENT_SINGLETON
                # reset / plan / run
                if req.get("reset"):
                    ag.reset()
                    self._json({"reset": True})
                    return
                instr = (req.get("instruction") or "").strip()
                if not instr:
                    self._json({"error": "instruction 不能为空"}, 400); return
                # 计划 + 执行 (用户在前端触发)
                plan_text = ""
                try:
                    plan_text = ag.plan(instr)
                except Exception as e:
                    plan_text = f"(plan 失败: {e})"
                # 上一轮上下文由 history 维护 (单例)
                t0 = time.time()
                try:
                    r = ag.run(instr)
                except Exception as e:
                    sys.stderr.write("[agent] ERR:\n" + _tb.format_exc() + "\n")
                    sys.stderr.flush()
                    self._json({"error": str(e)[:600]}, 500); return
                critic = None
                if r["cur"] and req.get("critic", True):
                    try:
                        c = ag.critic()
                        critic = c.get("review")
                    except Exception as e:
                        critic = f"(critic 失败: {e})"
                self._json({
                    "plan": plan_text,
                    "steps": [{"tool": t["tool"], "args": t["args"], "sec": t["sec"]}
                              for t in r["trace"]],
                    "n_steps": len(r["trace"]),
                    "n_iter": r["n_iter"],
                    "cur": r["cur"],
                    "cur_url": ("/file?path=" + quote(r["cur"])) if r["cur"] else None,
                    "final_text": r["final_text"],
                    "critic": critic,
                    "sec": round(time.time() - t0, 2),
                })
            elif self.path == "/api/effects/apply":
                sys.path.insert(0, str(ROOT / "ai-service" / "src" / "fx"))
                import fx_library as F
                img_p = safe_path(req["image"])
                arr = np.array(Image.open(img_p).convert("RGB"))
                # region: 用户圈选 (box/poly) -> 自动布局; effect 类型
                out_arr, meta = F.fx_auto.place(arr, req["effect"], req.get("region"), req.get("params"))
                pth = OUT / "fx_auto"
                pth.mkdir(parents=True, exist_ok=True)
                out_f = pth / f"{req['effect']}_{int(time.time())}.png"
                Image.fromarray(out_arr).save(out_f)
                meta["image_b64"] = to_png_b64(out_arr)
                meta["image_path"] = str(out_f.relative_to(ROOT))
                meta["size"] = list(out_arr.shape[:2])
                meta["sec"] = round(time.time() - t0, 2)
                self._json(meta)
            elif self.path == "/api/remove":
                import cv2
                img_p = safe_path(req["image"])
                arr = np.array(Image.open(img_p).convert("RGB"))
                sel = req["sel"]
                mask = np.zeros(arr.shape[:2], np.uint8)
                if sel["type"] == "rect":
                    x1, y1, x2, y2 = [int(max(0, v)) for v in sel["xyxy"]]
                    mask[y1:y2, x1:x2] = 255
                else:
                    pts = np.array(sel["points"], np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))     # 外扩, 覆盖边缘光晕
                # --- 背景填补策略 (丝滑衔接原图背景) ---
                # 0) 全图绿幕判定: 绿幕图上删除一律填绿幕主色 (洞周围是脸/头发也丝滑,
                #    杜绝 Telea 把皮肤色拖进绿幕区的"马赛克"伪影)
                f32 = arr.astype(np.float32)
                excess_all = f32[..., 1] - np.maximum(f32[..., 0], f32[..., 2])
                green_mask = excess_all > 25
                green_ratio = float(green_mask.mean())
                # 环带检测: 洞口边缘一圈的颜色是否均匀 (绿幕/纯色墙 -> 均匀)
                mfloat = (mask.astype(np.float32) / 255.0)
                rng = np.random.default_rng(7)
                if green_ratio > 0.15:
                    # 绿幕图: 全图绿幕主色 + 匹配噪声 + 羽化 (无条件, 最丝滑)
                    ref = np.median(arr[green_mask], axis=0)
                    gstd = float(arr[green_mask].std(axis=0).mean())
                    noise = rng.normal(0, gstd * 0.5, (arr.shape[0], arr.shape[1], 1)).astype(np.float32)
                    fill = np.clip(ref + noise, 0, 255)
                    soft = cv2.GaussianBlur(mfloat, (0, 0), 15)[..., None]
                    out = np.clip(arr * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
                else:
                    ring = cv2.dilate(mask, np.ones((61, 61), np.uint8)) - mask
                    ring_px = arr[ring > 0].astype(np.float32)
                    # 主色簇过滤: 环带里占比最大的颜色簇内部均匀 -> 视为均匀背景
                    ref = np.median(ring_px, axis=0)
                    close = np.linalg.norm(ring_px - ref, axis=1) < 35
                    fill_std = float(ring_px[close].std(axis=0).mean()) if close.sum() > 100 else 999
                    if close.mean() > 0.5 and fill_std < 18:
                        ref = ring_px[close].mean(axis=0)
                        noise = rng.normal(0, fill_std * 0.5, (arr.shape[0], arr.shape[1], 1)).astype(np.float32)
                        fill = np.clip(ref + noise, 0, 255)
                        soft = cv2.GaussianBlur(mfloat, (0, 0), 12)[..., None]
                        out = np.clip(arr * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
                    else:
                        out = cv2.inpaint(arr, mask, 7, cv2.INPAINT_TELEA)
                out_p = OUT / "remove" / "removed.png"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out).save(out_p)
                self._json({"image_path": str(out_p.relative_to(ROOT)),
                            "removed_ratio": round(float((mask > 0).mean()), 3),
                            "sec": round(time.time() - t0, 2)})
            elif self.path == "/api/greenscreen":
                r = studio_cli.task_greenscreen(req["image"], req["bg"], str(OUT / "greenscreen"))
                if "error" in r:
                    self._json(r, 400)
                    return
                r["image_b64"] = to_png_b64(np.array(Image.open(r["output"]).convert("RGB")))
                r["image_path"] = str(Path(r["output"]).relative_to(ROOT))
                self._json(r)
            elif self.path == "/api/composite":
                from pathlib import Path as _P
                bg_stem = _P(req["bg"]).stem
                r = studio_cli.task_composite(req["image"], req["bg"], str(OUT / "composite" / bg_stem),
                                              alpha=req.get("alpha"),
                                              harmonize=req.get("harmonize", True),
                                              relight=req.get("relight", True),
                                              shadow=req.get("shadow", True))
                r["final_b64"] = to_png_b64(np.array(Image.open(r["final"]).convert("RGB")))
                self._json(r)
            elif self.path == "/api/fx":
                img_p = safe_path(req["image"])
                arr = np.array(Image.open(img_p).convert("RGB"))
                r = studio_cli.task_fx_pil(arr, str(OUT / "fx"), req["effect"],
                                           float(req.get("intensity", 0.6)), req.get("region"))
                if "error" in r:
                    self._json(r, 400)
                    return
                r["image_b64"] = to_png_b64(r["image"])
                r["image_path"] = str(Path(r["out_path"]).relative_to(ROOT))
                r.pop("image")          # ndarray 不可 JSON 序列化
                self._json(r)
            elif self.path == "/api/save":
                src = safe_path(req["src"])
                dst = OUT / f"result_{int(time.time())}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.open(src).save(dst)
                self._json({"saved": str(dst)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>演播室图像合成工作台</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: "Microsoft YaHei", sans-serif; background: #20242b; color: #e8eaed;
         display: flex; height: 100vh; overflow: hidden; }
  #panel { width: 300px; padding: 14px; background: #2a2f38; overflow-y: auto; flex-shrink: 0; }
  #panel h3 { font-size: 14px; margin: 14px 0 6px; color: #8ab4f8; }
  #panel h3:first-child { margin-top: 0; }
  button { background: #3c4043; color: #e8eaed; border: 1px solid #5f6368; border-radius: 6px;
           padding: 7px 10px; cursor: pointer; width: 100%; margin: 3px 0; font-size: 13px; }
  button:hover { background: #4a4d51; }
  button.primary { background: #1a73e8; border-color: #1a73e8; }
  select, input[type=file] { width: 100%; margin: 3px 0; padding: 5px; background: #3c4043;
           color: #e8eaed; border: 1px solid #5f6368; border-radius: 6px; font-size: 13px; }
  label.chk { display: block; font-size: 13px; margin: 3px 0; color: #bdc1c6; }
  #stage { flex: 1; display: flex; flex-direction: column; align-items: center;
           justify-content: center; padding: 14px; position: relative; }
  #cv { background: #3a3a3a; cursor: crosshair; max-width: 100%; max-height: 100%;
        border: 1px solid #5f6368; }
  #status { position: absolute; bottom: 0; left: 0; right: 0; background: #111;
            padding: 8px 14px; font-size: 13px; color: #9aa0a6; }
  .hint { font-size: 12px; color: #9aa0a6; margin: 4px 0; }
  #intensity { width: 100%; }
</style>
</head>
<body>
<div id="panel">
  <h3>① 选择绿幕图片</h3>
  <input type="file" id="file" accept="image/*">
  <div class="hint">或使用预置素材:</div>
  <select id="fgList"></select>
  <div class="hint">抠图引擎:</div>
  <select id="engine">
    <option value="hybrid">hybrid (色度键+BiRefNet 补洞, 推荐)</option>
    <option value="chroma">chroma (纯色度键, 最快)</option>
    <option value="refiner">refiner (BiRefNet+精修, 非绿幕图用)</option>
  </select>

  <h3>② 圈选与抠图</h3>
  <div class="hint">在图上拖拽鼠标画红框。框的用途由下面选择:</div>
  <label class="chk"><input type="radio" name="selmode" value="fx" checked> 圈选 = 特效区域</label>
  <label class="chk"><input type="radio" name="selmode" value="grabcut"> 圈选 = GrabCut 抠图范围</label>
  <button onclick="doMatting()">自动抠图 (整图)</button>
  <button onclick="doGrabcut()">按圈选 GrabCut 抠图</button>

  <h3>③ 背景叠加</h3>
  <select id="bgList"></select>
  <label class="chk"><input type="checkbox" id="optHarm" checked> 和谐化 (防压黑)</label>
  <label class="chk"><input type="checkbox" id="optLight" checked> 重打光 (方向光)</label>
  <label class="chk"><input type="checkbox" id="optShadow" checked> 接触阴影</label>
  <button class="primary" onclick="doComposite()">背景叠加</button>

  <h3>④ 添加特效</h3>
  <select id="fxName">
    <option value="spotlight">聚光灯</option>
    <option value="bokeh">光斑散景</option>
    <option value="fog">舞台雾效</option>
    <option value="vignette">暗角</option>
    <option value="color_temp">色温滤镜</option>
    <option value="depth_blur">景深虚化</option>
  </select>
  <div class="hint">强度: <span id="iv">0.60</span></div>
  <input type="range" id="intensity" min="0.1" max="1" step="0.05" value="0.6"
         oninput="document.getElementById('iv').textContent=this.value">
  <div class="hint">勾选上面"圈选 = 特效区域"后画框, 特效只作用框内。</div>
  <button onclick="doFx()">添加特效</button>
  <button onclick="undo()">↩ 撤销上一步</button>

  <h3>⑤ 输出</h3>
  <button onclick="doSave()">保存结果 PNG</button>
  <div class="hint">同一引擎与 CLI 通用: app/studio_cli.py</div>
</div>
<div id="stage">
  <canvas id="cv" width="900" height="620"></canvas>
  <div id="status">就绪。选择一张绿幕图片开始。</div>
</div>
<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let cur = null;          // 当前显示 b64
let curPath = null;      // 当前图在服务器的路径
let sel = null;          // 圈选 {x1,y1,x2,y2} 显示坐标
let drag = null;
let hist = [];

function log(s){ document.getElementById('status').textContent = s; }
function setImage(b64, path){ cur = b64; curPath = path; hist = []; draw(); }

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if (!cur) return;
  const im = new Image();
  im.onload = () => {
    const s = Math.min((cv.width-20)/im.width, (cv.height-20)/im.height, 1);
    cv._scale = s; cv._iw = im.width; cv._ih = im.height;
    cv._dw = im.width*s; cv._dh = im.height*s;
    ctx.drawImage(im, 10, 10, cv._dw, cv._dh);
    if (sel){
      ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 2; ctx.setLineDash([7,5]);
      ctx.strokeRect(10+sel.x1*s, 10+sel.y1*s, (sel.x2-sel.x1)*s, (sel.y2-sel.y1)*s);
      ctx.setLineDash([]);
    }
  };
  im.src = 'data:image/png;base64,' + cur;
}

// 鼠标圈选
cv.addEventListener('mousedown', e=>{
  const r = cv.getBoundingClientRect();
  drag = {x: e.clientX-r.left, y: e.clientY-r.top};
});
cv.addEventListener('mousemove', e=>{
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  const x2 = e.clientX-r.left, y2 = e.clientY-r.top;
  if (cur) draw();
  ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 2; ctx.setLineDash([7,5]);
  ctx.strokeRect(drag.x, drag.y, x2-drag.x, y2-drag.y);
  ctx.setLineDash([]);
});
cv.addEventListener('mouseup', e=>{
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  const s = cv._scale || 1;
  const x1 = (Math.min(drag.x, e.clientX-r.left)-10)/s, y1 = (Math.min(drag.y, e.clientY-r.top)-10)/s;
  const x2 = (Math.max(drag.x, e.clientX-r.left)-10)/s, y2 = (Math.max(drag.y, e.clientY-r.top)-10)/s;
  drag = null;
  if (x2-x1 > 8 && y2-y1 > 8){ sel = {x1,y1,x2,y2}; log('圈选完成: '+JSON.stringify(sel)); draw(); }
});

async function api(path, body){
  log('处理中…');
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body)});
  const j = await r.json();
  if (j.error){ log('错误: '+j.error); throw new Error(j.error); }
  return j;
}

async function loadLists(){
  const j = await fetch('/api/list').then(r=>r.json());
  const fg = document.getElementById('fgList');
  j.fgs.forEach(n=>{ const o=document.createElement('option'); o.value='data/ai_generated/green_fg/'+n; o.textContent=n; fg.appendChild(o); });
  const bg = document.getElementById('bgList');
  j.bgs.forEach(n=>{ const o=document.createElement('option'); o.value='data/ai_generated/studio_bg/'+n; o.textContent=n; bg.appendChild(o); });
  // 默认加载第一张
  fg.value = 'data/ai_generated/green_fg/' + j.fgs[0];
  await loadPreset();
}
async function loadPreset(){
  const p = document.getElementById('fgList').value;
  const im = new Image();
  im.onload = ()=>{ cur = null; curPath = p; hist = [];
    const s = Math.min((cv.width-20)/im.width, (cv.height-20)/im.height, 1);
    cv._scale = s; cv._iw = im.width; cv._ih = im.height;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(im, 10, 10, im.width*s, im.height*s);
    sel = null;
    log('已加载 ' + p.split('/').pop() + '。可圈选, 或直接点自动抠图。');
  };
  im.src = '/file?path=' + encodeURIComponent(p);
}
document.getElementById('fgList').onchange = loadPreset;

document.getElementById('file').onchange = async e=>{
  const f = e.target.files[0]; if (!f) return;
  const b64 = await new Promise(res=>{ const rd=new FileReader();
    rd.onload=()=>res(rd.result); rd.readAsDataURL(f); });
  const j = await api('/api/upload', {data: b64});
  setImage(j.b64, j.path);
  log('已上传 ' + f.name + ' (' + j.w + 'x' + j.h + ')');
};

async function doMatting(){
  if (!curPath) return log('请先选择图片');
  const j = await api('/api/matting', {image: curPath, engine: document.getElementById('engine').value});
  hist.push({b64: cur, path: curPath});
  setImage(j.fg_b64, j.fg); cur._alphaPath = j.alpha;
  window._lastAlpha = j.alpha;
  log('抠图完成 (engine=' + j.engine + ', 前景 ' + Math.round(j.fg_ratio*100) + '%, ' + j.sec + 's)。');
}
async function doGrabcut(){
  if (!curPath) return log('请先选择图片');
  if (!sel) return log('请先在图上画圈选框, 并选中"圈选 = GrabCut 抠图范围"');
  const j = await api('/api/grabcut', {image: curPath, rect: [sel.x1, sel.y1, sel.x2, sel.y2]});
  hist.push({b64: cur, path: curPath});
  setImage(j.fg_b64, j.fg); window._lastAlpha = j.alpha;
  log('圈选 GrabCut 抠图完成。');
}
async function doComposite(){
  if (!curPath) return log('请先选择图片');
  const j = await api('/api/composite', {
    image: curPath, bg: document.getElementById('bgList').value,
    alpha: window._lastAlpha || null,
    harmonize: document.getElementById('optHarm').checked,
    relight: document.getElementById('optLight').checked,
    shadow: document.getElementById('optShadow').checked});
  hist.push({b64: cur, path: curPath});
  setImage(j.final_b64, j.final);
  log('背景叠加完成 (' + j.total_ms + 'ms)。可继续添加特效。');
}
async function doFx(){
  if (!curPath) return log('没有可加特效的图');
  const mode = document.querySelector('input[name=selmode]:checked').value;
  let region = null;
  if (mode === 'fx'){
    if (!sel) return log('特效圈选模式需要先画框');
    region = {type:'box', xyxy:[sel.x1, sel.y1, sel.x2, sel.y2]};
  }
  const j = await api('/api/fx', {image: curPath, effect: document.getElementById('fxName').value,
    intensity: parseFloat(document.getElementById('intensity').value), region});
  hist.push({b64: cur, path: curPath});
  setImage(j.image_b64, j.image_path);
  log('特效已添加 (' + (region ? '圈选区域' : '全图') + ')。');
}
function undo(){
  if (hist.length){ const h = hist.pop(); setImage(h.b64, h.path); log('已撤销。'); }
  else log('没有可撤销的步骤。');
}
async function doSave(){
  if (!curPath) return log('没有可保存的结果');
  const j = await api('/api/save', {src: curPath});
  log('已保存: ' + j.saved);
}
loadLists();
</script>
</body>
</html>
"""


def main():
    if not (ROOT / "web" / "studio_web.html").exists():
        (ROOT / "web" / "studio_web.html").write_text(HTML, encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[studio_web] http://127.0.0.1:{PORT}  (Ctrl+C 退出)")
    webbrowser.open(f"http://127.0.0.1:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[studio_web] bye")


if __name__ == "__main__":
    main()
