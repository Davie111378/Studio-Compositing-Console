# -*- coding: utf-8 -*-
"""
adapter.py — A–B 对接适配层 (《A_B接口对接说明_V1.0》§2/§3 落地)
规范 8 工具 (T01–T08) → B 组现有函数, 统一四件套信封:

  { node_id, tool, ok, data, error{code,message,retryable}|null,
    latency_ms, model_version, quality }

要点 (接口文档 §1):
  - 文件即契约: 全路径交互
  - fg 白底预乘: img*a + 255*(1-a)
  - 错误统一捕捉: B 组 fx 返回 error dict, 其余抛异常 → 适配层统一归一
  - 质量档位: draft/normal/fine → B 内部参数映射 (Schema quality_tiers)
  - T06 吸收规范入参: 适配层内部先 alpha_over 合成再对 composite 调 harmonize
"""
from __future__ import annotations
import sys, time, json, shutil, zipfile, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sp in [str(ROOT), str(ROOT / "ai-service"), str(ROOT / "ai-service" / "src"),
           str(ROOT / "ai-agent"), str(ROOT / "ai-service" / "src" / "fx"),
           str(ROOT / "training" / "scripts")]:
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np
from PIL import Image

OUT = ROOT / "outputs" / "agent_dag"
OUT.mkdir(parents=True, exist_ok=True)


# ---- T01 fine 档 AlphaRefiner 权重注册位 (训练方案_真实人像P3M10k_HM1k §7) ----
# 优先级: 环境变量 REFINER_WEIGHT(绝对路径) > checkpoints/active_refiner.txt 指定域 > refiner_studio
# 切换域只需改写 active_refiner.txt 内容(如 "refiner_real")，无需改代码。
def _active_refiner_name() -> str:
    ptr = ROOT / "training" / "checkpoints" / "active_refiner.txt"
    if ptr.exists():
        n = ptr.read_text(encoding="utf-8").strip()
        if n and (ROOT / "training" / "checkpoints" / n / "best.pt").exists():
            return n
    return "refiner_studio"


def refiner_weight() -> Path:
    import os
    env = os.environ.get("REFINER_WEIGHT")
    if env and Path(env).exists():
        return Path(env)
    return ROOT / "training" / "checkpoints" / _active_refiner_name() / "best.pt"


TOOL_VERSIONS = {
    "T01_matting": "birefnet_official@512 + " + _active_refiner_name() + "(fine)",
    "T02_background_generate": "assets-v1 | agnes-image-2.5-flash",
    "T03_lighting_estimate": "lambert-hemisphere-approx v0.9",
    "T04_relight": "relight directional v1.0",
    "T05_shadow_generate": "shadow procedural v1.0",
    "T06_harmonize": "harmonize v2 (FDR 防压黑) + composite alpha_over",
    "T07_enhance": "fx 兜底 v0.5",
    "T08_export": "export v1.0",
}


class ToolError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


# --------------------------------------------------------------- T01 matting
def t01_matting(p: dict, quality: str, ctx: dict) -> dict:
    img = p.get("image") or ""
    ip = Path(ROOT / img if not Path(img).is_absolute() else img)
    if not ip.exists():
        # 允许素材短名
        for d in ("data/ai_generated/green_fg", "data/ai_generated/green_fg_hard",
                  "data/ai_generated/studio_bg"):
            c = ROOT / d / img
            if c.exists():
                ip = c
                break
        else:
            raise ToolError("E_INPUT_MISSING", f"输入图不存在: {img}")
    mode = p.get("mode", "auto")
    keep = p.get("keep", "subject")
    refine = (quality == "fine")
    out_dir = OUT / "t01" / ctx["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ip.stem
    alpha_p, fg_p = out_dir / f"{stem}_alpha.png", out_dir / f"{stem}_fg.png"

    if mode == "interactive":
        from matting.interactive_matting import InteractiveMatting
        prompt = {}
        if p.get("box"):
            prompt = {"type": "box", "xyxy": p["box"]}
        elif p.get("points"):
            prompt = {"type": "points", "positive": p["points"]}
        elif p.get("polygon"):
            prompt = {"type": "polygon", "points": p["polygon"]}
        else:
            raise ToolError("E_MODE_CONFLICT", "interactive 模式需要 box/points/polygon")
        im = InteractiveMatting(refiner_weight=str(refiner_weight())
                                if refine else None)
        r = im.matte(str(ip), str(alpha_p), prompt)
        fg_ratio = r.get("fg_ratio", 0)
        # interactive 只出 alpha, 需补 fg (白底预乘)
        arr = np.array(Image.open(ip).convert("RGB")).astype(np.float32)
        a = np.array(Image.open(alpha_p).convert("L")).astype(np.float32) / 255.0
        fg = arr * a[..., None] + 255.0 * (1 - a[..., None])
        Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8)).save(fg_p)
        return {"alpha_path": str(alpha_p), "fg_path": str(fg_p),
                "fg_ratio": round(float(fg_ratio), 3), "engine": "grabcut+refiner" if refine else "grabcut"}

    # ---- MODNet 人像抠图引擎 (用户指定 engine=humanmatting* 时启用) ----
    # 来源: PaddleSeg/Matting 官方 MODNet 双 backbone (ONNX, onnxruntime 推理)。
    # 语义级 alpha, 适合非绿幕通用人像; 绿幕图仍建议默认链 (chroma/hybrid 带 despill)。
    # keep=background 不走此分支 (删主体逻辑依赖 BiRefNet 连通域分析)。
    _engine = p.get("engine") or ""
    if _engine.startswith("humanmatting") and keep != "background" and mode != "interactive":
        import cv2 as _cvmod
        from matting.humanmatting_engine import human_matting
        arr0 = np.array(Image.open(ip).convert("RGB"))
        alpha = human_matting(_cvmod.cvtColor(arr0, _cvmod.COLOR_RGB2BGR),
                              model="mobilenet" if _engine.endswith("fast") else "hrnet")
        Image.fromarray((alpha * 255).astype(np.uint8)).save(alpha_p)
        fg = arr0.astype(np.float32) * alpha[..., None] + 255.0 * (1 - alpha[..., None])
        Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8)).save(fg_p)
        model_name = "humanmatting-" + ("mobilenetv2" if _engine.endswith("fast") else "hrnet_w18")
        return {"alpha_path": str(alpha_p), "fg_path": str(fg_p),
                "fg_ratio": round(float((alpha > 0.5).mean()), 3), "engine": model_name,
                "keep": keep}

    # auto 模式: BiRefNet
    from matting.matting_backend import build_matting_tool
    tool, model_name = build_matting_tool()
    r = tool.predict(str(ip), str(alpha_p), str(fg_p))
    alpha = np.array(Image.open(alpha_p).convert("L")).astype(np.float32) / 255.0
    if refine:
        # fine 档: AlphaRefiner 残差精修 (refiner_studio 域内权重)
        try:
            import torch, cv2
            from train_refiner import AlphaRefiner
            w = refiner_weight()
            if w.exists():
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                net = AlphaRefiner(base=32).to(dev)
                ck = torch.load(w, map_location=dev, weights_only=False)
                net.load_state_dict(ck["model"]); net.eval()
                arr = np.array(Image.open(ip).convert("RGB").resize((512, 512))).astype(np.float32) / 255.0
                co = cv2.resize(alpha, (512, 512)).astype(np.float32)
                rgb_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
                co_t = torch.from_numpy(co).unsqueeze(0).unsqueeze(0).to(dev)
                with torch.no_grad():
                    pred = net(rgb_t, co_t)
                a_s = pred[0, 0].float().clamp(0, 1).cpu().numpy()
                alpha = cv2.resize(a_s, (alpha.shape[1], alpha.shape[0]))
                arrf = np.array(Image.open(ip).convert("RGB")).astype(np.float32)
                fg = arrf * alpha[..., None] + 255.0 * (1 - alpha[..., None])
                Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8)).save(fg_p)
                Image.fromarray((alpha * 255).astype(np.uint8)).save(alpha_p)
                model_name += "+" + w.parent.name
        except Exception:
            # 精修失败 → 保持 coarse (程序化降级), 但必须留痕便于排查
            sys.stderr.write("[T01.fine] refiner 精修失败:\n" + traceback.format_exc() + "\n")
            sys.stderr.flush()

    # ---- alpha 后处理: 填充空洞 + 保持躯干连通性 ----
    # BiRefNet 在低分辨率/高压缩图上常产生颈部/躯干空洞 → 轮廓填充 + 形态学修复
    if keep == "subject":
        import cv2 as _cv2
        h_a, w_a = alpha.shape[:2]
        bin_a = (alpha > 0.3).astype(np.uint8)
        # 1) 大核闭运算: 合并头部与躯干碎片 (颈部断裂通常 30-80px)
        bin_closed = _cv2.morphologyEx(bin_a, _cv2.MORPH_CLOSE,
                                        _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (31, 31)))
        # 2) 取最大连通域 (去碎噪)
        n_cc, lbl = _cv2.connectedComponents(bin_closed)
        if n_cc > 2:
            areas = [(i, int((lbl == i).sum())) for i in range(1, n_cc)]
            areas.sort(key=lambda x: -x[1])
            largest = areas[0][0]
            bin_clean = (lbl == largest).astype(np.uint8)
        else:
            bin_clean = bin_closed
        # 3) 轮廓填充: 用外轮廓填充凹陷 (包括与背景连通的颈部空洞)
        contours, _ = _cv2.findContours(bin_clean, _cv2.RETR_EXTERNAL,
                                         _cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            big = max(contours, key=_cv2.contourArea)
            filled = np.zeros((h_a, w_a), np.uint8)
            _cv2.fillPoly(filled, [big], 1)
            concavity = filled & (1 - bin_a)
            # 只有显著凹陷 (>3% 画面) 才填充, 避免对 P3M 等干净 alpha 的过度修改
            if concavity.sum() > 0.03 * h_a * w_a:
                alpha = np.maximum(alpha, filled.astype(np.float32) * 0.92)
                arr_f = np.array(Image.open(ip).convert("RGB")).astype(np.float32)
                fg = arr_f * alpha[..., None] + 255.0 * (1 - alpha[..., None])
                Image.fromarray((alpha * 255).astype(np.uint8)).save(alpha_p)
                Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8)).save(fg_p)
                model_name += "+postproc(contour_fill)"

    # keep=background: 反转 alpha —— 删除主体(人物), 保留背景。
    # 用于 "扣去/去掉/移除图中人物" 类指令 (语义与"抠出人物"相反)。
    if keep == "background":
        import cv2
        h, w = alpha.shape[:2]
        arr_orig = np.array(Image.open(ip).convert("RGB")).astype(np.float32)
        orig_soft_alpha = alpha.copy()   # 保留软 alpha, 用于覆盖抗锯齿边缘
        # ---- 人/实物分离 ----
        # BiRefNet 常把"人物 + 身前桌台"并成一个连通前景; 不能靠亮度(人脸/衬衫也亮),
        # 也不能只靠宽度(人像肩胸同样下宽)。稳健判据 = **宽度曲线是否出现"骤增后平台期"**:
        #   桌台: 在某一行宽度**骤增**(≥1.35×), 且之后宽度基本恒定(平台: 后半段标准差小) → 是横向家具;
        #   人像: 宽度自上而下**渐进增大**(梯形躯干), 无骤增+平台结构。
        # 只有同时满足"骤增 + 平台"才切出保留段, 否则(纯人像)整块删除。
        # 先做形态学闭运算: BiRefNet 在人物与桌台交界处常有细缝/毛刺 → 会把人+桌台切成
        # 多个连通域(人被当独立小域全删、桌台域又拿不到"骤增"上下文)。闭运算把小缝焊上。
        person = (alpha > 0.5).astype(np.uint8)
        person = cv2.morphologyEx(person, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n_lbl, lbl = cv2.connectedComponents(person)
        keep_obj = np.zeros((h, w), np.uint8)
        protect_from = None
        for i in range(1, n_lbl):
            comp = (lbl == i)
            area = int(comp.sum())
            if area < 0.01 * h * w:
                continue
            rows_i = np.where(comp.any(axis=1))[0]
            if len(rows_i) < 8:
                continue
            top, bot = int(rows_i[0]), int(rows_i[-1])
            widths = comp.sum(axis=1).astype(np.float32) / max(w, 1)
            span = max(bot - top, 1)
            cand = None
            for r in range(top + 2, bot + 1):
                base_end = max(top + 1, r - int(span * 0.10))
                base = float(widths[top:base_end].max()) if base_end > top else 0.0
                if widths[r] < 0.42 or base < 1e-6:
                    continue
                if widths[r] < 1.35 * base:          # 相对远上方基线骤增
                    continue
                tail = widths[r:bot + 1]
                if len(tail) < 0.06 * h:
                    continue
                # ---- 关键鉴别: 下半段是"平台"(桌台) 还是"持续变宽"(人像梯形躯干) ----
                # 线性拟合斜率: 平台≈0/负; 躯干明显为正(每行都在变宽)。实测 studio=-0.22, 人像=+0.28/+0.43
                xs = np.arange(len(tail), dtype=np.float32)
                slope = float(np.polyfit(xs, tail, 1)[0]) if len(tail) > 2 else 0.0
                slope_per_span = slope * len(tail) / max(float(tail.mean()), 1e-6)
                if slope_per_span > 0.25:            # 持续变宽 → 不是桌台
                    continue
                # 主体一直延伸到画面底部(触底) → 更像人像/大件, 不用保护
                if bot > 0.97 * h:
                    continue
                if float(tail.std()) / max(float(tail.mean()), 1e-6) > 0.30:
                    continue
                if float(widths[top:base_end].min()) > 0.25:   # 上方须有窄段(头/颈)
                    continue
                cand = r
                break
            if cand is None:
                continue
            seg = np.zeros((h, w), np.uint8)
            seg[cand:, :] = comp[cand:, :].astype(np.uint8) * 255
            keep_obj = cv2.bitwise_or(keep_obj, seg)
            protect_from = cand if protect_from is None else min(protect_from, cand)
        # ---- 凸包补洞: 把桌台区域内被算作前景的"人物残段"(躯干/头发)一起并入保留区 ----
        # 成因: 人在桌台后方时, 桌面以下的躯干段与桌面段同属一个连通域, 但 mask 上是
        #       "桌台盘面 + 两个三角形躯干"的锯齿边界; ^ 只保留 cand 行以下 → 躯干残角留下。
        # 用保留区轮廓的凸包包住这些凹口(仅限 cand 行以下, 不越界到桌面上方)。
        if keep_obj.any() and protect_from is not None:
            cnts, _ = cv2.findContours(keep_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                big = max(cnts, key=cv2.contourArea)
                hull = cv2.convexHull(big)
                hull_m = np.zeros((h, w), np.uint8)
                cv2.drawContours(hull_m, [hull], -1, 255, -1)
                hull_m[:protect_from, :] = 0            # 不越过保护起点, 避免糊掉上半身
                if int(hull_m.sum()) - int(keep_obj.sum()) < 0.35 * int(keep_obj.sum()):
                    keep_obj = hull_m                    # 凸包膨胀合理才采用(防极端外扩)
        # 真正要删的 = 主体 - 保留实物
        erase = cv2.bitwise_and(person * 255, cv2.bitwise_not(keep_obj))
        rem_p = out_dir / f"{stem}_removedbg.png"
        # 用**软 alpha** 覆盖抗锯齿边缘(二值化 >0.5 会漏掉半透明边缘 → 残留人形轮廓)。
        soft_a = np.clip(orig_soft_alpha, 0, 1)
        erase_soft = np.maximum(erase.astype(np.float32) / 255.0, soft_a * (keep_obj == 0))
        # 从邻域扩散填充(绿墙局部明暗自适应)
        m_erase = cv2.dilate((erase_soft > 0.15).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        removed = cv2.inpaint(arr_orig.astype(np.uint8), m_erase, 7, cv2.INPAINT_TELEA).astype(np.float32)
        # ---- 绿色域校正: Telea 会从紧邻的白色桌台顶边取色, 把"人贴桌台"处填成白三角/白残影 ----
        # 判据: 被删区里"不够绿"(G - max(R,B) 小)的像素 = 白/灰残影, 视为 bad。
        # 填充: 用 2D 最近"够绿"像素(距离变换)补色 —— 比逐列插值更稳, 因为桌台就在下方
        #       会污染列参考, 而横向邻域一定还有真绿幕。
        if int(m_erase.sum()) > 0:
            er = (m_erase > 0)
            g = removed
            gd = g[..., 1] - np.maximum(g[..., 0], g[..., 2])
            bad = (gd < 12) & er
            if bad.any():
                # 参考绿: 画面顶部横带(远离人物)的中位色, 作为兜底
                pad = np.zeros((h, w), bool); pad[: int(0.25 * h), :] = True
                pad &= (person == 0)
                gref = np.median(arr_orig[pad], axis=0) if pad.sum() > 2 else \
                    np.array([40.0, 170.0, 60.0], np.float32)
                # 可信绿像素 = 被删区内的"够绿"像素; 若无 → 用顶部绿带像素
                src = (~bad) & er
                if src.sum() < 50:
                    src = pad
                if not src.any():
                    removed = np.where(bad[..., None], gref[None, None, :], g)
                else:
                    # 距离变换: 每个像素到最近可信源的索引
                    inv = (~src).astype(np.uint8)
                    dist, labels = cv2.distanceTransformWithLabels(
                        inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                    src_idx = np.flatnonzero(src.ravel())
                    # labels 里的 0 对应"最靠近的 src 像素"编号(从1开始, 与 src_idx 顺序一致)
                    lut = src_idx[labels.ravel() - 1].reshape(h, w)
                    nearest = g.reshape(-1, 3)[lut.ravel()].reshape(h, w, 3)
                    # 平滑一下避免邻块跳变
                    nearest = cv2.GaussianBlur(nearest.astype(np.float32), (0, 0), 2.0)
                    removed = np.where(bad[..., None], nearest, g)
        # 保留实物区域贴回原图(桌面/麦克风零损伤)
        if keep_obj.any():
            ko = cv2.GaussianBlur((keep_obj.astype(np.float32) / 255.0), (0, 0), 2)[..., None]
            removed = removed * (1 - ko) + arr_orig * ko
            erase_soft = erase_soft * (1 - ko[..., 0])
        # 边界羽化
        soft = cv2.GaussianBlur(erase_soft, (0, 0), 1.5)[..., None]
        removed = np.clip(arr_orig * (1 - soft) + removed * soft, 0, 255)
        # 保留掩码 = 背景(非人) ∪ 保留实物
        keep_mask = np.where((person == 0) | (keep_obj > 0), 255, 0).astype(np.uint8)
        alpha = keep_mask.astype(np.float32) / 255.0
        fg = arr_orig * alpha[..., None] + 255.0 * (1 - alpha[..., None])
        Image.fromarray((alpha * 255).astype(np.uint8)).save(alpha_p)
        Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8)).save(fg_p)
        Image.fromarray(removed.astype(np.uint8)).save(rem_p)
        model_name += f"+inverse(keep=bg,protect_from={protect_from})"
        return {"alpha_path": str(alpha_p), "fg_path": str(fg_p),
                "removed_path": str(rem_p), "fg_ratio": round(float((alpha > 0.5).mean()), 3),
                "engine": model_name, "keep": keep}

    return {"alpha_path": str(alpha_p), "fg_path": str(fg_p),
            "fg_ratio": round(float((alpha > 0.5).mean()), 3), "engine": model_name,
            "keep": keep}


# --------------------------------------------------------------- T02 bg
def _green_ratio(path: str) -> float:
    """绿幕色度占比: G 通道显著超 R/B 的像素比例 (中文路径安全)。

    色度绿幕场景图通常 >15%; 普通实景 (篮球场/街道) ≈0。用于 green_key
    模式的角色物理判别——LLM 把 app_fg/file 填反时自动纠偏。
    """
    try:
        import cv2 as _cv2
        _arr = _cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if _arr is None:
            return 0.0
        _f = _arr.astype(np.float32)
        return float((_f[..., 1] - np.maximum(_f[..., 0], _f[..., 2]) > 25).mean())
    except Exception:
        return 0.0


def t02_background_generate(p: dict, quality: str, ctx: dict) -> dict:
    sys.path.insert(0, str(ROOT / "ai-agent"))
    import assets
    # ---- mode=green_key: 色度键直合成 (来自 skills/green-screen-composite) ----
    # 适合: 用户给了绿幕前景 + 背景, 想一步到位出合成图; 或 BiRefNet 抠不净绿边时用 HSV 键更稳。
    if p.get("mode") == "green_key" or p.get("app_fg"):
        # 防角色填反 (2026-09-12): app_fg/file 都给且 file 更"绿" → 互换。
        # 绿幕场景图 (色度绿占比高) 必须是 app_fg; 物理判据, 不依赖 LLM 自觉。
        if p.get("app_fg") and p.get("file"):
            _fa, _fi = _abs(p["app_fg"]), _abs(p["file"])
            if _fa != _fi and Path(_fa).exists() and Path(_fi).exists():
                _gra, _gri = _green_ratio(_fa), _green_ratio(_fi)
                if _gri > _gra + 0.10:      # file 侧绿幕占比显著更高 → 它才是场景
                    p["app_fg"], p["file"] = p["file"], p["app_fg"]
        fg_in = p.get("app_fg") or p.get("file")
        if not fg_in:
            raise ToolError("E_INPUT_MISSING", "green_key 模式需要 app_fg (绿幕前景)")
        fg_abs = _abs(fg_in)
        if not Path(fg_abs).exists():
            raise ToolError("E_INPUT_MISSING", f"前景不存在: {fg_abs}")
        # 背景优先级: bg(显式) → file(用户上传图, LLM 常写这里) → semantic 素材库 → 纯绿兜底
        bg_abs = None
        for _k in ("bg", "file"):
            _v = p.get(_k)
            if _v and str(_v) != str(fg_abs):
                _cand = _abs(_v)
                if Path(_cand).exists():
                    bg_abs = _cand
                    break
        if bg_abs is None and p.get("semantic"):
            hit = assets.find_background(str(p["semantic"]).strip())
            if hit:
                bg_abs = str(ROOT / hit)
        # 用户明确指定了背景(bg/file)但没解析出来 → 显式报错, 绝不静默退化/T2I 幻觉
        _bg_hint = p.get("bg") or p.get("file")
        if bg_abs is None and _bg_hint:
            raise ToolError("E_INPUT_MISSING",
                            f"green_key 指定的背景图不存在或与前景同源: {_bg_hint}")
        if bg_abs is None and not p.get("semantic"):
            raise ToolError("E_INPUT_MISSING",
                            "green_key 模式需要背景 (bg/file 路径 或 semantic 语义), 不支持纯绿兜底合成")
        out_dir = OUT / "t02" / ctx["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            skill_dir = ROOT / ".workbuddy/skills/green-screen-composite/scripts"
            if str(skill_dir) not in sys.path:
                sys.path.insert(0, str(skill_dir))
            import green_key as GK

            kw = dict(lower=int(p.get("lower", 26)), upper=int(p.get("upper", 99)),
                      median_blur=int(p.get("median_blur", 1)),
                      erode_iter=int(p.get("erode_iter", 1)),
                      dilate_iter=int(p.get("dilate_iter", 1)),
                      feather=int(p.get("feather", 1)),
                      filter_map=GK.resolve_filter(p.get("media_filter")))
            # green_scope (2026-09-12): 只在绿幕区域填背景, 演播室桌台/LED屏等实体保留。
            # 未指定时自动: 有摆位意图(center/fg_scale) → False 传统抠人合成; 否则 True(演播室常态)。
            gs_raw = p.get("green_scope")
            if gs_raw is None:
                green_scope = not (p.get("center") or p.get("fg_scale"))
            else:
                green_scope = gs_raw if isinstance(gs_raw, bool) else \
                    str(gs_raw).strip().lower() not in ("false", "0", "no", "off")
            kw["green_scope"] = green_scope
            media = (p.get("media_type") or "image").lower()
            is_video = media == "video" or Path(fg_abs).suffix.lower() in (
                ".mp4", ".avi", ".mov", ".mkv", ".webm")
            if is_video:
                if not bg_abs:
                    raise ToolError("E_INPUT_MISSING", "视频合成需要背景视频 (bg 或 semantic)")
                out_p = out_dir / f"gkey_{int(time.time())}.mp4"
                kw["with_audio"] = True
                GK.key_video(fg_abs, bg_abs, str(out_p), **kw)
                return {"bg_path": bg_abs or str(out_p), "source": "green_key",
                        "composite_path": str(out_p), "media_type": "video",
                        "green_scope": green_scope}
            # 图片: 位置/缩放参数 (green_scope=True 时画布=场景, key_image 内部会剔除摆位参数)
            if not green_scope:
                if p.get("fg_scale"):
                    s = float(p["fg_scale"])
                    kw["fg_scale_x"] = kw["fg_scale_y"] = s
                if p.get("center"):
                    cx_, cy_ = p["center"]
                    kw["center_x"], kw["center_y"] = int(cx_), int(cy_)
                elif bg_abs:
                    # 未指定位置 → 前景居中 (按**缩放后**尺寸计算)
                    s = float(p.get("fg_scale") or 1.0)
                    fg_im = GK.imread(fg_abs); bg_im = GK.imread(bg_abs)
                    if fg_im is not None and bg_im is not None:
                        fh = int(fg_im.shape[0] * s); fw = int(fg_im.shape[1] * s)
                        kw["center_x"] = max(0, (bg_im.shape[1] - fw) // 2)
                        kw["center_y"] = max(0, bg_im.shape[0] - fh)   # 人物贴底(演播室常规构图)
            out_p = out_dir / f"gkey_{int(time.time())}.png"
            GK.key_image(fg_abs, bg_abs, str(out_p), **kw)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError("E_GREENKEY_FAIL", f"{type(e).__name__}: {e}", retryable=True)
        return {"bg_path": bg_abs or str(out_p), "source": "green_key",
                "composite_path": str(out_p),
                "fx": p.get("media_filter"), "media_type": "image",
                "green_scope": green_scope}
    # 用户上传的背景图直通 (优先级最高)
    if p.get("file"):
        fp = _abs(p["file"])
        if not Path(fp).exists():
            raise ToolError("E_INPUT_MISSING", f"用户背景图不存在: {fp}")
        return {"bg_path": fp, "source": "user"}
    semantic = (p.get("semantic") or "").strip()
    # draft/normal: 素材库语义检索
    if semantic and quality in ("draft", "normal"):
        hit = assets.find_background(semantic)
        if hit:
            return {"bg_path": str(ROOT / hit), "source": "assets"}
        if not p.get("prompt"):
            raise ToolError("E_SEMANTIC_MISS", f"素材库无匹配: {semantic}", retryable=True)
    # fine 或无匹配: 文生图 (走 ctx 选定的 provider; 失败自动回退另一后端)
    prompt = p.get("prompt") or f"photorealistic TV studio background: {semantic}, professional broadcast lighting, no people"
    from agnes_client import AgnesClient
    out = OUT / "t02" / ctx["run_id"] / f"bg_{int(time.time())}.png"
    provider = ctx.get("provider")
    used = None
    errs = []
    # 候选顺序: 指定 provider → 另一可用 provider
    cands = [provider] if provider else []
    for alt in ("agnes", "qwen"):
        if alt not in cands:
            cands.append(alt)
    for pv in cands:
        try:
            cli = AgnesClient(provider=pv)
            if not cli.key:
                errs.append(f"{pv}: key 未配置")
                continue
            cli.generate_image(prompt, str(out), size=p.get("size", "1024x1024"))
            used = pv
            break
        except Exception as _e:
            errs.append(f"{pv}: {str(_e)[:120]}")
    if used is None:
        raise ToolError("E_T2I_FAIL", "文生图全部后端失败: " + " | ".join(errs), retryable=True)
    return {"bg_path": str(out), "source": "t2i", "prompt": prompt, "provider": used}


# --------------------------------------------------------------- T03 lighting
def t03_lighting_estimate(p: dict, quality: str, ctx: dict) -> dict:
    from lighting.lighting_estimate import estimate
    bg = _abs(p["bg_path"])
    if not Path(bg).exists():
        raise ToolError("E_INPUT_MISSING", f"背景不存在: {bg}")
    return estimate(bg, quality=quality)


# --------------------------------------------------------------- T04 relight
def t04_relight(p: dict, quality: str, ctx: dict) -> dict:
    from relighting.relight_tool import RelightTool
    fg, bg = _abs(p["fg_path"]), _abs(p["bg_path"])
    for k, v in (("fg", fg), ("bg", bg)):
        if not Path(v).exists():
            raise ToolError("E_INPUT_MISSING", f"{k} 不存在: {v}")
    hint = {}
    if p.get("light_dir"):
        hint["direction"] = [float(x) for x in p["light_dir"]]
    if p.get("color_temp") and p["color_temp"] != "neutral":
        hint["color_temp"] = p["color_temp"]
        # 色温强度上限 0.15: LLM 幻觉出大值会把人脸染成蓝色 (工具层另有 0.25 硬上限)
        hint["color_temp_strength"] = min(float(p.get("color_temp_strength", 0.12)), 0.15)
    tier_intensity = {"draft": 0.6, "normal": 0.8, "fine": 1.0}[quality]
    hint["intensity"] = float(p.get("intensity", tier_intensity))
    out = OUT / "t04" / ctx["run_id"] / f"relit_{int(time.time()*1000)%100000}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    method = "mkl" if p.get("method") == "mkl" else "directional"
    r = RelightTool(method=method).relight(fg, bg, str(out), light_hint=hint or None)
    return {"relit_fg_path": r["relit_fg_path"], "direction": r.get("direction", hint.get("direction"))}


# --------------------------------------------------------------- T05 shadow
def t05_shadow_generate(p: dict, quality: str, ctx: dict) -> dict:
    from shadow.shadow_tool import ShadowTool
    alpha, bg = _abs(p["alpha_path"]), _abs(p["bg_path"])
    for k, v in (("alpha", alpha), ("bg", bg)):
        if not Path(v).exists():
            raise ToolError("E_INPUT_MISSING", f"{k} 不存在: {v}")
    tier = {"draft": (8, 0.4), "normal": (12, 0.5), "fine": (16, 0.6)}[quality]
    hint = {"blur_radius": float(p.get("blur_radius", tier[0])),
            "opacity": float(p.get("opacity", tier[1]))}
    if p.get("light_dir"):
        d = [float(x) for x in p["light_dir"]]
        mag = float(np.hypot(*d)) or 1.0
        base = {"draft": 12, "normal": 18, "fine": 24}[quality]
        hint["direction"] = [d[0] / mag * base, d[1] / mag * base]
    elif p.get("offset"):
        hint["direction"] = [float(x) for x in p["offset"]]
    out = OUT / "t05" / ctx["run_id"] / f"shadowbg_{int(time.time()*1000)%100000}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = ShadowTool().generate(alpha, bg, str(out), shadow_hint=hint)
    return {"shadow_bg_path": r.get("shadow_path") or str(out)}


# --------------------------------------------------------------- T06 底部播放器UI自动裁剪
def _crop_bottom_inplace(path: str, keep_ratio: float) -> bool:
    """把画布裁到 keep_ratio 高度 (顶部对齐), 原地覆盖。失败返回 False 不影响主流程。"""
    try:
        im = Image.open(path)
        w, h = im.size
        nh = max(1, int(h * keep_ratio))
        if nh >= h:
            return False
        im.crop((0, 0, w, nh)).save(path)
        return True
    except Exception:
        return False


def detect_bottom_ui_crop(fg_path: str, alpha_path: str):
    """检测前景图底部烧录的播放器 UI, 返回保留高度比例 (0~1); 无需裁剪返回 None。

    双判据 (实测 B站截图布局: 暂停键+时间码 0.837-0.862 → 进度条 0.90 → 黑面板 0.91-1.0):
      1) alpha 行覆盖率: 主体底边 ab<=0.96 (BiRefNet 在 UI 压暗区提前"松手")
      2) 播放器签名 (二选一):
         a) 进度条: 底部 20% 内 strict白(>165且|R-B|<25, 排黄字幕)行占比>=0.25 的细亮带,
            且自身与上下邻行都暗 (白衬衫行 strict 也高但邻行不暗, 靠此排除)
         b) 黑色控制面板: >=5%h 的整行近黑带(行均值<45) 且延伸到 0.90h 以下
      裁剪线 = 进度条顶 - 0.07h (标准布局控件行在进度条上方 ~0.065h) / 无进度条时=面板顶。
    注意: 特征必须量在**原始上传图**上 (T01 输出 fg 为白底预乘, 渐变已被冲成纯白),
    调用方需传 T01 的原始入参图。任何异常返回 None (零回归风险)。
    """
    try:
        import cv2
        fg = cv2.imdecode(np.fromfile(str(fg_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        al = cv2.imdecode(np.fromfile(str(alpha_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if fg is None or al is None:
            return None
        h = fg.shape[0]
        if al.shape[0] != h:
            al = cv2.resize(al, (fg.shape[1], h), interpolation=cv2.INTER_LINEAR)
        # 门1: 主体提前离开底边 (正常全身/半身照主体都顶到下边缘)
        cov = (al > 128).mean(axis=1)
        rows = np.where(cov >= 0.02)[0]
        if rows.size == 0:
            return None
        ab = float(rows[-1]) / (h - 1)
        if ab > 0.96:
            return None
        g = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY).astype(np.float32)
        row_mean = g.mean(axis=1)
        r16 = fg.astype(np.int16)
        strict = ((g > 165) & (np.abs(r16[..., 2] - r16[..., 0]) < 25)
                  & (np.abs(r16[..., 2] - r16[..., 1]) < 25)).mean(axis=1)
        cut = None
        # 签名a: 进度条 (细亮带宽占>=25%, 且邻域暗 —— 衬衫亮行邻域不暗, 排除)
        lo, hi = int(h * 0.80), int(h * 0.97)
        line_rows = [y for y in range(lo, hi)
                     if strict[y] >= 0.25 and row_mean[y] < 120.0
                     and row_mean[max(0, y - 4)] < 90.0 and row_mean[min(h - 1, y + 4)] < 90.0]
        if line_rows:
            cut = float(min(line_rows)) - 0.07 * h    # 控件行在进度条上方 ~0.065h
        else:
            # 签名b: 黑色控制面板带
            dark = row_mean < 45.0
            run_len = run_start = 0
            i = int(h * 0.78)
            top = min(h - 1, int(h * 0.995))
            while i <= top:
                if dark[i]:
                    j = i
                    while j <= top and dark[j]:
                        j += 1
                    if j - i > run_len:
                        run_len, run_start = j - i, i
                    i = j
                else:
                    i += 1
            if run_len >= 0.05 * h and (run_start + run_len) >= 0.90 * h:
                cut = float(run_start)
        if cut is None:
            return None
        keep = min(ab + 0.02, cut / h)
        keep = max(0.5, min(1.0, keep))
        if 1.0 - keep < 0.03:
            return None                       # 裁掉太少不值得动构图
        return keep
    except Exception:
        return None


# --------------------------------------------------------------- T06 harmonize (吸收规范入参)
def t06_harmonize(p: dict, quality: str, ctx: dict) -> dict:
    from composite.composite_tool import CompositeTool
    from harmonization.harmonize_tool import HarmonizeTool
    fg, alpha, bg = _abs(p["fg_path"]), _abs(p["alpha_path"]), _abs(p["bg_path"])
    for k, v in (("fg", fg), ("alpha", alpha), ("bg", bg)):
        if not Path(v).exists():
            raise ToolError("E_INPUT_MISSING", f"{k} 不存在: {v}")
    run = OUT / "t06" / ctx["run_id"]
    run.mkdir(parents=True, exist_ok=True)
    tier_strength = {"draft": 0.55, "normal": 0.75, "fine": 0.9}[quality]
    feather = 3 if quality == "fine" else 2
    strength = float(p.get("strength", tier_strength))
    ts = int(time.time() * 1000) % 100000

    if p.get("mode") == "greenscreen":
        # 虚拟演播室键控: 只换绿幕区域, 保留非绿实物 (收敛 greenscreen → T06)
        # 2026-09-12: 从 studio_cli.task_greenscreen(主连通域策略) 换成 green_key
        # 绿幕范围算法(面积+饱和度双判据)——主连通域策略在"绿幕被横杆切成多块"
        # 时会漏掉小块绿域, 留下绿线; 双判据靠饱和度把真绿幕全收进来。
        skill_dir = ROOT / ".workbuddy/skills/green-screen-composite/scripts"
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        import green_key as GK
        scene = GK.imread(fg)
        bgim = GK.imread(bg)
        if scene is None:
            raise ToolError("E_INPUT_MISSING", f"场景图读取失败: {fg}")
        if bgim is None:
            raise ToolError("E_INPUT_MISSING", f"背景图读取失败: {bg}")
        out_p = run / f"gs_{ts}.png"
        comp = GK.composite_green_scope(scene, bgim,
                                        feather=feather, erode_iter=1, despill=True)
        GK.imwrite(str(out_p), comp)
        if not Path(out_p).exists():
            raise ToolError("E_NO_GREENSCREEN", "greenscreen 合成输出失败", retryable=True)
        return {"composite_path": str(out_p), "harmonized_fg_path": str(out_p),
                "fdr": 1.0, "fdr_ok": True, "mode": "greenscreen"}

    # 1) alpha_over 合成 (含可选 T05 阴影 bg)
    comp_raw = run / f"comp_{ts}.png"
    shadow_bg = p.get("shadow_bg_path")
    CompositeTool(feather_radius=feather).composite(
        fg, alpha, bg, str(comp_raw),
        shadow_path=_abs(shadow_bg) if shadow_bg and Path(_abs(shadow_bg)).exists() else None)
    # 2) 对 composite 做和谐化 (alpha 作为前景 mask, FDR 防压黑)
    comp_out = run / f"harm_{ts}.png"
    hr = HarmonizeTool(method="v2").harmonize(str(comp_raw), bg, str(comp_out), alpha_path=alpha)
    # harmonize v2 内部 FDR 自动回退; 读取指标
    info = hr if isinstance(hr, dict) else {}
    fdr = float(info.get("fdr", 1.0))
    fdr_ok = bool(info.get("fdr_ok", 0.70 <= fdr <= 1.30))
    # 3) 底部播放器UI自动裁剪 (确定性后处理, 不经 LLM): 前景为带烧录进度条/控制键的
    #    视频截图时, UI 渐变压暗会让主体在底边前被截断, 成片底部残留 UI —— 裁到主体底边+2%。
    #    特征必须量在原始上传图上 (T01 输出 fg 为白底预乘, 渐变已被冲成纯白),
    #    原图从 engine ctx.step_params 的 T01 入参取; 取不到则不裁 (零回归风险)。
    _cb = p.get("crop_bottom")
    keep = float(_cb) if isinstance(_cb, (int, float)) and _cb > 0 else None
    if keep is None:
        _orig = None
        for _v in (ctx.get("step_params") or {}).values():
            if isinstance(_v, dict) and _v.get("tool") == "T01_matting":
                _orig = (_v.get("params") or {}).get("image")
                if _orig:
                    break
        if _orig:
            keep = detect_bottom_ui_crop(_abs(_orig), alpha)
    crop_info = None
    if keep:
        if _crop_bottom_inplace(str(comp_out if fdr_ok else comp_raw), keep):
            crop_info = {"keep": round(keep, 3), "removed": round(1 - keep, 3)}
    if not fdr_ok:
        # 降级: 用未和谐化的合成图 (保证可用, 程序化降级)
        r = {"composite_path": str(comp_raw), "harmonized_fg_path": str(comp_raw),
             "fdr": fdr, "fdr_ok": False, "degraded": True, "mode": "alpha_over"}
    else:
        r = {"composite_path": str(comp_out), "harmonized_fg_path": str(comp_out),
             "fdr": round(fdr, 3), "fdr_ok": True, "mode": "alpha_over"}
    if crop_info:
        r["bottom_crop"] = crop_info
    return r


# --------------------------------------------------------------- T07 enhance (fx 兜底)
def t07_enhance(p: dict, quality: str, ctx: dict) -> dict:
    """增强/风格化: fx_library 全库兜底 (景深/暗角/颗粒/锐化/色温/贴纸/水印)。"""
    img = _abs(p["image_path"])
    if not Path(img).exists():
        raise ToolError("E_INPUT_MISSING", f"输入不存在: {img}")
    mode = p.get("mode", "depth_blur")
    tier_i = {"draft": 0.3, "normal": 0.45, "fine": 0.45}[quality]
    intensity = float(p.get("intensity", tier_i))
    # mode → fx_library key
    key_map = {"depth_blur": "depth_blur", "vignette": "vignette",
               "grain": "film_grain", "film_grain": "film_grain",
               "warm": "warm", "cool": "cool", "bloom": "bloom"}
    sticker_map = {"heart": "heart", "star": "star", "crown": "crown",
                   "flower": "flower", "cat": "cat", "rainbow": "rainbow"}
    if mode == "sticker":
        key = sticker_map.get(str(p.get("sticker", "heart")).lower(), "heart")
    elif mode == "watermark_text":
        key = "watermark_text"
    else:
        key = key_map.get(mode, "depth_blur")
    params = {"frac": intensity, "intensity": intensity}
    if mode == "watermark_text" and p.get("text"):
        params["text"] = str(p["text"])
    out = OUT / "t07" / ctx["run_id"] / f"enh_{int(time.time()*1000)%100000}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from fx_library import fx_auto
        arr = np.array(Image.open(img).convert("RGB"))
        out_arr, meta = fx_auto.place(arr, key, params=params)
        Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8)).save(out)
    except Exception as e:
        raise ToolError("E_ENHANCE_FAIL", f"{type(e).__name__}: {e}", retryable=True)
    enhanced = str(out)
    # 锐化 = USM (fine 档的 depth_blur 额外补锐)
    do_sharpen = (mode == "sharpen") or (quality == "fine" and mode == "depth_blur")
    if do_sharpen:
        import cv2
        buf = np.fromfile(enhanced, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        blur = cv2.GaussianBlur(arr, (0, 0), 2.0)
        sharp = cv2.addWeighted(arr, 1.35, blur, -0.35, 0)
        ok2, enc = cv2.imencode(Path(enhanced).suffix, sharp)
        enc.tofile(enhanced)
    return {"enhanced_path": enhanced, "mode": mode}


# --------------------------------------------------------------- T08 export
def t08_export(p: dict, quality: str, ctx: dict) -> dict:
    final = _abs(p["final_path"])
    if not Path(final).exists():
        raise ToolError("E_INPUT_MISSING", f"成片不存在: {final}")
    exp_dir = OUT / "t08" / ctx["run_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    exported = exp_dir / f"final_{ctx['run_id']}.png"
    shutil.copy(final, exported)
    meta_p = None
    zip_p = None
    if quality in ("normal", "fine"):
        meta = {"run_id": ctx["run_id"], "tool_versions": TOOL_VERSIONS,
                "exported": str(exported), "time": time.strftime("%F %T"),
                **(p.get("meta") or {})}
        meta_p = exp_dir / "meta.json"
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if quality == "fine" or p.get("make_zip"):
        zip_p = exp_dir / "export.zip"
        with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(exported, exported.name)
            if meta_p:
                z.write(meta_p, meta_p.name)
    o = {"exported_path": str(exported)}
    if meta_p:
        o["meta_path"] = str(meta_p)
    if zip_p:
        o["zip_path"] = str(zip_p)
    return o


# --------------------------------------------------------------- 注册表 + 信封
ADAPTERS = {
    "T01_matting": t01_matting, "T02_background_generate": t02_background_generate,
    "T03_lighting_estimate": t03_lighting_estimate, "T04_relight": t04_relight,
    "T05_shadow_generate": t05_shadow_generate, "T06_harmonize": t06_harmonize,
    "T07_enhance": t07_enhance, "T08_export": t08_export,
}


def _abs(v: str) -> str:
    """项目相对/绝对路径 → 绝对路径 (容忍 Windows 反斜杠)。"""
    if not v:
        return v
    p = Path(v)
    if p.is_absolute():
        return str(p)
    return str(ROOT / v)


def call_tool(tool: str, node_id: str, params: dict, quality: str = "draft",
              ctx: dict | None = None) -> dict:
    """统一入口: 执行工具并包四件套信封。绝不抛异常。"""
    ctx = dict(ctx or {})
    ctx.setdefault("run_id", time.strftime("%Y%m%d_%H%M%S"))
    t0 = time.time()
    base = {"node_id": node_id, "tool": tool, "quality": quality}
    fn = ADAPTERS.get(tool)
    if fn is None:
        return {**base, "ok": False, "data": None,
                "error": {"code": "E_UNKNOWN_TOOL", "message": f"未知工具 {tool}", "retryable": False},
                "latency_ms": 0, "model_version": None}
    try:
        data = fn(params, quality, ctx)
        return {**base, "ok": True, "data": data, "error": None,
                "latency_ms": int((time.time() - t0) * 1000),
                "model_version": TOOL_VERSIONS.get(tool)}
    except ToolError as e:
        return {**base, "ok": False, "data": None,
                "error": {"code": e.code, "message": e.message, "retryable": e.retryable},
                "latency_ms": int((time.time() - t0) * 1000),
                "model_version": TOOL_VERSIONS.get(tool)}
    except Exception as e:
        return {**base, "ok": False, "data": None,
                "error": {"code": "E_INTERNAL", "message": f"{type(e).__name__}: {e}",
                          "retryable": False, "trace": traceback.format_exc()[-500:]},
                "latency_ms": int((time.time() - t0) * 1000),
                "model_version": TOOL_VERSIONS.get(tool)}


if __name__ == "__main__":
    # 冒烟: T03 + T07 快速链
    ctx = {"run_id": "smoke"}
    r3 = call_tool("T03_lighting_estimate", "n1", {"bg_path": "data/ai_generated/studio_bg/bg_02_interview.png"})
    print("T03:", r3["ok"], (r3["data"] or {}).get("color_temp"), r3["latency_ms"], "ms",
          "" if r3["ok"] else r3["error"])
    # 找一张真实存在的图
    cand = [ROOT / "outputs/agent/composite/composite/final.png",
            ROOT / "outputs/agent/fx/teal_orange_1788965371.png",
            ROOT / "data/ai_generated/studio_bg/bg_02_interview.png"]
    img = next((str(c) for c in cand if c.exists()), None)
    r7 = call_tool("T07_enhance", "n2", {"image_path": img, "mode": "depth_blur", "quality": "fine"})
    print("T07:", r7["ok"], (r7["data"] or {}).get("enhanced_path"), r7["latency_ms"], "ms",
          "" if r7["ok"] else r7["error"])
