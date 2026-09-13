# -*- coding: utf-8 -*-
"""
演播室图像合成 · 统一任务触发接口 (studio_cli)

四个任务:
  [1] matting    抠图     : 绿幕图 -> alpha / 去溢出前景 / RGBA  (engine: chroma|birefnet|refiner)
  [2] composite  背景叠加 : 抠图 + 合成 + 可选(和谐化/重打光/接触阴影)
  [3] fx         添加特效 : 单图特效, 支持 region 圈选 + 链式
  [4] full       一键全流程: 绿幕图 -> 抠图 -> 叠加 -> 光影 -> 特效 -> 成片

用法 (命令行):
  python app/studio_cli.py                              # 交互式菜单
  python app/studio_cli.py list                         # 任务列表
  python app/studio_cli.py matting --image fg.png
  python app/studio_cli.py composite --image fg.png --bg bg.png --relight --shadow
  python app/studio_cli.py fx --image in.png --fx spotlight --intensity 0.6 --region 900,0,1536,600
  python app/studio_cli.py full --image fg.png --bg bg.png --fx spotlight:0.5:900,0,1536,600 --fx vignette:0.35
输出: 每步产物落盘 + JSON 摘要 (stdout), 便于 Agent 调用。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # project root (app/ -> root)
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-service"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

DEMO_FG = ROOT / "data" / "ai_generated" / "green_fg" / "fg_02_anchor_female.png"
DEMO_BG = ROOT / "data" / "ai_generated" / "studio_bg" / "bg_01_news_led.png"
DEFAULT_OUT = ROOT / "outputs"

TASK_LIST = """\n==== 演播室图像合成 · 任务接口 ====
[1] matting    抠图        绿幕图 -> alpha / 去溢出前景 / RGBA
[2] composite  背景叠加    抠图 + 合成 + (可选)和谐化/重打光/接触阴影
[3] fx         添加特效    聚光灯/散景/雾效/暗角/色温/景深, 支持圈选区域
[4] full       一键全流程  绿幕图 -> 抠图 -> 叠加 -> 光影 -> 特效 -> 成片
[h] 查看本菜单, [q] 退出"""


# ---------------------------------------------------------------- chroma strong
def chroma_key_strong(img: "np.ndarray") -> "np.ndarray":
    """增强版色度键 (生产链路用, 比 ai_material_eval 的 GT 版更稳):
    1) 边缘带取绿幕参考色 (Lab);
    2) 像素-参考色 Lab 距离场 -> 软阈值 (自适应 t1/t2);
    3) 硬截断 + 强连通域清理 (非最大前景域一律归零) + 闭运算填洞。
    对偏暗/不均匀绿幕稳健, 消除低 alpha 大面积污染。"""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    # 注意: cv2 LAB 转换必须用 uint8 输入 (float32 会被假定 0-1 值域, Lab 全乱)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge = np.concatenate([lab[:20].reshape(-1, 3), lab[-20:].reshape(-1, 3),
                           lab[:, :20].reshape(-1, 3), lab[:, -20:].reshape(-1, 3)])
    # 参考色只从"绿色占优"的边缘像素估计, 防前景占满边缘时污染 (如半身像贴边)
    edge_rgb = np.concatenate([img[:20].reshape(-1, 3), img[-20:].reshape(-1, 3),
                               img[:, :20].reshape(-1, 3), img[:, -20:].reshape(-1, 3)]).astype(np.float32)
    greenish = edge_rgb[..., 1] - np.maximum(edge_rgb[..., 0], edge_rgb[..., 2]) > 25
    ref = np.median(edge[greenish] if greenish.sum() > 100 else edge, axis=0)
    d = np.linalg.norm(lab - ref, axis=-1)
    med_d = float(np.percentile(d, 25))   # 25 分位 = 纯背景内部典型距离 (全图中位会被前景拉高)
    t1 = max(med_d * 1.9, 22.0)          # 距离 < t1 -> 背景 (近参考绿)
    t2 = t1 + 28.0                        # 距离 > t2 -> 前景
    alpha = np.clip((d - t1) / max(t2 - t1, 1.0), 0, 1)
    alpha = np.where(alpha < 0.12, 0.0, np.where(alpha > 0.75, 1.0, alpha))  # 硬截断
    from scipy.ndimage import label, binary_closing, binary_fill_holes
    solid = alpha > 0.5
    if solid.any():
        lb, n = label(solid)
        sizes = np.bincount(lb.ravel()); sizes[0] = 0   # 排除背景 label=0
        keep = int(sizes.argmax()) if sizes.sum() > 0 else 0
        alpha[(lb != keep)] = 0.0          # 强清理: 非最大前景域一律归零
        fg_any = alpha > 0.05
        lb2, _ = label(fg_any)
        keep_ids = set(np.unique(lb2[solid])) - {0}
        alpha[fg_any & ~np.isin(lb2, list(keep_ids))] = 0.0
        s2 = binary_closing(alpha > 0.2, np.ones((5, 5), bool))
        alpha[binary_fill_holes(s2) & ~s2] = 1.0
    return alpha


def despill_strong(img: "np.ndarray", alpha: "np.ndarray") -> "np.ndarray":
    """强绿幕溢出压制: 前景区 g 上限压到 (r+b)/2+12 (对发丝混色区更狠)。"""
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    g2 = np.minimum(g, (r + b) / 2 + 12)
    return np.clip(np.stack([r, g2, b], -1), 0, 255).astype(np.uint8)


def _birefnet_alpha(img: "np.ndarray", tmp_dir: Path) -> "np.ndarray":
    """BiRefNet 结构先验 alpha (512 推理)。"""
    from PIL import Image
    import torch
    from matting.matting_backend import BiRefNetMatting
    tool = BiRefNetMatting(device="cuda" if torch.cuda.is_available() else "cpu", seg_mask_size=512)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p_in, p_a = tmp_dir / "_b_in.png", tmp_dir / "_b_alpha.png"
    Image.fromarray(img).save(p_in)
    tool.predict(str(p_in), str(p_a), original_size=True)
    return np.array(Image.open(p_a).convert("L")).astype(np.float32) / 255.0


# ---------------------------------------------------------------- [1] matting
def task_matting(image: str, out_dir: str, engine: str = "chroma") -> dict:
    import numpy as np
    from PIL import Image
    from ai_material_eval import fix_watermark, chroma_key_alpha, despill

    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    img = fix_watermark(np.array(Image.open(image).convert("RGB")))
    info = {"task": "matting", "engine": engine, "image": image}

    if engine in ("chroma", "hybrid"):
        alpha = chroma_key_strong(img)
        if engine == "hybrid":
            # BiRefNet 结构先验补洞: 深色头发/暗绿背景色距近时色度键会误抠,
            # 取二者逐像素最大值 (色度键保边缘精度, BiRefNet 保语义完整)
            alpha = np.maximum(alpha, _birefnet_alpha(img, out) * 0.95)
        fg = despill_strong(img, alpha)                # 强绿幕溢出压制 (绿边去除)
    elif engine in ("humanmatting", "humanmatting_fast"):
        # MODNet 人像语义抠图 (PaddleSeg 官方权重→ONNX, onnxruntime 推理)
        # 适合非绿幕通用人像; 绿幕图无 despill, 建议仍用 chroma/hybrid
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "ai-service" / "src"))
        from matting.humanmatting_engine import human_matting
        alpha = human_matting(img[..., ::-1],            # RGB->BGR
                              model="mobilenet" if engine == "humanmatting_fast" else "hrnet")
        fg = (img.astype(np.float32) * alpha[..., None] + 255.0 * (1 - alpha[..., None])).astype(np.uint8)
        info["note"] = "humanmatting 无 despill, 绿幕图建议 chroma/hybrid"
    else:
        import torch
        from matting.matting_backend import BiRefNetMatting
        tool = BiRefNetMatting(device="cuda" if torch.cuda.is_available() else "cpu", seg_mask_size=512)
        tmp_in = out / f"{stem}_clean.png"
        Image.fromarray(img).save(tmp_in)
        tool.predict(str(tmp_in), str(out / f"{stem}_alpha.png"), original_size=True)
        alpha = np.array(Image.open(out / f"{stem}_alpha.png").convert("L")).astype(np.float32) / 255.0
        fg = (img.astype(np.float32) * alpha[..., None] + 255.0 * (1 - alpha[..., None])).astype(np.uint8)
        if engine == "refiner":                        # AlphaRefiner 残差精修
            from ai_finetune_refiner import CKPT_OUT as _CK
            w = _CK / "best.pt"
            if w.exists():
                from train_refiner import AlphaRefiner
                import torch, cv2
                net = AlphaRefiner(base=32).to("cuda" if torch.cuda.is_available() else "cpu")
                ck = torch.load(w, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
                net.load_state_dict(ck["model"]); net.eval()
                S = 512
                rgb_t = torch.from_numpy(np.array(Image.fromarray(fg).resize((S, S))).astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
                co_t = torch.from_numpy(cv2.resize(alpha.astype(np.float32), (S, S))).unsqueeze(0).unsqueeze(0).cuda()
                with torch.no_grad():
                    pred = net(rgb_t, co_t)
                a_s = pred[0, 0].float().clamp(0, 1).cpu().numpy()
                alpha = cv2.resize(a_s, (img.shape[1], img.shape[0]))
                fg = (img.astype(np.float32) * alpha[..., None] + 255.0 * (1 - alpha[..., None])).astype(np.uint8)
                info["refiner"] = str(w)
        info["note"] = "engine 非 chroma 时无 despill, 绿幕图建议默认 chroma"

    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([fg, a8])
    Image.fromarray(a8).save(out / f"{stem}_alpha.png")
    Image.fromarray(fg).save(out / f"{stem}_fg.png")
    Image.fromarray(rgba, "RGBA").save(out / f"{stem}_rgba.png")
    info.update({"alpha": str(out / f"{stem}_alpha.png"), "fg": str(out / f"{stem}_fg.png"),
                 "rgba": str(out / f"{stem}_rgba.png"),
                 "fg_ratio": round(float((alpha > 0.5).mean()), 3),
                 "sec": round(time.time() - t0, 2)})
    return info


def fit_bg_cover(bg_rgb: "np.ndarray", tw: int, th: int) -> "np.ndarray":
    """背景适配前景画布: 等比缩放到完全覆盖 (cover) 后居中裁剪。
    输出尺寸恒等于前景尺寸, 背景不变形, 前景尺寸不变。"""
    h, w = bg_rgb.shape[:2]
    sc = max(tw / w, th / h)
    nw, nh = max(tw, int(round(w * sc))), max(th, int(round(h * sc)))
    im = Image.fromarray(bg_rgb).resize((nw, nh), Image.LANCZOS)
    x = (nw - tw) // 2
    y = (nh - th) // 2
    return np.array(im)[y:y + th, x:x + tw]


# ---------------------------------------------------------------- [2] composite
def task_composite(image: str, bg: str, out_dir: str, alpha: str | None = None,
                   harmonize: bool = True, relight: bool = True, shadow: bool = True,
                   light_hint: dict | None = None, engine: str = "chroma") -> dict:
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # 背景适配: 画布 = 原图(前景)尺寸, 背景 cover 裁剪 -> 输出尺寸永不改变
    fg0 = Image.open(image)
    tw, th = fg0.size
    bg_rgb = np.array(Image.open(bg).convert("RGB"))
    bg_fit = fit_bg_cover(bg_rgb, tw, th)
    bg_tmp = Path(out_dir) / "_bg_fitted.png"
    bg_tmp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bg_fit).save(bg_tmp)
    bg = str(bg_tmp)
    if alpha is None:                                   # 无 alpha -> 按 engine 抠图 (非绿幕图务必用 birefnet/refiner/humanmatting)
        m = task_matting(image, str(out / "matting"), engine=engine)
        alpha = m["alpha"]
        # 用 despill 后的前景替换原图, 避免绿色溢出进入合成
        image = m["fg"]
    from pipeline import run_pipeline
    r = run_pipeline(image, bg, str(out / "composite"), device=None,
                     precomputed_alpha=alpha, enable_harmonize=harmonize,
                     enable_relight=relight, enable_shadow=shadow, light_hint=light_hint)
    return {"task": "composite", "final": r["final"], "stages": r["stages"],
            "total_ms": round(r["total_time_ms"]), "sec": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- [2b] greenscreen
def chroma_key_scene(arr: "np.ndarray") -> "np.ndarray":
    """场景绿幕键控 (演播室大图): 绿幕可能在图中任意位置, 不能用边缘带估计参考色。
    1) 全图取最绿的颜色簇 (excess 高分位) -> Lab 参考色
    2) 簇内距离 90 分位为 t1 (宽容光照渐变/褶皱), 距离场 -> 软 alpha
    3) 硬截断 + 最大绿幕连通域外的孤立绿块保留为前景 (防误伤)"""
    import cv2
    f = arr.astype(np.float32)
    excess = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    thr = max(float(np.percentile(excess, 80)), 15.0)
    greenish = excess > thr
    if greenish.sum() < 200:
        return np.ones(arr.shape[:2], np.float32)     # 无绿幕 -> 全保留
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[greenish], axis=0)
    d = np.linalg.norm(lab - ref, axis=-1)
    t1 = float(np.percentile(d[greenish], 90)) + 4.0
    t2 = t1 + 26.0
    alpha = np.clip((d - t1) / max(t2 - t1, 1.0), 0, 1)
    alpha = np.where(alpha < 0.10, 0.0, np.where(alpha > 0.85, 1.0, alpha))
    from scipy.ndimage import label, binary_closing
    bg = alpha < 0.5                                   # 绿幕区域
    lb, n = label(bg)
    if n > 1:
        sizes = np.bincount(lb.ravel()); sizes[0] = 0
        keep = sizes.argmax()                          # 最大绿幕连通域
        # 不在主绿幕域内的"绿块"(如桌面上的绿色贴纸)保留为前景, 防误伤
        alpha[(bg) & (lb != keep)] = np.maximum(alpha[(bg) & (lb != keep)], 0.9)
    return alpha


def task_greenscreen(image: str, bg: str, out_dir: str,
                     feather: int = 3, despill_strength: int = 12) -> dict:
    """绿幕区域替换 (虚拟演播室键控):
    保留整张原图 (桌子/灯光/话筒等一切非绿内容), 仅把绿幕像素替换为新背景。
    2026-09-12: 内部改用 green_key 绿幕范围算法 (面积+饱和度双判据)——
    旧 chroma_key_scene 主连通域策略在绿幕被横杆/装饰条切成多块时会漏掉
    小块绿域 (顶部留绿线); 双判据按饱和度把真绿幕全收进来, 同时排除
    LED 屏里的低饱和绿色画面。despill_strength 参数保留兼容旧签名,
    新算法内部用自适应 band despill (0.85)。"""
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import cv2
    # 0) 绿幕检测保护: 非绿幕图直接报错 (前端已锚定绿幕原图, 此为兜底)
    arr = np.array(Image.open(image).convert("RGB"))
    f32 = arr.astype(np.float32)
    green_ratio_raw = float((f32[..., 1] - np.maximum(f32[..., 0], f32[..., 2]) > 25).mean())
    if green_ratio_raw < 0.03:
        return {"task": "greenscreen", "error":
                "未检测到绿幕 (绿幕占比 <3%)。请确认当前是绿幕原图——已自动撤回原图的操作请重新上传绿幕图。"}
    # 1) 双判据绿幕范围合成 (场景画布 = 原图, 实物保留)
    skill_dir = ROOT / ".workbuddy" / "skills" / "green-screen-composite" / "scripts"
    if str(skill_dir) not in sys.path:
        sys.path.insert(0, str(skill_dir))
    import green_key as GK
    scene = GK.imread(image)                      # BGR
    bgim = GK.imread(bg)
    if scene is None or bgim is None:
        return {"task": "greenscreen", "error": f"图像读取失败: {image if scene is None else bg}"}
    comp_bgr = GK.composite_green_scope(scene, bgim,
                                        feather=int(feather), erode_iter=1, despill=True)
    out_p = Path(out_dir) / "greenscreen_composited.png"
    GK.imwrite(str(out_p), comp_bgr)
    _, keep = GK.make_green_scope_mask(scene, erode_iter=1, feather=int(feather))
    Image.fromarray(keep).save(Path(out_dir) / "greenscreen_alpha.png")
    return {"task": "greenscreen", "output": str(out_p),
            "alpha": str(Path(out_dir) / "greenscreen_alpha.png"),
            "green_ratio": round(float((keep > 0).mean()), 3),
            "size": [scene.shape[1], scene.shape[0]],
            "sec": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- [3] fx
def task_fx(image: str, out_dir: str, fx_specs: list[dict]) -> dict:
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    from fx.fx_tool import FxTool, EFFECTS
    chain = []
    for spec in fx_specs:
        name = spec["effect"]
        if name not in EFFECTS:
            return {"error": f"unknown effect {name}, valid: {EFFECTS}"}
        chain.append({"effect": name, "params": {"intensity": spec.get("intensity", 0.6)},
                      "region": spec.get("region")})
    r = FxTool().chain(image, str(out / "fx_result.png"), chain, str(out))
    return {"task": "fx", **r, "sec": round(time.time() - t0, 2)}


def task_fx_pil(img_arr, out_dir: str, effect: str, intensity: float = 0.6,
                region: dict | None = None) -> dict:
    """特效的内存数组版 (供 GUI 调用): 直接对 numpy RGB 数组加特效。"""
    from fx.fx_tool import FxTool, build_region_mask, _DISPATCH
    if effect not in _DISPATCH:
        return {"error": f"unknown effect {effect}"}
    mask = build_region_mask(img_arr.shape, region)
    out = _DISPATCH[effect](img_arr, mask, intensity=intensity)
    out = np.clip(out, 0, 255).astype(np.uint8)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(out_dir) / f"fx_{effect}.png"
    Image.fromarray(out).save(p)
    return {"out_path": str(p), "image": out, "effect": effect,
            "intensity": intensity, "region": region or "full"}


# ---------------------------------------------------------------- [4] full
def task_full(image: str, bg: str, out_dir: str, fx_specs: list[dict] | None = None,
              harmonize: bool = True, relight: bool = True, shadow: bool = True,
              light_hint: dict | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    comp = task_composite(image, bg, str(out), harmonize=harmonize,
                          relight=relight, shadow=shadow, light_hint=light_hint)
    result = {"task": "full", "composite": comp}
    if fx_specs:
        fxr = task_fx(comp["final"], str(out / "fx"), fx_specs)
        result["fx"] = fxr
        result["final"] = fxr.get("out_path", comp["final"])
    else:
        result["final"] = comp["final"]
    result["sec"] = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------- helpers
def parse_fx(args_list: list[str] | None) -> list[dict]:
    """--fx name[:intensity[:x1,y1,x2,y2]] 可重复。"""
    specs = []
    for raw in (args_list or []):
        parts = raw.split(":")
        spec = {"effect": parts[0].strip()}
        if len(parts) > 1 and parts[1]:
            spec["intensity"] = float(parts[1])
        if len(parts) > 2 and parts[2]:
            xy = [float(v) for v in parts[2].split(",")]
            spec["region"] = {"type": "box", "xyxy": xy}
        specs.append(spec)
    return specs


def prompt_path(msg: str, default: Path) -> str:
    v = input(f"{msg} (回车=默认 {default.name}): ").strip()
    return v or str(default)


def interactive():
    print(TASK_LIST)
    while True:
        c = input("\n选择任务 [1/2/3/4, h 菜单, q 退出]: ").strip().lower()
        if c == "q":
            break
        if c == "h":
            print(TASK_LIST); continue
        out = str(DEFAULT_OUT / "interactive")
        try:
            if c == "1":
                img = prompt_path("绿幕图片路径", DEMO_FG)
                eng = input("engine [chroma/birefnet/refiner] (回车=chroma): ").strip() or "chroma"
                print(json.dumps(task_matting(img, out, eng), ensure_ascii=False, indent=2))
            elif c == "2":
                img = prompt_path("绿幕图片路径", DEMO_FG)
                bg = prompt_path("背景图片路径", DEMO_BG)
                flags = input("光影模块 (回车=全开; 输入如 relight,shadow 选择): ").strip()
                names = {x.strip() for x in flags.split(",") if x.strip()} if flags else {"harmonize", "relight", "shadow"}
                print(json.dumps(task_composite(img, bg, out,
                      harmonize="harmonize" in names, relight="relight" in names,
                      shadow="shadow" in names), ensure_ascii=False, indent=2))
            elif c == "3":
                img = prompt_path("待加特效图片", DEFAULT_OUT / "interactive" / "composite" / "final.png")
                fx = input("特效 (回车=spotlight; 格式 名:强度[:x1,y1,x2,y2], 多个用逗号): ").strip() or "spotlight:0.6"
                specs = parse_fx([s for s in fx.split(",") if s])
                print(json.dumps(task_fx(img, out, specs), ensure_ascii=False, indent=2))
            elif c == "4":
                img = prompt_path("绿幕图片路径", DEMO_FG)
                bg = prompt_path("背景图片路径", DEMO_BG)
                fx = input("特效链 (回车=默认聚光+暗角; 格式 名:强度[:region], 逗号分隔, no=不加): ").strip()
                specs = None if fx == "no" else (parse_fx([s for s in fx.split(",") if s]) if fx else
                    parse_fx(["spotlight:0.5", "vignette:0.35"]))
                print(json.dumps(task_full(img, bg, out, specs), ensure_ascii=False, indent=2))
            else:
                print("无效选项")
        except Exception as e:
            print(f"[error] {e}")


def main():
    ap = argparse.ArgumentParser(description="演播室图像合成任务接口")
    ap.add_argument("task", nargs="?", default=None,
                    help="matting | composite | fx | full | list (缺省=交互菜单)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--bg", default=None)
    ap.add_argument("--alpha", default=None, help="已有 alpha (跳过抠图)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default="chroma",
                    choices=["chroma", "hybrid", "birefnet", "refiner",
                             "humanmatting", "humanmatting_fast"])
    ap.add_argument("--fx", action="append", default=None,
                    help="特效 name[:intensity[:x1,y1,x2,y2]], 可重复")
    ap.add_argument("--no-harmonize", action="store_true")
    ap.add_argument("--no-relight", action="store_true")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--light-hint", default=None, help='JSON, 如 {"direction":[1,0],"color_temp":"warm"}')
    args = ap.parse_args()

    if args.task in (None, "list", "h", "help"):
        if args.task is None:
            interactive()
        else:
            print(TASK_LIST)
        return
    out = args.out or str(DEFAULT_OUT / args.task)
    lh = json.loads(args.light_hint) if args.light_hint else None
    if args.task == "matting":
        r = task_matting(args.image, out, args.engine)
    elif args.task == "composite":
        r = task_composite(args.image, args.bg, out, args.alpha,
                           not args.no_harmonize, not args.no_relight, not args.no_shadow, lh)
    elif args.task == "fx":
        r = task_fx(args.image, out, parse_fx(args.fx))
    elif args.task == "full":
        r = task_full(args.image, args.bg, out, parse_fx(args.fx),
                      not args.no_harmonize, not args.no_relight, not args.no_shadow, lh)
    else:
        r = {"error": f"unknown task {args.task}"}
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
