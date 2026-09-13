# -*- coding: utf-8 -*-
"""
habits_gptimage.py — my-gptimage-habits 技能接入 (habits × agnes API)
接入方式 (按官网要求 + 用户指定 agnes api):
  - 工作流模板/铁律 来自技能 references (逐字照抄用户文字/比例写进 prompt/n=1/失败不自动重试)
  - 文生图类 (W2字幕条 W9透明底 W6PPT封面 W7分镜): agnes /v1/images/generations
  - 图生图类 (W1封面加字 W3绿幕抠图提取 W4多图合成 W5改字): agnes edits 端点团队不可用
    → 路由到本地更强工具 (色度键抠图/BiRefNet/PIL加字/合成管线)
铁律 (来自 SKILL.md): 失败→报错不重试; 效果不好→交用户确认; 用户文字逐字照抄。
"""
from __future__ import annotations
import base64, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "ai-agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai-agent"))
if str(ROOT / "ai-service" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai-service" / "src"))

import numpy as np
from PIL import Image

OUT = ROOT / "outputs" / "habit"


# ---------------------------------------------------------------- 模板 (逐字来自技能 references)
def p_cover_overlay(title, subtitle, ratio="（16：9）"):
    """W1 封面加字 (本集成: 本地 PIL 文字叠加)。"""
    return {"template": "cover_overlay", "ratio": ratio, "title": title, "subtitle": subtitle}


def p_subtitle_bar(name, title, theme="演播室", transparent=True, one_line=True):
    """W2 字幕条 (文生图→agnes)。透明底=文生图直出真透明 PNG (铁律2)。"""
    layout = "一行排版" if one_line else "两行排版"
    bg = "输出真透明底" if transparent else "背景为方便抠图的绿色背景"
    return (f"设计一个视频用字幕条（主题是{theme}，{layout}）\n{name}\n{title}\n"
            f"注意字体颜色排版等等，要有高级感，设计感，尽可能紧凑一点\n{bg}")


def p_extract(element, keep_person=False):
    """W3 抠图提取 prompt (图生图→本地色度键/抠图)。"""
    s = f"将{element}完整的提取出来（不改变布局排版等等），背景设置为方便抠图的纯绿色背景"
    if not keep_person:
        s += "，不要人物等任何其他画面元素"
    return s + "，整体降噪处理"


def p_merge(img_a, img_b, position="画面右侧"):
    """W4 多图合成 prompt (图生图→本地合成管线)。"""
    return (f"将图二人物完美的融入到图一（{position}），人物都保持在一个水平面"
            f"（脚要在一个水平面，身高尽量匹配），注意脸部细节")


def p_change_text(new_text):
    """W5 就地改字 (图生图→本地 inpaint+文字)。"""
    return f"字改成{new_text}，其他不变"


def p_transparent_title(title, subtitle, style="高级感设计感", theme="科技"):
    """W9 透明底直出 (文生图→agnes, 铁律: 不传图+写“透明底”)。"""
    return (f"设计一个视频开头用字幕\n{title}\n{subtitle}\n"
            f"要有高级感设计感（符合主题，{style}，符合{theme}主题），输出真透明底")


# ---------------------------------------------------------------- 生成通道
def _load_font(px: int, prefer_bold: bool = True):
    """加载系统字体, 失败则用默认字体。"""
    from PIL import ImageFont
    candidates = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
                  r"C:\Windows\Fonts\simhei.ttf"] if prefer_bold else \
                 [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
                  r"C:\Windows\Fonts\simhei.ttf"]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, px)
            except Exception:
                pass
    return ImageFont.load_default()


def _local_typography_bar(name: str, role: str, out_path: Path, theme: str = "演播室",
                          transparent: bool = True, H: int = 220, W: int = 1400) -> dict:
    """字幕条本地排版 (W2) — 文字 100% 准确。"""
    from PIL import ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)) if transparent else \
        Image.new("RGBA", (W, H), (20, 160, 80, 255))
    dr = ImageDraw.Draw(im)
    f1 = _load_font(int(H * 0.34), prefer_bold=True)
    f2 = _load_font(int(H * 0.2), prefer_bold=False)
    bar_w = int(H * 0.12)
    dr.rectangle([0, 0, bar_w, H], fill=(75, 91, 215, 235))
    dr.text((bar_w + 40, int(H * 0.16)), name, font=f1, fill=(255, 255, 255, 245))
    w1 = dr.textbbox((0, 0), name, font=f1)[2]
    dr.text((bar_w + 40 + w1 + 36, int(H * 0.30)), role, font=f2, fill=(210, 214, 224, 230))
    im.save(out_path)
    return {"path": str(out_path), "provider": "local-typography",
            "note": "本地排版直出, 文字逐字准确" + ("" if transparent else " (绿底素材)"),
            "transparent": transparent}


def _local_title_card(title: str, subtitle: str, out_path: Path, theme: str = "科技",
                      transparent: bool = True, W: int = 1920, H: int = 1080) -> dict:
    """W9 片头字幕透明底排版 — 居中大字 + 副标题。"""
    from PIL import ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)) if transparent else \
        Image.new("RGBA", (W, H), (20, 160, 80, 255))
    dr = ImageDraw.Draw(im)
    ft = _load_font(int(H * 0.12), prefer_bold=True)
    fs = _load_font(int(H * 0.05), prefer_bold=False)
    # 主题色下划线
    color = (75, 91, 215, 235) if theme in ("科技", "科技蓝") else (200, 60, 60, 235)
    # 主标题居中
    wt = dr.textbbox((0, 0), title, font=ft)[2]
    xt = (W - wt) / 2
    yt = H * 0.36
    dr.text((xt, yt), title, font=ft, fill=(255, 255, 255, 245))
    # 副标题居中
    ws = dr.textbbox((0, 0), subtitle, font=fs)[2]
    xs = (W - ws) / 2
    ys = yt + int(H * 0.16)
    dr.text((xs, ys), subtitle, font=fs, fill=(220, 225, 235, 230))
    # 装饰下划线
    line_w = min(W * 0.25, max(wt, ws) * 0.6)
    dr.rectangle([W / 2 - line_w / 2, ys + int(H * 0.08), W / 2 + line_w / 2, ys + int(H * 0.08) + 6],
                 fill=color)
    im.save(out_path)
    return {"path": str(out_path), "provider": "local-title-card",
            "note": "本地排版直出, 文字逐字准确", "transparent": transparent}


def _agnes_t2i(prompt: str, out_path: Path, size: str = "1024x1024") -> dict:
    from agnes_client import AgnesClient
    AgnesClient().generate_image(prompt, str(out_path), size=size)
    _check_alpha(out_path, expect_transparent="透明底" in prompt)
    return {"path": str(out_path), "provider": "agnes-t2i", "prompt": prompt}


def _check_alpha(p: Path, expect_transparent: bool):
    """铁律: 文生图透明底必须返回 alpha; 无 alpha 时警告 (不自动重试)。"""
    img = Image.open(p)
    has_alpha = img.mode in ("RGBA", "LA") and np.array(img)[..., 3].min() < 250
    if expect_transparent and not has_alpha:
        print(f"[habit] 警告: 期望透明底但返回无 alpha (mode={img.mode})。"
              f"可改用绿底链路 + 本地色度键。", flush=True)
    return has_alpha


def _local_green_extract(img_path: str, out_path: Path) -> dict:
    """W3: 绿幕抠图提取 → 本地色度键 (比 API 生成更干净)。"""
    sys.path.insert(0, str(ROOT / "training" / "scripts"))
    from ai_material_eval import chroma_key_alpha, despill
    arr = np.array(Image.open(img_path).convert("RGB"))
    alpha = chroma_key_alpha(arr)
    fg = despill(arr, alpha)
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([fg, a8])
    out_p = Path(str(out_path).replace(".png", "_rgba.png"))
    Image.fromarray(rgba, "RGBA").save(out_p)
    Image.fromarray(fg).save(out_path)
    return {"path": str(out_path), "rgba_path": str(out_p), "provider": "local-chroma",
            "fg_ratio": round(float((alpha > 0.5).mean()), 3)}


def _local_cover_overlay(img_path: str, title: str, subtitle: str, out_path: Path,
                         ratio: str = "16:9") -> dict:
    """W1: 封面加字 → 本地 PIL 文字叠加 (标题+副标题, 底部渐变压暗)。"""
    import cv2
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(overlay)
    grad_h = int(H * 0.35)
    for i in range(grad_h):
        alpha = int(140 * (1 - i / grad_h))
        dr.line([(0, H - grad_h + i), (W, H - grad_h + i)], fill=(0, 0, 0, alpha))
    def load_font(px):
        for fp in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
                   r"C:\Windows\Fonts\simhei.ttf"):
            if Path(fp).exists():
                return ImageFont.truetype(fp, px)
        return ImageFont.load_default()
    f1 = load_font(int(H * 0.075)); f2 = load_font(int(H * 0.042))
    def center(draw, text, font, y, fill=(255, 255, 255)):
        w = draw.textbbox((0, 0), text, font=font)[2]
        draw.text(((W - w) / 2, y), text, font=font, fill=fill)
    center(dr, title, f1, H - grad_h + int(grad_h * 0.18))
    center(dr, subtitle, f2, H - grad_h + int(grad_h * 0.62), fill=(230, 230, 230))
    out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    out.save(out_path)
    return {"path": str(out_path), "provider": "local-overlay", "title": title}


def _local_composite(subject_img: str, bg_img: str, out_path: Path,
                     position: str = "画面中央") -> dict:
    """W4: 多图合成 → 透明叠加优先, 否则走抠图+合成管线。"""
    bg = Image.open(bg_img).convert("RGBA")
    sub = Image.open(subject_img)
    # 若 image2 本身带透明且为设计元素/字幕, 直接 alpha 叠加 (避免抠成黑底)
    has_alpha = sub.mode in ("RGBA", "LA")
    if has_alpha:
        a = np.array(sub.convert("RGBA"))[:, :, 3]
        if a.min() < 250 and (a < 250).mean() > 0.05:   # 有显著透明区域
            sub = sub.convert("RGBA")
            # 等比缩放, 宽度适配背景 0.6~0.9
            bw, bh = bg.size
            sw, sh = sub.size
            scale = min(bw / sw * 0.85, bh / sh * 0.65, 1.0)
            nw, nh = int(sw * scale), int(sh * scale)
            sub = sub.resize((nw, nh), Image.LANCZOS)
            # 根据 position 决定锚点
            pos = position.lower()
            if "右" in pos:
                x = bw - nw - int(bw * 0.05); y = (bh - nh) // 2
            elif "左" in pos:
                x = int(bw * 0.05); y = (bh - nh) // 2
            elif "上" in pos or "顶" in pos:
                x = (bw - nw) // 2; y = int(bh * 0.05)
            elif "下" in pos or "底" in pos:
                x = (bw - nw) // 2; y = bh - nh - int(bh * 0.05)
            else:
                x = (bw - nw) // 2; y = (bh - nh) // 2
            bg.paste(sub, (x, y), sub)
            bg.convert("RGB").save(out_path)
            return {"path": str(out_path), "provider": "local-alpha-overlay",
                    "note": f"透明元素直接叠加到背景({position})"}
    # 否则按人物/前景走完整合成管线
    sys.path.insert(0, str(ROOT / "app"))
    import studio_cli
    r = studio_cli.task_composite(subject_img, bg_img, str(OUT / "w4"))
    shutil.copy(r["final"], out_path)
    return {"path": str(out_path), "provider": "local-composite"}


def _local_change_text(img_path: str, new_text: str, out_path: Path) -> dict:
    """W5: 就地改字 (简版: 原字区 inpaint + 居中重写)。"""
    import cv2
    arr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    H, W = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th[: int(H * 0.5), :] = 0          # 只处理下半部 (字幕区)
    mask = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=2)
    arr = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(im)
    fp = r"C:\Windows\Fonts\msyh.ttc" if Path(r"C:\Windows\Fonts\msyh.ttc").exists() else None
    font = ImageFont.truetype(fp, int(H * 0.06)) if fp else ImageFont.load_default()
    w = dr.textbbox((0, 0), new_text, font=font)[2]
    dr.text(((W - w) / 2, H * 0.82), new_text, font=font, fill=(255, 255, 255))
    im.save(out_path)
    return {"path": str(out_path), "provider": "local-inpaint-text"}


import shutil  # noqa: E402  (W4 用)


# ---------------------------------------------------------------- 工作流路由
def generate_habit(workflow: str, params: dict, out_dir: str | None = None) -> dict:
    """
    habit 工作流路由 (铁律: 失败报错不重试; 用户文字逐字照抄)。
    workflow: W1 封面加字 | W2 字幕条 | W3 抠图提取 | W4 多图合成 | W5 改字
              | W6 PPT封面 | W7 分镜 | W9 透明底直出
    params 字段: image/image2(图生图类必传), title, subtitle, text, new_text, element,
                 name, role, theme, style, transparent, keep_person, position
    """
    out_dir = Path(out_dir) if out_dir else OUT / workflow
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000) % 100000
    out_p = out_dir / f"{workflow}_{stamp}.png"

    if workflow == "W2":       # 字幕条: 本地排版 (文字逐字准确; agnes 文字渲染不可靠)
        return _local_typography_bar(params.get("name", "姓名"), params.get("role", "头衔"),
                                     out_p, theme=params.get("theme", "演播室"),
                                     transparent=params.get("transparent", True))
    if workflow == "W9":       # 透明底直出: 本地排版 (片头字幕)
        return _local_title_card(params.get("title", "标题"), params.get("subtitle", "副标题"),
                                 out_p, theme=params.get("theme", "科技"),
                                 transparent=params.get("transparent", True))
    if workflow in ("W6", "W7"):   # PPT 单页封面 / 分镜: agnes 文生图
        prompt = (f"设计一张{'PPT 封面' if workflow=='W6' else '分镜图（多格宫格构图）'}\n"
                  f"{params.get('title','标题')}\n{params.get('subtitle','')}\n"
                  f"要有高级感设计感，让人眼前一亮")
        return _agnes_t2i(prompt, out_p)
    if workflow == "W1":       # 封面加字: 本地叠加 (截图必传)
        if not params.get("image"):
            raise ValueError("W1 需要参考截图 (params.image)")
        return _local_cover_overlay(params["image"], params.get("title", "标题"),
                                    params.get("subtitle", ""), out_p,
                                    ratio=params.get("ratio", "16:9"))
    if workflow == "W3":       # 抠图提取: 本地色度键
        if not params.get("image"):
            raise ValueError("W3 需要原图 (params.image)")
        return _local_green_extract(params["image"], out_p)
    if workflow == "W4":       # 多图合成: 本地抠图+合成
        if not (params.get("image") and params.get("image2")):
            raise ValueError("W4 需要 image(背景) 与 image2(人物)")
        return _local_composite(params["image2"], params["image"], out_p,
                                position=params.get("position", "画面中央"))
    if workflow == "W5":       # 就地改字: 本地 inpaint+文字
        if not params.get("image"):
            raise ValueError("W5 需要图 (params.image)")
        return _local_change_text(params["image"], params.get("new_text", params.get("text", "新文字")),
                                  out_p)
    raise ValueError(f"未知工作流 {workflow} (支持 W1/W2/W3/W4/W5/W6/W7/W9)")


if __name__ == "__main__":
    # 冒烟: W9 透明底直出 (agnes 文生图)
    r = generate_habit("W9", {"title": "演播室智能体", "subtitle": "一句话完成专业合成",
                              "theme": "科技", "style": "高级感设计感"})
    print("W9:", r)
