# -*- coding: utf-8 -*-
"""
全面图片特效引擎 (fx_library.py) — 参考市面主流特效 (醒图/美图秀秀/抖音/Photoshop等)
分类(全目录): 滤镜 / 风格化 / 美颜 / 边框纹理 / 光影特效 / 局部特效 / 贴纸与道具
自动智能布局: 由 fx_auto.apply 统一调度 — 自动定位 + 自动拉伸 + 自动设参 + 自动混合

API:
  ALL_EFFECTS          -> [{key,cat,cn,params:{p:[min,max,def]}}]   特效全目录
  AUTO_LAYOUT          -> {类别:[...特效keys]}                        自动可布局的特效
  fx_auto.apply(img, effect, region=None, mode="auto") -> uint8      自动放置+拉伸+设参
  fx_manual.apply(img, effect, params) -> uint8                       手动指定参数
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps

# ===================================================================== 基础工具
def _u8(x): return np.clip(x, 0, 255).astype(np.uint8)
def _f32(img): return np.array(img.convert("RGB"), np.float32)
def _alpha_region(mask, feather=25):  # mask 0-255, 返回羽化 0-1
    return cv2.GaussianBlur((mask.astype(np.float32)/255), (0, 0), feather)


def _fitbox(region, img, def_frac=0.25):
    """把 region(点/框/多边形) 规范化为 (x,y,w,h) + frac; 无 region 时默认中心占 25%。"""
    H, W = img.shape[:2]
    if region is None:
        w, h = int(W*def_frac), int(H*def_frac)
        return (W//2 - w//2, H//2 - h//2, w, h)
    t = region.get("type")
    if t == "box":
        x1, y1, x2, y2 = [int(v) for v in region["xyxy"]]
        return (min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
    if t == "polygon":
        xs = [int(p[0]) for p in region["points"]]; ys = [int(p[1]) for p in region["points"]]
        return (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
    return (0, 0, W, H)


def _box_mask(img, box, soft=0):
    x, y, w, h = box
    m = np.zeros(img.shape[:2], np.uint8)
    m[y:y+h, x:x+w] = 255
    if soft: m = cv2.GaussianBlur(m, (0, 0), soft)
    return m


# ===================================================================== 滤镜 (LUT / 颜色)
FILTERS = {}
def _register(name, fn): FILTERS[name] = fn

def _lut_lift(img, r, g, b):  # 线性通道乘加
    f = img.astype(np.float32)
    return _u8(f * np.array([r, g, b], np.float32)[None, None, :])

# ---- 胶片/复古
_register("vintage_1970s", lambda i: _u8(i.astype(np.float32)*np.array([1.12, 1.0, 0.82])[None,None,:]*0.92 + np.array([12, 6, 0])[None,None,:]))
_register("vintage_1980s", lambda i: _lut_lift(_u8(i.astype(np.float32)*np.array([1.18, 1.0, 1.22])[None,None,:]), 1.05, 1.0, 1.05))
_register("polaroid", lambda i: _lut_lift(i, 1.02, 1.0, 0.9))          # 冷调偏青
_register("sepia", lambda i: _apply_sepia(i))
def _apply_sepia(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    return np.stack([np.clip(g*1.05+18, 0, 255), np.clip(g*0.95+8, 0, 255), np.clip(g*0.8, 0, 255)], -1).astype(np.uint8)
_register("kodachrome", lambda i: _lut_lift(i, 1.1, 1.0, 0.85))
_register("portra", lambda i: _u8(i.astype(np.float32)*np.array([1.08, 1.0, 0.92])[None,None,:]+np.array([5, 0, 8])[None,None,:]))
_register("cinestill", lambda i: _lut_lift(i, 1.08, 1.0, 1.3))          # 暖钨
_register("bleach_bypass", lambda i: _u8(0.6*cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)[...,None] + 0.4*i.astype(np.float32)))
# ---- 黑白
def _bw(i, contrast=1.0, lift=0):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = np.clip((g-128)*contrast+128+lift, 0, 255)
    return np.stack([g]*3, -1).astype(np.uint8)
_register("bw_classic", lambda i: _bw(i, 1.0))
_register("bw_highcontrast", lambda i: _bw(i, 1.5))
_register("bw_noir", lambda i: _u8(_bw(i, 1.7).astype(np.float32)*1.1))
_register("infrared", lambda i: _bw_ir(i))
def _bw_ir(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sky = np.clip((i[..., 2].astype(np.float32) - g)*0.6, -30, 0)
    veg = np.clip((i[..., 1].astype(np.float32) - g)*0.6, 0, 60)
    out = np.clip(g + veg + sky, 0, 255)
    return np.stack([out]*3, -1).astype(np.uint8)
# ---- 现代/氛围
_register("clean_bright", lambda i: _u8(i.astype(np.float32)*1.06+10))
_register("moody_dark", lambda i: _moody(i))
def _moody(i):
    f = i.astype(np.float32)
    f = np.clip((f-128)*1.15+110, 0, 255)      # 对比高, 暗部压黑
    return _u8(f*np.array([0.92, 1.0, 1.14])[None, None, :])  # 冷调
_register("pastel", lambda i: _u8(np.clip((i.astype(np.float32)-128)*0.85+140, 0, 255)))
_register("vibrant_pop", lambda i: _u8(i.astype(np.float32)*np.array([1.2, 1.2, 1.2])[None,None,:]))
# ---- 电影
_register("teal_orange", lambda i: _teal_orange(i))
def _teal_orange(i):
    f = i.astype(np.float32)
    out = f.copy()
    out[..., 0] *= 1.08      # 高光偏橙 R+
    out[..., 2] *= 0.95      # 阴影偏青 B 微降 -> 用通道近似
    low = f.mean(-1) < 100
    out[low, 2] += 15        # 暗部加青蓝
    out[~low, 0] += 12       # 亮部加橙
    return _u8(out)
_register("film_golden", lambda i: _u8(i.astype(np.float32)*np.array([1.15, 1.02, 0.85])[None,None,:]+np.array([10, 0, 0])[None,None,:]))
# ---- 风格化色
_register("cyberpunk", lambda i: _u8(i.astype(np.float32)*np.array([1.3, 0.85, 1.5])[None,None,:]))
_register("vaporwave", lambda i: _u8(i.astype(np.float32)*np.array([1.1, 0.9, 1.6])[None,None,:]))
_register("nightvision", lambda i: _nv(i))
def _nv(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    return np.stack([np.clip(g*0.3, 0, 255), np.clip(g*1.2, 40, 255), np.clip(g*0.5, 0, 255)], -1).astype(np.uint8)
_register("thermal", lambda i: _thermal(i))
def _thermal(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    lut = np.array([[0, 0, 20], [60, 0, 120], [200, 60, 0], [255, 220, 60]], np.float32)
    idx = np.clip(g, 0, 1) * 3
    f = np.floor(idx).astype(int); c = np.ceil(idx).astype(int); fr = idx - f
    c = np.minimum(c, 3)
    return _u8(lut[f]*(1-fr[...,None]) + lut[c]*fr[...,None])
_register("duotone", lambda i: _duotone(i))
def _duotone(i, hi=(230, 120, 60), lo=(10, 10, 40)):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    return _u8(np.stack([lo[c]+(hi[c]-lo[c])*g for c in range(3)], -1))
# 暖色温滤镜 (独立, 供面板"滤镜-色温"用)
def color_temp_lut(i, warm=0.15):
    return _u8(i.astype(np.float32)*np.array([1+warm, 1, 1-warm*0.6])[None,None,:])


# ===================================================================== 风格化 (结构变换)
STYLES = {}
def _register_style(name, fn): STYLES[name] = fn
_register_style("mosaic_cell", lambda i, p: _pixellate(i, p.get("cell", 12)))
_register_style("pixelate", lambda i, p: _pixellate(i, p.get("cell", 16)))
def _pixellate(i, cell):
    h, w = i.shape[:2]
    small = cv2.resize(i, (max(1, w//max(cell, 1)), max(1, h//max(cell, 1))), cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), cv2.INTER_NEAREST)
_register_style("oil_painting", lambda i, p: _oil(i, p.get("radius", 6)))
def _oil(i, radius):
    out = i.copy()
    small = cv2.resize(i, (max(8, i.shape[1]//3), max(8, i.shape[0]//3)))
    small = cv2.bilateralFilter(small, 9, 60, 60)
    return cv2.resize(small, (i.shape[1], i.shape[0]), cv2.INTER_CUBIC)
_register_style("cartoon_comic", lambda i, p: _cartoon(i, p.get("level", 8)))
_register_style("toon", lambda i, p: _cartoon(i, p.get("level", 12)))
def _cartoon(i, level):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 5)
    edge = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
    q = cv2.bilateralFilter(i, 9, 60, 60)
    q = cv2.resize(q, (max(8, i.shape[1]//max(level, 2)), max(8, i.shape[0]//max(level, 2))), cv2.INTER_LINEAR)
    q = cv2.resize(q, (i.shape[1], i.shape[0]), cv2.INTER_LINEAR)
    out = q.copy()
    out[edge == 0] = [0, 0, 0]
    return out
_register_style("sketch_pencil", lambda i, p: _sketch(i))
def _sketch(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    inv = 255 - g
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    dodge = cv2.divide(g, 255 - blur, scale=256)
    return np.stack([dodge]*3, -1).astype(np.uint8)
_register_style("halftone", lambda i, p: _halftone(i, p.get("cell", 6)))
def _halftone(i, cell):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = i.shape[:2]
    out = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (yy//cell + 0.5)*cell, (xx//cell + 0.5)*cell
    d = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    lum = np.clip(g/255, 0.01, 1)
    rad = cell*0.55*(1 - lum)
    out[d < rad] = 0
    return out
_register_style("crystallize", lambda i, p: _crystal(i, p.get("cell", 18)))
def _crystal(i, cell):
    h, w = i.shape[:2]
    ys = np.arange(0, h, cell); xs = np.arange(0, w, cell)
    rng = np.random.default_rng(3)
    pts = [(x + int(rng.uniform(-cell//3, cell//3)), y + int(rng.uniform(-cell//3, cell//3)))
           for y in ys for x in xs]
    cells = np.full((h, w, 3), 0, np.float32)
    for (x, y) in pts:
        if not (0 <= x < w and 0 <= y < h): continue
        c = i[max(0, y-2):y+3, max(0, x-2):x+3].mean((0, 1))
        # 简单最近邻 cell 填充(降采样风格)
    # 简化: 用晶格均值重映射
    out = cv2.resize(i, (max(1, w//max(cell//2, 1)), max(1, h//max(cell//2, 1))), cv2.INTER_AREA)
    return cv2.resize(out, (w, h), cv2.INTER_NEAREST)
_register_style("stained_glass", lambda i, p: _pixellate(i, p.get("cell", 20)))
_register_style("pointillism", lambda i, p: _pixellate(i, p.get("cell", 8)))
_register_style("watercolor", lambda i, p: cv2.resize(cv2.bilateralFilter(cv2.resize(i, (i.shape[1]//2, i.shape[0]//2)), 15, 40, 40), (i.shape[1], i.shape[0]), cv2.INTER_CUBIC))
_register_style("vhs", lambda i, p: _vhs(i))
def _vhs(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    scan = (np.arange(i.shape[0]) % 6 < 1)
    out = i.copy()
    out[scan] = np.clip(out[scan].astype(np.float32)*0.6, 0, 255).astype(np.uint8)
    out = np.roll(out, 4, axis=1).copy() if False else out
    return out
_register_style("glitch", lambda i, p: _glitch(i))
def _glitch(i):
    out = i.copy()
    h, w = i.shape[:2]
    out[..., 0] = np.roll(out[..., 0], 6, axis=1)
    out[..., 2] = np.roll(out[..., 2], -6, axis=1)
    return out
_register_style("comic_halftone", lambda i, p: _cartoon(i, p.get("level", 8)))


# ===================================================================== 扩充特效 (v2 追加)
def _hsv_shift(i, dh=0, ds=0, dv=0):
    f = cv2.cvtColor(i, cv2.COLOR_RGB2HSV).astype(np.float32)
    f[..., 0] = (f[..., 0] + dh) % 180
    f[..., 1] = np.clip(f[..., 1] + ds, 0, 255)
    f[..., 2] = np.clip(f[..., 2] + dv, 0, 255)
    return cv2.cvtColor(_u8(f), cv2.COLOR_HSV2RGB)

def _blend_lut(i, r, g, b, warm=0, cool=0):
    out = i.astype(np.float32) * np.array([r, g, b], np.float32)[None, None, :]
    out[..., 0] += warm; out[..., 2] += cool
    return _u8(out)
def _curve_s(i, power=0.8):  # S 曲线
    f = i.astype(np.float32)/255
    f = f*f*(3-2*f) if power < 1 else f
    return _u8(f*255)
def _grain(i, amt): return fx_grain(i, amt)
def _sharpen(i, amt=0.5):
    blur = cv2.GaussianBlur(i, (0, 0), 2)
    return cv2.addWeighted(i, 1 + amt, blur, -amt, 0)
def _punch(i, sat=1.2):
    return _hsv_shift(i, ds=(sat-1)*80)
def _lomo(i):
    out = _u8(i.astype(np.float32)*np.array([1.05,1.0,1.1])[None,None,:])
    return fx_vignette(out, 0.4)
def _faded(i, lift=18):
    return _u8(np.clip((i.astype(np.float32)-128)*0.85+128+lift, 0, 255))
def _contrast(i, c=1.3):
    return _u8(np.clip((i.astype(np.float32)-128)*c+128, 0, 255))
def _teal(i):
    return _blend_lut(i, 1.02, 1.05, 1.2, warm=0, cool=20)
def _orange(i): return _blend_lut(i, 1.2, 1.05, 0.95, warm=25, cool=0)
def _green(i): return _blend_lut(i, 0.9, 1.25, 0.95)
def _purple(i): return _blend_lut(i, 1.2, 0.9, 1.25)
def _rose(i): return _blend_lut(i, 1.15, 0.9, 0.95, warm=15, cool=0)
def _dusk(i): return _blend_lut(i, 1.1, 0.95, 1.05, warm=20, cool=5)
def _dawn(i): return _blend_lut(i, 1.15, 1.0, 1.0, warm=30, cool=10)
def _neon(i): return _hsv_shift(i, ds=40, dv=10)
def _retro(i): return _blend_lut(i, 1.3, 0.9, 1.1, warm=30, cool=20)
def _solarize(i):
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
    out = np.where(g < 128, i.astype(np.float32), 255 - i.astype(np.float32))
    return _u8(out)
def _edge_glow(i): return _u8(np.clip(i.astype(np.float32)*0.6 + cv2.Canny(i, 100, 200)[...,None]*1.5, 0, 255))
def _emboss(i):
    k = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.filter2D(g, -1, k); return np.stack([np.clip(e+128,0,255)]*3,-1).astype(np.uint8)
def _cross_process(i): return _blend_lut(i, 1.1, 0.85, 1.25, warm=-5, cool=30)
def _xpro(i):
    out = _blend_lut(i, 1.15, 0.95, 1.1, warm=10, cool=10)
    return _u8(_contrast(out, 1.25))
def _acid(i): return _hsv_shift(i, ds=60, dv=10)
def _invert_filter(i): return 255 - i
def _sepia2(i): return _apply_sepia(i)
def _desat(i, f=0.5): return _u8(i.astype(np.float32)*(1-f) + cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)[...,None]*f)

# v2 滤镜注册 (滤镜-更多)
_more = {
    "noir": lambda i: _bw(i, 1.3), "soft_bw": lambda i: _bw(i, 0.8, 20),
    "s_curve": lambda i: _contrast(i, 1.25), "faded_film": lambda i: _faded(i, 25),
    "lomo": lambda i: _lomo(i), "teal_shadow": lambda i: _teal(i), "orange_highlight": lambda i: _orange(i),
    "fresh_green": lambda i: _green(i), "royal_purple": lambda i: _purple(i),
    "rose_gold": lambda i: _rose(i), "dusk_mood": lambda i: _dusk(i), "dawn_warm": lambda i: _dawn(i),
    "neon_city": lambda i: _neon(i), "retro_synth": lambda i: _retro(i), "solarize": lambda i: _solarize(i),
    "cross_process": lambda i: _cross_process(i), "xpro": lambda i: _xpro(i), "acid_pop": lambda i: _acid(i),
    "invert_color": lambda i: _invert_filter(i), "emboss_light": lambda i: _emboss(i),
    "edge_glow": lambda i: _edge_glow(i), "sharp_photo": lambda i: _sharpen(i),
    "grainy_film": lambda i: _grain(i, 12), "desat_mood": lambda i: _desat(i, 0.6),
    "cream_cream": lambda i: _u8(i.astype(np.float32)*np.array([1.0,0.98,0.92])[None,None,:]+15),
    "lavender": lambda i: _blend_lut(i, 1.15, 0.9, 1.3, cool=30),
    "baby_blue": lambda i: _blend_lut(i, 0.95, 1.05, 1.2, cool=25),
    "mint": lambda i: _blend_lut(i, 0.9, 1.15, 1.05, cool=10),
    "peach": lambda i: _blend_lut(i, 1.25, 0.95, 0.9, warm=25),
    "warm_honey": lambda i: _blend_lut(i, 1.15, 1.05, 0.85, warm=20),
    "cool_steel": lambda i: _blend_lut(i, 0.9, 1.0, 1.1, cool=25),
    "vintage_lomography": lambda i: _u8(_faded(_lomo(i), 15)),
    "night_walk": lambda i: _u8(np.clip((i.astype(np.float32)-128)*1.1+100,0,255) * np.array([0.8,1,1.2])[None,None,:]),
}
for _n, _f in _more.items(): _register(_n, _f)

# v2 风格注册
def _mirror(i, parts=3):
    # 万花筒
    import math
    return i
def _texture_warp(i):
    h, w = i.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    # 波浪形变形
    mapx = (xx + 15*np.sin(yy/25)).astype(np.float32)
    mapy = (yy + 15*np.sin(xx/30)).astype(np.float32)
    return cv2.remap(i, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def _liquify(i): return _texture_warp(i)
def _lens_distort(i, strength=0.3):
    h, w = i.shape[:2]
    cx, cy = w/2, h/2
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = (xx-cx)/cx, (yy-cy)/cy
    r = np.sqrt(dx*dx + dy*dy)
    k = strength
    mapx = (dx*(1+k*r*r)+1)*cx
    mapy = (dy*(1+k*r*r)+1)*cy
    return cv2.remap(i, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def _twirl(i, a=0.5):
    h, w = i.shape[:2]; cx, cy = w/2, h/2
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = (xx-cx), (yy-cy)
    r = np.sqrt(dx*dx+dy*dy)
    ang = np.arctan2(dy, dx) + (r/ (w/2)) * a
    mapx = cx + r*np.cos(ang); mapy = cy + r*np.sin(ang)
    return cv2.remap(i, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def _popart(i):
    q = _pixellate(i, 12)
    return _u8(q.astype(np.float32)*np.array([1.3,1.0,1.3])[None,None,:])
def _psychedelic(i): return _hsv_shift(_pixellate(i, 8), dh=40, ds=50)
def _scanlines(i):
    out = i.copy(); out[::4] = np.clip(out[::4].astype(np.float32)*0.5, 0, 255).astype(np.uint8)
    return out
def _dot_screen(i, cell=4): return _halftone(i, cell)
def _mirror_kaleido(i):
    # 简单镜面万花筒
    half = i[:, :i.shape[1]//2]
    mirrored = cv2.flip(half, 1)
    out = np.hstack([half, mirrored])
    return out
_style_more = {
    "dot_screen": lambda i, p: _halftone(i, p.get("cell", 5)),
    "comic_popart": lambda i, p: _popart(i),
    "psychedelic": lambda i, p: _psychedelic(i),
    "scanlines": lambda i, p: _scanlines(i),
    "twirl": lambda i, p: _twirl(i, p.get("a", 0.8)),
    "lens_distort": lambda i, p: _lens_distort(i, p.get("strength", 0.3)),
    "liquid": lambda i, p: _texture_warp(i),
    "mirror_kaleido": lambda i, p: _mirror_kaleido(i),
    "emboss_art": lambda i, p: _emboss(i),
    "neon_edges": lambda i, p: _edge_glow(i),
    "smoke_bw": lambda i, p: _bw(_sketch(i), 1.2),
    "pencil_color": lambda i, p: _u8(i.astype(np.float32)*0.4 + _sketch(i).astype(np.float32)*0.6),
    "water_dream": lambda i, p: _pixellate(cv2.GaussianBlur(i, (0,0), 3), 6),
}
for _n, _f in _style_more.items(): _register_style(_n, _f)


# ===================================================================== 美颜 / 美妆
def beautify_skin(i, smooth=10, amount=0.7, whiten=8):
    """磨皮(双边) + 提亮。整体脸部肤色处理(近似美颜)。"""
    blurred = cv2.bilateralFilter(i, 9, smooth, smooth)
    out = cv2.addWeighted(blurred, 1-amount, i, amount, 0)
    return _u8(out.astype(np.float32) + whiten)
BEAUTY = {
    "natural": lambda i: beautify_skin(i, 10, 0.6, 6),
    "fair": lambda i: beautify_skin(i, 12, 0.7, 20),
    "smooth": lambda i: beautify_skin(i, 14, 0.8, 4),
    "sharp": lambda i: cv2.addWeighted(cv2.GaussianBlur(i, (0, 0), 0), -0.4, cv2.bilateralFilter(i, 9, 8, 8), 1.4, 0),
    "blur_skin": lambda i: cv2.bilateralFilter(i, 15, 20, 20),
}
# 美妆 LUT (近似): 口红/腮红/眼影 用局部 HSV
def makeup_lip(i, strength=30):
    f = cv2.cvtColor(i, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = f[..., 0], f[..., 1], f[..., 2]
    # 偏红肤区(近似唇/腮) 增红
    warm = (h < 25) | (h > 150)
    s[warm] = np.minimum(s[warm] + strength, 255)
    f[..., 1] = s
    return cv2.cvtColor(_u8(f), cv2.COLOR_HSV2RGB)
def makeup_cheek(i, strength=20):
    return makeup_lip(i, strength)
def makeup_eye(i, strength=25):
    f = cv2.cvtColor(i, cv2.COLOR_RGB2HSV).astype(np.float32)
    dark = f[..., 2] < 120
    f[dark, 0] = (f[dark, 0] + 5) % 180   # 微调
    f[dark, 2] = np.minimum(f[dark, 2] * 0.9, 255)
    return cv2.cvtColor(_u8(f), cv2.COLOR_HSV2RGB)


# ===================================================================== 光影/局部特效
def make_effect(name, img, params=None):
    """动态叠加式光影特效 (区域可控)。返回混合后图。"""
    p = params or {}
    H, W = img.shape[:2]
    if name == "vignette":
        return fx_vignette(img, p.get("strength", 0.6))
    if name == "bokeh":
        return fx_bokeh(img, p.get("n", 40), p.get("strength", 0.7))
    if name == "spotlight":
        x, y = p.get("cx", W//2), p.get("cy", H//3)
        return fx_spotlight(img, (x, y), p.get("radius", W*0.45), p.get("intensity", 0.5))
    if name == "bloom":
        return fx_bloom(img, p.get("strength", 0.8))
    if name == "god_rays":
        return fx_godrays(img, p.get("strength", 0.5))
    if name == "film_grain":
        return fx_grain(img, p.get("amount", 15))
    if name == "tilt_shift":
        return fx_tiltshift(img)
    if name == "blur":
        r = int(p.get("radius", 8))
        return cv2.GaussianBlur(img, (0, 0), r)
    if name == "warm": return color_temp_lut(img, 0.15)
    if name == "cool": return color_temp_lut(img, -0.15)
    if name == "light_leak": return fx_lightleak(img)
    return img


def fx_vignette(i, strength=0.6):
    h, w = i.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx-w/2)/(w/2))**2 + ((yy-h/2)/(h/2))**2)
    m = 1 - np.clip(d-0.5, 0, 1)**1.8 * strength
    return _u8(i.astype(np.float32) * m[..., None])
def fx_bokeh(i, n=40, strength=0.7):
    rng = np.random.default_rng(2)
    h, w = i.shape[:2]
    layer = np.zeros_like(i, np.float32)
    g = i.mean(-1)
    hot = g > g.mean()*1.4
    for _ in range(n):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        r = int(rng.uniform(5, 22)); b = rng.uniform(0.3, 0.9)*strength
        tint = rng.uniform(0.85, 1.15, 3)
        cv2.circle(layer, (x, y), r, (255*b)*tint, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 8)
    return _u8(255 - (255 - i.astype(np.float32))*(255 - layer)/255)
def fx_spotlight(i, center, radius, intensity):
    yy, xx = np.mgrid[0:i.shape[0], 0:i.shape[1]]
    d = np.sqrt((xx-center[0])**2 + (yy-center[1])**2)
    glow = np.clip(1 - d/radius, 0, 1)**2 * intensity
    layer = glow[..., None]*np.array([255, 248, 225], np.float32)
    return _u8(255 - (255 - i.astype(np.float32))*(255-layer)/255)
def fx_bloom(i, strength=0.8):
    g = i.astype(np.float32)
    bright = np.clip(g.mean(-1)[..., None] - 160, 0, 95)/95
    blurred = cv2.GaussianBlur(g, (0, 0), 9)
    return _u8(np.clip(g + bright*blurred*strength, 0, 255))
def fx_godrays(i, strength=0.5):
    h, w = i.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - h*0.2, xx - w*0.5)
    rays = (np.cos(ang*20) + 1) / 2
    layer = rays[..., None]*np.array([255, 250, 235], np.float32)*strength*0.4
    return _u8(255 - (255 - i.astype(np.float32))*(255 - cv2.GaussianBlur(layer, (0, 0), 6))/255)
def fx_grain(i, amount=15):
    noise = np.random.default_rng(5).normal(0, amount, i.shape[:2])[..., None]
    return _u8(i.astype(np.float32) + noise)
def fx_tiltshift(i, focus=0.5):
    h = i.shape[:2][0]
    yy = np.arange(h)[:, None]
    m = np.exp(-((yy - focus*h)**2) / (2*(h*0.12)**2)).astype(np.float32)
    blur = cv2.GaussianBlur(i, (0, 0), 12)
    return _u8(i.astype(np.float32)*m[..., None] + blur.astype(np.float32)*(1-m[..., None]))
def fx_lightleak(i):
    h, w = i.shape[:2]
    grad = np.linspace(1.15, 0.9, w, dtype=np.float32)[None, :, None]
    out = _u8(i.astype(np.float32)*grad)
    out[..., 2] = _u8(out[..., 2].astype(np.float32)*1.05)
    return out


# ===================================================================== 贴纸 / 水印 (可自动布局)
def overlay_image(i, sticker_path, box, alpha=1.0, opacity=1.0):
    """把贴纸图贴到 box (x,y,w,h), 支持圆形蒙版(头像/水印常为圆形)。"""
    st = np.array(Image.open(sticker_path).convert("RGBA"))
    sw, sh = st.shape[1], st.shape[0]
    bw, bh = max(box[2], 1), max(box[3], 1)
    st2 = cv2.resize(st, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    x, y = box[0], box[1]
    out = i.copy()
    roi = out[y:y+bh, x:x+bw]
    if roi.shape[0] < 1 or roi.shape[1] < 1: return out
    a = (st2[..., 3:4].astype(np.float32)/255)*opacity
    out[y:y+bh, x:x+bw] = _u8(roi.astype(np.float32)*(1-a) + st2[..., :3].astype(np.float32)*a)
    return out


def _text_layer(i, text, box, font_size, color, opacity):
    x, y, w, h = box
    pil = Image.fromarray(i)
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont_default(font_size)
    except Exception:
        from PIL import ImageFont as _F
        try: font = _F.truetype("arial.ttf", font_size)
        except Exception: font = _F.load_default()
    # 居中
    bb = dr.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx, ty = x + (w - tw)//2 - bb[0], y + (h - th)//2 - bb[1]
    dr.text((tx, ty), text, font=font, fill=tuple(color)+(int(255*opacity),))
    pil = Image.alpha_composite(pil.convert("RGBA"), ov)
    return np.array(pil.convert("RGB"))


def ImageFont_default(sz):
    from PIL import ImageFont
    try: return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", sz)
    except Exception:
        try: return ImageFont.truetype("arial.ttf", sz)
        except Exception: return ImageFont.load_default()


def _rect_layer(i, box, color, fill, opacity, radius=0):
    x, y, w, h = box
    pil = Image.fromarray(i)
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    f = tuple(color)+(int(255*opacity),) if fill else None
    dr.rounded_rectangle([x, y, x+w, y+h], radius=radius, outline=f if not fill else None,
                         width=3 if not fill else 0, fill=f if fill else None)
    pil = Image.alpha_composite(pil.convert("RGBA"), ov)
    return np.array(pil.convert("RGB"))


# ===================================================================== 局部特效 (某区域做马赛克/模糊/水印/边框等)
def local_effect(img, box, name, params=None):
    """对 box 区域内做特效, 羽化边缘自然衔接。name: mosaic|blur|warm|cool|brighten|pencil_region|pixelate|invert"""
    p = params or {}
    x, y, w, h = [int(v) for v in box]
    x = max(0, x); y = max(0, y); w = min(w, img.shape[1]-x); h = min(h, img.shape[0]-y)
    if w < 2 or h < 2: return img
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y:y+h, x:x+w] = 255
    m = _alpha_region(mask, feather=p.get("feather", 12))
    reg = img[y:y+h, x:x+w]
    if name == "mosaic":
        cell = max(int(p.get("cell", 20) * w / 200), 4)
        eff = _pixellate(reg, cell)
    elif name == "blur": eff = cv2.GaussianBlur(reg, (0, 0), int(p.get("r", 12)))
    elif name == "brighten": eff = _u8(reg.astype(np.float32) + p.get("amt", 60))
    elif name == "invert": eff = 255 - reg
    elif name == "pixelate": eff = _pixellate(reg, max(int(p.get("cell", 30)*w/200), 4))
    elif name in FILTERS: eff = FILTERS[name](reg)
    elif name == "style_oil": eff = STYLES["oil_painting"](reg, p)
    else: eff = reg
    out = img.copy()
    # eff 是 box 区域结果, 铺回全图对齐后再按羽化 mask 混合
    eff_full = out.astype(np.float32).copy()
    eff_full[y:y+h, x:x+w] = eff.astype(np.float32)
    blend = m[..., None]
    out = _u8(out.astype(np.float32) * (1 - blend) + eff_full * blend)
    return out


# ===================================================================== 自动智能布局
# 自动布局预设: 每种贴纸/道具默认位置分数, 由 auto 拉伸
STICKER_PRESETS = {   # name -> (category, placement fn 返回 box 分数, 默认拉伸比)
}
# 文字水印/边框/徽章等"可自动放角落"的特效
# _AUTOSEG 在下方模块级构建

AUTO_DEF_FRAC = {   # 特效 -> 默认占图比例 (自动拉伸用)
    "sticker": 0.22, "heart": 0.12, "star": 0.12, "crown": 0.12, "bubble": 0.2,
    "emoji": 0.12, "cat": 0.25, "flower": 0.18, "rainbow": 0.4, "music": 0.15,
    "watermark_text": 0.5, "brand_logo": 0.18, "date_stamp": 0.45, "qr_code": 0.12,
    "polaroid_frame": 0.9, "film_border": 0.98, "photo_white": 0.9,
    "circle_avatar": 0.3, "rect_badge": 0.2, "light_ring": 0.7, "camera_lens": 0.25,
    "mosaic": 0.25, "blur_face": 0.25,
}


class fx_auto:
    """统一自动入口: 自动定位 + 自动拉伸 + 自动设参 + 自动混合"""
    @staticmethod
    def locate(img, mode):
        """自动位置: corner 类放四角, 默认中心。"""
        h, w = img.shape[:2]
        corners = {"tr": (0, 0), "tl": (0, 0), "br": (w, h), "bl": (w, h),
                   "tc": ((w - w*0.4)//2, 0), "bc": ((w - w*0.4)//2, h)}
        return corners.get(mode, (0, 0))

    @staticmethod
    def place(img, effect, region=None, params=None):
        """自动放置 + 拉伸 + 设参 -> (img, meta)"""
        p = dict(params or {})
        H, W = img.shape[:2]
        # 1) 位置: region 用户圈选优先, 否则按 effect 类别默认
        if region:
            box = _fitbox(region, img)
        else:
            cat, pres = AUTO_PRESET.get(effect, ("corner", "br"))
            frac = p.get("frac", AUTO_DEF_FRAC.get(effect, 0.25))
            if pres == "center":
                bw, bh = int(W*frac), int(H*frac)
                box = ((W-bw)//2, (H-bh)//2, bw, bh)
            elif pres in ("top", "bottom"):
                bw = int(W*frac); bh = int(H*frac*0.5)
                y = 0 if pres == "top" else max(0, H-bh)
                box = ((W-bw)//2, y, bw, bh)
            else:  # corner tl/tr/bl/br
                bw, bh = int(W*frac), int(H*frac)
                cx, cy = fx_auto.locate(img, pres)
                box = (cx, cy, bw, bh) if pres[0] == "t" else (cx-bw, cy-bh, bw, bh)
        # 2) 用所选类别执行
        return fx_auto._exec(img, effect, box, p)

    @staticmethod
    def _exec(img, effect, box, p):
        from pathlib import Path
        H, W = img.shape[:2]
        meta = {"effect": effect, "box": [int(v) for v in box]}
        # --- 装饰/贴纸类 ---
        if effect in DECOR:
            d = DECOR[effect]
            if d["kind"] == "sticker":
                stp = Path(__file__).parent / "assets" / f"{d['file']}.png"
                if stp.exists():
                    img = overlay_image(img, str(stp), box, opacity=d.get("opacity", 1.0))
                    meta["asset"] = str(stp)
                else: meta["warn"] = f"素材 {d['file']}.png 未生成"
            elif d["kind"] == "text":
                img = _text_layer(img, p.get("text", d.get("text", "WATERMARK")),
                                  box, int(box[2]*d.get("fs", 0.12)), d.get("color", (255, 255, 255)), d.get("opacity", 0.5))
            elif d["kind"] == "border":
                img = _border_effect(img, box, p)
            elif d["kind"] == "rect":
                img = _rect_layer(img, box, d.get("color", (255, 255, 255)), d.get("fill", True), d.get("opacity", 0.8), d.get("radius", 20))
            elif d["kind"] == "frame":
                img = _frame_polaroid(img, box)
            elif d["kind"] == "lens":
                img = _lens_overlay(img, box)
            elif d["kind"] == "avatar":
                img = _round_avatar(img, box, d.get("opacity", 0.9))
            elif d["kind"] == "diag":
                img = _diag_watermark(img, p.get("text", d.get("text", "WATERMARK")),
                                      int(img.shape[1]*0.06), d.get("opacity", 0.25), d.get("color", (255, 255, 255)))
            elif d["kind"] == "corners":
                img = _corner_marks(img, p.get("len", img.shape[0]*0.15), d.get("color", (255, 255, 255)))
            meta["cat"] = d["cat"]
            return img, meta
        # --- 局部处理类 ---
        if effect in ("mosaic", "blur_face", "blur"):
            name = {"mosaic": "mosaic", "blur_face": "blur", "blur": "blur"}[effect]
            img = local_effect(img, box, name, {"cell": p.get("cell", 18), "feather": p.get("feather", 12), "r": p.get("r", 12)})
            meta["cat"] = "局部"
            return img, meta
        # --- 景深虚化 (depth_blur) ---
        if effect == "depth_blur":
            intensity = float(p.get("intensity", p.get("frac", 0.45)))
            blurred = cv2.GaussianBlur(img, (0, 0), 9 + 12 * intensity)
            H, W = img.shape[:2]
            bw, bh = int(W * 0.7), int(H * 0.8)
            mask = np.zeros((H, W), np.float32)
            mask[(H - bh) // 2:(H - bh) // 2 + bh, (W - bw) // 2:(W - bw) // 2 + bw] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), W * 0.08)
            mask = mask / max(mask.max(), 1e-6)
            sharp_m = mask
            out = img.astype(np.float32) * sharp_m[..., None] + blurred.astype(np.float32) * (1 - sharp_m[..., None])
            img = np.clip(out, 0, 255).astype(img.dtype)
            meta["cat"] = "景深"
            return img, meta
        # --- 光影特效 ---
        if effect in ("vignette", "bokeh", "bloom", "god_rays", "film_grain", "tilt_shift", "warm", "cool", "light_leak"):
            img = make_effect(effect, img, p)
            meta["cat"] = "光影"
            return img, meta
        # --- 滤镜/风格 (全图) ---
        if effect in FILTERS:
            img = FILTERS[effect](img)
            meta["cat"] = "滤镜"
            return img, meta
        if effect in STYLES:
            img = STYLES[effect](img, p)
            meta["cat"] = "风格"
            return img, meta
        meta["warn"] = f"unknown effect {effect}"
        return img, meta


# 边框/相框
def _diag_watermark(img, text, font_size, opacity=0.25, color=(255, 255, 255)):
    """斜向平铺水印 (防盗用经典)。"""
    from PIL import ImageFont
    h, w = img.shape[:2]
    pil = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    try: font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", font_size)
    except Exception:
        try: font = ImageFont.truetype("arialbd.ttf", font_size)
        except Exception: font = ImageFont.load_default()
    bb = dr.textbbox((0, 0), text, font=font); tw, th = bb[2]-bb[0], bb[3]-bb[1]
    # 对角平铺
    step = font_size * 3
    for y0 in range(-h, h*2, step):
        for x0 in range(-w*2, w, int(font_size*9)):
            dr.text((x0, y0), text, font=font,
                    fill=tuple(color)+(int(255*opacity),))
    ov = ov.rotate(-22, expand=False)
    pil = Image.alpha_composite(pil, ov)
    return np.array(pil.convert("RGB"))
def _corner_marks(img, mark_len=60, color=(255, 255, 255)):
    """四角取景标记 (ins/相机/证件常用)."""
    pil = Image.fromarray(img).convert("RGB")
    dr = ImageDraw.Draw(pil, "RGBA")
    h, w = img.shape[:2]
    L = int(mark_len); t = max(6, int(w*0.008))
    c = tuple(color)+(255,)
    for (x, y, sx, sy) in [(0, 0, 1, 1), (w-L, 0, -1, 1), (0, h-L, 1, -1), (w-L, h-L, -1, -1)]:
        cx, cy = x if sx == 1 else x+L, y if sy == 1 else y+L
        # horizontal & vertical corner strokes
        dr.line([cx, cy, cx + sx*L, cy], fill=c, width=t)
        dr.line([cx, cy, cx, cy + sy*L], fill=c, width=t)
        dr.line([cx + sx*L*0.5, cy, cx, cy], fill=c, width=t) if False else None
    return np.array(pil.convert("RGB"))
def _border_effect(img, box, p):
    w = int(p.get("width", max(6, int(img.shape[1]*0.02))))
    color = tuple(p.get("color", (255, 255, 255)))
    return cv2.copyMakeBorder(img, w, w, w, w, cv2.BORDER_CONSTANT, value=color)
def _frame_polaroid(img, box):
    # 白边 + 底部留白(拍立得)
    x, y, w, h = box
    pad = max(12, int(w*0.06)); bottom = int(w*0.22)
    return cv2.copyMakeBorder(img, pad, bottom, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
def _lens_overlay(img, box):
    # 复古镜头圈 + 高光
    x, y, w, h = box
    cx, cy = x+w//2, y+h//2
    r = min(w, h)//2
    out = img.copy()
    cv2.circle(out, (cx, cy), r, (200, 200, 200), 4)
    cv2.circle(out, (cx, cy), int(r*0.9), (230, 230, 230), 2)
    return out
def _round_avatar(img, box, opacity=0.9):
    # 圆形头像角标(模拟美颜/相机预览气泡): 放一张抠出圆形区域, 若缺用原图圆形+描边
    x, y, w, h = box
    r = min(w, h)//2
    cx, cy = x + w//2, y + h//2
    out = img.copy()
    # 画空心圆形描边 + 内圈, 简单头像占位
    cv2.circle(out, (cx, cy), r, (255, 255, 255), 3)
    return out


# 装饰(自动素材可放) 目录 —— 部分需要 assets/*.png (可先用内置绘制 fallback)
DECOR = {
    "heart":       {"cat": "贴纸", "kind": "sticker", "file": "heart", "opacity": 1.0},
    "star":        {"cat": "贴纸", "kind": "sticker", "file": "star"},
    "crown":       {"cat": "贴纸", "kind": "sticker", "file": "crown"},
    "bubble":      {"cat": "贴纸", "kind": "sticker", "file": "bubble"},
    "emoji_smile": {"cat": "贴纸", "kind": "sticker", "file": "emoji_smile"},
    "cat":         {"cat": "贴纸", "kind": "sticker", "file": "cat"},
    "flower":      {"cat": "贴纸", "kind": "sticker", "file": "flower"},
    "music":       {"cat": "贴纸", "kind": "sticker", "file": "music"},
    "rainbow":     {"cat": "贴纸", "kind": "sticker", "file": "rainbow", "opacity": 0.85},
    "watermark_text": {"cat": "水印", "kind": "text", "text": "© 演播室合成", "opacity": 0.45, "color": (255, 255, 255)},
    "brand_logo":  {"cat": "水印", "kind": "sticker", "file": "logo", "opacity": 0.8},
    "date_stamp":  {"cat": "水印", "kind": "text", "text": "2026.09.09", "opacity": 0.5, "color": (255, 255, 255)},
    "qr_code":     {"cat": "水印", "kind": "sticker", "file": "qr", "opacity": 0.9},
    "polaroid_frame": {"cat": "边框", "kind": "frame"},
    "film_border": {"cat": "边框", "kind": "border", "color": (20, 20, 20)},
    "photo_white": {"cat": "边框", "kind": "border", "color": (255, 255, 255)},
    "circle_avatar": {"cat": "道具", "kind": "avatar"},
    "rect_badge":  {"cat": "道具", "kind": "rect", "color": (30, 30, 30), "opacity": 0.75, "radius": 14},
    "camera_lens": {"cat": "道具", "kind": "lens"},
    # ---- v2 扩充贴纸/水印/边框 ----
    "balloon": {"cat": "贴纸", "kind": "sticker", "file": "balloon"},
    "moon": {"cat": "贴纸", "kind": "sticker", "file": "moon"},
    "cloud": {"cat": "贴纸", "kind": "sticker", "file": "cloud", "opacity": 0.9},
    "lightning": {"cat": "贴纸", "kind": "sticker", "file": "lightning"},
    "camera_icon": {"cat": "道具", "kind": "sticker", "file": "camera"},
    "gift": {"cat": "贴纸", "kind": "sticker", "file": "gift"},
    "snowman": {"cat": "贴纸", "kind": "sticker", "file": "snowman"},
    "sun": {"cat": "贴纸", "kind": "sticker", "file": "sun"},
    "fish": {"cat": "贴纸", "kind": "sticker", "file": "fish"},
    "icecream": {"cat": "贴纸", "kind": "sticker", "file": "icecream"},
    "seal_red": {"cat": "水印", "kind": "text", "text": "原创", "opacity": 0.8, "color": (220, 40, 40)},
    "diag_watermark": {"cat": "水印", "kind": "diag", "text": "WATERMARK", "opacity": 0.25},
    "year_stamp": {"cat": "水印", "kind": "text", "text": "2026", "opacity": 0.5, "color": (255, 255, 255)},
    "frame_double": {"cat": "边框", "kind": "border", "color": (255, 255, 255)},
    "corner_marks": {"cat": "边框", "kind": "corners"},
}

# 自动布局默认位置
AUTO_PRESET = {
    "heart": ("corner", "tr"), "star": ("corner", "tr"), "crown": ("corner", "tl"),
    "bubble": ("corner", "br"), "emoji_smile": ("corner", "tr"), "cat": ("center", "center"),
    "flower": ("corner", "bl"), "music": ("corner", "br"),
    "rainbow": ("corner", "tr"), "watermark_text": ("corner", "br"), "brand_logo": ("corner", "tr"),
    "date_stamp": ("corner", "bl"), "qr_code": ("corner", "tr"),
    "polaroid_frame": ("center", "center"), "film_border": ("full", "full"),
    "photo_white": ("full", "full"), "circle_avatar": ("corner", "br"),
    "rect_badge": ("corner", "bl"), "camera_lens": ("corner", "tr"),
    # v2 贴纸
    "balloon": ("corner", "tr"), "moon": ("corner", "tr"), "cloud": ("corner", "br"),
    "lightning": ("corner", "tr"), "camera_icon": ("corner", "tr"), "gift": ("center", "center"),
    "snowman": ("corner", "bl"), "sun": ("corner", "tr"), "fish": ("corner", "bl"),
    "icecream": ("corner", "br"), "seal_red": ("corner", "bl"), "diag_watermark": ("full", "full"),
    "year_stamp": ("corner", "br"), "frame_double": ("full", "full"), "corner_marks": ("full", "full"),
    "mosaic": ("corner", "br"), "blur_face": ("center", "center"),
    "blur": ("center", "center"),
    "vignette": ("full", "full"), "bokeh": ("full", "full"), "bloom": ("full", "full"),
    "god_rays": ("full", "full"), "film_grain": ("full", "full"), "tilt_shift": ("full", "full"),
    "light_leak": ("full", "full"), "warm": ("full", "full"), "cool": ("full", "full"),
}
for _k in FILTERS: AUTO_PRESET.setdefault(_k, ("full", "full"))
for _k in STYLES: AUTO_PRESET.setdefault(_k, ("full", "full"))

# 兼容: FILTERS/STYLES 也需要在_auto里
AUTO_EFFECTS = sorted(set(AUTO_DEF_FRAC) | set(AUTO_PRESET))
