# -*- coding: utf-8 -*-
"""gen_schemas.py — 依据《A_B接口对接说明_V1.0》§2/§3 生成 agent/schema/T01–T08.json
统一信封: input / output / error{code,message,retryable} / latency_ms / model_version
质量档位: quality ∈ {draft, normal, fine} → B 组内部档位映射
运行: python agent/schema/_gen.py  (幂等, 重跑覆盖)
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent

ENV = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "additionalProperties": False,
}

def err(code, msg, retryable):
    return {"code": code, "message": msg, "retryable": retryable}

SCHEMAS = {
"T01_matting": {
  "id": "T01", "name": "matting", "version": "1.1",
  "b_mapping": "ai-service/src/matting/matting_backend.build_matting_tool().predict(); interactive_matting.InteractiveMatting.matte(box/points); refiner=agent/adapter.refiner_weight() 统一注册位 (env REFINER_WEIGHT > training/checkpoints/active_refiner.txt > refiner_studio), 当前 T01 fine 档权重由 active_refiner.txt 决定",
  "desc": "抠图: 分离前景与背景, 输出 alpha 与白底预乘 fg (img*a+255*(1-a))。keep=subject 默认(保留主体/人物); keep=background 时反转 alpha(删除主体, 保留背景), 用于'扣去/去掉图中人物'类指令",
  "quality_tiers": {"draft": "seg512, refine=false", "normal": "seg512, refine=false",
                    "fine": "seg512, refine=true (AlphaRefiner 精修)"},
  "input": {**ENV, "type": "object",
    "properties": {
      "image": {"type": "string", "description": "输入图路径 (项目相对或绝对)"},
      "mode": {"type": "string", "enum": ["auto", "interactive"], "description": "auto=整图自动; interactive=需 box/points/polygon"},
      "engine": {"type": "string", "enum": ["humanmatting", "humanmatting_fast"],
                 "description": "抠图引擎 (可选, 默认 BiRefNet)。humanmatting=MODNet-HRNetW18 语义人像抠图(精度, ~0.3s); humanmatting_fast=MODNet-MobileNetV2(更快, ~0.15s)。适合非绿幕通用人像; 用户点名 MODNet/humanmatting/人像语义抠图时使用"},
      "keep": {"type": "string", "enum": ["subject", "background"], "default": "subject",
               "description": "保留哪一侧: subject=保留主体/人物(默认, 抠出人物); background=删除主体只留背景(扣去人物/去掉人物/移除人物)"},
      "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
      "points": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
      "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"}},
    "required": ["image"]},
  "output": {"type": "object",
    "properties": {"alpha_path": {"type": "string"}, "fg_path": {"type": "string"},
                   "fg_ratio": {"type": "number"}}, "required": ["alpha_path", "fg_path"]},
  "errors": [err("E_INPUT_MISSING", "输入图不存在", False),
             err("E_MATTING_FAIL", "抠图推理失败(显存/权重)", True),
             err("E_MODE_CONFLICT", "interactive 模式缺 box/points", False)],
  "latency_budget_ms": {"draft": 1500, "fine": 3500},
},
"T02_background_generate": {
  "id": "T02", "name": "background_generate", "version": "1.1",
  "b_mapping": "A 组负责: 素材库 ai-agent/assets.py (semantic 匹配, 0s) 或 agnes-image 文生图 (fine 档); 色度键直合成见 skills/green-screen-composite/scripts/green_key.py",
  "desc": "背景生成/检索: 按 semantic 关键词检索素材库; fine 档或无匹配时文生图。mode=green_key 时直接把粗抠前景色度键合成到背景(适合绿幕边缘残留/需精确位置摆放的场景), 输出即为合成成片",
  "quality_tiers": {"draft": "assets 语义检索 (0s)", "normal": "assets 语义检索",
                    "fine": "agnes-image 文生图 (10~20s)"},
  "input": {**ENV, "type": "object",
    "properties": {
      "file": {"type": "string", "description": "直接指定背景文件路径 (用户上传的背景图), 优先级最高"},
      "semantic": {"type": "string", "description": "中文语义关键词: 访谈/新闻LED/全景/综艺/播客/天气 或 0~5 序号"},
      "prompt": {"type": "string", "description": "文生图提示词 (fine 档或素材库无匹配时)"},
      "size": {"type": "string", "default": "1024x1024"},
      "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"},
      "mode": {"type": "string", "enum": ["semantic", "green_key"], "default": "semantic",
               "description": "green_key=色度键直接合成(app_fg 必填): 用 HSV 绿幕键把 app_fg 抠净后叠到背景, 输出 composite_path"},
      "green_scope": {"type": "boolean", "default": true,
               "description": "mode=green_key: true=只在绿幕区域填背景, 场景画布保留(演播室桌台/LED屏不动, 画布=原图尺寸); false=传统模式把前景抠出摆到背景上(可用 fg_scale/center)。未指定时自动: 有 fg_scale/center → false, 否则 true"},
      "app_fg": {"type": "string", "description": "mode=green_key 时的绿幕前景图 (人物原图, 非抠好的 fg)"},
      "fg_scale": {"type": "number", "description": "mode=green_key: 前景相对背景的缩放比例, 如 0.55"},
      "center": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                 "description": "mode=green_key: 前景放置位置 [x,y] (左上角像素坐标), 默认居中"},
      "media_filter": {"type": "string",
                       "description": "色卡滤镜文件名 (如 '5小纸条.png'), 对合成结果叠加 LUT 色调; 也支持 'auto' 走素材库滤镜"},
      "media_type": {"type": "string", "enum": ["image", "video"], "default": "image",
                     "description": "green_key 合成媒体类型; video 时输出 mp4 (需 app_fg 为视频)"}},
    "anyOf": [{"required": ["file"]}, {"required": ["semantic"]}, {"required": ["prompt"]},
              {"required": ["app_fg"]}]},
  "output": {"type": "object",
    "properties": {"bg_path": {"type": "string"}, "source": {"type": "string", "enum": ["assets", "t2i", "green_key"]},
                   "composite_path": {"type": "string", "description": "mode=green_key 时的合成成片路径"}},
    "required": ["bg_path", "source"]},
  "errors": [err("E_SEMANTIC_MISS", "素材库无匹配且未给 prompt (可降级 t2i)", True),
             err("E_T2I_FAIL", "文生图失败(网络/限流)", True),
             err("E_GREENKEY_FAIL", "色度键合成失败(前景读取/滤镜缺失)", True)],
  "latency_budget_ms": {"draft": 50, "fine": 20000},
},
"T03_lighting_estimate": {
  "id": "T03", "name": "lighting_estimate", "version": "0.9",
  "b_mapping": "ai-service/src/lighting/lighting_estimate.py.estimate() (B 组新补独立工具)",
  "desc": "光照估计: 从背景估计主光方向(θ,φ 与 2D dx,dy)、色温、强度与 SH9 系数 (Lambert 拟合近似)",
  "quality_tiers": {"draft": "128px 快速估计", "normal": "同 draft", "fine": "256px + 平滑"},
  "input": {**ENV, "type": "object",
    "properties": {"bg_path": {"type": "string"},
                   "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"}},
    "required": ["bg_path"]},
  "output": {"type": "object",
    "properties": {"light_dir": {"type": "array", "items": {"type": "number"}, "description": "2D [dx,dy] 归一化, 供 T04/T05 直接使用"},
                   "theta_phi": {"type": "array", "items": {"type": "number"}, "description": "[θ方位角deg, φ仰角deg]"},
                   "color_temp": {"type": "string", "enum": ["warm", "cool", "neutral"]},
                   "intensity": {"type": "number"},
                   "sh_coeff": {"type": "array", "items": {"type": "number"}, "minItems": 9, "maxItems": 9}},
    "required": ["light_dir", "theta_phi", "color_temp", "intensity", "sh_coeff"]},
  "errors": [err("E_INPUT_MISSING", "背景图不存在", False)],
  "latency_budget_ms": {"draft": 100, "fine": 250},
  "note": "SH9 由背景亮度半球 Lambert 近似投影; θ/φ 由梯度方向换算, 精度有限(降级可用, 版本 v0.9)",
},
"T04_relight": {
  "id": "T04", "name": "relight", "version": "1.0",
  "b_mapping": "ai-service/src/relighting/relight_tool.RelightTool(method=directional).relight(fg,bg,out,light_hint)",
  "desc": "前景重打光: 方向光渐变+色温; light_dir 可由 T03 输出传入 (规范 light_dir ↔ 实机 direction 已在适配层映射)",
  "quality_tiers": {"draft": "directional 强度0.6", "normal": "directional 强度0.8", "fine": "directional 强度1.0"},
  "input": {**ENV, "type": "object",
    "properties": {
      "fg_path": {"type": "string"}, "bg_path": {"type": "string"},
      "light_dir": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                    "description": "规范参数; 适配层映射为 B 的 direction:[dx,dy]"},
      "color_temp": {"type": "string", "enum": ["warm", "cool", "neutral"]},
      "color_temp_strength": {"type": "number"}, "intensity": {"type": "number"},
      "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"}},
    "required": ["fg_path", "bg_path"]},
  "output": {"type": "object",
    "properties": {"relit_fg_path": {"type": "string"}, "direction": {"type": "array", "items": {"type": "number"}}},
    "required": ["relit_fg_path"]},
  "errors": [err("E_INPUT_MISSING", "fg/bg 不存在", False),
             err("E_RELIGHT_FAIL", "重打光失败", True)],
  "latency_budget_ms": {"draft": 400, "fine": 600},
},
"T05_shadow_generate": {
  "id": "T05", "name": "shadow_generate", "version": "1.0",
  "b_mapping": "ai-service/src/shadow/shadow_tool.ShadowTool().generate(alpha,bg,out,shadow_hint)",
  "desc": "接触阴影: 程序化软阴影 (alpha 高斯模糊+偏移+衰减) 合入背景",
  "quality_tiers": {"draft": "blur8/opacity0.4", "normal": "blur12/opacity0.5", "fine": "blur16/opacity0.6"},
  "input": {**ENV, "type": "object",
    "properties": {
      "alpha_path": {"type": "string"}, "bg_path": {"type": "string"},
      "light_dir": {"type": "array", "items": {"type": "number"}, "description": "规范参数 → 适配层映射为 shadow_hint.direction"},
      "offset": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
      "blur_radius": {"type": "number"}, "opacity": {"type": "number"},
      "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"}},
    "required": ["alpha_path", "bg_path"]},
  "output": {"type": "object", "properties": {"shadow_bg_path": {"type": "string"}},
             "required": ["shadow_bg_path"]},
  "errors": [err("E_INPUT_MISSING", "alpha/bg 不存在", False)],
  "latency_budget_ms": {"draft": 150, "fine": 250},
},
"T06_harmonize": {
  "id": "T06", "name": "harmonize", "version": "2.0",
  "b_mapping": "ai-service/src/harmonization/harmonize_tool.HarmonizeTool(v2).harmonize; 适配层吸收规范入参: 内部先 alpha_over 合成(composite_tool)再对 composite 调 harmonize; mode=greenscreen 时走 chroma_key_scene 只换绿幕(保留桌台实物)",
  "desc": "和谐化+合成: v2=Lab 色度对齐+低频亮度迁移+FDR 防压黑; 输入为 fg+alpha (B 现行为), 适配层产出 composite",
  "quality_tiers": {"draft": "strength0.55 feather2", "normal": "strength0.75 feather2", "fine": "strength0.9 feather3"},
  "input": {**ENV, "type": "object",
    "properties": {
      "fg_path": {"type": "string"}, "alpha_path": {"type": "string"}, "bg_path": {"type": "string"},
      "mode": {"type": "string", "enum": ["alpha_over", "greenscreen"], "default": "alpha_over",
               "description": "greenscreen=虚拟演播室键控(只换绿幕区域, 保留非绿实物)"},
      "strength": {"type": "number"},
      "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"}},
    "required": ["fg_path", "alpha_path", "bg_path"]},
  "output": {"type": "object",
    "properties": {"composite_path": {"type": "string", "description": "合成图(和谐化后)"},
                   "harmonized_fg_path": {"type": "string"}, "fdr": {"type": "number"}, "fdr_ok": {"type": "boolean"}},
    "required": ["composite_path", "fdr", "fdr_ok"]},
  "errors": [err("E_INPUT_MISSING", "输入缺失", False),
             err("E_NO_GREENSCREEN", "mode=greenscreen 但图无绿幕(占比<3%)", False),
             err("E_FDR_FAIL", "FDR 越界(自动回退后仍压黑/过增强)", True)],
  "latency_budget_ms": {"draft": 500, "fine": 900},
},
"T07_enhance": {
  "id": "T07", "name": "enhance", "version": "0.5",
  "b_mapping": "fx_tool 兜底 (接口文档 §2: T07 用 fx 兜底): depth_blur 景深 + 可选 USM 锐化",
  "desc": "增强: 景深虚化/锐化/颗粒 (fx 兜底实现, version v0.5)",
  "quality_tiers": {"draft": "depth_blur 0.3", "normal": "depth_blur 0.45", "fine": "depth_blur 0.45 + 锐化"},
  "input": {**ENV, "type": "object",
    "properties": {"image_path": {"type": "string"},
                   "mode": {"type": "string", "enum": ["depth_blur", "sharpen", "grain"], "default": "depth_blur"},
                   "intensity": {"type": "number"},
                   "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "draft"}},
    "required": ["image_path"]},
  "output": {"type": "object", "properties": {"enhanced_path": {"type": "string"}},
             "required": ["enhanced_path"]},
  "errors": [err("E_INPUT_MISSING", "输入缺失", False), err("E_ENHANCE_FAIL", "增强失败", True)],
  "latency_budget_ms": {"draft": 200, "fine": 400},
},
"T08_export": {
  "id": "T08", "name": "export", "version": "1.0",
  "b_mapping": "A 组负责: 落盘 png + meta json (+可选 zip)",
  "desc": "导出: 最终成片 + 元数据(json) 归档, 可选 zip",
  "quality_tiers": {"draft": "png only", "normal": "png+meta", "fine": "png+meta+zip"},
  "input": {**ENV, "type": "object",
    "properties": {"final_path": {"type": "string"}, "meta": {"type": "object"},
                   "make_zip": {"type": "boolean", "default": False},
                   "quality": {"type": "string", "enum": ["draft", "normal", "fine"], "default": "normal"}},
    "required": ["final_path"]},
  "output": {"type": "object",
    "properties": {"exported_path": {"type": "string"}, "meta_path": {"type": "string"}, "zip_path": {"type": "string"}},
    "required": ["exported_path"]},
  "errors": [err("E_INPUT_MISSING", "成片不存在", False)],
  "latency_budget_ms": {"draft": 50, "fine": 300},
},
}

for name, s in SCHEMAS.items():
    p = D / f"{name}.json"
    p.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", p.name)

# 元 Schema: 校验 Planner 输出的 DAG
META = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExecutionPlanDAG",
  "type": "object",
  "additionalProperties": False,
  "required": ["intent", "nodes"],
  "properties": {
    "intent": {"type": "string"},
    "nodes": {"type": "array", "minItems": 1, "maxItems": 10,
      "items": {"type": "object", "additionalProperties": False,
        "required": ["id", "tool", "params"],
        "properties": {
          "id": {"type": "string", "pattern": "^n[0-9]+$"},
          "tool": {"type": "string", "enum": [f"T0{i}_{n}" for i, n in
                    [(1,"matting"),(2,"background_generate"),(3,"lighting_estimate"),
                     (4,"relight"),(5,"shadow_generate"),(6,"harmonize"),(7,"enhance"),(8,"export")]]},
          "params": {"type": "object"},
          "depends_on": {"type": "array", "items": {"type": "string", "pattern": "^n[0-9]+$"}}}}},
    "outputs": {"type": "array", "items": {"type": "string", "pattern": "^n[0-9]+$"}}
  }
}
(D / "_dag_meta.json").write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote _dag_meta.json")
