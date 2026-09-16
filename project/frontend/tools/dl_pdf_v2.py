# -*- coding: utf-8 -*-
import os
import requests
from pathlib import Path

OUT = r"G:\myself\作业\生产实习\imagecompose-site\ImageCompose-首页v2-Editorial-Studio-设计稿.pdf"


def _safe_out(base: str, name: str = "") -> Path:
    """输出路径校验：解析后必须仍位于基准目录内（防路径穿越）。"""
    base_p = Path(base).resolve()
    p = (base_p / name).resolve() if name else base_p
    if p != base_p and base_p not in p.parents:
        raise SystemExit(f"路径越界: {p}")
    return p


url = ("https://ai.d.gtimg.com/mcp/exports/724747602162280/"
       "ImageCompose%20%E9%A6%96%E9%A1%B5%20v2%20%C2%B7%20Editorial%20Studio-20260911_223004050.pdf"
       "?sign=f37f9aa0aab057e3569f0d4d1b7f9acb&t=1791729004")
s = requests.Session()
s.trust_env = False
r = s.get(url, timeout=300)
out = _safe_out(OUT)
out.write_bytes(r.content)
print(r.status_code, round(len(r.content) / 1024), "KB")

note = """
## ImageCompose 首页 v2（Editorial Studio）· 按 Shopify 式反馈重做

- 新画布：`https://ardot.tencent.com/file/724747602162280`（1440 宽、16 屏）
- 交付 PDF：`imagecompose-site/ImageCompose-首页v2-Editorial-Studio-设计稿.pdf`
- v1 保留在 `https://ardot.tencent.com/file/724742044579898`，可并排对照

### v2 相对 v1 的结构性改动
1. 拆开 ILLUMINATE → **04 LIGHT（环境光分析，安静）** + **05 RELIGHT（全页唯一视觉高潮，图占 70%）**
2. 新增 **08 ENHANCE（微距章节，三张局部特写）**，解决"最终像不像照片靠细节"的问题
3. **09 CRITIC 改为深色章节**（#1F2427），逐项检查 + `SHADOW 78 < 80 → reroll T04 回到 GROUND 重跑`
4. **10 FINAL 改为满宽静屏**（stage 1264×760），首屏去掉 eyebrow 与底部进度条（做减法）
5. **每章换视觉体裁**：图解 / 科学 / 电影 / 极简 / 视觉高潮 / 特写(976×470 横幅) / 慢 / 微距 / 深色技术 / 满屏静 / 交互
6. 06 SHADOW 换成 976×470 的脚部特写条 —— 用画幅比例本身换体裁
7. 11 CREATE 的 drop zone 去掉边框/阴影/玻璃，只留 `DROP AN IMAGE` 文字
8. Agent 字幕只在 01/03/05/10 出现，其余章节刻意静音

### 素材（全部来自真实照片，无生成式模型）
新增：`stage_relight.jpg`（左侧强暖光 + 主体左缘轮廓光）、`detail_feet.jpg`（脚部接触阴影特写）、
`macro_hair/water/cloud.jpg`（三张真实微距裁切）。
脚本：`imagecompose-site/tools/compose_stage_v2.py`

### 环境踩坑（重要）
- **COS 预签名 PUT 必须"一个文件一个进程"**：同一 python 进程内连续 PUT 多个文件会被 `AccessDenied`。
  可靠做法 = bash `for` 循环，每个文件调一次 `up_one.py`（独立进程）。见 `tools/up_one.py`。
- 预签名链接有效期约 16 分钟，注册后要尽快上传。
- 仍要 `requests.Session().trust_env = False` 绕过系统代理。
"""

with open(r"G:\myself\作业\生产实习\.workbuddy\memory\2026-09-11.md", "a", encoding="utf-8") as f:
    f.write(note)
print("memory appended")
