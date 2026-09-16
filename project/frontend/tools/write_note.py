# -*- coding: utf-8 -*-
import os

D = r"G:\myself\作业\生产实习\.workbuddy\memory"
os.makedirs(D, exist_ok=True)
p = os.path.join(D, "2026-09-11.md")

note = """
## ImageCompose 首页 · 纵向叙事重构设计稿（Ardot）

- 交付物：Ardot 设计稿 `https://ardot.tencent.com/file/724742044579898`（1440 宽、14 屏）
- 结构：Hero + 01 UNDERSTAND / 02 EXTRACT / 03 COMPOSE / 04 ILLUMINATE / 05 GROUND / 06 HARMONIZE / 07 CRITIC / 08 FINAL / 09 CREATE / 10 UPLOAD / 11 YOUR TURN + Footer + 附录（动效与固定元素规格）
- 核心结构决策：
  - 取消「每章一横排缩略图」的作品集式布局；改为中央 ImageStage 固定、同一张图连续 morph。
  - 左栏统一为「序号 / EN 衬线标题 / 中文标题 / 一句说明 / MOTION 规格盒」，全页重复同一骨架（编辑器式节奏）。
  - 三处固定元素：sticky 顶栏、右侧章节索引、视口底部完成度条（表示 Agent 完成度，不是播放条）。
- 设计令牌：底色 #F3F2EE、雾蓝 #E8EFF2、柔蓝 #C4D7DE、主文字 #20272B、辅助 #7D8588、Agent #7EA7B4
- 字体：Instrument Serif（英文标题）+ Inter（UI）+ Noto Sans SC（中文）+ Geist Mono（数字/规格标注）
- 视觉资产：本地合成，未使用生成式模型。matting 得到透明主体；PIL 合成 stage_compose / stage_ground / stage_final（左侧暖光、接触阴影、水面倒影、颗粒）。脚本：`imagecompose-site/tools/compose_stage.py`
- 导出：`imagecompose-site/ImageCompose-首页纵向叙事重构-设计稿.pdf`

## Ardot 踩坑（重要，下次直接复用）

- **lineHeight 单位是 PIXELS 不是倍数**：写 1.85 会被当成 1.85px，多行文字被压扁成一条线。必须填 `字号 × 倍数` 的 px 值（13.5px 正文配 25）。
- `cornerRadius` 不接受数组，不支持单角圆角（气泡尾巴只能做成统一圆角）。
- `layout:"none"` 的父节点里，子节点不能用 `width:"fill_container"`，必须给具体数值，否则会回退成固定尺寸并报 potentialIssue。
- 图片资源上传链路：`register_assets` 取 uploadUrl/downloadUrl → 用 python requests PUT（**必须 `trust_env=False` 绕过系统代理**，并带重试）→ `upload_images` 传 downloadUrl。
  偶发 `SignatureDoesNotMatch`，重新 register 一次即可。
- 抠图/去水印走内置 `buddy-image-processing` 脚本（`--operation matting|erase`），需要先 `connect_cloud_service` 取 token。
"""

with open(p, "a", encoding="utf-8") as f:
    f.write(note)
print("appended ->", p, os.path.getsize(p), "bytes")
