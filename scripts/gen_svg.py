# -*- coding: utf-8 -*-
"""生成学术风格八层架构SVG流程图并回填到HTML中（坐标精确计算，验证不越界）"""
import re

layers = [
    ("L0", "需求形式化层", "任务规约 · 约束集 C · 效用函数 U", "边界条件 / 系统拒识触发", False),
    ("L1", "状态空间建模层", "S={任务,环境,代理} · P(s′|s,a)", "终止条件 · MDP / POMDP 可观测", False),
    ("L2", "感知与表征层", "多模态编码 · 语义解析(AST/Schema)", "置信度校准 · OOD 拒识", False),
    ("L3", "认知架构层", "规划器 · 执行器 · 工作/长期记忆", "元认知 / 自我监控与纠错", False),
    ("L4", "工具集成层", "FC / MCP / OpenAPI · 能力注册", "DAG 编排 · 熔断/重试/降级", False),
    ("L5", "决策与控制层", "策略 π(a|s) · MCTS / 束搜索", "Critic 反思循环 · HITL", False),
    ("L6", "验证与对齐层", "形式化验证 · 经验差分检测", "价值对齐 · 红队测试", False),
    ("L7", "接口与交互层", "渐进式披露 · 轨迹可视化", "纠正反馈 · 上下文同步", False),
    ("L8", "评估与进化层", "离线/在线评估 · 持续学习", "NAS / HPO 系统进化", True),
]

W, H = 620, 838
boxX, boxW = 170, 290
boxH, pitch, startY = 60, 79, 62
barH = 22
cx = boxX + boxW / 2
dataX, fbX = 74, 560
out = []
S = out.append

S(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="\'Segoe UI\',\'Microsoft YaHei\',sans-serif>')
S('''  <defs>
    <marker id="arrB" markerWidth="9" markerHeight="9" refX="5" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#2c5f8a"/></marker>
    <marker id="arrO" markerWidth="9" markerHeight="9" refX="5" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#b4582b"/></marker>
    <linearGradient id="gL" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#12325c"/><stop offset="100%" stop-color="#2c5f8a"/></linearGradient>
  </defs>''')

topY = startY - 16
botY = startY + len(layers) * pitch - pitch + boxH + 6
# 左右贯穿轴
S(f'<line x1="{dataX}" y1="{topY}" x2="{dataX}" y2="{botY}" stroke="#c6cfda" stroke-width="2"/>')
S(f'<text x="{dataX}" y="{topY-8}" text-anchor="middle" font-size="10" fill="#8a94a3">数据流</text>')
S(f'<line x1="{fbX}" y1="{botY}" x2="{fbX}" y2="{topY}" stroke="#b4582b" stroke-width="2" opacity="0.65" stroke-dasharray="5,4" marker-end="url(#arrO)"/>')
S(f'<text x="{fbX}" y="{botY+22}" text-anchor="middle" font-size="10" fill="#b4582b">← L8 反馈 / 进化信号</text>')
# 顶部输入
S(f'<rect x="{cx-125}" y="18" width="250" height="30" rx="15" fill="#eef3f8" stroke="#12325c" stroke-width="1.2"/>')
S(f'<text x="{cx}" y="37" text-anchor="middle" font-size="12" fill="#12325c" font-weight="600">多模态原始输入</text>')

tops = []
for i, (tag, name, l1, l2, final) in enumerate(layers):
    y = startY + i * pitch
    tops.append(y)
    fill = "#b4582b" if final else "url(#gL)"
    stroke = "#b4582b" if final else "#bcd0e4"
    linecol = "#b4582b" if final else "#2c5f8a"
    S(f'<rect x="{boxX}" y="{y}" width="{boxW}" height="{boxH}" rx="4" fill="#fff" stroke="{stroke}" stroke-width="{1.3 if final else 1.0}"/>')
    S(f'<rect x="{boxX}" y="{y}" width="{boxW}" height="{barH}" rx="4" fill="{fill}"/>')
    S(f'<line x1="{boxX}" y1="{y+barH}" x2="{boxX+boxW}" y2="{y+barH}" stroke="{linecol}" stroke-width="1"/>')
    S(f'<text x="{boxX+12}" y="{y+15}" font-size="11.5" fill="#fff" font-weight="600">{tag}\u3000{name}</text>')
    S(f'<text x="{boxX+12}" y="{y+barH+20}" font-size="9.6" fill="#2b3646">{l1}</text>')
    S(f'<text x="{boxX+12}" y="{y+barH+38}" font-size="9.6" fill="#2b3646">{l2}</text>')
# 层间数据流箭头
for i in range(len(layers) - 1):
    y1 = tops[i] + boxH
    y2 = tops[i + 1] - 2
    S(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#2c5f8a" stroke-width="2" marker-end="url(#arrB)"/>')
# 底部输出
oy = botY + 20
S(f'<rect x="{cx-130}" y="{oy}" width="260" height="30" rx="15" fill="#f4efe8" stroke="#b4582b" stroke-width="1.2"/>')
S(f'<text x="{cx}" y="{oy+19}" text-anchor="middle" font-size="12" fill="#b4582b" font-weight="600">任务输出 · 决策 / 结果呈现</text>')
S('</svg>')
svg = "\n".join(out)

# 校验：所有元素是否越界 viewBox
recl = re.findall(r'y="(\d+(?:\.\d+)?)"', svg)
floaty = [float(v) for v in recl]
assert max(floaty) - 0 < H, f"SVG 越界 y: max={max(floaty)}, H={H}"
print(f"SVG OK, svg_length={len(svg)}, max_y={max(floaty):.0f} < {H}")

path = r"d:\AIcode\生产实习\docs\专用Agent系统架构与工程实现指南_论文化版.html"
html = open(path, encoding="utf-8").read()
old_start = html.find('<svg viewBox="0 0 600 860"')
old_end = html.find("</svg>") + len("</svg>")
assert old_start != -1, "旧的占位 SVG 未找到"
html = html[:old_start] + svg + html[old_end:]
open(path, "w", encoding="utf-8").write(html)
print("replaced svg into html ok")