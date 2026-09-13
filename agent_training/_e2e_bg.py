# -*- coding: utf-8 -*-
"""用户报告的两个 bug 的端到端验证。

Bug1: 模型不用我提供的背景图替换绿幕 (而是去了素材库/文生图)
Bug2: 提示词里出现"城市"这类词 → 幻觉出训练过的图, 而不是用我喂的背景图

场景: 会话已有绿幕前景 → 用户上传城市背景图 + 指令 "绿幕图片用城市背景图替换"
"""
import json, urllib.request, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = "http://127.0.0.1:8765"
FG = str(ROOT / "data/ai_generated/green_fg/fg_01_anchor_male.png")
CITY = str(ROOT / "data/ai_generated/studio_bg/bg_03_panorama.png")


def post(path, body, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12] if p and Path(p).exists() else "N/A"


INSTR = "绿幕图片用城市背景图替换"
print("前景:", Path(FG).name, "| 用户背景:", Path(CITY).name)
print("指令:", INSTR)
print("=" * 72)

post("/api/agent/dag", {"reset": True})
# 上一轮: 绿幕前景进入会话
post("/api/agent/dag", {"instruction": "把这个绿幕人物作为当前待处理的对象",
                        "image": FG, "critic": False, "provider": "qwen"})
# 本轮: 上传城市背景 + 指令
t0 = time.time()
r = post("/api/agent/dag", {"instruction": INSTR, "image": CITY,
                            "critic": False, "provider": "qwen"})
print(f"plan_source={r.get('plan_source')}  {time.time()-t0:.1f}s")
for s in (r.get("steps") or []):
    print(f"  {s.get('node_id')} {str(s.get('tool')):26s} {str(s.get('status')):7s} {str(s.get('latency_ms')):>6s}ms {s.get('error','')}")
final = r.get("final")
print("成片:", final)

if not final:
    print("\n❌ 未出成片")
    raise SystemExit(1)

# 核心校验: 成片背景必须是用户上传的那张图
import cv2, numpy as np
def rd(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

comp, ref = rd(final), rd(CITY)
print(f"成片 {comp.shape[1]}x{comp.shape[0]} | 用户背景 {ref.shape[1]}x{ref.shape[0]}")
rr = cv2.resize(ref, (comp.shape[1], comp.shape[0]))
# 头部/顶层条带 (人物不触及的极顶部) 逐像素比对
band = max(6, int(comp.shape[0] * 0.035))
diff_top = np.abs(comp[:band].astype(np.int16) - rr[:band].astype(np.int16)).mean()
print(f"\n[校验] 顶部 {band}px 条带平均像素差 = {diff_top:.2f}")
print("  < 3  → ✓ 成片背景就是用户上传的图" if diff_top < 3 else
      "  >= 3 → ✗ 背景不是用户那张图 (仍在使用素材库/文生图)")
cv2.imwrite(str(ROOT / "outputs/_verify_userbg_final.png"), np.hstack([comp, rr]))
print("对比图: outputs/_verify_userbg_final.png")
