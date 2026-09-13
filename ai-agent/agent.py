# -*- coding: utf-8 -*-
"""
agent.py — 演播室图像合成 Agent (文本驱动 / 多模态 Critic)
依据《精编版.pptx》四层架构之"Agent 智能体层 + 工具层"落地:

  用户文本指令 → [agnes-3.0-flash 规划器]
      │  输出 JSON 工具调用 (function calling)
      ▼
  [本地图像工具层] studio_cli / fx_library / agnes-image / video
      │  执行并返回结果文件
      ▼
  结果回填 → 规划器 → (可选 VLM Critic 自评最终图) → 直到完成

特性:
  - 会话状态: ctx['cur'] = 当前工作图, 多步自动串联(抠图→换背景→加特效)
  - 素材语义检索(背景/人像按中文挑选)
  - Critic: 生成完成后让 VLM 看图评分, 触发自动修正(最多 redo 次)
  - 两步出图: plan(dry-run 展示规划) / run(实际执行)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sp in [str(ROOT / "ai-agent"), str(ROOT), str(ROOT / "ai-service" / "src"),
           str(ROOT / "ai-service" / "src" / "fx")]:
    if sp not in sys.path:
        sys.path.insert(0, sp)

import tools
from agnes_client import AgnesClient, _img_data_url
import assets

SYSTEM_PROMPT = f"""你是"智能演播室图像合成 Agent"(Chroma)。你根据用户的一句话中文/英文指令, 自主规划并调用下面的图像/视频工具, 完成完整的合成任务流, 然后向用户简要说明结果。

可用资源背景(供 bg 参数做语义挑选):
{assets.list_backgrounds()}

可用绿幕人像(供 matting/composite 的 image 参数挑选或让用户指定):
{assets.list_persons()}

工具调用与状态规则:
1. 每次调用后, 系统会记住"当前工作图"(通常为最新成片)。若下个工具不给 image, 自动作用于当前图, 便于"抠图→换背景→加特效"多步串联。
2. 背景 bg 参数可填: 素材中文语义(访谈/新闻LED/全景/综艺/播客/天气) 或 素材序号(0~5) 或 文件路径 或 不填(默认访谈背景)。
3. 特效 effect 参数可填英文 key 或中文(黑白/青橙/赛博朋克/油画/漫画/暗角/聚光灯/爱心/水印/美颜/复古/霓虹...); 不确定时先调 list_effects 看全部可用特效再选。
4. 对"含绿幕的演播室照片"(既有人物又有桌台话筒), 换背景用 greenscreen(只换绿幕区域、保留实物); 对"纯绿幕人像"换背景用 composite(抠出人换整背景)。
5. 用最少的必要工具完成目标; 一次规划多个独立工具时尽量合并, 避免冗余重做。
6. 设计类生成: 可用 gen_subtitle_bar(字幕条), gen_title_card(片头标题), gen_ppt_cover(PPT封面), gen_storyboard(分镜), add_cover_text(给图加字), replace_text(改字)。
7. 区域特效用 region_fx (只作用于主体/指定区域, 比 apply_fx 更精准): 聚光灯 spotlight_on / 主体遮罩 spotlight_off / 局部滤镜 region_filter / 区域调色 region_color / 背景虚化 bg_blur / 主体辉光 region_glow / 边缘描边 edge_highlight。region 默认 auto 自动抠主体。
8. 若要"只改天空/只虚化地面/只调整树木"这类语义级操作, 用 semantic_fx (group=天空/建筑/树木植物/地面/水/山石/雪/人物/车辆...); 不确定画面构成时先调 analyze_scene 看语义占比。仅 COCO val2017 图支持。

工作准则(依据PPT: 规划器只出计划不生成像素、工具层才执行):
- 先想清楚实现用户意图需要哪几步(如: 选人物→选背景→合成→(可选)加特效/滤镜), 再逐个调用。
- 关键产物图路径会由工具返回; 最终向用户用 1~2 句话汇报成片路径与做了什么。
"""


class StudioAgent:
    def __init__(self, client: AgnesClient | None = None,
                 max_iter: int = 12, enable_critic: bool = True, redo: int = 1):
        self.client = client or AgnesClient()
        self.max_iter = max_iter
        self.enable_critic = enable_critic
        self.redo = redo
        self.ctx: dict = {}
        self.history: list[dict] = []

    # ---------- 规划(dry run) ----------
    def plan(self, instruction: str) -> list[dict]:
        """让 LLM 先输出一段文字计划(不执行), 返回步骤描述。"""
        msg = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content":
                f"用户指令: {instruction}\n\n请只用文字输出你的执行计划(打算调用哪几个工具、每步做什么、背景/特效怎么选), 不要实际调用工具。"}]
        d = self.client.chat(msg, max_tokens=500, temperature=0.3)
        return d["choices"][0]["message"].get("content") or "(无计划)"

    # ---------- 执行 ----------
    def run(self, instruction: str, tool_history: list | None = None) -> dict:
        """执行: function-calling 循环直到无工具调用或达到上限。"""
        tools_schema = tools.build_tool_schemas()
        # 会话历史重建: 首次
        if not self.history:
            self.history = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": instruction}]
        else:
            self.history.append({"role": "user", "content": instruction})

        trace = []
        it = 0
        while it < self.max_iter:
            it += 1
            d = self.client.chat(self.history, tools=tools_schema, max_tokens=1200)
            msg = d["choices"][0]["message"]
            self.history.append(msg)          # assistant msg (可能含 tool_calls)

            calls = msg.get("tool_calls") or []
            if not calls:
                break                          # 规划器不再要工具, 结束

            for call in calls:
                fn = call["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                t0 = time.time()
                result = tools.call_tool(name, args, self.ctx)
                dt = round(time.time() - t0, 2)
                trace.append({"tool": name, "args": args, "sec": dt,
                              "result": _brief(result)})
                # 回填 tool 消息
                tool_body = json.dumps(result, ensure_ascii=False, default=str)[:1500]
                self.history.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": tool_body})
                print(f"  [step{it}] {name}{args} -> {dt}s", flush=True)

        final_text = self._last_text()
        return {"final_text": final_text, "trace": trace,
                "cur": self.ctx.get("cur"), "n_iter": it,
                "steps": len(trace)}

    # ---------- VLM Critic 自评与修正 ----------
    def critic(self, image_path: str | None = None) -> dict:
        """把当前/指定成片发给 VLM, 让其按 5 维评分, 若不合格给出修正建议。"""
        img = image_path or self.ctx.get("cur")
        if not img or not Path(img).exists():
            return {"critic_ok": False, "error": "无成片可评"}
        prompt = ("请审视这张演播室合成图, 从5维打分(每维0-100, 一句话): "
                  "Lighting光照一致性 / Shadow接触阴影 / Color色彩和谐 / Edge边缘发丝 / Overall自然度。"
                  "若 Overall < 80, 用一句话指出最该改的1点, 并说明应调用哪个工具(如 apply_fx/composite/regreenscreen)。")
        d = self.client.chat_with_image(prompt, [img], max_tokens=300)
        text = d["choices"][0]["message"].get("content") or ""
        return {"critic_ok": True, "image": img, "review": text}

    # ---------- 工具 ----------
    def _last_text(self) -> str:
        for m in reversed(self.history):
            if m.get("role") == "assistant" and m.get("content"):
                return m["content"]
        return ""

    def reset(self):
        self.ctx = {}
        self.history = []

    def close(self):
        pass


def _brief(result: dict, limit: int = 220) -> str:
    s = json.dumps(result, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


# ---------- CLI 入口 ----------
def main():
    agent = StudioAgent()
    print("=" * 60)
    print("演播室图像合成 Agent (agnes-3.0-flash) | 输入中文指令, 'exit'退出, 'reset'清空")
    print("例: '把绿幕女主播 fg_02 放到访谈背景并加青橙滤镜' | '给当前图加爱心和文字水印'")
    print("=" * 60)
    while True:
        try:
            instr = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not instr:
            continue
        if instr.lower() in ("exit", "quit", "退出"):
            break
        if instr.lower() == "reset":
            agent.reset(); print("已清空会话"); continue
        print("[计划] ", agent.plan(instr), "\n")
        t0 = time.time()
        r = agent.run(instr)
        print(f"\n[完成] 用时{time.time()-t0:.1f}s, 调用{len(r['trace'])}个工具")
        print(f"[成片] {r['cur']}")
        print(f"[总结] {r['final_text']}")
        if agent.enable_critic and r["cur"]:
            c = agent.critic()
            print(f"[Critic] {c.get('review', '')[:300]}")


if __name__ == "__main__":
    main()
