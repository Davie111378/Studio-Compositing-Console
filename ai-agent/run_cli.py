# -*- coding: utf-8 -*-
"""
run_cli.py — 文本交互 CLI 入口
用法:
  python ai-agent/run_cli.py                    # 启动 REPL
  python ai-agent/run_cli.py "一句中文指令"      # 单次执行并打印
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import StudioAgent


def single_shot(instruction: str, with_critic: bool = True):
    ag = StudioAgent(enable_critic=with_critic, max_iter=12)
    print(f"\n>>> 指令: {instruction}")
    print("\n[计划]")
    print(ag.plan(instruction))
    t0 = time.time()
    r = ag.run(instruction)
    print(f"\n[完成] {time.time()-t0:.1f}s, 调用 {len(r['trace'])} 个工具, "
          f"迭代 {r['n_iter']} 次")
    print(f"[成片] {r['cur']}")
    if r["final_text"]:
        print(f"[LLM 总结]\n{r['final_text']}")
    if with_critic and r["cur"]:
        c = ag.critic()
        if c.get("review"):
            print(f"\n[VLM Critic]\n{c['review']}")


def repl():
    ag = StudioAgent(enable_critic=True, max_iter=12)
    print("=" * 60)
    print("演播室图像合成 Agent  |  输入中文/英文指令")
    print("  - 输入一句指令: 自动 抠图/换背景/加特效/视频处理")
    print("  - 'reset': 清空会话 | 'exit' / 'quit': 退出")
    print("=" * 60)
    while True:
        try:
            raw = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "退出"):
            break
        if raw.lower() == "reset":
            ag.reset(); print("(已清空)"); continue
        # 显示计划
        try:
            print("\n[计划]")
            print(ag.plan(raw))
        except Exception as e:
            print(f"(计划失败: {e})")
        t0 = time.time()
        try:
            r = ag.run(raw)
        except Exception as e:
            print(f"(执行异常: {e})"); continue
        print(f"\n[完成] {time.time()-t0:.1f}s, {len(r['trace'])} 个工具")
        print(f"[成片] {r['cur']}")
        if r["final_text"]:
            print(f"[LLM]\n{r['final_text'][:600]}")
        # VLM Critic
        if r["cur"]:
            try:
                c = ag.critic()
                if c.get("review"):
                    print(f"\n[Critic]\n{c['review'][:500]}")
            except Exception as e:
                print(f"(Critic 异常: {e})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        single_shot(" ".join(sys.argv[1:]))
    else:
        repl()