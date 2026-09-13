# -*- coding: utf-8 -*-
"""
run.py — 端到端编排器 (冲刺计划 D4-D6 主链路)
一句话指令 → Planner(DAG) → Executor(真实 B 组工具) → Critic(VLM 5维) →
  不达标 → 定位最低维节点打补丁重跑(≤2 次) → 成片
多轮会话: AgentSession 保留 DAG/产物/版本树, 支持 "换回背景A保留光" 条件回滚。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dag as dagmod
import engine, planner as planner_mod, critic as critic_mod, rollback as rb_mod
from artifact import ArtifactTable

ROOT = Path(__file__).resolve().parent.parent


class AgentSession:
    """多轮会话: 共享产物表 + 版本树 (5 轮不崩 + 条件回滚)。"""

    def __init__(self, quality: str = "draft", enable_critic: bool = True,
                 critic_threshold: int = 75, verbose: bool = True,
                 provider: str | None = None):
        self.quality = quality
        self.enable_critic = enable_critic
        self.threshold = critic_threshold
        self.verbose = verbose
        self.provider = provider          # "agnes" | "qwen" | None(LLM_PROVIDER)
        self.tree = rb_mod.VersionTree()
        self.last_dag: dict | None = None
        self.last_run: dict | None = None
        self.cur_image: str | None = None
        self.history: list[str] = []          # cur_image 撤销栈 (面板直通 undo 用)
        self.orig_image: str | None = None    # 会话首张源图 (面板直通 reset 用)
        self._llm = None
        # B3 夜间扫描选优的 few-shot 自动生效
        self.fewshot_variant = "base"
        best_p = Path(__file__).resolve().parent / "night" / "out" / "best_fewshot.json"
        if best_p.exists():
            try:
                self.fewshot_variant = json.loads(
                    best_p.read_text(encoding="utf-8")).get("best", "base")
            except Exception:
                pass

    @property
    def llm(self):
        if self._llm is None:
            self._llm = planner_mod.AgnesLLM(provider=self.provider)
        return self._llm

    def set_provider(self, provider: str | None):
        """热切换后端: 换 provider 并让 LLM 客户端重建。"""
        self.provider = provider
        self._llm = None
        return self.provider

    def set_cur(self, path: str | None):
        """更新当前图: 旧图压入撤销栈; 首次写入记录为会话原图。"""
        if self.cur_image and self.cur_image != path:
            self.history.append(self.cur_image)
        if path and self.orig_image is None:
            self.orig_image = path
        self.cur_image = path
        return self.cur_image

    def _client(self):
        """按当前 provider 取 LLM 客户端 (供 critic 等独立调用方复用)。"""
        sys.path.insert(0, str(ROOT / "ai-agent"))
        from agnes_client import AgnesClient
        return AgnesClient(provider=self.provider)

    def _log(self, s: str):
        if self.verbose:
            print(s, flush=True)

    # ---------- 单轮 ----------
    def process(self, instruction: str, critic_max: int | None = None,
                bg_image: str | None = None, bg_from_upload: bool = False) -> dict:
        t0 = time.time()
        import constraints as cons
        import verifier
        critic_max = critic_max if critic_max is not None else cons.HARD["critic_max_replan"]
        # L0 拒识 (Abstention): OOD 指令不硬套工具
        why = cons.abstain_reason(instruction)
        if why:
            self._log(f"[abstain] {why}")
            return {"plan_source": "abstain", "dag": None, "run": None,
                    "critic": None, "final": None, "utility": None,
                    "abstain": why, "sec": round(time.time() - t0, 2), "vid": None}
        # 1) 规划 (多轮: 提示上次成片路径)
        dag, src = planner_mod.plan(instruction, cur_image=self.cur_image,
                                    llm=self.llm, verbose=self.verbose,
                                    fewshot_variant=getattr(self, "fewshot_variant", "base"),
                                    bg_image=bg_image, bg_from_upload=bg_from_upload)
        # L0 约束集 C: 计划级校验 (预算/规模/软约束) — 违规自动降档重试一次
        plan_errs = cons.check_plan(dag, self.quality)
        if plan_errs and self.quality != "draft":
            self._log(f"[plan] 约束违规: {plan_errs} → 降级 draft 重规划")
            old_q = self.quality
            self.quality = "draft"
            dag, src = planner_mod.plan(instruction, cur_image=self.cur_image,
                                        llm=self.llm, verbose=self.verbose,
                                        fewshot_variant=getattr(self, "fewshot_variant", "base"),
                                        bg_image=bg_image, bg_from_upload=bg_from_upload)
            plan_errs = cons.check_plan(dag, "draft")
            self.quality = old_q if not plan_errs else "draft"
        if plan_errs:
            self._log(f"[plan] 约束违规(兜底放行): {plan_errs}")
        self._log(f"[plan|{src}] {dagmod.summary(dag)}")
        # 2) 执行
        run = engine.execute(dag, quality_default=self.quality,
                             ctx={"cur_image": self.cur_image, "provider": self.provider})
        self._log(f"[exec] {run['status']} {run.get('total_ms', 0)}ms final={run.get('final_path')}")
        # 3) Critic 闭环 (≤critic_max 次)
        critic_out = None
        applied_patches: dict[str, dict] = {}   # node_id → 上次补丁 (防重复无效补丁)
        if self.enable_critic and run["final_path"] and run["status"] == "done":
            for i in range(critic_max):
                critic_out = self._critic_once(dag, run)
                self._log(f"[critic.{i}] {json.dumps(critic_out['scores'], ensure_ascii=False)}"
                          f" fix={critic_out.get('fix')}")
                if not critic_out.get("needs_replan") or not critic_out.get("replan"):
                    break
                patch_info = critic_out["replan"]
                node_id = self._find_node(dag, patch_info["tool"])
                if node_id is None:
                    self._log(f"[critic] 无 {patch_info['tool']} 节点可修, 跳过重跑")
                    break
                # 迭代增强: 补丁与上次相同 → 数值加大 (否则重跑无意义)
                patch = self._escalate_patch(dag, node_id, patch_info["patch"],
                                             applied_patches.get(node_id))
                applied_patches[node_id] = patch
                dag = rb_mod.swap_param(dag, node_id, patch)
                self._log(f"[replan] {node_id}({patch_info['tool']}) patch={patch}")
                run = engine.execute(dag, quality_default=self.quality,
                                     ctx={"cur_image": self.cur_image, "provider": self.provider})
                self._log(f"[re-exec] {run['status']} {run.get('total_ms', 0)}ms final={run.get('final_path')}")
                if run["status"] != "done":
                    break
        # 4) 版本树提交
        if run["status"] in ("done", "partial"):
            snap = run["artifacts"].snapshot()
            vid = self.tree.commit(dag, snap, note=instruction[:60])
            self.last_dag, self.last_run = dag, run
            if run["final_path"]:
                self.set_cur(run["final_path"])
        # 5) L6 成片终验 + R 效用函数 (多目标: 质量/延迟/成本)
        final_v = verifier.verify_final(run.get("final_path"), self.quality)
        if not final_v["ok"]:
            self._log(f"[verify-final] 违例: {final_v['violations']}")
        n_t2i = sum(1 for n in dag["nodes"]
                    if n["tool"] == "T02_background_generate"
                    and n.get("params", {}).get("quality") == "fine")
        overall = (critic_out or {}).get("scores", {}).get("Overall") if critic_out else None
        util = cons.utility(overall, run.get("total_ms", 0), self.quality, n_t2i=n_t2i)
        self._log(f"[utility] U={util['U']} (质量{util['quality_term']} / "
                  f"延迟{util['latency_term']} / 成本{util['cost_term']})")
        return {"plan_source": src, "dag": dag, "run": run, "critic": critic_out,
                "video": None, "sec": round(time.time() - t0, 2),
                "final": run.get("final_path"), "final_verify": final_v,
                "utility": util, "vid": self.tree.cur}

    # ---------- 条件回滚 (Demo04: 换回背景A保留光) ----------
    def rollback_background(self, new_bg_semantic: str, keep_light: bool = True) -> dict:
        if not self.last_dag:
            return {"error": "无可回滚的 DAG"}
        bg_node = next((n for n in self.last_dag["nodes"]
                        if n["tool"] == "T02_background_generate"), None)
        if bg_node is None:
            return {"error": "DAG 中无 T02 背景节点"}
        keep_tools = {"T03_lighting_estimate", "T04_relight"} if keep_light else set()
        new_dag, rerun_ids, keep_map = rb_mod.conditional_rerun(
            self.last_dag, self.last_run["artifacts"], bg_node["id"],
            {"semantic": new_bg_semantic}, keep_tools=keep_tools)
        self._log(f"[rollback] rerun={sorted(rerun_ids)} keep={sorted(keep_map)} (keep_light={keep_light})")
        run = engine.execute(new_dag, quality_default=self.quality,
                             ctx={"cur_image": None, "provider": self.provider}, keep_map=keep_map)
        self.last_dag, self.last_run = new_dag, run
        if run["final_path"]:
            self.set_cur(run["final_path"])
        self.tree.commit(new_dag, run["artifacts"].snapshot(),
                         note=f"rollback bg→{new_bg_semantic} keep_light={keep_light}")
        return {"status": run["status"], "final": run.get("final_path"),
                "rerun": sorted(rerun_ids), "kept": sorted(keep_map),
                "total_ms": run.get("total_ms", 0)}

    # ---------- 内部 ----------
    def _critic_once(self, dag: dict, run: dict) -> dict:
        try:
            return critic_mod.review_image(self._client(), run["final_path"])
        except Exception as e:
            return {"scores": {}, "needs_replan": False, "replan": None,
                    "error": f"critic 失败: {e}"}

    def _find_node(self, dag: dict, tool: str) -> str | None:
        for n in dag["nodes"]:
            if n["tool"] == tool:
                return n["id"]
        return None

    @staticmethod
    def _escalate_patch(dag: dict, node_id: str, patch: dict,
                        last_patch: dict | None) -> dict:
        """补丁迭代增强: 与上次补丁相同 → 数值参数加大, 避免无效重跑。"""
        cur_params = next((n.get("params", {}) for n in dag["nodes"] if n["id"] == node_id), {})
        out = {}
        for k, v in patch.items():
            if isinstance(v, (int, float)):
                cur = float(cur_params.get(k, v))
                if last_patch and last_patch.get(k) == v and cur <= v:
                    cap = 0.8 if k == "opacity" else 1.2
                    v = round(min(cur + 0.15, cap), 3)
                out[k] = v
            else:
                out[k] = v
        return out


# ---------------- CLI ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="演播室合成 Agent (Planner-DAG 版)")
    ap.add_argument("instruction", nargs="?", help="一句中文指令; 省略进入 REPL")
    ap.add_argument("--quality", default="draft", choices=["draft", "normal", "fine"])
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--provider", default=None, choices=["agnes", "qwen"],
                    help="LLM 后端 (默认读 LLM_PROVIDER)")
    a = ap.parse_args()

    s = AgentSession(quality=a.quality, enable_critic=not a.no_critic,
                     provider=a.provider)
    if a.instruction:
        r = s.process(a.instruction)
        print(json.dumps({k: r[k] for k in ("plan_source", "sec", "final")},
                         ensure_ascii=False, indent=2))
        if r.get("critic"):
            print("[critic]", json.dumps(r["critic"].get("scores"), ensure_ascii=False))
        return
    print("Agent REPL | 指令 / rollback:<背景语义> / exit")
    while True:
        try:
            raw = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("exit", "quit"):
            break
        if raw.startswith("rollback:"):
            print(json.dumps(s.rollback_background(raw.split(":", 1)[1]),
                             ensure_ascii=False, indent=2))
            continue
        r = s.process(raw)
        print(f"[final] {r['final']}  ({r['sec']}s, plan={r['plan_source']})")
        if r.get("critic"):
            print("[critic]", json.dumps(r["critic"].get("scores"), ensure_ascii=False))


if __name__ == "__main__":
    main()
