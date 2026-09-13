# -*- coding: utf-8 -*-
"""
engine.py — DAG Executor (冲刺计划 A3 最小可用版)
- 状态机: pending → running → done / failed / skipped
- 失败重试: retryable 错误自动重试 1 次 (e_retry), 不可重试 → failed (下游 skipped)
- 引用解析: "$nN.key" → ArtifactTable
- 结构化回放日志: eval/log 可复现 (D8 A8 最小版)
- 幂等: 同 run_id 重跑安全 (输出带时间戳目录)
"""
from __future__ import annotations
import json, time
from pathlib import Path

from artifact import ArtifactTable
import dag as dagmod
import adapter
import verifier

LOG_DIR = Path(__file__).resolve().parent / "eval" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def execute(dag: dict, quality_default: str = "draft",
            ctx: dict | None = None, on_event=None, keep_map: dict | None = None) -> dict:
    """执行 DAG。返回 {status, artifacts, node_results, log_path, final_path, total_ms}
    keep_map: {node_id: 旧产物} — 条件回滚时复用缓存 (如"换背景保留光照")。"""
    errs = dagmod.validate(dag)
    if errs:
        return {"status": "invalid", "errors": errs, "node_results": []}

    run_id = (ctx or {}).get("run_id") or time.strftime("%Y%m%d_%H%M%S")
    ctx = {"run_id": run_id, **(ctx or {})}
    table = ArtifactTable()
    for nid, data in (keep_map or {}).items():   # 回滚缓存预登记
        table.store(nid, data)
    states = {n["id"]: "pending" for n in dag["nodes"]}
    for nid in (keep_map or {}):
        states[nid] = "done"
    node_results: list[dict] = []
    t0 = time.time()

    def emit(ev):
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass

    emit({"type": "run_start", "run_id": run_id, "intent": dag.get("intent", "")})

    # 就绪层按工具号排序 (LLM 会交叉命名节点 id, 纯 id 序会让 T02 先于 T01 执行)
    _by_id = {n["id"]: n for n in dag["nodes"]}

    def _tool_rank(nid_: str):
        import re as _re
        m = _re.match(r"T(\d+)", _by_id.get(nid_, {}).get("tool") or "")
        return (int(m.group(1)) if m else 99, nid_)

    for node in dagmod.topo_order(dag, key=_tool_rank):
        nid, tool = node["id"], node["tool"]
        # 条件回滚: 缓存复用节点直接记 done (保留光照等语义)
        if nid in (keep_map or {}):
            node_results.append({"node_id": nid, "tool": tool, "status": "kept",
                                 "reason": "条件回滚缓存复用"})
            emit({"type": "node_kept", "node": nid})
            continue
        # 下游跳过: 任一依赖未 done
        deps = node.get("depends_on", [])
        if any(states[d] != "done" for d in deps):
            states[nid] = "skipped"
            node_results.append({"node_id": nid, "tool": tool, "status": "skipped",
                                 "reason": f"依赖未完成: {deps}"})
            emit({"type": "node_skipped", "node": nid})
            continue
        states[nid] = "running"
        try:
            params = table.resolve(node.get("params", {}))
            # "$cur" 引用: 用户当前输入图
            params = {k: (ctx.get("cur_image") if v == "$cur" else v)
                      for k, v in params.items()}
            params = {k: v for k, v in params.items() if v}  # 去掉无 cur 且 $cur 的参数
        except KeyError as e:
            states[nid] = "failed"
            node_results.append({"node_id": nid, "tool": tool, "status": "failed",
                                 "error": {"code": "E_REF_DANGLING", "message": str(e), "retryable": False}})
            emit({"type": "node_failed", "node": nid, "error": str(e)})
            continue

        # 质量档位: **以会话档位为准**, 节点级 quality 直接丢弃。
        # 历史 bug: planner/fewshot 模板把 "draft" 写进 params 当样板值, 若直接
        #   params.pop("quality", quality_default) 会让样板值覆盖用户在 UI 选的
        #   fine/normal → refiner 精修永不启用 (E3 失败根因)。
        # 也不可用 max(会话, 节点): LLM 会在节点里幻觉出 "fine", 把 draft 会话
        #   意外拉高 (E1 假失败根因)。质量档位只由用户显式选择的会话档位决定。
        # 注: T01.adapter 另有"节点显式 quality=fine"的支持, 但引擎层不再透传。
        if isinstance(params, dict):
            params.pop("quality", None)
        # 记录各节点解析后入参 (工具后处理需要跨节点取原始输入, 如 T06 底部UI裁剪
        # 必须量在 T01 的原始上传图上 — T01 输出的 fg 是白底预乘, 渐变特征已冲掉)
        ctx.setdefault("step_params", {})[nid] = {"tool": tool, "params": params}
        quality = quality_default
        attempts, result = 0, None
        while attempts < 2:                       # e_retry: retryable 自动重试 1 次
            attempts += 1
            result = adapter.call_tool(tool, nid, params, quality=quality, ctx=ctx)
            if result["ok"]:
                # L6 经验差分检测: 形式化不变量校验 (产物存在/尺寸/FDR/可解码)
                vr = verifier.verify_node(tool, result["data"])
                if vr["ok"]:
                    break
                result = {**result, "ok": False,
                          "error": {"code": "E_VERIFY_FAIL",
                                    "message": "; ".join(vr["violations"])[:300],
                                    "retryable": vr["retryable"]}}
                emit({"type": "node_verify_fail", "node": nid,
                      "violations": vr["violations"]})
                if not vr["retryable"]:
                    break
                continue
            if not (result.get("error") or {}).get("retryable"):
                break
            emit({"type": "node_retry", "node": nid, "attempt": attempts})

        if result["ok"]:
            states[nid] = "done"
            table.store(nid, result["data"])
            node_results.append({**result, "status": "done"})
            emit({"type": "node_done", "node": nid, "tool": tool,
                  "latency_ms": result["latency_ms"]})
        else:
            states[nid] = "failed"
            node_results.append({**result, "status": "failed"})
            emit({"type": "node_failed", "node": nid,
                  "error": (result.get("error") or {}).get("message", "")})

    total_ms = int((time.time() - t0) * 1000)
    run_status = "done" if all(states[n["id"]] == "done" for n in dag["nodes"]) else \
                 ("partial" if any(s == "done" for s in states.values()) else "failed")

    # 最终产物: outputs 指定节点的主产物 / 否则最后一个 done 节点
    # key 优先级: 导出 > 删主体成片 > 合成 > 增强 > 打光前景 > 背景 > 抠图前景/alpha
    _FINAL_KEYS = ("exported_path", "composite_path", "removed_path", "enhanced_path",
                   "relit_fg_path", "bg_path", "fg_path", "alpha_path")
    final_path = None
    outs = dag.get("outputs") or []
    if outs and table.get(outs[-1]):
        d = table.get(outs[-1])
        final_path = next((d[k] for k in _FINAL_KEYS if d.get(k)), None)
    if final_path is None:
        for n in reversed(dagmod.topo_order(dag, key=_tool_rank)):
            d = table.get(n["id"])
            if d:
                final_path = next((d[k] for k in _FINAL_KEYS if d.get(k)), None)
                if final_path:
                    break

    log = {"run_id": run_id, "intent": dag.get("intent", ""), "dag": dag,
           "states": states, "node_results": node_results,
           "total_ms": total_ms, "final_path": final_path, "time": time.strftime("%F %T")}
    log_p = LOG_DIR / f"run_{run_id}.json"
    log_p.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    emit({"type": "run_end", "run_id": run_id, "status": run_status, "total_ms": total_ms})

    return {"status": run_status, "run_id": run_id, "states": states,
            "node_results": node_results, "artifacts": table,
            "final_path": final_path, "total_ms": total_ms, "log_path": str(log_p)}


def load_log(run_id: str) -> dict:
    """回放: 读取结构化日志。"""
    return json.loads((LOG_DIR / f"run_{run_id}.json").read_text(encoding="utf-8"))
