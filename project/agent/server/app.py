"""Agent 服务 REST + WebSocket（C 组对接面，完整契约见 docs/api-spec.md）。

启动：python scripts/run_agent.py  ->  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.dag.models import Quality, now_ms
from agent.errors import (
    E_INVALID_INPUT,
    E_SESSION_NOT_FOUND,
    AgentError,
)
from agent.planner import PlanningRequest
from agent.rollback.versions import build_rollback_dag, select_plan
from agent.schema import load_registry, public_artifact_url
from agent.server.state import AppContext, build_context

logger = logging.getLogger("agent.server")


def serialize_run(state) -> dict[str, Any]:
    data = state.model_dump(mode="json")
    dag = data["dag"]
    for node, live in zip(dag["nodes"], state.dag.nodes):
        node["artifact_urls"] = [public_artifact_url(a) for a in live.artifacts]
    dag["edges"] = state.dag.edges()
    return data


def serialize_session(sess: dict[str, Any]) -> dict[str, Any]:
    out = dict(sess)
    out["uploads"] = {
        aid: {**asset, "url": public_artifact_url(asset["uri"])}
        for aid, asset in sess.get("uploads", {}).items()
    }
    return out


# ---- 请求体 ----

class InstructionBody(BaseModel):
    text: str = Field(min_length=1, description="自然语言指令（ASR 转写结果或键入文本）")
    asset_ids: Optional[list[str]] = Field(default=None, description="使用哪些上传图；缺省取最近一张")
    spatial: Optional[dict] = Field(default=None, description='空间指代：{"click":{"x":..,"y":..}} 或 {"box":[x1,y1,x2,y2]}')
    quality: str = Field(default="normal", description="draft(草稿级快) | normal | fine(精修级)")


class RollbackBody(BaseModel):
    role: str = Field(description='要回滚的节点角色，如 "background_generate"')
    version: int = Field(description="要恢复到的版本号")
    preserve: list[str] = Field(default_factory=list, description='保留当前版本的 role 列表，如 ["lighting_estimate","relight"]')


def create_app(context: AppContext | None = None) -> FastAPI:
    ctx = context or build_context()
    app = FastAPI(
        title="多模态图像合成 Agent Service",
        version="0.1.0",
        description="Planner / DAG Executor / Critic / Rollback —— A 组 Agent 服务",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ctx.settings.ensure_dirs()
    app.mount("/artifacts", StaticFiles(directory=str(ctx.settings.artifacts_dir)), name="artifacts")

    @app.exception_handler(AgentError)
    async def agent_error_handler(_: Request, exc: AgentError) -> JSONResponse:
        code = exc.code
        status = 404 if code in (E_SESSION_NOT_FOUND, "E_RUN_NOT_FOUND", "E_ASSET_NOT_FOUND", "E_ARTIFACT_NOT_FOUND") \
            else 400 if code in (E_INVALID_INPUT, "E_STATE_CONFLICT", "E_ROLLBACK_INVALID", "E_SCHEMA_VALIDATION",
                                 "E_DAG_INVALID", "E_TOOL_UNKNOWN", "E_TIMEOUT", "E_CANCELLED") \
            else 500
        return JSONResponse(status_code=status, content={"error": exc.to_dict()})

    # ---------- 基础 ----------

    @app.get("/healthz", tags=["meta"])
    async def healthz():
        return {
            "status": "ok",
            "service": "agent",
            "version": "0.1.0",
            "planner": ctx.settings.planner_mode,
            "critic": type(ctx.critic).__name__,
            "provider": type(ctx.provider).__name__,
            "ai_service_url": ctx.settings.ai_service_url or "(embedded)",
            "critic_threshold": ctx.settings.critic_threshold,
        }

    @app.get("/api/v1/tools", tags=["meta"])
    async def list_tools():
        """8 个工具 Schema 全文（B 组实现契约 / C 组 DAG 渲染用）。"""
        return {"tools": load_registry()}

    # ---------- 会话 ----------

    @app.post("/api/v1/sessions", tags=["session"])
    async def create_session():
        sess = ctx.sessions.create()
        return {"session_id": sess["session_id"]}

    @app.get("/api/v1/sessions/{session_id}", tags=["session"])
    async def get_session(session_id: str):
        return serialize_session(ctx.sessions.get(session_id))

    @app.post("/api/v1/sessions/{session_id}/uploads", tags=["session"])
    async def upload(session_id: str, request: Request, file: UploadFile | None = File(default=None)):
        if file is not None:
            data = await file.read()
            filename = file.filename or "upload.png"
        else:
            data = await request.body()
            filename = request.headers.get("x-filename", "upload.png")
        if not data:
            raise AgentError(E_INVALID_INPUT, "请求体为空：请用 multipart(file=) 或原始字节上传")
        asset = ctx.sessions.add_upload(session_id, filename, data)
        return {**asset, "url": public_artifact_url(asset["uri"])}

    # ---------- 指令 -> 计划 -> 执行 ----------

    @app.post("/api/v1/sessions/{session_id}/instructions", tags=["agent"])
    async def instruct(session_id: str, body: InstructionBody):
        sess = ctx.sessions.get(session_id)
        uploads = sess.get("uploads") or {}
        if body.asset_ids:
            asset_uris = [ctx.sessions.get_asset(session_id, a)["uri"] for a in body.asset_ids]
        elif uploads:
            last = sorted(uploads.values(), key=lambda a: a["ts"])[-1]
            asset_uris = [last["uri"]]
        else:
            asset_uris = []
        try:
            quality = Quality(body.quality)
        except ValueError:
            raise AgentError(E_INVALID_INPUT, f"非法 quality: {body.quality}")

        req = PlanningRequest(
            session_id=session_id, text=body.text, asset_uris=asset_uris,
            round_index=len(sess.get("runs") or []) + 1, quality=quality,
            spatial=body.spatial, available_roles=ctx.sessions.available_roles(session_id),
        )
        result = await ctx.planner.plan(req)

        rollback_meta = None
        if result.kind == "rollback":
            rb = result.rollback or {}
            base_dag = select_plan(ctx.sessions.plan_history(session_id), rb["role"])
            versions = {
                role: {v["version"]: v for v in hist}
                for role, hist in (sess.get("node_versions") or {}).items()
            }
            dag, rollback_meta = build_rollback_dag(
                base_dag, versions,
                rb["role"], int(rb["version"]), list(rb.get("preserve") or []),
            )
            text = f"{body.text}（回滚 {rb['role']} -> v{rb['version']}，重跑 {len(rollback_meta['rerun'])} 个下游节点）"
        else:
            dag = result.dag
            text = body.text

        state = await ctx.executor.start(session_id, dag, instruction=text, kind=result.kind)
        if result.kind == "rollback" and result.rollback:
            ctx.sessions.set_current(session_id, result.rollback["role"], int(result.rollback["version"]))
        ctx.sessions.add_conversation(session_id, "user", body.text)
        assistant = (f"已生成执行计划（{len(dag.nodes)} 个节点，planner={result.planner}），开始执行。"
                     if result.kind != "rollback" else f"已按回滚语义重建计划（{len(dag.nodes)} 个节点），开始执行。")
        ctx.sessions.add_conversation(session_id, "assistant", assistant)
        ctx.bus.publish(session_id, {"type": "message", "ts": now_ms(),
                                     "run_id": state.run_id, "data": {"role": "assistant", "text": assistant}})
        return {
            "run_id": state.run_id,
            "session_id": session_id,
            "kind": result.kind,
            "planner": result.planner,
            "rollback": rollback_meta,
            "plan": serialize_run(state)["dag"],
        }

    # ---------- 运行查询与控制 ----------

    @app.get("/api/v1/runs/{run_id}", tags=["run"])
    async def get_run(run_id: str):
        return serialize_run(ctx.executor.get_run(run_id))

    @app.get("/api/v1/sessions/{session_id}/runs", tags=["run"])
    async def list_runs(session_id: str):
        sess = ctx.sessions.get(session_id)
        runs = []
        for rid in sess.get("runs") or []:
            if rid in ctx.executor.runs:
                runs.append(serialize_run(ctx.executor.runs[rid].state))
            else:
                cached = ctx.recorder.load_result(rid)
                if cached:
                    runs.append(cached)
        return {"runs": runs}

    @app.post("/api/v1/runs/{run_id}/pause", tags=["run"])
    async def pause_run(run_id: str):
        return serialize_run(ctx.executor.pause(run_id))

    @app.post("/api/v1/runs/{run_id}/resume", tags=["run"])
    async def resume_run(run_id: str):
        return serialize_run(ctx.executor.resume(run_id))

    @app.post("/api/v1/runs/{run_id}/cancel", tags=["run"])
    async def cancel_run(run_id: str):
        return serialize_run(await ctx.executor.cancel(run_id))

    @app.post("/api/v1/runs/{run_id}/nodes/{node_id}/retry", tags=["run"])
    async def retry_node(run_id: str, node_id: str):
        return serialize_run(ctx.executor.retry_node(run_id, node_id))

    @app.post("/api/v1/runs/{run_id}/nodes/{node_id}/skip", tags=["run"])
    async def skip_node(run_id: str, node_id: str):
        return serialize_run(ctx.executor.skip_node(run_id, node_id))

    @app.post("/api/v1/runs/{run_id}/nodes/{node_id}/rerun", tags=["run"])
    async def rerun_node(run_id: str, node_id: str, higher_quality: bool = False):
        return serialize_run(ctx.executor.rerun_node(run_id, node_id, higher_quality))

    # ---------- 版本树与回滚 ----------

    @app.get("/api/v1/sessions/{session_id}/versions", tags=["rollback"])
    async def versions(session_id: str):
        sess = ctx.sessions.get(session_id)
        return {"node_versions": sess.get("node_versions") or {}, "last_plan": sess.get("last_plan")}

    @app.post("/api/v1/sessions/{session_id}/rollback", tags=["rollback"])
    async def rollback(session_id: str, body: RollbackBody):
        sess = ctx.sessions.get(session_id)
        base_dag = select_plan(ctx.sessions.plan_history(session_id), body.role)
        versions = {
            role: {v["version"]: v for v in hist}
            for role, hist in (sess.get("node_versions") or {}).items()
        }
        dag, meta = build_rollback_dag(base_dag, versions,
                                       body.role, body.version, body.preserve)
        text = f"[回滚] {body.role} -> v{body.version}，保留 {body.preserve or '无'}"
        state = await ctx.executor.start(session_id, dag, instruction=text, kind="rollback")
        ctx.sessions.set_current(session_id, body.role, body.version)
        ctx.sessions.add_conversation(session_id, "assistant",
                                      f"已回滚 {body.role} 到 v{body.version}，重跑下游：{meta['rerun'] or '无'}")
        return {"run_id": state.run_id, "rollback": meta, "plan": serialize_run(state)["dag"]}

    # ---------- A8 回放 ----------

    @app.get("/api/v1/runs/{run_id}/replay", tags=["eval"])
    async def replay(run_id: str):
        result = ctx.recorder.load_result(run_id)
        events_path = ctx.recorder._dir(run_id) / "events.jsonl"
        events = []
        if events_path.exists():
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
        return {"result": result, "events": events}

    # ---------- WebSocket ----------

    @app.websocket("/ws/{session_id}")
    async def ws_endpoint(ws: WebSocket, session_id: str):
        await ws.accept()
        queue = ctx.bus.subscribe(session_id)
        try:
            snapshot_runs = []
            sess = ctx.sessions.get(session_id)
            for rid in sess.get("runs") or []:
                if rid in ctx.executor.runs:
                    snapshot_runs.append(serialize_run(ctx.executor.runs[rid].state))
            await ws.send_json({
                "type": "snapshot", "ts": now_ms(),
                "data": {"session": serialize_session(sess), "runs": snapshot_runs},
            })
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    await ws.send_json(event)
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "ping", "ts": now_ms()})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("ws %s 断开: %s", session_id, e)
        finally:
            ctx.bus.unsubscribe(session_id, queue)

    return app


app = create_app()
