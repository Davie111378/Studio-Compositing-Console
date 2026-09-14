"""API 集成测试（C 组视角）：会话 -> 上传 -> 指令 -> 轮询 -> 产物下载 -> 回滚 -> WebSocket。"""

import asyncio
import time

from starlette.testclient import TestClient


def make_client(context):
    from agent.server.app import create_app
    return TestClient(create_app(context))


def test_end_to_end_rest_flow(context, sample_image_bytes):
    with make_client(context) as client:
        # 1. 会话
        sid = client.post("/api/v1/sessions").json()["session_id"]
        # 2. 上传
        r = client.post(f"/api/v1/sessions/{sid}/uploads",
                        files={"file": ("person.png", sample_image_bytes, "image/png")})
        assert r.status_code == 200
        asset = r.json()
        assert asset["url"].startswith("/artifacts/")
        assert client.get(asset["url"]).status_code == 200
        # 3. 指令（Demo 01 剧本）
        r = client.post(f"/api/v1/sessions/{sid}/instructions",
                        json={"text": "把这个人物放进傍晚的咖啡馆，光从左边照过来"})
        assert r.status_code == 200
        body = r.json()
        run_id, plan = body["run_id"], body["plan"]
        assert [n["tool"] for n in plan["nodes"]][0] == "matting"
        # 4. 轮询至完成
        deadline = time.time() + 120
        while True:
            state = client.get(f"/api/v1/runs/{run_id}").json()
            if state["status"] in ("done", "failed", "cancelled"):
                break
            assert time.time() < deadline, "运行超时"
            time.sleep(0.1)
        assert state["status"] == "done"
        assert state["critic"]["passed"] is True
        # 5. 导出产物可下载
        export = next(n for n in state["dag"]["nodes"] if n["tool"] == "export")
        file_url = export["outputs"]["file_url"].replace("artifact://", "/artifacts/")
        assert client.get(file_url).status_code == 200
        # 6. 版本树
        versions = client.get(f"/api/v1/sessions/{sid}/versions").json()
        assert "background_generate" in versions["node_versions"]
        # 7. 回滚接口（Demo 04）
        r = client.post(f"/api/v1/sessions/{sid}/rollback",
                        json={"role": "background_generate", "version": 1,
                              "preserve": ["lighting_estimate", "relight"]})
        assert r.status_code == 200
        rb = r.json()
        rb_run = rb["run_id"]
        deadline = time.time() + 120
        while True:
            state = client.get(f"/api/v1/runs/{rb_run}").json()
            if state["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert state["status"] == "done"
        # 8. 会话详情含对话记录
        sess = client.get(f"/api/v1/sessions/{sid}").json()
        assert any(m["role"] == "assistant" for m in sess["conversation"])
        # 9. 回放接口（A8）
        replay = client.get(f"/api/v1/runs/{run_id}/replay").json()
        assert replay["result"]["status"] == "done"
        assert any(e["event"] == "critic" for e in replay["events"])


def test_instruction_without_upload_rejected(context):
    with make_client(context) as client:
        sid = client.post("/api/v1/sessions").json()["session_id"]
        r = client.post(f"/api/v1/sessions/{sid}/instructions", json={"text": "把人物放进咖啡馆"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "E_INVALID_INPUT"


def test_unknown_session_404(context):
    with make_client(context) as client:
        r = client.get("/api/v1/sessions/nope")
        assert r.status_code == 404


def test_tools_endpoint_returns_schemas(context):
    with make_client(context) as client:
        tools = client.get("/api/v1/tools").json()["tools"]
        assert set(tools.keys()) >= {"matting", "relight", "export"}
        assert tools["matting"]["id"] == "T01"


def test_websocket_snapshot_and_events(context, sample_image_bytes):
    with make_client(context) as client:
        sid = client.post("/api/v1/sessions").json()["session_id"]
        client.post(f"/api/v1/sessions/{sid}/uploads",
                    files={"file": ("p.png", sample_image_bytes, "image/png")})
        with client.websocket_connect(f"/ws/{sid}") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            # 触发一次运行，应能收到实时事件
            client.post(f"/api/v1/sessions/{sid}/instructions",
                        json={"text": "把这个人物放进咖啡馆"})
            got_node_update = False
            deadline = time.time() + 30
            while time.time() < deadline and not got_node_update:
                msg = ws.receive_json()
                if msg["type"] in ("node_update", "run_finished", "critic"):
                    got_node_update = True
            assert got_node_update, "未收到实时推送"
