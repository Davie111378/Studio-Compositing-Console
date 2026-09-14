# Agent 服务 API 契约 V1.0（C 组对接面）

> Base URL：`http://127.0.0.1:8000`（内嵌模式）。
> 交互式文档：`/docs`（Swagger UI），原始 OpenAPI：`/openapi.json`。
> 所有请求/响应均为 JSON（上传除外）。错误统一格式见文末。

## 0. 快速接入（三步出图）

```js
// 1. 建会话
const { session_id } = await POST("/api/v1/sessions");
// 2. 上传图片（multipart，字段名 file）
const asset = await POST(`/api/v1/sessions/${session_id}/uploads`, form);
// 3. 一句话 -> 自动计划 + 执行
const { run_id } = await POST(`/api/v1/sessions/${session_id}/instructions`,
                             { text: "把这个人物放进傍晚的咖啡馆，光从左边照过来" });
// 之后：WS 实时收节点状态，或轮询 GET /api/v1/runs/{run_id}
```

## 1. REST 接口

### 元信息

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 服务健康 + 当前配置（planner/critic/provider 形态） |
| GET | `/api/v1/tools` | T01–T08 八个工具 Schema 全文（DAG 节点渲染/参数提示用） |

### 会话与上传

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/sessions` | 创建会话 → `{session_id}` |
| GET | `/api/v1/sessions/{sid}` | 会话全量：uploads / runs / node_versions / current / conversation / plans |
| POST | `/api/v1/sessions/{sid}/uploads` | 上传图片（multipart `file`；也支持原始字节 + `x-filename` 头）→ `{asset_id, uri, url, width, height}` |

### 指令（核心入口）

`POST /api/v1/sessions/{sid}/instructions`

```json
{
  "text": "把人物放到咖啡馆，光从左边照过来",
  "asset_ids": ["a1b2c3d4"],          // 可选，缺省用最近上传
  "spatial": {"click": {"x": 120, "y": 200}},  // 可选空间指代；或 {"box":[x1,y1,x2,y2]}
  "quality": "normal"                  // draft(草稿级快) | normal | fine(精修)
}
```

响应：`{run_id, kind: "plan"|"rollback", planner, plan(DAG), rollback?}`，执行已异步开始。

**多轮语义**：同一 session 内连续发指令即多轮；"改成傍晚 / 光线太冷了 / 阴影轻一点" 等会自动生成增量 DAG（冻结上游产物，只重跑受影响下游）。"换回上一版背景，保留现在的光" 自动走条件回滚。

### 运行查询与控制（A3）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/runs/{run_id}` | 运行全量状态（DAG 各节点 status/version/artifacts/latency + critic） |
| GET | `/api/v1/sessions/{sid}/runs` | 会话全部运行 |
| POST | `/api/v1/runs/{run_id}/pause` | 暂停（当前 wave 跑完后停） |
| POST | `/api/v1/runs/{run_id}/resume` | 恢复 |
| POST | `/api/v1/runs/{run_id}/cancel` | 取消 |
| POST | `/api/v1/runs/{run_id}/nodes/{nid}/retry` | 失败节点重试（下游重置，已完成节点不动） |
| POST | `/api/v1/runs/{run_id}/nodes/{nid}/skip` | 跳过 pending 节点（下游连带跳过） |
| POST | `/api/v1/runs/{run_id}/nodes/{nid}/rerun?higher_quality=true` | 手动重跑（版本 +1） |

### 版本树与回滚（A5 / C6）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/sessions/{sid}/versions` | 节点版本树：`node_versions[role] = [{version, run_id, artifacts, quality, ts}]` |
| POST | `/api/v1/sessions/{sid}/rollback` | 条件回滚：`{"role":"background_generate","version":1,"preserve":["lighting_estimate","relight"]}` → 新 run |

角色（role）取值 = 工具名：`matting / background_generate / lighting_estimate / relight / shadow_generate / harmonize / enhance / export`。

### 回放（A8）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/runs/{run_id}/replay` | `{result(终态), events[](全事件流)}` —— DAG 可视化回放直接用它 |

## 2. WebSocket 实时推送

`GET(ws) /ws/{session_id}`

连接后立即收到一条 `snapshot`（会话 + 活跃运行全量），此后推送增量事件：

```json
{"type": "snapshot",     "ts": 0, "data": {"session": {...}, "runs": [...]}}
{"type": "node_update",  "ts": 0, "run_id": "...", "data": {"id":"n4","tool":"relight","role":"relight",
                 "label":"重打光","status":"running","version":1,"attempts":1,
                 "artifacts":["artifact://..."],"artifact_urls":["/artifacts/..."],"latency_ms":180}}
{"type": "critic",       "ts": 0, "run_id": "...", "data": {"scores":{"lighting":86,"shadow":93,"color":86,"edge":84},
                 "overall":87.3,"threshold":85,"passed":true,"action":"pass","comment":"..."}}
{"type": "run_status",   "ts": 0, "run_id": "...", "data": {"status": "replanning"}}
{"type": "run_finished", "ts": 0, "run_id": "...", "data": {"status": "done", "critic": {...}}}
{"type": "message",      "ts": 0, "run_id": "...", "data": {"role":"assistant","text":"已生成执行计划（7 个节点）..."}}
{"type": "ping",         "ts": 0}
```

- `status` 取值：节点 `pending/running/done/failed/skipped/cancelled`；运行 `running/paused/critiquing/replanning/done/failed/cancelled`。
- `artifact_urls` 是同源 HTTP 路径，`<img src>` 直接可用。
- 15s 无事件发一次 `ping` 作心跳。

## 3. 资源 URI 约定

| URI | 含义 | HTTP 形式 |
|---|---|---|
| `asset://uploads/<sid>/<aid>_<name>.png` | 用户上传原图 | `/artifacts/uploads/...` |
| `artifact://runs/<run_id>/<nid>/v<n>/<name>.png` | 节点产物（中间态可视化就点这里） | `/artifacts/runs/...` |

## 4. 错误格式

```json
{"error": {"code": "E_INVALID_INPUT", "message": "...", "retryable": false, "detail": [...]}}
```

| HTTP | 场景 |
|---|---|
| 400 | 参数/状态错误：E_INVALID_INPUT、E_STATE_CONFLICT、E_ROLLBACK_INVALID、E_DAG_INVALID… |
| 404 | 资源不存在：E_SESSION_NOT_FOUND、E_RUN_NOT_FOUND、E_ASSET_NOT_FOUND |
| 500 | 服务内部错误 |

## 5. C 组工作台对接建议（对照 C2/C5/C6 验收）

1. **右栏 Agent 面板**：渲染 `message` 事件（对话）+ `critic` 事件（五维分数条）。
2. **底部 DAG 区**：`run.nodes` 建图（`edges` 已在响应中）；`node_update` 事件刷状态色。
3. **点击节点看中间产物**：直接展示该节点 `artifact_urls`（抠取有 rgba/alpha/mask 三张）。
4. **历史/回滚 UI**：`GET /versions` 画版本树；对选中版本 `POST /rollback`，带 preserve 列表（"保留现在的光" = 勾选 lighting_estimate + relight）。
5. **语音**：ASR 得到文本后调 `instructions`；`message` 事件的 assistant 文本直接喂 TTS。
