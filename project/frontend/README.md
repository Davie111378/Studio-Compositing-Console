# frontend（C 组主战场）—— ImageCompose 工作台（已合并）

原独立站点 `imagecompose-site` 已于 2026-09-15 并入本目录（见 docs/merge-report-2026-09-15.md）。

## 启动

```bash
python tools/dev_server.py    # :8899，静态站点 + 同源反代 /api /healthz /artifacts -> agent :8000，/speech -> :8200
```

结构：`index.html`（13 章叙事首页 + CREATE/RUN/SUMMARY 工作台）· `app.js` · `styles.css` ·
`media/`（演示素材与自托管字体）· `tools/`（dev_server + 六类自动化测试）·
`legacy-v1-canvas/`（v1 存档）· `design_v2/`、`shots/`、`*.pdf`（设计资产，本地保留不入库）。

## 测试（需 Node ≥20：D:\codex-tools\node-v22.17.0-win-x64\node.exe）

| 脚本 | 覆盖 |
|---|---|
| `tools/smoke_test.js` | 13 章滚动回归 + 深链 |
| `tools/drift_test.js` | 轮询断连/缺字段/run failed 三类异常优雅降级 |
| `tools/real_flow_test.js` | 真实 job + 离线 demo 双流程端到端 |
| `tools/voice_flow_test.js` | 语音 UI / healthz 探针 / 降级 / 文本兜底（C3） |
| `tools/multiround_test.js` | 规范 §3.5 五轮连续编辑剧本（C4，含条件回滚轮） |
| `tools/demo_captures.js` | §3.4 四个 Demo 剧本截图归档 demo/screenshots/ |
| `tools/backend_flow_test.py` | 后端全链路冒烟（API 视角） |

- 接入契约：`docs/api-spec.md`（REST + WebSocket 全表）、`docs/integration-site.md`（站点接入手册）、`docs/integration-c.md`
- 语音链路：`speech/run.py`(:8200) + 前端麦克风/播报（C3），语音不可用时文本输入兜底
- 交互式调试：启动 Agent 服务后开 `http://127.0.0.1:8000/docs`
