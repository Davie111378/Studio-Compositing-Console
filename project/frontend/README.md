# frontend（C 组主战场）—— ImageCompose 工作台（已合并）

原独立站点 `imagecompose-site` 已于 2026-09-15 并入本目录（见 docs/merge-report-2026-09-15.md）。

## 启动

```bash
python tools/dev_server.py    # :8899，静态站点 + 同源反代 /api /healthz /artifacts -> agent :8000，/speech -> :8200
```

结构：`index.html`（13 章叙事首页 + CREATE/RUN/SUMMARY 工作台 + 悬浮球快捷工作台）· `app.js` · `styles.css` ·
`media/`（演示素材与自托管字体）· `tools/`（dev_server + 自动化测试）·
`legacy-v1-canvas/`（v1 存档）· `design_v2/`、`shots/`、`*.pdf`（设计资产，本地保留不入库）。

## 悬浮球一键操作（2026-09-16 新增）

右下角悬浮球，点开即达全部能力，Agent 对话与一键按钮双通道：

| 入口 | 行为 |
|---|---|
| AGENT · 一键生成 | 走既有 CREATE → RUN → SUMMARY 流水线 |
| IMAGE · 上传图片 | 选图后直接进滤镜工作台 |
| FX · 滤镜 | 本地画布 8 种模式（按键 1-8）：原图/暖色温/冷色温/暗角/聚光/雾效/散景/景深虚化，强度可调、可叠加，与 `ai-service/bsrc/fx` 特效链同源；「让 Agent 接管」交后端继续 |
| LASSO · 圈选抠图 | 套索/矩形两种模式（按键 1/2）+ 羽化，输出透明 PNG；「送去背景合成」直通背景台；「让 AI 精抠」把圈选框作为 spatial box 交给后端 SAM2 |
| SCENE · 背景加图 | 内置场景/纯色/上传三种背景来源（按键 1/2/3），主体缩放合成；「让 Agent 生成场景」走完整 Agent 链 |
| REPLAY · 回看过程 | job 回放或演示滚动 |
| EXPORT · 下载成片 | 下载当前工作台结果或最近一次成片 |

工作台内 `ESC` 关闭；Agent 离线时本地一键操作不受影响。

## 测试（需 Node ≥20：D:\codex-tools\node-v22.17.0-win-x64\node.exe）

| 脚本 | 覆盖 |
|---|---|
| `tools/smoke_test.js` | 13 章滚动回归 + 深链 |
| `tools/drift_test.js` | 轮询断连/缺字段/run failed 三类异常优雅降级 |
| `tools/real_flow_test.js` | 真实 job + 离线 demo 双流程端到端 |
| `tools/voice_flow_test.js` | 语音 UI / healthz 探针 / 降级 / 文本兜底（C3） |
| `tools/multiround_test.js` | 规范 §3.5 五轮连续编辑剧本（C4，含条件回滚轮） |
| `tools/workbench_test.js` | 悬浮球 + 三个一键工作台（模式按键/画布特效/圈选抠图/背景合成，离线可跑） |
| `tools/wb_agent_test.js` | 工作台 → Agent 桥接 E2E（需 dev_server + agent 后端） |
| `tools/demo_captures.js` | §3.4 四个 Demo 剧本截图归档 demo/screenshots/ |
| `tools/backend_flow_test.py` | 后端全链路冒烟（API 视角） |

- 接入契约：`docs/api-spec.md`（REST + WebSocket 全表）、`docs/integration-site.md`（站点接入手册）、`docs/integration-c.md`
- 语音链路：`speech/run.py`(:8200) + 前端麦克风/播报（C3），语音不可用时文本输入兜底
- 交互式调试：启动 Agent 服务后开 `http://127.0.0.1:8000/docs`
