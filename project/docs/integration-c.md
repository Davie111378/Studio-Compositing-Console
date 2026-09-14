# C 组前端/语音接入指南

> Agent 服务已可用 mock 全链路跑通，你现在就可以对接真实后端，不必等算法。

## 环境

- 服务地址：`http://127.0.0.1:8000`（启动：`python scripts/run_agent.py`）
- 接口契约全文：`docs/api-spec.md`；可交互调试：`http://127.0.0.1:8000/docs`
- 建议在 `frontend/` 用任意框架（Vite+React/Vue 均可）；**组件 PascalCase，文件与目录全英文**（规范 N7）

## 最小对接清单（对照你的任务卡）

| 任务卡 | 你要做的 | 后端已提供 |
|---|---|---|
| C1 首页 | 上传控件 + 一句话输入框 | `POST /sessions`、`POST /sessions/{sid}/uploads`、`POST /instructions` |
| C2 三栏工作台 | 左素材/中 Canvas/右 Agent + 底部 DAG | `GET /runs/{rid}`（节点+edges+产物 URL）、WS 事件流 |
| C3 语音可见 | 识别文本 + Agent 动作上屏 | `message` 事件（assistant 文本可直喂 TTS）；低置信度时把文本弹确认框再发 `instructions` |
| C5 DAG 可视化 | 节点状态实时染色 + 点击看中间产物 | `node_update` 事件；每个节点 `artifact_urls`（抠取含 rgba/alpha/mask 三张） |
| C6 历史回滚 UI | 版本树 + 条件回滚交互 | `GET /sessions/{sid}/versions`；`POST /rollback {role, version, preserve[]}` |
| C7 导出 | 多尺寸/格式下载 | 对最终图再发一次 T08：`POST /instructions` 后……或直接用 export 节点产物（PNG）；JPG/多尺寸走 `GET /api/v1/tools` 看 T08 schema，后端可按需补一个直连导出端点 |

## 联调顺序建议

1. 先用 `/docs` 手工把"建会话→上传→指令→轮询→下载"点通；
2. 接 WS：`ws://127.0.0.1:8000/ws/{sid}`，把 `node_update` 打到 console 验证实时性；
3. 搭三栏静态布局，把真实数据填进去；
4. 语音链路（ASR 文本 -> instructions，message -> TTS）；
5. 版本树与回滚 UI；
6. 每周三/周日集成日跑 `python scripts/smoke_test.py` 回归。

## 备注

- 语音模块放 `speech/asr/`、`speech/tts/`（全本地部署，规范 5.5）；它只产出/消费文本，与 Agent 之间没有别的耦合。
- CORS 已全开（开发态）；封版前可以收紧。
- 需要后端加字段/端点随时在站会提，接口改动会同步更新 `docs/api-spec.md` 并保持向后兼容。
