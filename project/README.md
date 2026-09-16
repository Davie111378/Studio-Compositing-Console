# 多模态图像合成 Agent —— 项目仓库

> 面向真实感图像合成的图像抠取与光影一致性融合方法研究——多模态交互式 Agent 系统
> 团队：A（Agent / 后端 / 系统集成）· B（算法 / 训练 / 实验）· C（前端 / 语音 / Demo）
> 总规范见《项目总规范_多模态图像合成Agent_V1.0.md》（仓库外 `生产实习/` 目录）

## 当前状态（2026-09-16 千问全量接入）

**四档千问模型已实测接入并全链路验证**：Planner=qwen-plus · Critic=qwen-vl-max · ASR=qwen3-asr-flash · T02 背景生成=wanx2.1-t2i-turbo。`.env` 只填 `DASHSCOPE_API_KEY` 即全部启用（LLM/VLM 端点与模型自动取千问缺省），失败逐级回落规则档，离线仍可跑：

```text
上传/语音输入 -> Planner(qwen-plus / 规则兜底) -> DAG -> Executor(暂停/恢复/重试/跳过/重跑)
     -> ai-service 8 工具(T02 真实文生图 + bsrc + mock 回落) -> Critic(VLM qwen-vl-max / 规则兜底 + 自动重规划)
     -> 条件回滚(节点级版本) -> 语音+字幕反馈 -> SUMMARY 继续编辑(5 轮剧本) / 版本树回滚
```

- 2026-09-16：千问 API 全量接入实测通过（key 实测：qwen-plus/qwen-vl-max/qwen3-asr-flash/wanx2.1-t2i-turbo 均 200）；T02 背景生成实装千问文生图真实档（`ai-service/aiservice/diffusion/t2i.py`，约 10-15s 出图 + 亮度近似深度图，失败回落渐变 mock）；LLM Planner 增加计划期输入类型校验 + 失败反馈重试；修复 VLM Critic comment 混入 scores 的 pydantic 崩溃；`scripts/smoke_test.py` 9/9 通过、`pytest tests/` 37 例全绿。
- 2026-09-15：`imagecompose-site` 前端整体并入 `frontend/`（13 章叙事首页 + CREATE/RUN/SUMMARY 工作台 + 回放）；新增 `speech/`（ASR 双档：千问 qwen3-asr-flash / 本地 faster-whisper；TTS 浏览器本地引擎）；模型接入统一走千问（DashScope，`.env` 填 key 即用，详见 `.env.example`）。
- 2026-09-13：B 组 Studio-Compositing-Console 算法引擎已合并进 `ai-service/bsrc/`（上游 b3107bc）。T03/T04/T05/T06/T07 为 B 组真实算法；T01 四级引擎链（BiRefNet→rembg→GrabCut→色彩距离）；缺 opencv 自动回落 mock，HTTP 信封契约不变（`IMC_ENGINE` 可切换）。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 DASHSCOPE_API_KEY（可选，不填全部自动降级离线可跑）

# 三进程形态（推荐，贴近生产）
python scripts/run_ai_service.py    # 终端 1，ai-service :8100
python scripts/run_agent.py         # 终端 2，agent :8000（读 .env）
python speech/run.py                # 终端 3，语音 :8200（可选）
python frontend/tools/dev_server.py # 终端 4，前端 :8899（/api /speech /artifacts 同源反代）

# 一体化形态（单进程，内嵌 mock 工具）
python scripts/run_agent.py

# 冒烟与验收
python scripts/smoke_test.py                     # 后端全链路（建会话→上传→合成→回滚）
python -m pytest tests/ -q                       # 单元/集成测试（37 例）
python scripts/planner_eval.py                   # Planner 指令集评测（107 条，规范 §2.2A）
python frontend/tools/backend_flow_test.py       # 前端视角后端冒烟
D:\codex-tools\node-v22.17.0-win-x64\node.exe frontend/tools/smoke_test.js       # 13 章滚动回归
D:\codex-tools\node-v22.17.0-win-x64\node.exe frontend/tools/drift_test.js       # 异常容错
D:\codex-tools\node-v22.17.0-win-x64\node.exe frontend/tools/real_flow_test.js   # 真实+离线端到端
D:\codex-tools\node-v22.17.0-win-x64\node.exe frontend/tools/voice_flow_test.js  # 语音链路
D:\codex-tools\node-v22.17.0-win-x64\node.exe frontend/tools/multiround_test.js  # 5 轮多轮剧本
D:\codex-tools\node-v22.17.0-win-x64\node.exe frontend/tools/demo_captures.js    # 4 个 Demo 截图归档
```

启动后：前端 http://127.0.0.1:8899 · API 文档 http://127.0.0.1:8000/docs

## 量化验收核销表（规范 §2.2，逐条如实）

| 指标 | 目标 | 实测 | 结论 | 证据 |
|---|---|---|---|---|
| A. Tool Selection（指令集） | ≥95% | **95.33%** | ✅ 达成 | `python scripts/planner_eval.py` → data/eval/planner_eval_report.json（107 条） |
| A. DAG Order | ≥95% | **98.13%** | ✅ 达成 | 同上 |
| A. Parameter | ≥90% | **96.26%** | ✅ 达成 | 同上 |
| A. Spatial Reference | ≥80% | **100%**（规划级；画布点选入口待 C 组扩展） | ✅ 达成 | 同上 G6 组 |
| A. 计划生成延迟 | ≤3s | **<1ms**（规则档，纯内存路由） | ✅ 达成 | 同上 |
| A. 端到端出图（草稿级） | ≤10s | **≈5-10s**（mock 引擎 5s；bsrc 引擎 15-20s） | ⚠️ mock 档达成 | multiround_test R3=5.3s |
| E. 5 轮连续编辑不崩溃 | 100% | **5/5 通过**（含条件回滚轮） | ✅ 达成 | frontend/tools/multiround_test.js |
| E. 条件回滚（换背景保留光） | 100% | **通过**，下游自动重跑 | ✅ 达成 | tests/test_rollback.py + multiround R5 |
| E. ASR 中文（本地/云端） | ≥95% 字准 / ≤800ms | **云端档已实测**：qwen3-asr-flash 转写准确（"把人物放进傍晚的咖啡馆"），延迟 ≈1.7s；本地档待 faster-whisper 实测 | ⚠️ 云端可用（链路含 VAD 降级与二次确认） | speech/ + voice_flow_test.js |
| B. 抠取精度（微调 vs baseline） | SAD ↓≥10% | **未训**（本机无 GPU/权重）；四级引擎链可用 | ❌ 如实标注待办 | training/README.md |
| C. 光影一致性四组实验 | D 组自然度≥4.0 | **未开实验**（数据集未入库） | ❌ 如实标注待办 | experiments/ 模板就绪 |
| D. Critic 可信度（Spearman） | ≥0.6 | VLM（qwen-vl-max）已实测在线打分并可触发重规划；与人工分 Spearman 标定待 30 组样本 | ⚠️ 在线可用，标定待办 | agent/critic/vlm_critic.py |
| T02 背景生成 | 语义可控 | **千问文生图真实档已上线**（wanx2.1-t2i-turbo，实测约 10-15s 出图）；安全拦截/断网自动回落渐变 mock | ✅ 真实档达成（含 L3 回落） | ai-service/aiservice/diffusion/t2i.py |

> 成果包 17 项对照：01 可运行系统✅ / 02 Agent 源码✅ / 03 工具链源码✅ / 05 Docker✅ / 06 API 文档✅ /
> 17 Git 仓库✅；04 微调权重、07 数据集、08 训练配置、09-12 实验/对比、13 用户研究、14 Demo 视频、
> 15 技术报告、16 答辩 PPT 依赖 GPU 训练与人工环节，按规范三级降级线如实标注（见 docs/merge-report）。

## 目录结构（规范 §4.7，不得自创顶层目录）

| 目录 | Owner | 说明 |
|---|---|---|
| `agent/` | A | Planner / Executor / Critic / Rollback / Schema / API 服务 |
| `ai-service/` | B | 8 个图像工具实现（当前 mock，按 Schema 逐个替换） |
| `backend/springboot/` | A | 业务 API 层（规划 09-22，见目录内 README） |
| `frontend/` `speech/` | C | Web 工作台与语音链路（接入指南见 docs/integration-c.md） |
| `training/` | B | 训练脚本（checkpoint/resume/log/config 四件套） |
| `data/` | B | 数据集（artifacts/ 与 runs/ 为运行产物，不入库） |
| `experiments/` | B | 实验记录（N5 五件套） |
| `docs/` | 共同 | 架构 / API 契约 / 接入指南 / 部署 / 周报 |
| `demo/` | C | 截图 / 视频 / 最终版 |

## 三条硬规则（来自总规范，违反不予合并）

1. **接口先行**：任何模块开工前先把 Input/Output/Error/Latency 四件套写进 `agent/schema/`。
2. **模型与 Agent 解耦**：Agent 只依赖工具 HTTP 契约，禁止直接依赖模型权重或推理框架。
3. **一切可恢复**：训练 checkpoint+resume+log+config 四件套；系统任何单点失败不得白屏。

## 对接入口速查

| 你是谁 | 你要接什么 | 去看 |
|---|---|---|
| B（算法） | 把 mock 工具换成真模型 | `docs/integration-b.md` + `agent/schema/T01–T08.json` |
| C（前端） | REST + WebSocket 契约 | `docs/api-spec.md` + `docs/integration-c.md` |
| 新机器部署 | 一键启动 | `docs/deployment.md` + `docker-compose.yml` |
