# 多模态图像合成 Agent —— 项目仓库

> 面向真实感图像合成的图像抠取与光影一致性融合方法研究——多模态交互式 Agent 系统
> 团队：A（Agent / 后端 / 系统集成）· B（算法 / 训练 / 实验）· C（前端 / 语音 / Demo）
> 总规范见《项目总规范_多模态图像合成Agent_V1.0.md》（仓库外 `生产实习/` 目录）

## 当前状态（B 组算法引擎已合并）

**全链路已可用**：上传图片 → 一句话指令 → Planner 生成 DAG → Executor 按拓扑执行 T01–T08 工具链 → Critic 五维打分 → 不达标自动重跑 → 结果导出。

**2026-09-13：B 组 Studio-Compositing-Console 算法引擎已合并进 `ai-service/bsrc/`**（上游 b3107bc）。T03 光照估计（Lambert 半球）、T04 方向性重打光、T05 程序化接触阴影、T06 harmonize-v2 和谐化、T07 深度增强 + 特效链已切换为 B 组真实算法；T01 离线档升级为 GrabCut 自动分割，torch + `BIREFNET_WEIGHT_PATH` 就绪后自动切换真实 BiRefNet。缺 opencv 时自动回落原 mock 引擎，**HTTP 信封契约与产物命名完全不变**（`IMC_ENGINE` 可强制切换，验收脚本 `python scripts/bsrc_engine_check.py`）。

```text
上传 → Planner(规则兜底/LLM) → DAG → Executor(暂停/恢复/重试/跳过/重跑)
     → ai-service 8 工具(B 组算法引擎 bsrc + mock 回落) → Critic(五维+自动重规划) → 条件回滚(节点级版本)
```

## 快速开始

```bash
pip install -r requirements.txt

# 方式一：一体化启动（Agent 服务内嵌 mock 工具，单进程）
python scripts/run_agent.py

# 方式二：分离启动（贴近生产形态，B 组替换真模型时用这种）
python scripts/run_ai_service.py    # 终端 1，默认 :8100
python scripts/run_agent.py         # 终端 2，设置 AI_SERVICE_URL=http://127.0.0.1:8100

# 冒烟验证（对运行中的服务跑全链路）
python scripts/smoke_test.py
python -m pytest tests/ -q          # 单元测试
```

启动后：交互式 API 文档 http://127.0.0.1:8000/docs

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
