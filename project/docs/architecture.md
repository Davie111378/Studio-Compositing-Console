# 系统架构说明 V1.0

## 1. 分层（对应总规范 §4.1 六层中的 L2/L3/L4）

```text
C 组前端/语音                A 组 Agent 服务                B 组 ai-service
┌──────────────┐   REST+WS   ┌──────────────────────┐  HTTP 信封  ┌─────────────────┐
│ Home/Workbench│ ─────────> │  FastAPI (agent/server)│ ─────────> │ FastAPI(aiservice)│
│ DAG 可视化    │ <───────── │  Planner               │            │  T01 matting     │
│ 语音链路      │  实时事件   │  ┌────────────────┐   │            │  T02 background  │
└──────────────┘             │  │ RulePlanner(兜底)│   │            │  T03/T04 light   │
                             │  │ LLMPlanner(可选) │   │            │  T05 shadow      │
                             │  └────────────────┘   │            │  T06/T07 harmo   │
                             │  Executor(DAG 引擎)    │            │  T08 export      │
                             │  Critic(五维+重规划)   │            │ (mock -> 真模型) │
                             │  Rollback(版本树)      │            └─────────────────┘
                             │  SessionManager/Recorder│
                             └──────────────────────┘
```

三种部署形态（`agent/config.py` 环境变量切换，代码不变）：

| 形态 | 触发条件 | 场景 |
|---|---|---|
| 内嵌 | `AI_SERVICE_URL` 为空 | 单机 Demo / 开发默认；Agent 经 ASGI 进程内调 ai-service，仍走完整 HTTP 信封 |
| 分离 | `AI_SERVICE_URL=http://...:8100` | 生产形态；ai-service 独立进程/容器（B 组换真模型用这种） |
| 分离+兜底 | 分离 + `AI_FALLBACK_TO_EMBEDDED=1`（默认开） | ai-service 挂了自动回落内嵌 mock，Demo 永不白屏（规范 5.7） |

## 2. 一次指令的生命周期

```text
POST /instructions {text}
  → Planner(规则/LLM) 产出 PlanDAG ──validate_dag──> Executor.start
      → wave 并发执行就绪节点（matting ∥ background_generate 同 wave）
      → 每节点: 解析@引用 → Schema 校验 → 调 provider(信封) → 产物落盘 → 事件广播
      → L1 降级: retryable 失败 → 重试并降 quality 档位
  → 全部完成 → Critic 五维打分
      → overall ≥ 阈值 → done
      → 低于阈值且 replan<2 → 最低维映射工具升档重跑(下游连带) → 再评
      → 重规划耗尽 → force_pass（提示用户）
  → 版本树登记（session.node_versions + current 指针）
  → result.json + events.jsonl 落盘（可回放）
```

## 3. 关键设计决策

| 决策 | 理由 |
|---|---|
| Agent 与工具间只有 HTTP 信封（含内嵌模式） | N2 解耦；B 组换模型 Agent 零改动；内嵌也走同一契约，形态切换无代码差异 |
| 节点级版本化（role + version），非整图版本化 | 条件回滚的前提（规范 4.4）；`current` 指针支持"换回 v1 保留光" |
| 计划历史保留 30 份，回滚选"最近含目标角色的计划" | 小步修改轮（只重跑阴影）不含背景节点，直接用 last_plan 会找不到回滚目标 |
| 显式 skip 优先于 Critic | 用户跳过是明确意图；Critic 不覆盖（跳过后无完整产物则跳过评审） |
| Critic 阈值/权重为占位口径 | 规范 6.1-1：09-21 前用 30 组样本标定后更新 `agent/critic/` |
| 会话/运行状态：内存索引 + JSON 落盘 | 实习体量够用、零依赖、可回放；Spring Boot + MySQL 是后续业务层（A6）的事 |
| 规则 Planner 是默认而非 LLM | 离线确定性可测、零成本、永不阻塞；LLM 配好即在 auto 模式自动接管、失败回落 |

## 4. 与总规范交付物的对应

| 规范 WBS | 本仓库落点 | 状态 |
|---|---|---|
| A1 Schema | `agent/schema/T01–T08.json` + `ENVELOPE.md` + 校验器 | 完成 |
| A2 Planner | `agent/planner/`（规则模板覆盖 5 轮剧本/4 Demo；LLM 插槽） | mock 完成，LLM 待配 |
| A3 Executor | `agent/executor/engine.py`（Run/Pause/Resume/Retry/Skip/Rerun/Cancel） | 完成 |
| A4 Critic | `agent/critic/`（规则版 + VLM 插槽 + 维度→工具映射） | 规则完成，VLM 待标定 |
| A5 Rollback | `agent/rollback/versions.py` + `/rollback` API | 完成 |
| A6 Spring Boot | `backend/springboot/README.md`（规划：用户/项目/任务/产物四类业务接口包住 Agent 服务） | 待做（需 Maven） |
| A7 异步队列 | FastAPI asyncio + 事件总线 + WS 推送（单机形态）；compose 分容器 | 完成（单机版） |
| A8 评测回放 | `agent/eval/recorder.py` + `/replay` API | 完成 |
| A9 部署 | `docker-compose.yml` + `docs/deployment.md` | 完成 |
