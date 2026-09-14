# A 组阶段汇报 —— Agent 框架与后端接口

> 汇报人：A（Agent / 后端 / 系统集成）
> 日期：2026-09-10　|　对应 WBS：A1–A5、A7、A8、A9　|　依据：《项目总规范 V1.0》+《任务看板 V1.0》

## 一、一句话结论

**Agent 全链路已可用且全部验证通过**：上传图片 → 一句话指令 → 自动生成执行计划 DAG → 按拓扑执行 8 个工具 → Critic 五维打分 → 不达标自动升档重跑 → 条件回滚（"换回背景 v1 保留现在的光"）→ 运行全程可回放。当前 8 个工具为程序化 mock（丑但能跑），B 组按固定契约逐个替换真模型即可，Agent 代码零改动。

## 二、交付物清单与验收对照

| WBS | 交付物 | 位置 | 验收状态 |
|---|---|---|---|
| A1 | T01–T08 工具 Schema（input/output/error/latency 四件套） | `agent/schema/T01.json–T08.json` + `ENVELOPE.md` | ✅ 8/8 提交，内置校验器，B 组可直接按 Schema 实现 |
| A2 | Planner（指令→JSON DAG） | `agent/planner/`（规则模板 + LLM 插槽） | ✅ 规则版确定性覆盖 5 轮多轮剧本 + 4 个 Demo 场景；LLM 配 `LLM_API_BASE/LLM_MODEL` 即启用，失败自动回落（L2） |
| A3 | Executor 执行引擎 | `agent/executor/engine.py` | ✅ 拓扑分层执行、无依赖节点并发；Run/Pause/Resume/Retry/Skip/Re-run/Cancel 全实现；L1 失败降精度重试 |
| A4 | VLM Critic | `agent/critic/` | ✅ 规则版闭环（五维 + 权重 0.3/0.25/0.2/0.25 + 阈值 85 + 最低维→工具映射自动重跑，上限 2 次 force_pass）；VLM 插槽留好，阈值标定（30 组样本）待 B5 产出后做 |
| A5 | 条件回滚 | `agent/rollback/versions.py` + `/rollback` API | ✅ 节点级版本化 + `current` 指针；"换回背景 A 保留现在的光"实测通过，下游自动重跑 |
| A7 | 异步任务 + 状态推送 | FastAPI asyncio + 事件总线 + WebSocket | ✅ 长任务不阻塞；`/ws/{sid}` 实时推节点状态/Critic/对话消息（前端可直接上屏） |
| A8 | 评测与日志 | `agent/eval/recorder.py` + `/replay` API | ✅ 每次运行落盘 plan.json + events.jsonl + result.json，可完整回放 |
| A9 | 部署 | `docker-compose.yml` + `docker/*.Dockerfile` + `docs/deployment.md` | ✅ 单机/分离/Docker 三形态；ai-service 挂掉自动回落内嵌 mock，Demo 永不白屏 |
| A6 | Spring Boot 业务层 | `backend/springboot/README.md`（规划与模块边界） | ⏸ 待 Maven（见"需要协调"） |

**不在本轮范围**（按 72 小时行动清单节奏）：Spring Boot 编码（09-22 截止）、Critic 阈值人工标定（09-21 前需 B5 样本）。

## 三、怎么验证的（全部实测，非"应该能跑"）

1. **单元/集成测试 37 条全部通过**（`python -m pytest tests/ -q`）：
   - 8 个 Schema 结构与校验器、DAG 拓扑/环检测/引用校验、Planner 5 轮剧本路由；
   - 执行引擎：全链路成功、可重试失败自动降级恢复、不可重试失败不污染已完成节点、暂停/恢复、跳过、取消、手动重跑版本 +1、根节点并发；
   - 条件回滚端到端：两轮换背景后回滚 v1 → 仅阴影/和谐化/导出重跑，光照保留，版本链 [1,2] 正确；
   - API 集成：会话→上传→指令→轮询→产物下载→回滚→回放→WebSocket 快照与实时推送。
2. **真实进程冒烟 9 项全 PASS**（`python scripts/smoke_test.py`），两种形态都验过：
   - 内嵌模式（单进程）；分离模式（ai-service :8100 + agent :8001，HTTP 信封）；
   - 覆盖 Demo01（一句话合成，Critic 87.3 分，自动重规划 1 次）与 Demo04（条件回滚）。
3. **5 轮多轮剧本 100% 通过**（规范 §3.5，实测日志）：
   ```text
   Round 1 [done] 把人物放到咖啡馆            nodes=7 replan=1 critic=87.3
   Round 2 [done] 改成傍晚                    nodes=7 replan=1 critic=87.3
   Round 3 [done] 光线太冷了，暖一点           nodes=5 replan=0 critic=78.1
   Round 4 [done] 阴影轻一点                  nodes=3 replan=0 critic=70.3
   Round 5 [done] 背景换回上一版，保留现在的光线 kind=rollback nodes=7 replan=1 critic=87.3
   ```
   背景版本链 [1, 2]、回滚后当前生效版本正确回指 v1，状态无丢失（对应 E 指标"5 轮不崩溃"与 Demo04）。

## 四、B / C 现在就能开工（不用等对方）

- **B（算法）**：看 `docs/integration-b.md`。8 个工具每个就是一个纯函数（`ai-service/aiservice/<模块>/impl.py`），换模型只改函数体；错误码/超时/输出键名全部由 `agent/schema/T01–T08.json` 规定。程序化阴影会长期保留作 B6-A 基线与降级方案。
- **C（前端/语音）**：看 `docs/api-spec.md` + `docs/integration-c.md`。REST+WS 契约已定：三栏工作台要的节点状态/中间产物 URL/五维分数/对话消息/版本树/回滚接口全部就绪，可用真实后端联调，mock 数据都不用自己造。
- 双方联调回归：`python scripts/smoke_test.py` 一键确认链路健康。

## 五、需要协调 / 我需要的东西

1. **Maven**：A6 Spring Boot 需要本机装 Maven 3.9+（JDK 17 已就绪）。装好后我补工程骨架 + Flyway 迁移脚本。
2. **LLM API（可选但建议）**：给我一个 OpenAI 兼容的 `LLM_API_BASE / LLM_API_KEY / LLM_MODEL`（如组内中转），Planner 即从规则模板升级为真语义解析（规范 A2 的 100 条指令集评测需要它）；不配也不影响演示。
3. **Git 仓库（P1，需你拍板）**：仓库还没 `git init`（在哪建、建远端没定）。确认后我做分支保护 + PR 模板 + LFS 首提交。`.gitignore` 已按规范备好（大文件/产物不入库）。
4. **VLM（A4 后半）**：Critic 的 VLM 打分需要一个多模态模型端点（可与 LLM 同一个）；之后用 30 组样本做阈值标定，写 `agent/critic/threshold.md`。
5. **需要 B 确认**：T01–T08 Schema 里我改了一处——T05 的输入拆为 `foreground_png/background_png/mask_png/light_dir`（合成在 T05 内部完成，输出仍是 `shadow_png + composited_png`），避免"先合成后加影"的排序耦合；规范口径的其余字段不变，请按此实现。
6. **需要 C 确认**：导出多尺寸目前走 T08 单次一档，如需要"一次请求多尺寸打包"我再加端点。

## 六、风险与下一步

| 事项 | 状态/计划 |
|---|---|
| mock 语义与真实模型差异 | 契约层已冻结（信封+Schema），差异只可能出现在工具内部质量；B5 每替换一个工具跑一次冒烟回归 |
| Critic 占位口径 | 09-21 前用 30 组样本标定（已写进 `agent/critic/rule_critic.py` 注释与架构文档） |
| 中文路径跑 Docker | 本机验证用 Python 直跑通过；容器化统一跑在 Linux/英文路径（已在部署文档标注） |
| 下一步（我） | ① Maven 就绪后落 Spring Boot 骨架（A6）；② 100 条测试指令集建表进 `data/instruction/`，跑 Planner 离线评测脚本；③ 组织 09-13 前的 mock 全链路集成日（规范改进建议 2 的"丑但能跑"已提前达成） |

## 七、给评审的 30 秒版本

接口先行完成（A1，8 个 Schema 冻结）；Agent 内核完成（Planner/Executor/Critic/Rollback/回放/推送）；全链路 mock 跑通并有 37 条测试 + 双形态冒烟 + 5 轮剧本 100% 的实测证据；B、C 都已有明确的接入文档和稳定契约，可以完全并行。生死的 09-20"全链路贯通"线在 mock 层面已提前达成，风险集中在算法侧模型替换，已用降级链和固定契约兜住。
