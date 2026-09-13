# A 组 · 文字输入 Agent 完善任务清单 V1.0

> 分析依据：《精编版.pptx》（四层架构）+《项目总规范_多模态图像合成Agent_V1.0.md》A 组职责（A1–A9）+
> 《任务看板_认领表_V1.0.md》+ 现有 `ai-agent/` 源码
> 分析对象：**目前已经实现的文字输入 Agent**（`ai-agent/agent.py` + `run_cli.py` + `tools.py` + `agnes_client.py` + `assets.py`）
> 分析时间：2026-09-09（W1 冻结周）
> 建议 Owner：**A**（赵梓涵）
> ⚠️ 全文使用中文路径仅为文档可读性，**落地代码/文件一律禁止中文文件名**（规范 N7）

---

## 0. 一句话结论

当前文字 Agent 是一条**可跑的线性 function-calling 演示链路**，但尚未达到《总规范》对 Agent 研究点（创新三：执行计划 DAG + 自评 + 条件回滚）的结构要求；
**距离 09-20 生死线 2（语音→Agent→DAG→工具→图像 贯通）约差 8~13 个工作日**的 A 组工作，需立即从"收敛 8 原子工具 Schema"起步。

---

## 1. 现状盘点（已实现，勿重复造轮子）

| 现有文件 | 已实现能力 | 复用价值 |
|---|---|---|
| `ai-agent/agnes_client.py` | OpenAI 兼容客户端：文本对话 + **function calling** + **VLM 看图（critic）** + 文生图；HTTP 429/5xx 自动重试 | ✅ 可直接复用，仅需把硬编码 API Key 改为环境变量 |
| `ai-agent/agent.py` | StudioAgent：`plan()`（文字计划）+ `run()`（function-calling 循环）+ `critic()`（5 维 VLM 评分 + redo）；会话态 `ctx['cur']` 多步串联 | ⚠️ `run()` 是线性循环，需**重构为 DAG 执行器**；`critic()` 需**结构化评分** |
| `ai-agent/tools.py` | 9 个工具注册表：matting / composite / greenscreen / apply_fx / remove_region / gen_image / list_effects / list_assets / video_process；参数错误捕获不抛错 | ⚠️ 属"高层组合工具"，需**收敛/映射到规范 8 原子工具**并补四件套 |
| `ai-agent/assets.py` | 背景/人像素材语义索引（中文标签）+ 语义匹配 `find_background()` | ✅ 直接复用，作 T02/T11 参考 |
| `ai-service/pipeline.py` | `run_pipeline` + `TOOL_SCHEMAS`（仅含 matting/relight/shadow/harmonize/composite 五个，编号与规范冲突、字段缺 error/latency 结构）+ `export_tool_schemas()` | ⚠️ 是"接口先行"的雏形，**需按规范重排为 T01–T08 并补全四件套** |
| `ai-service/src/{matting,relighting,shadow,harmonization,composite}` | B 组落地的真实图像处理能力（BiRefNet+AlphaRefiner、MKL 光影、程序化阴影等） | ✅ 作为 DAG 各节点**真实执行底座** |
| `studio_cli.py` | `task_matting / task_composite / task_greenscreen / task_fx / task_full` | ✅ Agent 工具现绑定这些高层任务；DAG 化时应拆到原子级 |
| `ai-agent/run_cli.py` | 交互 REPL / 单次执行，含 plan→run→critic 打印 | ✅ 文本输入本就是语音降级兜底的 MVP 入口 |

**缺口信号（必须补）**
- `ai-service/api/`、`ai-service/configs/`、`ai-service/tests/` 目录**空** → A1 的 8 工具 Schema 文件尚未提交。
- 顶层**无 git 仓库**（P1/Git、N6 分支保护未落地）。
- `agnes_client.py` 第 14 行**硬编码 API Key**，交付前必须抽离（安全问题）。

---

## 2. 需求对照（A 组应交付 ≠ 现状）

| 规范任务 | 要求 | 现状 | 差距 |
|---|---|---|---|
| **A1** | 8 个原子工具 `agent/schema/T01–T08.json`（input/output/error/latency 四件套），09-09 冻结 | `ai-service/api` 空；`tools.py` 9 个高层工具无四件套 | ❌ 全缺 |
| **A2 Planner** | 指令 → **JSON DAG**；Tool Sel ≥95% / DAG Order ≥95% / Param ≥90%；≤3s | `plan()` 只出**文字计划**；`run()` 走线性 function-calling | ❌ 需重构为 DAG 输出 + 强校验 + 兜底 |
| **A3 Executor** | DAG 拓扑执行；Run/Pause/Resume/Retry/Skip/Re-run；节点幂等；单节点失败不拖累已完成节点 | 无 DAG 模型，线性循环 | ❌ 全缺（最大缺口） |
| **A4 Critic** | 五维结构评分；最低维度→映射工具；阈值 30 组标定；Spearman≥0.6 | `critic()` 单 prompt 文本评分 + 1 次 redo | ⚠️ 缺结构化解析/标定/闭环重跑/可信度评测 |
| **A5 回滚** | 节点级版本化；"换回背景A保留光" 100% 通过；下游自动重跑 | 无版本树；仅内存 `self.history` | ❌ 全缺 |
| **A7 队列** | 异步长任务不阻塞 + 状态推送 | 无 | ❌（若走纯文本 CLI 可后置到 Web） |
| **A8 评测/日志** | 每次运行可回放（输入/DAG/产物/Critic/耗时），可导出数据集 | 无结构化日志 | ❌ 全缺 |
| **A9 部署** | docker-compose 一键启动 ≤15min | 无 | ❌ 后置 |

**三大创新对应检查**：创新三（DAG 多轮稳定）＝A2/A3/A5，**目前近乎空白**；创新一（光影一致性）B 组已交付 T03–T06 底座，A 组需接成原子工具；创新二（语音+空间指代）主责 C 组，但 **Agent 层需预留空间指代输入（点击/框选坐标→T01 refine 提示与 region 参数）**，本次文字版本先保证文本指令全流程，接口向后兼容。

---

## 3. 精细任务清单（按优先级与依赖排序）

> 耗时口径：单人多日（8h/日），未含联调/返工。**P0 为不做则 09-20 生死线 2 必挂。**

### Phase 0 — 工程基座（当天，P0）

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A0.1 | 初始化 git：分支保护(main)/LFS/.gitignore/README 骨架/PR 模板 | 顶层仓库 + 规范 N6 | `main` 不可直接 push，`*.safetensors` 走 LFS | 0.5 天 |
| A0.2 | **抽离硬编码 API Key**：`agnes_client.py` 读环境变量 `AGNES_API_KEY`，提供 `.env.example` | `.env.example` | 代码无明文 Key；本地冒烟可跑 | 0.5 天 |
| A0.3 | 统一 8 原子工具命名，消灭编号冲突（`pipeline.py` 的 T05_shadow vs 规范 T05_shadow_generate；**以规范为准**） | 编号对照表 | 全项目一处编号源 | 0.5 天 |

### Phase 1 — A1：8 原子工具 Schema（今天完成，全组接口冻结，P0）

| # | 工具 | 关键输入 | 关键输出 | 复用底座 |
|---|---|---|---|---|
| T01 | matting | image,[point/box],mode | rgba/alpha/mask | `src/matting/`（含难例档位） |
| T02 | background_generate | prompt,[ref],size,style | bg_png | 现 `gen_image`/B 组扩散 |
| T03 | lighting_estimate | bg_png | sh9,light_dir,color_temp,intensity | **新增**（用现成估计器） |
| T04 | relight | rgba,light_dir,color_temp,intensity | relit_png | `src/relighting/` |
| T05 | shadow_generate | composite,mask,light_dir,geometry | shadow+composited | `src/shadow/`（程序化降级） |
| T06 | harmonize | composite,mask | harmonized | `src/harmonization/` |
| T07 | enhance | image,[depth],strength | enhanced | 现 `apply_fx`/降噪部分 |
| T08 | export | image,format,[size] | file_url,meta | 现合成/保存 |

任务：

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A1.1 | 每个工具写 `agent/schema/T0N.json`，含**统一信封**（tool/version/request_id/inputs/options[quality]）+ 响应（status/outputs/error{code,retryable}/latency_ms/artifacts） | 8 个 json | 能被另一个不看文档的人直接实现；B 组确认可实现 | 1 天 |
| A1.2 | 把现有 `tools.py` 的 9 个高层工具**映射/拆分**到 8 原子（greenscreen/apply_fx/remove_region 如何处理，会议定：并入 T01/T07 或保留为 T0XX 扩展但**不计入核心 8 个**） | 映射表 | 每个原子工具只有一种职责 | 0.5 天 |
| A1.3 | JSON Schema **强校验器**（Python + jsonschema 库）：校验失败→结构化错误→可重试 | `agent/schema/validate.py` | 非法 DAG/参数返回 `E_SCHEMA` + retryable | 0.5 天 |

### Phase 2 — A3：DAG 执行引擎（09-16 前，P0）

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A3.1 | DAG 数据结构 + 拓扑排序（用 `networkx` 或自实现），支持规范 §4.3 的 `nodes[{id,tool,depends_on}]` 与 `edges` | `agent/executor/dag.py` | 能解析规范示例 DAG 并得到正确执行序 | 1 天 |
| A3.2 | 节点状态机 pending→running→done/failed + Run/Pause/**Resume**/Retry/**Skip**/Re-run + **节点级幂等**（同参数重跑不重算） | `agent/executor/engine.py` | 5 轮脚本中途可停可续；单节点失败重试不影响已完成节点 | 1.5 天 |
| A3.3 | 产物注册表：每个节点 `output_artifact` 落 `outputs/agent/` 并记路径（复用现有输出结构） | `agent/executor/artifact.py` | 中间产物可回看（支撑 C5） | 0.5 天 |
| A3.4 | **真实工具适配层**：把 8 原子映射到 `ai-service/src/*` / `studio_cli` 真实执行；草稿级(draft)快速档位 | `agent/executor/adapters/*` | 单节点真实跑通（matting+composite） | 1 天 |
| A3.5 | 单节点超时/OOM → **L1 降级**（重试1次→低精档→程序化方案），节点并行（抠图与背景生成可并行） | 降级钩子 | 任何单点失败不白屏 | 1 天 |

### Phase 3 — A2：Planner（指令→DAG）（09-18 前，P0）

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A2.1 | 重写 SYSTEM_PROMPT：让模型输出**结构化 DAG JSON**（含工具选择、依赖、参数、光向/强度/区域），并以规范 §4.3 示例做 few-shot | `agent/planner/prompt.py` a+ few-shot | 文本/中文指令→合法 DAG | 1 天 |
| A2.2 | **JSON 强校验**（A1.3 校验器）：失败重试 ≤2 → 再失败走**关键词模板 DAG 兜底**（L2 降级） | `agent/planner/fallback.py` | 100 条指令无一次返回非法 DAG | 1.5 天 |
| A2.3 | 争议工具决策落地：composite/greenscreen/apply_fx 等高层工具在 DAG 里如何表达（拆原子 or 保留）；与 B/C 评审定稿 | 决策记录 | 工具收敛到 8（核心） | 0.5 天 |
| A2.4 | 100 条测试指令集 + 指标脚本（Tool Selection / DAG Order / Parameter 三个准确率） | `agent/eval/instructions.json` + `metrics.py` | 有可复测基线与分数 | 1 天 |
| A2.5 | 迭代 prompt/few-shot 达标：**Tool ≥95% / Order ≥95% / Param ≥90%**；优化至 Plan ≤3s | 达标报告 | 三指标齐达标 | 1~2 天 |

### Phase 4 — A4：Critic 结构化（09-23 前，P0）＋ A5：回滚（09-25 前，P0）

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A4.1 | `critic()` 结构化：解析 5 维分数→`{Lighting,Shadow,Color,Edge,Overall}`，定位最低维度→按规范 §3.1 A4 映射表触发工具（Light→T04/Shadow→T05/Color→T06/Edge→T01） | `agent/critic/parse.py` | 输出结构化分 + 触发工具 | 1 天 |
| A4.2 | 阈值用 30 组样本标定（ROC 选阈值，避免疯狂重跑/形同虚设） | `agent/critic/threshold.md` + 标定脚本 | 有数据支撑的阈值 | 1 天 |
| A4.3 | re-plan 闭环：触发后**重跑该节点及下游**，最多重规划 2 次，第 3 次强制 PASS | 回调集成进 executor | 自动修正链路通 | 1 天 |
| A4.4 | Critic 可信度评测：与人工分 **Spearman≥0.6**、自动修正成功率≥70%、误触发≤20% | 评测脚本 | 三指标达标 | 1 天 |
| A5.1 | **节点级版本化**：每个节点存 `version` + `output_artifact`（非整图版本化，才能部分回滚） | `agent/rollback/version.py` | 换权重/换背景→版本号递增 | 1 天 |
| A5.2 | 编辑历史树（分叉）+ 条件回滚语义：恢复某节点旧版本→保留其他当前版→**自动重跑下游** | `agent/rollback/tree.py` | 5 轮剧本第 5 轮"换回背景A保留光" 100% 通过 | 1.5 天 |

### Phase 5 — A8：评测与日志（09-26 前，P1）＋ 端到端延迟达标

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A8.1 | 结构化日志：每次运行记录 输入/DAG/各节点产物/Critic 分/耗时，可回放、可导出为数据集（供后续 few-shot 与报告） | `agent/eval/log.py` | "丑但能跑→录→回放" | 1 天 |
| A8.2 | 5 轮连续编辑完整剧本评测（规范 §3.5） | `agent/eval/multi_round.py` | 5 轮不崩溃、状态不丢失 | 0.5 天 |
| A8.3 | **两级出图**：草稿(draft ≤10s) / 精修(fine ≤30s)；负载/延迟实测并调优 | 延迟报告 | 端到端草稿 ≤10s | 1~2 天 |

### Phase 6 — 集成到 09-20 生死线 2（P0，与 Phase2/3 并线）

| # | 任务 | 交付物 | 验收 | 预计 |
|---|---|---|---|---|
| A-I1 | 把 DAG executor + planner 接入现有 `studio_cli`/`ai-service`，跑通"一句话→DAG→真实工具→图" | `ai-agent` 重构版 | 演示能出真实成片 | 1.5 天（并行） |
| A-I2 | 与 C 组约定"喂给 Agent 的统一输入"（文本/语音转文本/点击坐标），保证语音与空间指代后接入不重构 Agent | 输入契约 | 文本指令全流程可作语音/指代降级兜底 | 0.5 天 |

---

## 4. 预计耗时与排期

### 4.1 各阶段合计（单人 A）

| 阶段 | 内容 | 预计（人日） | 依赖 | 对应截止 |
|---|---|---|---|---|
| Phase 0 | 基座(git/Key/编号) | 0.5–1 | — | 09-09 |
| Phase 1 | A1 工具 Schema | 1–1.5 | Phase0 | **09-09（今天）** |
| Phase 2 | A3 Executor | 3.5–5 | Phase1+A1.3 | 09-16 |
| Phase 3 | A2 Planner | 4–6 | Phase1+A3 | 09-18 |
| Phase 4 | A4 Critic + A5 回滚 | 3.5–5 | A3 | 09-23/09-25 |
| Phase 5 | A8 + 延迟 | 1.5–3 | A3/A4 | 09-26 |
| Phase 6 | 生死线2集成 | 1.5–2 | P2+P3 | 09-20 |

### 4.2 关键路径与总量

- **到 09-20（生死线 2）关键路径** = P0/P1 + P3(Executor) + P4(Planner) + P6(集成) ≈ **8–13 个工作日**。
  09-09→09-20 约 9 个工作日，**排期极紧、无缓冲**；必须 09-09 当天冻结 8 工具 Schema（A1），否则整体后移。
- **到 09-30（封版）含 A4/A5/A8/降级** ≈ 总计 **14–22 个人日**。
- 任务 A6（Spring Boot API）、A7（任务队列）、A9（Docker 部署）属 P1，**与文字 Agent 的 P0 主线并行**，不在本清单主路径，但 A9 是 09-29 截止，需预留 1–2 天收尾。

### 4.3 并行化建议（压缩日历时间）

- A1 Schema（今天）→ 立即交 B/C 评审，**不等做完**就启动 A3。
- A3（Executor）与 A4/A5 数据模型**一次设计到位**（DAG 节点模型同时支撑执行、critic 节点定位、版本化），避免二次返工。
- Critic 可视化/日志（C5/C2 依赖）与 A3.3 产物注册表一起做。
- 语音/空间指代由 C 组并行，A 组只保证**输入契约稳定**。

---

## 5. 关键风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| 09-20 排期无缓冲，A1 延迟即全盘后移 | 中 | 高 | **今天冻结 8 工具 Schema**；A3/A2 用 mock DAG 先跑通再换真工具 |
| DAG 重构破坏现有可演示的线性链路 | 高 | 中 | 分阶段：先加"执行器壳"包住现有 `run()`，验证 DAG 版不回归再替换 |
| 工具数超 8 导致 Planner 准确率下降 | 中 | 高 | 严格收敛到 8 原子；greenscreen/apply_fx 并入 T01/T07/T08，不进核心集 |
| hard-case 抠图在 Agent 中重跑导致超 10s | 中 | 中 | draft 档走低分辨率；抠图与背景生成并行 |
| agnes API 在演示环境不可用 | 低 | 高 | 保留 `run_cli.py` 纯文本兜底 + 模板 DAG 降级（规范 L2） |
| **Key 硬编码泄露** | 中 | 高 | P0 就改环境变量（已列入 Phase0） |

---

## 6. 安全与合规提示（强制）

1. `ai-service/...model.safetensors`（444MB）与 `refiner/best.pt`（40MB）**禁止** `git add` 直接入库 → 走 LFS 或只提交路径/元信息（规范 N7 / 6.3）。
2. `agnes_client.py` 硬编码 Key → 立即环境变量化。
3. 交付前检查：代码库无 `sk-` 明文、无 >50MB 文件入库。

---

## 7. 完成标准（本清单闭环定义）

- [ ] `agent/schema/T01–T08.json` 8 份全提交并通过 B/C 评审（09-09）
- [ ] 一句话中文指令 → 强校验后的合法 DAG → 真实工具执行 → 成片（09-20）
- [ ] Tool Selection≥95% / DAG Order≥95% / Parameter≥90%（09-18 前基线 + 达标）
- [ ] Critic 结构化评分 + 最低维度映射重跑 + Spearman≥0.6（09-23）
- [ ] 5 轮剧本 + "换回背景A保留光" 条件回滚 100% 通过（09-25）
- [ ] 草稿端到端 ≤10s / 精修 ≤30s；运行可回放日志（09-26）

> 复审建议：以上 4.2 排期关键路径已计入该阶段所有 P0，**请明天（09-10 站会）确认 A1 Schema 冻结与是否接受 09-20 无缓冲风险**；若可加人手（B/C 协助回滚/日志），关键路径可压缩至 8 个工作日。