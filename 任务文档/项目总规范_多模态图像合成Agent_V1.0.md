# 多模态图像合成 Agent 项目总规范 V1.0

> **项目全称**：面向真实感图像合成的图像抠取与光影一致性融合方法研究——多模态交互式 Agent 系统
> **适用周期**：2026-09-08 — 2026-10-10（有效工期 24 个工作日，10/01–10/07 国庆不计入）
> **团队规模**：3 人（A / B / C 三个 Owner 制）
> **文档性质**：项目总体架构 + 分工 + 研究实验规范 + 开发部署规范 + 验收标准，五合一
> **配套文件**：`任务看板_认领表_V1.0.md`（WBS 认领用）、`项目总规范_多模态图像合成Agent_V1.0.html`（网页版）

---

## 0. 如何使用本文档

| 你是谁 | 你该读哪几节 | 你今天要做的第一件事 |
|---|---|---|
| **A：Agent / 后端 / 系统集成** | §3.1 任务清单 → §4.3 工具 Schema → §4.4 DAG 与回滚 → §5.4 Agent 实现 | 建 `agent/schema/` 目录，把 T01–T08 八个工具的 `input_schema / output_schema` 写成 JSON 文件提交 |
| **B：算法 / 训练 / 实验** | §3.2 任务清单 → §5.1 抠取 → §5.3 光影 → §1.5 训练规范 → §2.2 指标 | 跑通 BiRefNet 推理冒烟，输出第一张 alpha 图并存档到 `experiments/baseline/` |
| **C：前端 / 语音 / Demo** | §3.3 任务清单 → §4.2 数据流 → §5.5 语音 → §5.6 前端 | 搭出首页"上传 + 一句话"双入口静态原型，接 mock 数据跑通界面 |
| **指导教师 / 评审** | §2 预期效果与指标 → §6 改进建议与风险 | 确认 §2.2 指标口径与 §2.4 三条生死线 |

**三条硬规则（写在最前面，因为最容易翻车）：**

1. **接口先行**：任何模块开工前，先把 `Input / Output / Error / Latency` 四项写进 `agent/schema/`，评审通过后才允许写业务代码。
2. **训练与主线解耦**：训练失败时生产环境自动回落到 Base Model，Demo 永远不能因为训练挂掉而不可演示。
3. **一切可恢复**：训练必须 checkpoint + resume + log + config 四件套齐全；租卡只是算力，不能成为项目状态。

---

# 一、总体规范与标准

## 1.1 项目定义

不是"上传图片 → AI 生成图片"的工具，而是一个**以执行计划 DAG 为中心、可自评修正、可条件回滚的多模态图像合成 Agent 系统**。

主链路（与开题报告一致）：

```text
高精度抠取 → 背景生成 → 光影一致性融合 → 结果自评修正 → 语音反馈
```

四个研究方向锚点（不得漂移）：

| 方向 | 具体研究问题 |
|---|---|
| R1 抠取 | 透明 / 半透明区域 alpha 估计（玻璃、薄纱、发丝、复杂背景） |
| R2 光影 | 光照 + 阴影 + 几何联合一致的合成（消除"贴纸感"） |
| R3 Agent | 基于执行计划 DAG 的稳定多轮交互（缓解多轮编辑退化） |
| R4 交互 | 语音 + 空间指代（点击/框选）的中文多模态交互范式 |

## 1.2 开发规范（8 条，强制执行）

### N1 接口先行

所有模块（含内部脚本）先定四件套，再开发：

```text
Input   —— 明确的 JSON Schema / 文件路径 / 张量形状
Output  —— 明确的 JSON Schema / 产物路径
Error   —— 错误码 + 可重试性标记（retryable: true/false）
Latency —— P50 / P95 目标，超时即降级
```

### N2 模型与 Agent 解耦

**禁止** Agent 直接依赖具体模型权重或具体推理框架。强制三层：

```text
Agent  →  Tool Interface (JSON Schema)  →  Model / Script
```

换模型只改 Tool 内部实现，Agent 代码零改动。违反此条的代码不予合并。

### N3 训练与主线解耦

```text
Training → Checkpoint → Validation → Model Registry → Production
                                                   ↘ (失败) Base Model
```

Model Registry 用 `model_registry.yaml` 维护，字段：`name / version / path / metrics / status(staging|prod) / fallback`。

### N4 所有训练必须可恢复

四件套缺一不可：`checkpoint`（每 20–30 分钟）+ `resume` + `train.log` + `config.yaml`。
单次租卡任务 **≤ 24 小时**，每个 epoch 做一次 evaluation，最终保留 best + last 两个 checkpoint。

### N5 实验可复现

每个实验目录必须包含：

```text
experiments/<exp_id>/
├── config.yaml        # 完整超参
├── train.log          # 原始日志
├── eval.csv           # 逐样本指标
├── metrics.json       # 汇总指标
├── checkpoints/       # best / last
├── visual/            # 对比图（before / after / gt）
└── README.md          # 一句话结论 + 失败案例
```

`exp_id` 命名：`YYYYMMDD-<方向>-<序号>`，例：`20260912-matting-01`。

### N6 代码与分支规范

| 分支 | 用途 | 保护级别 |
|---|---|---|
| `main` | 可演示版本，任何时刻都能启动 | 受保护，仅 PR 合入，需 1 人 review |
| `dev` | 集成分支 | 需冒烟通过 |
| `feat/<wbs>-<简述>` | 个人开发 | 例：`feat/B3-matting-finetune` |
| `exp/<方向>-<序号>` | 实验脚本，不保证可复用 | 允许不合并 |

提交信息：`[WBS] 动词: 简述`，例：`[B3] feat: 添加透明物体难例集`。
大文件（权重、数据集）**必须**走 Git LFS 或只存 `data/` 外的对象存储路径，**禁止**直接 `git add` 超过 50MB 的文件。

### N7 目录与命名规范

- 目录结构固定为 §4.7，任何人不得自创顶层目录。
- Python 文件 / 变量：`snake_case`；前端组件：`PascalCase`；常量与错误码：`UPPER_SNAKE_CASE`。
- 图片产物命名：`<task_id>_<node_id>_<version>.png`，例：`t20260910a_n04_v3.png`。
- **全项目禁用中文路径与中文文件名**（避免 Windows / Linux 跨端编码问题，见 §6.3 环境新坑）。

### N8 文档与交付规范

- 每个 WBS 任务完成时必须更新三处：任务看板状态、对应 `docs/` 文档、`experiments/` 记录（若涉及实验）。
- 每周日 21:00 前提交《周进展简报》到 `docs/weekly/`，格式固定：本周完成 / 下周计划 / 风险 / 需要协助。

## 1.3 研究与实验规范

### E1 数据集划分

```text
train / validation / test / hard-case test
```

四个集合**必须物理隔离**，禁止样本泄漏。hard-case test 单独建集：

| 难例类别 | 目标数量 | 说明 |
|---|---:|---|
| Hair（发丝） | ≥ 30 | 细碎边缘 |
| Glass（玻璃） | ≥ 30 | 高透明 + 折射 |
| Veil（薄纱） | ≥ 20 | 大面积半透明 |
| Transparent（透明体） | ≥ 20 | 塑料瓶、水杯 |
| Semi-transparent（半透明） | ≥ 20 | 烟雾、水花 |
| Fine Edge（细边缘） | ≥ 20 | 羽毛、树枝 |
| Complex Background（复杂背景） | ≥ 30 | 前景背景颜色相近 |

### E2 基线必须先行

**任何"我们的方法"之前，必须先有基线数据。** 禁止在没有 baseline 的情况下汇报改进。

### E3 指标口径统一

- 抠取：Composition-1k 协议的 **SAD / MSE / Grad / Conn**（SAD、MSE、Grad 越低越好，Conn 越低越好）。
- 光影：主观 5 分制（光照一致性 / 阴影一致性 / 色彩一致性 / 整体自然度），**双盲、随机顺序、n ≥ 10**。
- Agent：Tool Selection / DAG Order / Parameter / Spatial Reference 四个准确率，测试集 ≥ 100 条指令。

### E4 失败案例必须留档

`experiments/failure/` 目录，每个案例一张对比图 + 一句话归因。**这是论文与答辩最有价值的素材，不要只存成功案例。**

## 1.4 租卡与算力规范

```text
本地实验室服务器 A6000
    ↓  1/10 数据冒烟（≤30 分钟）
Docker 镜像构建（锁定 torch / cuda / 依赖版本）
    ↓
租卡平台正式训练（单次 ≤ 24h）
    ↓  每 20–30 分钟 checkpoint
    ↓  每 epoch evaluation
下载 best checkpoint → 本地验证 → Model Registry
```

**租卡前检查清单（缺一项不准下单）**：

- [ ] 冒烟脚本在本地跑通且 loss 下降
- [ ] Docker 镜像可离线构建，依赖版本锁死（写 `requirements-lock.txt`）
- [ ] `config.yaml` 里 `resume_from` 字段已配置
- [ ] 产物自动同步到对象存储 / 网盘的脚本已验证
- [ ] 预算与时长上限已确认（建议单次 ≤ 24h，总预算提前报备）

## 1.5 例会与节点机制

| 机制 | 频率 | 内容 |
|---|---|---|
| 站会 | 每日 10 分钟（异步文字即可） | 昨天做了什么 / 今天做什么 / 是否被阻塞 |
| 集成日 | 每周三、周日 | 三人代码合并到 `dev`，跑通端到端冒烟 |
| 周会 | 每周日 21:00 | 对照 §3 任务看板逐条核销，更新风险表 |
| 封版 | 09-30 | Demo Freeze，之后只修 bug 不增功能 |

---

# 二、预期达成效果与成果指标

## 2.1 最终系统形态（用户视角）

用户完成一次完整操作：

> 上传人物照片 → 说："把这个人物放进傍晚的咖啡馆，光从左边照过来，阴影自然一点。"

系统自动执行：

```text
语音识别 → 理解对象 → 消解空间指代 → 生成执行计划 DAG
  → 人物抠取 → 背景生成 → 环境光分析 → 人物重打光
  → 接触阴影 → 色彩/景深和谐化 → VLM Critic 打分
  → 发现问题 → 自动修正 → 输出结果 → 语音播报
```

**开发者全程不需要手动介入。**

## 2.2 量化验收指标（写进 README，逐条核销）

### A. Agent 能力（≥100 条测试指令集）

| 指标 | 目标值 | 测量方式 |
|---|---:|---|
| Tool Selection（工具选择准确率） | **≥ 95%** | 100 条指令，人工标注期望工具集，比对 |
| DAG Order（依赖顺序正确率） | **≥ 95%** | 拓扑排序正确性 + 必要依赖不缺失 |
| Parameter（参数解析准确率） | **≥ 90%** | 关键参数（风格、光向、强度、区域）抽取正确 |
| Spatial Reference（空间指代准确率） | **≥ 80%** | 点击/框选坐标 + 语音"左边/这里"联合消解 |
| 计划生成延迟 | **≤ 3 s** | 语音结束到 Plan 展示 |
| 端到端出图（草稿级） | **≤ 10 s** | 语音结束到首张结果图 |

### B. 抠取算法

| 指标 | 要求 |
|---|---|
| Baseline（BiRefNet / SAM 2 组合） | 必须给出 SAD / MSE / Grad / Conn 四项数值 |
| Fine-tuned | 同四项数值 + 相对 baseline 提升幅度 |
| 目标（建议口径） | 整体 SAD 相对 baseline **下降 ≥ 10%**；难例子集（透明/半透明）**下降 ≥ 15%** |
| 难例集 | §1.3 E1 七类，每类达标数量 + 单类指标 |
| 视觉对比 | ≥ 20 组 before/after 对比图 |

### C. 光影一致性

四组必做实验：

| 组 | 配置 |
|---|---|
| **A** | Direct Composite（直接合成） |
| **B** | Composite + Harmonization |
| **C** | Composite + Relighting |
| **D（Ours）** | Composite + Relighting + Shadow + Harmonization |

评价维度：**光照一致性 / 阴影一致性 / 色彩一致性 / 整体自然度**（5 分制，双盲）。
目标：D 组整体自然度 **≥ 4.0**，且相对 A 组 **提升 ≥ 0.5 分**。
案例量：**20–50 组代表性案例**，含 ≥ 5 组失败案例分析。

### D. Critic 可信度

| 指标 | 目标 |
|---|---|
| Critic 打分与人工主观分的 Spearman 相关系数 | **≥ 0.6** |
| 自动修正成功率（触发修正后指标提升的比例） | **≥ 70%** |
| 误触发率（人工认为无需修正却触发） | **≤ 20%** |

### E. 系统性能与交互

| 指标 | 目标 |
|---|---|
| ASR 中文识别（安静环境字准） | ≥ 95% |
| ASR 端到端延迟（含 VAD） | ≤ 800 ms |
| 5 轮连续编辑不崩溃、状态不丢失 | 100% 通过 |
| 条件回滚（"换回背景 A，保留现在的光"） | 100% 通过，且后续节点自动重跑 |
| 精修级出图延迟 | ≤ 30 s |

## 2.3 成果包清单（10-10 交付，17 项）

```text
01  可运行系统（Web + Agent + AI Service 一键启动）
02  Agent 服务源码（Planner / Executor / Critic / DAG / Rollback）
03  图像工具链源码（T01–T08）
04  微调模型权重（含 Model Registry 记录）
05  Docker 环境（推理镜像 + 训练镜像）
06  API 文档（OpenAPI / Markdown）
07  数据集说明（来源、规模、划分、授权）
08  训练配置（全部 config.yaml）
09  实验指标（eval.csv + metrics.json）
10  Baseline 对比表（抠取 + 光影）
11  视觉对比图（≥20 组抠取 + 20–50 组光影）
12  失败案例分析（≥5 组）
13  用户测试结果（n ≥ 10，含原始问卷）
14  Demo 视频（3 分钟）
15  技术报告（含完整实验记录）
16  答辩 PPT
17  Git 仓库（含完整提交历史与 README）
```

## 2.4 三条生死线

| 日期 | 必须达到的状态 | 没达标的后果 |
|---|---|---|
| **09-15** | 训练模型 + Baseline + 指标表 + 视觉对比图 | 项目退化为纯工程 Demo，学术价值归零 |
| **09-20** | 语音 → Agent → DAG → 工具 → 图像 完整跑通 | 交互创新点无法验证 |
| **09-30** | 系统封版 + 实验完成 + 录屏完成 + 用户测试完成 | 10-08 后无缓冲，交付风险极高 |

## 2.5 阶段性效果对照表

| 阶段 | 日期 | 阶段末必须能演示什么 | 不能演示就说明什么 |
|---|---|---|---|
| W1 冻结 | 09-08 ~ 09-09 | 单张图跑通 BiRefNet 抠图并看到 alpha 图；8 个工具 Schema 全部提交 | 环境 / 数据未就绪，全员阻塞 |
| W2 训练 | 09-10 ~ 09-12 | 第一次正式训练启动并产出 checkpoint | 训练链路不通，后面全盘延后 |
| **W2 成果** | **09-15** | **抠取微调前后对比表 + 对比图** | **生死线 1** |
| W3 贯通 | 09-16 ~ 09-20 | 说一句话 → 出图（功能可以不完美，链路必须通） | **生死线 2** |
| W4 智能 | 09-21 ~ 09-25 | Critic 自动发现问题并修正；5 轮多轮；条件回滚 | Agent 研究点落空 |
| W5 评测 | 09-26 ~ 09-29 | 四组光影实验完成 + 用户研究 n≥10 完成 | 无量化证据 |
| **W5 封版** | **09-30** | **完整 Demo 录屏成功** | **生死线 3** |
| W6 交付 | 10-08 ~ 10-10 | 报告 + PPT + 视频 + 仓库归档 | — |

---

# 三、任务清单（WBS）

> **认领规则**：每个任务必须有且仅有一个 Owner。跨组依赖在任务卡"依赖"列标明，依赖方负责催办，不是等。
> 完整可勾选版本见 `任务看板_认领表_V1.0.md`。
> 优先级：**P0 = 不做项目失败；P1 = 不做验收扣分；P2 = 有余力再做。**

## 3.1 A 组：Agent / 后端 / 系统集成

**Owner 核心责任**：Planner、Executor、Critic、DAG、Rollback、API、任务队列、系统集成、部署。

| WBS | 任务 | 交付物 | 验收标准 | 截止 | 优先级 |
|---|---|---|---|---|---|
| **A1** | 8 个原子工具接口定义 | `agent/schema/T01–T08.json`（含 input/output/error/latency 四件套） | 8 个 Schema 全部评审通过；能被 B 组按 Schema 直接实现 | 09-09 | P0 |
| **A2** | Planner（指令 → JSON DAG） | `agent/planner/` + prompt + few-shot 示例 | 100 条测试集上 Tool Selection ≥95%、DAG Order ≥95%、Parameter ≥90% | 09-18 | P0 |
| **A3** | Executor（DAG 执行引擎） | `agent/executor/` | 支持 Run / Pause / Resume / Retry / Skip / Re-run；单节点失败可重试且不影响已完成节点 | 09-16 | P0 |
| **A4** | VLM Critic | `agent/critic/` + 评分 rubric | 五维打分（Lighting/Shadow/Color/Edge/Overall）；与人工分 Spearman ≥0.6；能定位最低分维度 | 09-23 | P0 |
| **A5** | 版本 / DAG / 条件回滚 | `agent/rollback/` + 状态树 | "换回背景 A 但保留现在的光" 场景 100% 通过，后续节点自动重跑 | 09-25 | P0 |
| **A6** | Backend API（Spring Boot） | `backend/springboot/` | 用户/项目/任务/产物四类接口；OpenAPI 文档；Flyway 迁移脚本 | 09-22 | P1 |
| **A7** | 异步任务队列 | Job Queue + 状态轮询 / WebSocket 推送 | 长任务不阻塞；前端能实时看到节点状态 | 09-20 | P1 |
| **A8** | 评测与日志体系 | `agent/eval/` + 结构化日志 | 每次运行可回放（输入、DAG、各节点产物、Critic 分数、耗时） | 09-26 | P1 |
| **A9** | 部署（Docker Compose 一键启动） | `docker-compose.yml` + 部署文档 | 新机器 15 分钟内启动完整系统 | 09-29 | P1 |

### A1 详解：8 个原子工具（固定，不得随意扩张）

| ID | 工具 | 作用 | 关键输入 | 关键输出 |
|---|---|---|---|---|
| T01 | `matting` | 前景抠取 | image, [point/box prompt], mode(auto/trimap) | rgba_png, alpha_png, mask_png |
| T02 | `background_generate` | 背景生成 | prompt, [ref_image], size, style | bg_png |
| T03 | `lighting_estimate` | 环境光分析 | bg_png | sh_coeff(9), light_dir(θ,φ), color_temp, intensity |
| T04 | `relight` | 前景重打光 | rgba_png, light_dir, color_temp, intensity | relit_png |
| T05 | `shadow_generate` | 接触阴影 | composite_png, mask_png, light_dir, geometry | shadow_png, composited_png |
| T06 | `harmonize` | 前后景和谐化 | composite_png, mask_png | harmonized_png |
| T07 | `enhance` | 景深/噪点/细节增强 | image, [depth_map], strength | enhanced_png |
| T08 | `export` | 最终输出 | image, format, [size] | file_url, meta |

> **工具不宜超过 8 个。** 每增加一个工具，Planner 准确率下降且测试集要同步扩，收益递减。

### A4 详解：Critic 决策逻辑

```text
打分 → Overall ≥ Threshold ? → PASS → 输出
                    ↓ no
              找最低分维度 → 映射到对应 Tool → 重跑该节点及下游 → 重新评估
              （最多重规划 2 次，第 3 次强制 PASS 并提示用户）
```

维度到工具的映射（固定）：

| 最低维度 | 触发工具 |
|---|---|
| Lighting | T04 `relight` |
| Shadow | T05 `shadow_generate` |
| Color | T06 `harmonize` |
| Edge | T01 `matting`（重抠，提高精度档位） |

Threshold **先用 30 组样本人工标定**，不得拍脑袋（见 §6.1 改进建议 1）。

## 3.2 B 组：算法 / 训练 / 实验

**Owner 核心责任**：Matting、Fine-tuning、Lighting、Relighting、Shadow、Harmonization、Evaluation、Training。

| WBS | 任务 | 交付物 | 验收标准 | 截止 | 优先级 |
|---|---|---|---|---|---|
| **B1** | 环境与基线跑通 | `experiments/baseline/` 首张 alpha 图 + 指标 | BiRefNet 推理跑通；SAM 2 交互分割跑通；记录显存/耗时 | 09-09 | P0 |
| **B2** | 数据集与难例集构建 | `data/matting/` 四集划分 + 难例清单 | train/val/test 物理隔离；七类难例达标数量（§1.3 E1） | 09-09 | P0 |
| **B3** | 抠取域微调训练 | checkpoints + config + log | 支持断点续训；单次 ≤24h；产出 best/last | 09-12 | P0 |
| **B4** | 抠取正式实验表 | SAD/MSE/Grad/Conn 对比表 + 视觉对比 | Baseline vs Fine-tuned 完整四指标；≥20 组对比图；难例子集单独出数 | **09-15** | P0 |
| **B5** | 光影一致性 Pipeline | `ai-service/` 下 lighting/relight/shadow/harmonize 串联 | 单张图跑通全链路，中间态可导出 | 09-20 | P0 |
| **B6** | 光影四组实验 | A/B/C/D 四组结果 + 20–50 组案例 | 四组齐全；主观四维评分；D 组相对 A 组提升 ≥0.5 分 | 09-27 | P0 |
| **B7** | LoRA 训练 | 1 个稳定风格 LoRA | 风格一致性人工通过；非核心，时间不足可砍 | 09-26 | P1 |
| **B8** | 消融与失败案例 | `experiments/failure/` | ≥5 组失败案例 + 归因；至少 1 组消融（去掉 shadow / 去掉 relight） | 09-29 | P1 |

### B4 必须产出的正式实验表（模板）

| Model | SAD ↓ | MSE ↓ | Grad ↓ | Conn ↓ |
|---|---:|---:|---:|---:|
| Baseline（BiRefNet） | | | | |
| Fine-tuned | | | | |
| **Improvement** | | | | |

配套必须保存：`train.log`、`eval.csv`、`config.yaml`、`checkpoint`、视觉对比图。

### B6 四组实验定义（严格按此执行，不得中途改口径）

| 组 | 流程 |
|---|---|
| A | 抠图 → 直接合成 |
| B | 抠图 → 合成 → T06 Harmonization |
| C | 抠图 → 合成 → T03+T04 Relighting |
| D | 抠图 → 合成 → T03+T04 → T05 Shadow → T06 Harmonization |

## 3.3 C 组：前端 / 语音 / Demo

**Owner 核心责任**：Web UI、Canvas、ASR、TTS、Agent 交互、DAG 可视化、历史、回滚 UI、Demo。

| WBS | 任务 | 交付物 | 验收标准 | 截止 | 优先级 |
|---|---|---|---|---|---|
| **C1** | 首页（上传 + 一句话双入口） | `frontend/pages/Home` | 首屏只有"上传图片"和"说一句话"两个主入口，无传统工具栏 | 09-14 | P0 |
| **C2** | 三栏工作台 | `frontend/pages/Workbench` | 左素材 / 中 Canvas / 右 Agent（对话+Plan+Critic+状态）；底部 DAG 区 | 09-20 | P0 |
| **C3** | 语音链路（ASR + VAD + TTS） | `speech/` | 端到端 ≤800ms；字准 ≥95%；全本地部署；界面能显示"你说了什么 / AI 在做什么" | 09-18 | P0 |
| **C4** | 多轮交互 | 会话状态 + 轮次管理 | 5 轮连续编辑完整剧本 100% 通过（见 §3.5） | 09-24 | P0 |
| **C5** | DAG 可视化 | 节点图组件 | 实时显示节点状态（pending/running/done/failed）；可点击节点查看中间产物 | 09-22 | P0 |
| **C6** | 历史与回滚 UI | 版本树 + 条件回滚交互 | 用户能选中历史节点回滚；支持"只回滚某一步，保留后续" | 09-25 | P0 |
| **C7** | 导出 | 导出组件 | PNG / JPG 多尺寸；带元数据 | 09-26 | P1 |
| **C8** | Demo 录制与素材 | 3 分钟视频 + 截图 | 4 个 Demo 剧本全部录成（§3.4）；截图归档 `demo/screenshots/` | **09-30** | P0 |

### C2 工作台布局（固定）

```text
┌──────────┬──────────────────┬────────────┐
│  素材     │      Canvas      │   Agent    │
│  原图     │                  │  对话       │
│  背景     │    最终合成       │  Plan      │
│  参考图   │                  │  Critic    │
│  Mask    │                  │  状态       │
├──────────┴──────────────────┴────────────┤
│              Execution DAG               │
└───────────────────────────────────────────┘
```

### C3 语音交互必须可见

```text
你：「把人物左侧调暖。」
AI：「已识别人物和左侧光照区域，正在重新计算光照。」
```

**禁止出现"用户说完没有任何反馈"的黑盒状态。**

## 3.4 四个 Demo 剧本（提前定死，C8 按此录）

| Demo | 剧本 | 验证的研究点 |
|---|---|---|
| **Demo 01** | 「把这个人物放进咖啡馆。」→ 语音 → Planner → DAG → 抠图 → 背景 → 光影 → 结果 | 主链路 + 语音入口 |
| **Demo 02** | 用户点击人物左侧：「这里增加一点暖光。」→ 系统识别 Target=Person / Region=Left / Action=Relight | 空间指代（创新点 2） |
| **Demo 03** | Critic 打出 Lighting 93 / Shadow 71 / Color 94 → 自动判断 Shadow 不达标 → 重跑 T05 → Shadow 91 → PASS | 自评修正（主链路第 4 环） |
| **Demo 04** | 「换回第一版背景，但保留现在的光。」→ Background Node V1 + 当前 Lighting Node → 重跑 → 新结果 | 条件回滚（创新点 3，最重要） |

## 3.5 多轮测试剧本（C4 验收用，必须 5 轮全过）

```text
Round 1  把人物放到咖啡馆
Round 2  改成傍晚
Round 3  光线太冷了
Round 4  阴影轻一点
Round 5  背景换回上一版，但是保留现在的光线
```

## 3.6 公共任务（三人共同，Owner 轮值）

| WBS | 任务 | 交付物 | 截止 | 优先级 |
|---|---|---|---|---|
| **P1** | Git 仓库初始化（分支保护 + LFS + README + PR 模板） | 仓库 | 09-08 | P0 |
| **P2** | 环境与数据冻结（依赖锁版本、数据清单、许可证确认） | `docs/env.md` + `data/README.md` | 09-09 | P0 |
| **P3** | 周进展简报 | `docs/weekly/2026-W*.md` | 每周日 | P1 |
| **P4** | 用户研究组织（约人 n≥10、问卷设计、双盲流程） | `experiments/user_study/` | 09-26 | P1 |

---

# 四、总体架构与框架结构

## 4.1 六层架构

```text
┌────────────────────────────────────────────┐
│              Layer 1  用户交互               │
│   图片 / 语音 / 点击 / 框选 / 文本 / 参考图   │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│              Layer 2  Agent                 │
│   Planner / Tool Router / Critic / Re-plan  │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│              Layer 3  DAG                   │
│   Task Graph / State / Version / Rollback   │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│            Layer 4  图像工具                 │
│  Matting / Background / Lighting / Shadow   │
│  Relighting / Harmonization / Enhancement   │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│             Layer 5  模型                    │
│  BiRefNet / SAM 2 / Diffusion / IC-Light    │
│  LoRA / VLM / ASR / TTS                     │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│         Layer 6  数据与评测                  │
│  Dataset / Checkpoint / Metrics / Logs       │
│  Experiments / User Study / Reports          │
└────────────────────────────────────────────┘
```

## 4.2 控制流总图

```text
                        用户
                         │
             ┌───────────┼───────────┐
             │           │           │
           图片         语音       点击/框选
             │           │           │
             └───────────┼───────────┘
                         ↓
                  Multimodal Agent
                         ↓
                      Planner
                         ↓
                   Structured DAG
                         ↓
       ┌─────────────────┼─────────────────┐
       ↓                 ↓                 ↓
     Matting         Background        Lighting
       ↓                 ↓                 ↓
       └─────────────────┼─────────────────┘
                         ↓
                      Relight
                         ↓
                       Shadow
                         ↓
                   Harmonization
                         ↓
                       Critic
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
            PASS                  FAIL
              ↓                     ↓
            Final            Re-plan / Re-run
              ↓                     ↓
              └──────────┬──────────┘
                         ↓
                      TTS 语音反馈
```

## 4.3 工具 Schema 骨架

统一信封（所有工具共用）：

```json
{
  "tool": "matting",
  "version": "1.0",
  "request_id": "uuid",
  "inputs": { },
  "options": { "quality": "draft|normal|fine" }
}
```

统一响应：

```json
{
  "request_id": "uuid",
  "status": "success | failed | timeout",
  "outputs": { },
  "error": { "code": "E_MATTING_OOM", "message": "...", "retryable": true },
  "latency_ms": 1830,
  "artifacts": ["s3://.../alpha.png"]
}
```

Planner 输出的 DAG 示例：

```json
{
  "plan_id": "p_001",
  "nodes": [
    { "id": "n1", "tool": "matting",             "inputs": { "image": "@upload:0" } },
    { "id": "n2", "tool": "background_generate", "inputs": { "prompt": "傍晚的咖啡馆" } },
    { "id": "n3", "tool": "lighting_estimate",   "depends_on": ["n2"] },
    { "id": "n4", "tool": "relight",             "depends_on": ["n1", "n3"] },
    { "id": "n5", "tool": "shadow_generate",     "depends_on": ["n2", "n4"] },
    { "id": "n6", "tool": "harmonize",           "depends_on": ["n5"] }
  ],
  "edges": [["n1","n4"],["n2","n3"],["n3","n4"],["n4","n5"],["n5","n6"]]
}
```

## 4.4 版本 / DAG / 回滚模型

**版本链**（线性）：

```text
V1 → V2 → V3 → V4 → V5
```

**编辑历史树**（分叉 + 条件回滚）：

```text
              背景 A
               │
原图 → 抠图 ───┤
               │
               └→ 背景 B → 重打光 → 阴影
```

条件回滚语义：

```text
用户：「换回背景 A，但保留现在的灯光。」
→ 恢复 Background Node 到 V1 版本
→ 保留 Relighting Node 当前版本
→ 重跑 Background 的所有下游节点（Shadow / Harmonize）
→ 生成新版本 V6
```

实现要点：**节点级版本化**（每个 node 存 `version` + `output_artifact`），而非整图版本化。这样才可能做"部分回滚"。

## 4.5 Critic 评分模型

```text
Lighting      92
Shadow        84
Color         95
Edge          97
─────────────────
Overall       91   ← 加权，权重先按 (0.3/0.25/0.2/0.25) 初始化，用 30 组样本校准
```

## 4.6 部署架构（维持开题方向，不推翻后端）

```text
                    Web Frontend
                         │
                         ↓
                  Spring Boot API
                         │
            ┌────────────┼────────────┐
            ↓            ↓            ↓
          Auth        Project       Agent
                                      │
                                      ↓
                                 Python AI Service
                                      │
                ┌─────────────────────┼─────────────────┐
                ↓                     ↓                 ↓
              Matting              Lighting          Diffusion
```

| 组件 | 选型 |
|---|---|
| 数据库 | MySQL + Flyway 迁移 |
| 任务 | Job Queue（异步，WebSocket 推送状态） |
| 模型 | Model Registry（`model_registry.yaml`） |
| 对象 | Image / Mask / Result / Checkpoint 统一对象存储 |
| 语音 | ASR / TTS **全本地部署**（数据不出域） |

## 4.7 推荐目录结构（固定，不得自创顶层目录）

```text
project/
├── docs/
│   ├── architecture.md        # 架构说明
│   ├── api-spec.md            # 接口规范
│   ├── task-board.md          # 任务看板
│   ├── experiment.md          # 实验规范与记录
│   ├── env.md                 # 环境与依赖锁版本
│   ├── deployment.md          # 部署文档
│   └── weekly/                # 周进展简报
├── backend/springboot/
├── ai-service/
│   ├── matting/  lighting/  shadow/  harmonization/  diffusion/
├── agent/
│   ├── planner/  executor/  critic/  schema/  rollback/  eval/
├── frontend/
│   ├── pages/  components/  canvas/
├── speech/
│   ├── asr/  tts/
├── training/
│   ├── matting/  lora/  lighting/
├── data/
│   ├── matting/  composition/  lighting/  instruction/
├── experiments/
│   ├── baseline/  ours/  metrics/  visualization/  failure/  user_study/
└── demo/
    ├── screenshots/  videos/  final/
```

---

# 五、实现方案概述

## 5.1 抠取（B 组主责）

**路线**：BiRefNet（无引导发丝级） + SAM 2（交互指代） 组合

```text
输入 RGB
  ↓  SAM 2（点/框提示）→ 粗分割 mask
  ↓  trimap 构造
  ↓  BiRefNet / 轻量抠图头 → 细化半透明边缘
输出 Alpha
```

**训练数据**：按合成公式 `I = αF + (1−α)B` 自动构造训练对，前景来自已有标注数据，背景随机采样。
**训练策略**：单次 ≤24h、每 30 分钟存 checkpoint、支持断点续训与抢占式实例；先在本地 A6000 用 1/10 数据冒烟（≤30 分钟）再全量。
**评价**：Composition-1k 协议 SAD / MSE / Grad / Conn，含难例子集单独统计。

## 5.2 背景生成（B 组 + A 组接口）

以 **SDXL / FLUX + ControlNet（深度 / 边缘结构条件）** 为主，参考图驱动走 Paint-by-Example 思路。
关键约束：必须输出**可与前景对齐的尺寸与视角**，并同步产出深度图供 T03/T05/T07 使用。

## 5.3 光影一致性融合（B 组核心研究点）

```text
Background
    ↓  T03 lighting_estimate（球谐 SH9 / 方向光 / 色温 / 强度）
Lighting Params
    ↓  T04 relight（借鉴 IC-Light 潜空间光照一致性思想）
Relit Foreground
    ↓  T05 shadow_generate（借鉴物理推理阴影：光源方向估计 + 几何锚定）
Contact Shadow
    ↓  T06 harmonize（色彩 / 噪点 / 景深匹配）
Final
```

- **T03**：优先用现成光照估计器输出 SH 系数与光源方向，不要自研（时间不够）。
- **T04**：IC-Light 优先（背景条件模式最贴合本场景）；SwitchLight / DiLightNet 作为对比。
- **T05**：先做程序化降级版（按光源方向 + mask 形态生成软阴影），再做扩散版；程序化版同时是 B6 中 A 组基线的一部分与降级方案。
- **T06**：DoveNet / Harmonizer 思路，或直接用扩散 inpainting 做局部和谐化。

**一致性评分**：用光照 / 法线估计器对前后景分别估计，计算方向差与色温差，作为 Critic 的客观分量（与 VLM 主观分融合）。

## 5.4 Agent（A 组主责）

- **Planner**：调用成熟大模型 function calling 输出 JSON DAG，**不生成像素**。用 JSON Schema 强校验 + few-shot 示例约束自由度；校验失败最多重试 2 次，再失败走程序化降级（关键词映射到固定模板 DAG）。
- **Executor**：拓扑排序执行；节点级幂等；产物落对象存储并在 DAG State 中记录引用。
- **Critic**：VLM 打分（五维）+ 客观光照一致性分 → Overall；低于阈值则按 §3.1 A4 映射表触发重跑。
- **Rollback**：节点级版本化，支持部分回滚 + 下游重跑。
- **两级出图**：草稿级（低步数 / 低分辨率，≤10s）→ 精修级（≤30s），控制端到端延迟。

## 5.5 语音（C 组主责）

```text
Microphone → VAD（端点检测）→ 流式 ASR（本地）→ Text
   → Planner → DAG → Execution → TTS（本地）→ 播放
```

- 全本地部署（数据不出域）。
- **降级设计**：ASR 置信度低或解析出歧义指代时，弹出"我理解的是 XXX，确认吗？"二次确认，同时提供文本输入框兜底（对应开题里提到的语音一致劣于文本这一已知问题）。
- 界面必须实时显示识别文本与 Agent 当前动作。

## 5.6 前端（C 组主责）

- 首页：`上传图片` + `说一句话` 双入口，无传统工具栏。
- 工作台：三栏 + 底部 DAG（§3.3 C2 固定布局）。
- DAG 可视化：节点状态实时刷新，点击节点查看中间产物（**这个能力对 debug 的价值极大，见 §6.1 建议 4**）。
- 历史树：版本树 + 条件回滚交互。

## 5.7 降级方案（三级，必须实现）

| 级别 | 触发条件 | 行为 |
|---|---|---|
| L1 | 单工具超时 / OOM | 重试 1 次 → 换低精度档位 → 程序化方案 |
| L2 | Planner 连续 2 次校验失败 | 关键词模板 DAG 兜底 |
| L3 | 训练权重缺失 / 加载失败 | 自动回落 Base Model，日志告警 |

**原则：任何单点失败都不能让系统白屏。**

---

# 六、改进建议与风险

## 6.1 对当前方案的 10 条改进建议

> 以下均针对本规范 / 常见执行偏差提出，按影响面排序。

| # | 建议 | 为什么 | 落地动作 |
|---|---|---|---|
| **1** | **Critic 阈值必须先标定，不能拍脑袋** | 阈值定高了疯狂重跑（慢），定低了形同虚设。开题里没有给口径 | 09-21 前用 30 组样本做"Critic 分 vs 人工分"标定，用 ROC 选阈值，写进 `agent/critic/threshold.md` |
| **2** | **MVP Demo 与训练彻底解耦，09-13 就要有能跑的端到端** | 9/15 之前如果只有抠图没有链路，一旦训练延期，整个项目就只剩抠图 | 09-13 前用 Base Model + 程序化阴影搭出"丑但能跑"的全链路，之后逐段替换成训练模型 |
| **3** | **数据集与难例集必须 09-09 冻结** | 数据集不冻结，B 组会一直"再清洗一批"，训练无限延后 | 09-09 24:00 冻结 `data/`，之后新增数据只能进 v2 且需全组同意 |
| **4** | **中间态可视化是最高性价比的投入** | 合成链路 6~8 个节点，出图不对时肉眼无法定位是哪一步错 | C5 的"点击 DAG 节点看中间产物"提前到 09-16 完成，优先级等同 P0 |
| **5** | **Critic 不只是验收器，要做数据飞轮** | 现在只用于 PASS/FAIL，浪费了 | 把 (指令, DAG, 产物, Critic 分, 人工修正) 存成结构化日志，作为后续 Planner few-shot 与报告素材 |
| **6** | **语音链路必须有二次确认与文本兜底** | 开题已引用语音一致劣于文本的结论，不加兜底会在 Demo 现场翻车 | C3 内置"低置信度 → 确认弹窗 + 文本框"双通道 |
| **7** | **用户研究要提前设计，不能 09-27 才临时找人** | n≥10 双盲评测约人 + 问卷 + 统计至少要 5 天 | 09-20 前完成问卷设计与人员预约，数据 09-26 开始采集 |
| **8** | **Model Registry 必须带指标与 fallback 字段** | "换权重"如果没有记录，实验表对不上交付物 | `model_registry.yaml` 由 A 组维护，B 组每次产出权重提 PR 更新 |
| **9** | **失败案例集要当作正式交付物** | 答辩时"我们分析了 5 类失败"比"我们效果很好"更有说服力，也是后续论文素材 | B8 至少 5 组，含归因（数据 / 方法 / 参数 / 无效提示） |
| **10** | **成本与算力要建预算表** | 租卡无预算容易在 09-20 后发现超支且没成果 | 建 `docs/budget.md`，记录每次租卡：时长、单价、用途、产出 checkpoint、是否达标 |

## 6.2 风险登记表

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| 外部算力不可用 / 被抢占 | 中 | 高 | 短任务 ≤24h + 断点续训；实验室服务器夜间兜底；09-10 就启动第一次训练留缓冲 |
| 训练效果不达预期 | 中 | 中 | 指标口径分"整体"与"难例子集"，难例子集提升也可作为有效结论；失败案例同样可写 |
| 多轮编辑退化（EdiVal 已验证的现象） | 高 | 高 | 这正是本课题的研究点：DAG + 版本树 + 条件回滚；同时限制单轮 Planner 只做增量修改 |
| 端到端延迟超 10s | 中 | 中 | 草稿 / 精修两级出图；节点并行（抠图与背景生成可并行）；低精度档位 |
| 中文 ASR 在噪声环境退化 | 中 | 中 | 本地 VAD + 二次确认 + 文本兜底；Demo 现场用安静环境 |
| 人像数据授权合规 | 低 | 高 | 09-09 前确认数据来源与授权范围；公开数据集优先；Demo 使用自拍或授权素材 |
| 三人进度不同步导致集成失败 | 中 | 高 | 每周三 / 周日固定集成日；接口先行；`dev` 分支冒烟 |
| 国庆假期压缩交付工期 | 高 | 高 | 09-30 封版；10-01~10-07 只安排报告撰写类可异步任务 |

## 6.3 环境新坑与规避（实践层面，提前规避可省数天）

| 坑 | 表现 | 规避方式 |
|---|---|---|
| **中文路径 / 中文文件名** | Windows 上 Python / Docker 挂载报编码错误，Linux 训练机上路径乱码 | 全项目禁用中文路径与文件名（规范 N7） |
| **Windows / Linux 依赖不一致** | 本地跑通，租卡机器上 torch 版本冲突、CUDA 不匹配 | Docker 镜像锁死版本 + `requirements-lock.txt`；本地先冒烟 |
| **Git 误传大权重** | 仓库体积爆炸、push 失败 | `.gitignore` 提前配置 + Git LFS；权重只提交路径与元信息 |
| **显存不足（OOM）** | 抠图 + 扩散同进程抢占显存 | 各工具独立进程 / 独立容器；显存上限参数化；L1 降级自动切低档位 |
| **Docker 挂载路径权限** | 容器内无法写产物目录 | 统一挂载到 `/workspace/artifacts`，启动脚本 `chmod` 并验证可写 |
| **模型权重下载失败 / 慢** | 租卡机器无代理，HuggingFace 拉取超时 | 提前下载到对象存储，镜像启动脚本从内网拉取 |
| **ASR 与扩散模型抢 GPU** | 语音延迟飙升 | 语音模块独立进程 + 固定显存预留，或 CPU 小模型方案 |
| **实验记录了但复现不了** | 报告写不出确切超参 | 规范 N5：`config.yaml` + `train.log` + `eval.csv` 三件套，随实验提交 |

## 6.4 未来 72 小时行动清单（从今晚开始）

| 时间 | A 组 | B 组 | C 组 |
|---|---|---|---|
| **09-08 晚** | 建仓库、分支保护、PR 模板；起草 T01–T08 Schema 骨架 | 装环境，跑 BiRefNet 推理冒烟，出第一张 alpha 图 | 前端脚手架 + 首页静态原型（上传 + 一句话） |
| **09-09 上午** | Schema 评审会，B 组确认可实现；定 Planner 输出 JSON 格式 | 数据集四集划分完成，难例清单建表 | 语音模块技术选型验证（本地 ASR 跑通） |
| **09-09 下午** | A3 Executor 骨架（不接真实工具，先跑 mock DAG） | **数据集冻结**（24:00 硬截止） | 三栏工作台静态布局 |
| **09-09 晚** | 第一次集成日：`dev` 分支端到端冒烟（mock 版） | 准备 09-10 训练：冒烟脚本 + Docker 镜像 | 首页原型可点，接 mock 数据 |

**09-09 24:00 冻结检查（三项全绿才进入下一阶段）**：

- [ ] 8 个工具 Schema 全部提交并通过评审
- [ ] 数据集四集划分完成、难例清单达标、授权确认
- [ ] 训练集 Docker 镜像可离线构建，本地冒烟通过

---

## 附录 A：术语表

| 术语 | 含义 |
|---|---|
| **DAG** | 有向无环图，本项目中指执行计划（节点=工具调用，边=依赖） |
| **Planner** | 将用户多模态输入解析为结构化 DAG 的模块 |
| **Critic** | 对合成结果多维打分并触发重规划的模块（VLM + 客观光照一致性分） |
| **条件回滚** | 只回滚某个历史节点，保留其他节点的当前状态，并重跑其下游 |
| **Matting / Alpha** | 估计每像素前景不透明度 α，I = αF + (1−α)B |
| **Relighting** | 按目标环境光重新渲染前景光照 |
| **接触阴影** | 前景与背景接触处因遮挡产生的阴影，是消除"贴纸感"的关键 |
| **Harmonization** | 前后景色彩 / 噪点 / 景深统一 |
| **SH（球谐）** | 用 9 个系数低频近似环境光照的表示 |
| **VAD** | 语音端点检测，判断用户是否说完 |
| **Model Registry** | 模型版本登记表，含指标与 fallback 策略 |

## 附录 B：实验记录模板

```markdown
# EXP-YYYYMMDD-<方向>-<序号>

## 一句话结论
（改进 / 退步 / 无效，以及关键数字）

## 配置
- 数据：
- 超参：
- 硬件：
- 代码版本（commit）：

## 结果
| Model | SAD ↓ | MSE ↓ | Grad ↓ | Conn ↓ |
|---|---:|---:|---:|---:|

## 视觉对比
（before / after / gt 三图并列）

## 失败案例
（≥1 组，附归因）

## 下一步
```

## 附录 C：变更记录

| 版本 | 日期 | 变更内容 | 作者 |
|---|---|---|---|
| V1.0 | 2026-09-08 | 首版：总规范 + 分工 + 指标 + WBS + 架构 + 实现方案 + 改进建议 | — |

---

> **一句话收尾**：不要追求"功能最多"，要追求一条完整的证据链——
> **有数据 → 有 Baseline → 有训练 → 有改进 → 有量化指标 → 有光影方法 → 有 Agent → 有自动评估 → 有多轮交互 → 有完整系统。**
> 这样项目才从"三个人做了一个 AI 图片工具"，提升为"三个人完成了一个有算法实验、有 Agent 规划、有交互系统、有量化评价的多模态图像合成研究系统"。
