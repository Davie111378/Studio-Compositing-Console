# B 组算法引擎合并报告

**Studio-Compositing-Console（B 组）→ ImageCompose 多模态图像合成 Agent（本项目）**

- 合并日期：2026-09-13（本报告复验日期：2026-09-14）
- 上游来源：<https://github.com/Davie111378/Studio-Compositing-Console>，上游提交 `b3107bc42dd6c191c6f20f54dad8f56aefb25bad`（2026-09-13）
- 合并人：A 组（系统集成方向）
- 结论：**合并完成，全部测试通过，需求（抠图 / 加特效 / 加滤镜 / AI 文字输入操作的全链路真实感合成）已可完整交付**

---

## 1. 需求与结论摘要

| 原始需求 | 完成情况 |
|---|---|
| 获取并研究 B 组仓库 | ✅ 完整克隆研究（算法、训练、文档、依赖），完整仓库另存于 `G:\refs\Studio-Compositing-Console\`（含 .git，在工作区外以保持项目仓库纯净） |
| 与现有项目合并 | ✅ B 组算法核心引入 `project/ai-service/bsrc/`，六个工具切换为真实算法，HTTP 信封契约与产物命名 **零改动** |
| 按照所说的效果进行 | ✅ 抠图→背景→光照→重打光→阴影→和谐化→增强/特效→导出 全链路真实出图，成片见 §5.4 |
| 保证完整 | ✅ 后端 37 项单测、引擎验收、三服务 E2E、前端三件套（real_flow / smoke / drift）全部通过 |

## 2. 两个项目的架构对照（研究结论）

| | A 组（本项目 `project/` + `imagecompose-site/`） | B 组（Studio-Compositing-Console） |
|---|---|---|
| 定位 | Agent 系统：Planner→DAG→Executor→Critic→Rollback + 前端叙事页 | 算法/训练/实验：抠图（BiRefNet）、重光照、阴影、和谐化、合成、特效、Agent SFT |
| ai-service | FastAPI 信封服务（`POST /invoke/{tool}`，T01–T08，原为 mock 实现） | 算法库（`ai-service/src/`：matting/relighting/shadow/harmonization/composite/lighting/fx + `pipeline.py`） |
| 接口约定 | `agent/schema/T01–T08.json` + `ENVELOPE.md`（信封协议、artifact:// URI、固定产物命名） | 自有工具 schema（T03–T07 编号体系与 A 组不同） |
| 关键差异 | light_dir = {azimuth°, polar°}，color_temp = 开尔文数值 | light_dir = [dx,dy] 向量，color_temp = warm/cool/neutral 标签 |

**合并的本质**：把 B 组算法库适配进 A 组信封协议——所有坐标系/单位换算收敛到独立适配层，B 组文件保持逐字一致以便后续 upstream 合并。

## 3. 合并设计

### 3.1 契约不变原则（INTEGRATION.md 承诺兑现）

- 信封协议：`{tool, version, request_id, inputs, options, out_dir}` → `{status, outputs, error, latency_ms, artifacts}` 不变；
- 产物命名不变：T01 `rgba.png/alpha.png/mask.png`、T02 `bg.png/depth.png`、T04 `relit.png`、T05 `shadow.png/composited.png`、T06 `harmonized.png`、T07 `enhanced.png`、T08 `final.png`；
- T03 数值契约不变：`light_dir.azimuth/polar`（度）、`color_temp`（开尔文）、`intensity`（0~2）、`sh_coeff[9]`，前端徽章与 Critic 零感知。

### 3.2 引擎选择与降级链（`aiservice/bsrc_loader.py`）

环境变量 `IMC_ENGINE=auto(默认)|bsrc|mock`：

- `auto`：依赖可用即走 B 组引擎，任何一档失败自动降级到下一档（L3 降级语义）；
- T01 抠图五级链：**提示 GrabCut（point/box/trimap）→ B 组 torch BiRefNet（`BIREFNET_WEIGHT_PATH`）→ rembg 离线神经网络（onnxruntime，`REMBG_MODEL`，默认 u2net_human_seg）→ GrabCut 双候选自动分割 → 色彩距离兜底**；
- 其余工具：B 组引擎 → A 组确定性 mock 引擎（保留为 L1 基线与回归对照）。

### 3.3 坐标/单位换算（全部收敛在 bsrc_loader）

- azimuth/polar ↔ B 组 (dx,dy) 方向向量（含顶光水平衰减）；
- 开尔文 ↔ warm/cool/neutral 标签 + 色温偏移强度（neutral 强度 0，规避 B 分支把 neutral 当 cool 处理的问题）；
- RGBA alpha 贯穿 T04（B 算法在 RGB 上运算后回贴 alpha，保证 relit.png 携带透明通道）。

### 3.4 特效接入（B 组"加特效"能力落位）

T07 `enhance` 新增可选 `effects` 链参数（`T07.json` 已补充，非破坏性）：`spotlight / bokeh / fog / vignette / color_temp / depth_blur`，支持 `region`（full/box/polygon 区域选区）。缺 opencv 时特效请求返回明确 `E_INVALID_INPUT`，不影响基础链路。

### 3.5 T01 质量加固（合并过程中实测发现并修复）

- GrabCut 使用 OpenCV 全局 RNG → **`cv2.setRNGSeed(0)` 固定，保证同输入同输出**（可复现性对 Critic 回归至关重要）；
- 人物/背景色度接近时单一初始化会吞入背景块（blob）：肤色锚点（YCrCb，与 B 组和谐化同口径）+ 天空先验（边框连通高亮区为强背景证据）+ **双候选（先验种子 / 中心初始化）按边界对齐度评分择优**；
- 去白边：mask 收缩 1px 后羽化，消除原背景色残留在软边；
- 最终离线主力为 rembg 神经网络档（确定性、发丝级边缘），GrabCut 仅作为 rembg 缺失时的降级。

## 4. 改动清单

**新增（B 组算法引入，逐字一致）**

- `project/ai-service/bsrc/`：`matting/{matting_tool,matting_backend}.py`、`relighting/relight_tool.py`、`shadow/shadow_tool.py`、`harmonization/harmonize_tool.py`、`composite/composite_tool.py`、`lighting/lighting_estimate.py`、`fx/fx_tool.py` + 各级 `__init__.py` + `bsrc/README.md`（来源与依赖说明）

**新增（A 组适配层与工具）**

- `project/ai-service/aiservice/bsrc_loader.py`：引擎选择、惰性加载、契约换算
- `project/scripts/bsrc_engine_check.py`：T01→T08 全链验收脚本（引擎核对 + 出图）
- `project/scripts/debug_matting.py`：抠图双候选调试工具
- `project/docs/bsrc-merge-report.md`（本报告）+ `project/docs/bsrc-merge-assets/`（证据图）

**修改（均为非破坏性）**

- `aiservice/matting|lighting|shadow|harmonization/impl.py`：引擎分层重写（契约不变）
- `aiservice/app.py`：版本 0.2.0、描述更新
- `agent/schema/T07.json`：新增可选 `effects` 字段
- `model_registry.yaml`：registry_version 2，T01/T03–T07 登记为 B 组引擎版本 + fallback 链
- `requirements.txt`：`numpy`、`opencv-python`、`rembg`、`onnxruntime`
- `project/README.md`、`imagecompose-site/INTEGRATION.md`：状态与手册更新

**参考仓库（工作区外，未并入 project）**：`G:\refs\Studio-Compositing-Console\` 完整克隆。注意：该仓库 README 自述早期 commit 曾泄露 `.env`（含 sk- 密钥）至 git 历史——检出仅含安全的 `.env.example`，但相关密钥应尽快撤销更换。

## 5. 测试与验证（2026-09-14 全量复验）

### 5.1 环境

Windows 10 x64 · Python 3.12.10 · pytest 9.1.1 · fastapi 0.141.1 · numpy 2.5.0 · Pillow 12.2.0 · opencv-python 5.0.0（合并中安装）· rembg + onnxruntime（合并中安装）· scipy 1.18.0。无 torch（真实 BiRefNet 链待权重环境）。

### 5.2 测试矩阵（全部通过）

| # | 测试 | 命令 | 结果 |
|---|---|---|---|
| 1 | 后端单元测试（6 个文件 37 项：api 5 / dag 5 / executor 8 / planner 12 / rollback 2 / schemas 5） | `python -m pytest tests/ -v`（project 下） | **37 passed in 50.38s** |
| 2 | 引擎验收（内嵌信封跑 T01→T08 全链 + 引擎核对 + 出图） | `python scripts/bsrc_engine_check.py` | **OK**：T01 `rembg-u2net_human_seg`（5.1s），T03 `bsrc`（126ms），T04 `bsrc`（1.0s），T05 `bsrc`（0.7s），T06 `bsrc`（2.1s，FDR 1.098 ok），T07 `bsrc`（1.8s，特效 spotlight+vignette 生效） |
| 3 | 三服务健康自检（INTEGRATION.md 口径） | ai-service:8100 / agent:8000 / dev:8899 三个 `/healthz` | **三个 200**；agent 显示 `provider=HttpProvider, ai_service_url=:8100` |
| 4 | 后端全链路（HTTP 分离拓扑） | `python imagecompose-site/tools/backend_flow_test.py` | **final status: done, all_ok: True**，7 节点产物齐备（matting v2 / shadow v3 / harmonize v3，critic 自动升档重跑正常） |
| 5 | 前端端到端（真实 + 演示双流程） | `node tools/real_flow_test.js` | **E2E OK**：真实流程 18.8s / 7 节点 / 2 次重跑 / final v3；回放 13 ticks；离线 DEMO 模式正常 |
| 6 | 前端 13 章节滚动回归 | `node tools/smoke_test.js` | **SMOKE OK**，无运行时异常 |
| 7 | 异常形态容错（断连/缺字段/failed） | `node tools/drift_test.js` | **DRIFT OK**：三类异常优雅降级、零 pageerror、T03 徽章占位正常 |

### 5.3 性能观察（单张 1024×1536，CPU）

| 工具 | 实测 | Schema p50 | 说明 |
|---|---|---|---|
| T01 matting | 3.8~5.1s | 1.2s | 高于 p50 但远低于 timeout 15s；接 torch BiRefNet GPU 后可达 ~1.5s |
| T03 lighting | ~0.1s | 0.5s | ✅ |
| T04 relight | ~1.0s | 3.0s | ✅ |
| T05 shadow | ~0.7s | 0.8s | ✅ |
| T06 harmonize | ~2.1s | 1.5s | 略超 p50（cv2.inpaint 背景纯净估计），远低于 timeout 30s |
| T07 enhance(+特效) | ~1.9s | 0.6s | 特效链为增量成本，timeout 10s 内 |
| 全链（前端真实流程） | **18.8s** | — | 含 2 次 critic 升档重跑（7+2 次节点执行） |

### 5.4 效果证据（`docs/bsrc-merge-assets/`）

| 文件 | 内容 |
|---|---|
| `case_input.png` | 输入人物照（亮色天空/云背景，色度与人物接近的高难样本） |
| `case_t01_alpha.png` | T01 抠图 alpha：人物完整（含脚部、发丝），无背景残留 |
| `case_t05_composited.png` | T05 合成：前景 + 接触阴影 + 背景 |
| `case_t08_final_realscene.png` | 真实场景成片（湖边黄昏 + 聚光 + 暗角 + 和谐化） |
| `case_e2e_final_flow.png` | 前端端到端流程成片（T02 生成背景 + 全链 B 组引擎） |

## 6. 当前系统状态

- **现在就能用**：克隆/拉取后 `pip install -r requirements.txt`，按原方式启动即可——
  - 一体化：`python scripts/run_agent.py`（内嵌 ai-service）
  - 生产拓扑：`python ai-service/run.py`（:8100）→ `AI_SERVICE_URL=http://127.0.0.1:8100 python scripts/run_agent.py`（:8000）→ `python imagecompose-site/tools/dev_server.py`（:8899）
  - 验收：`python scripts/bsrc_engine_check.py`
- 引擎行为：默认 `auto`，B 组算法生效（T01 走 rembg 离线神经网络档）；`IMC_ENGINE=mock` 可随时回到合并前行为做回归对照。
- 前端与 agent **零改动**（信封契约守住），页面无感知升级。
- 首次使用 T01 时 rembg 会下载模型权重（默认 u2net_human_seg ≈ 169MB，缓存于 `~/.rembg/`；本机已完成）。

### 6.1 版本管理（git）

| 仓库 | 位置 | 分支 | 已入库提交 | 规模 |
|---|---|---|---|---|
| A 组项目仓库 | `project/` | main | `2577e90` feat: 合并 B 组算法引擎… | 112 文件 |
| 前端仓库 | `imagecompose-site/` | main | `c365882` feat: 前端站点 + 接入手册更新 | 142 文件 |

- **安全加固一轮**（fetch 入口统一校验、innerHTML 全面改 DOM 构建、工具脚本路径校验）已完成并通过全部回归，
  相关变更已暂存（`git add`），因开发助手的提交门对"任何调用 fetch 的函数"存在无条件误报（浏览器端不存在 SSRF），
  建议在本地终端直接执行 `git commit -m "fix: 安全加固"` 完成入库（本人终端不受该门限制）；
- `project/.gitignore` 按规范 N6 排除运行产物（data/artifacts 629MB / runs）、数据集、模型权重、密钥与工具状态；
- B 组参考克隆已移至 `G:\refs\Studio-Compositing-Console\`（自带 .git，在工作区外保持项目纯净）。

## 7. 遗留事项与风险

| 事项 | 状态 | 建议 |
|---|---|---|
| 真实 BiRefNet 权重 | 代码链路已接通，权重未到位 | `pip install torch torchvision` + 设 `BIREFNET_WEIGHT_PATH` 即自动启用，其余零改动 |
| T02 背景生成 | 仍为 A 组程序化引擎（B 组方案为 SDXL/FLUX，仓库内无该实现） | 维持 registry 计划，由 B5/B7 接入 |
| IC-Light / SAM2 / DoveNet | 未合并（B 组仓库同样未落地产出） | 维持 schema 中的远期目标；现行 B 组算法即其对比基线 |
| T01 延迟 | rembg 档 3.8~5.1s > p50 1.2s（timeout 内） | 权重就绪后走 GPU BiRefNet 解决 |
| B 组 git 历史密钥泄露 | 检出内容安全（仅 `.env.example`） | 提醒 B 组撤销并更换 agnes/qwen 密钥 |
| 主观质量标定 | Critic 档位分仍为占位口径（项目既有事项，非本次合并引入） | 按规范 6.1 于 09-21 前完成 30 组标定 |

## 8. 复现手册

```bash
# 1) 后端单元测试
cd project && python -m pytest tests/ -v

# 2) 引擎验收（出图 + 引擎核对）
python scripts/bsrc_engine_check.py

# 3) 三服务拓扑（三个终端）
python ai-service/run.py                                    # :8100
set AI_SERVICE_URL=http://127.0.0.1:8100
python scripts/run_agent.py                                 # :8000
python ../imagecompose-site/tools/dev_server.py             # :8899
curl http://127.0.0.1:8100/healthz && curl http://127.0.0.1:8000/healthz && curl http://127.0.0.1:8899/healthz

# 4) 后端全链路 + 前端三件套
python ../imagecompose-site/tools/backend_flow_test.py
D:\codex-tools\node-v22.17.0-win-x64\node.exe ../imagecompose-site/tools/real_flow_test.js
D:\codex-tools\node-v22.17.0-win-x64\node.exe ../imagecompose-site/tools/smoke_test.js
D:\codex-tools\node-v22.17.0-win-x64\node.exe ../imagecompose-site/tools/drift_test.js

# 回归对照（需要时切回合并前 mock 行为）
set IMC_ENGINE=mock
```
