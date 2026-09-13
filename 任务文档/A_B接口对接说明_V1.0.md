# A–B 接口对接说明 V1.0（基于代码级严核）

> 目的：严格核验 B 组算法实际完成情况，明确 A 组（Agent/后端）接入 B 组能力**所需的确切信息**，
> 使 A1 的 `agent/schema/T01–T08.json` 能"照着 B 组真实接口直接实现"，杜绝凭档案猜接口。
> 核验范围：`ai-service/` 全部工具源码 + `pipeline.py` + `docs/工作文档` + B4/B5 实验报告 + 规范 §4.3 + 看板 B5。
> 审核时间：2026-09-09。Owner：A（赵梓涵），对接人：B。
> 事实以代码为准（本文档已载明文件与函数名，可逐条复核）。

---

## 0. 结论摘要

- B 组已真实落地并可被 Agent 调用的原子能力 = **{ matting, interactive-matting, harmonize, relight, shadow, composite, fx(特效) }**，经 `pipeline.py::run_pipeline()` 可串成一条完整光影合成链。
- 但**与规范的 8 工具存在三处硬缺口**：① 无独立的 `T03 lighting_estimate`（仅 relight 内嵌粗估计，无 SH9 / 独立工具）；② 无 `T02 background_generate`（背景生成不在 B 组，需走 A 组素材库/文生图）；③ 无 `T07 enhance`（只有 fx 特效可部分充当）。`T08 export` 也无。
- **编号体系冲突（必须今天对齐）**：B 组沿用旧编号 `matting=T03, relight=T04, shadow=T05, harmonize=T06, composite=T07`；规范 `T03=lighting_estimate`、`T01=matting`、`T07=enhance`、`T08=export`。若不一致，Planner/DAG 与 B 组工具名将全部对不上。
- 建议：**A 组按规范出 Schema 后，B 组增加轻量"适配层"把规范工具名映射到现有函数**（改动小，约 0.5 天），正向对接。

---

## 1. B 组实际交付清单（代码级）

| 能力 | 入口（类.方法） | 真实后端/算法 | 关键参数（来自源码） | 产物 |
|---|---|---|---|---|
| Matting 抠图 | `matting_backend.build_matting_tool()` / `BiRefNetMatting.predict(image_path,out_alpha,out_fg,original_size)` | BiRefNet-real → carvekit → SimplifiedBiRefNet 三档回退 | 权重 `ai-service/src/matting/birefnet_official/model.safetensors`；推理尺寸 512(节省显存)/1024 | `alpha_path`,`fg_path`；fg 约定 = `img*a + 255*(1-a)`（白底预乘） |
| 圈选抠图(空间指代) | `interactive_matting.InteractiveMatting.matte(image,out_alpha,prompt,refine)` | GrabCut(CPU,零依赖) + 可选 AlphaRefiner 精修 | `prompt{type:box|polygon|points, xyxy/points/positive}`；`refine` 时权重 `training/checkpoints/refiner_studio/best.pt` | `alpha_path`,`fg_path`,`model`,`fg_ratio` |
| Harmonize 和谐化 | `harmonize_tool.HarmonizeTool(method=v2\|mkl).harmonize(fg,bg,out,alpha?)` | v2=Lab 色度对齐+低频亮度迁移+FDR 防压黑；mkl=旧版 | `alpha_path?`,`strength`（内部 0.75，自动降） | `harmonized_fg_path`,`method`,`fdr`,`fdr_ok` |
| Relight 重打光 | `relight_tool.RelightTool(method=directional\|mkl).relight(fg,bg,out,light_hint)` | directional=方向光渐变+色温（默认）；mkl=色彩迁移 | `light_hint{direction:[dx,dy], color_temp:warm\|cool, color_temp_strength, intensity, color:[r,g,b]}`；方向可据背景自动估 | `relit_fg_path`,`method`,`direction` |
| Shadow 阴影 | `shadow_tool.ShadowTool().generate(alpha,bg,out,shadow_hint)` | 程序化软阴影（alpha 高斯模糊+偏移+衰减） | `shadow_hint{direction, offset:[dx,dy], blur_radius, opacity}` | `shadow_path`（已含阴影的背景图） |
| Composite 合成 | `composite_tool.CompositeTool(method=alpha_over\|poisson, feather_radius).composite(fg,alpha,bg,out,shadow_path?)` | alpha over + 边界羽化；poisson 为近似 | `method`,`feather_radius=2` | `output_path`,`method` |
| 特效/风格化 | `fx_tool.FxTool().apply(image,out,effect,params,region)` / `.chain(...)` | 6 种：spotlight/bokeh/fog/vignette/color_temp/depth_blur | `effect`,`params{intensity,...}`,`region{type:box|polygon, xyxy/points}` | `out_path`,`error{E_FX_UNKNOWN,...}` |
| 整链 | `pipeline.run_pipeline(image,bg,out_dir,matting_weight?,enable_*?,light_hint?,shadow_hint?,composite_method?,precomputed_alpha?,fx_chain?)` | matting→harmonize→relight→shadow→composite(+fx) | 各子模块开关 + `precomputed_alpha` 复用 alpha | `stages[]`,`final`,`total_time_ms`,`intermediate_dir` |

要点（对接时务必遵守）：
- **文件即契约**：输入/输出全为本地路径（A 组用 `file_id` 索引路径），工具内部自动 `resize` 对齐分辨率。
- **fg 白底预乘约定**：alpha/fg 由抠图输出统一为 `img*a+255*(1-a)`，下游 relight/harmonize 均按此处理。
- **错误处理不一致**：fx 返回 `{"error":{code,message,retryable}}`；其余工具多直接抛异常（A 组封装时需统一捕捉）。这是 AB 对接第一项要统一的地方。
- **latency 未实测入库**：代码内无准实时延数据（`matting_backend` latency_ms 字段多为 0）；需 B 组按 8 工具各跑一次压测，产出"质量档位(draft/normal/fine)×延长表"供 ≤10s 预算。

---

## 2. 规范 8 工具 ↔ B 组现有能力 差距矩阵（AB 对接核心）

| 规范工具 | B 组现状 | 差距 | A 组需要用到的精确信息 |
|---|---|---|---|
| **T01 matting** | ✅ 有（BiRefNet + 圈选 + Refiner 三用） | 无独立"难例/质量档位"参数（现靠 hidden 推理尺寸与 refiner 开关） | `image`, `mode(auto\|trimap)`, 可选 `point/box`；需 b 确认：trimap 是否支持(auto 只支持 box/points)，1280 与原尺寸原样输出说明 |
| **T02 background_generate** | ❌ 无 | B 组不产背景；背景来自 A 组素材库(`assets.py`)或 B 组程序化背景(`make_demo_assets`) | A 侧决定：素材库 or 文生图(agnes)作为 T02 实现，写明输入输出口径 |
| **T03 lighting_estimate** | ⚠️ 部分 | relight 仅内嵌 `estimate_direction`(2D)+`color_temp`(warm/cool) 粗估；**无 SH9、无独立工具、非 θ/φ** | 需 B 补：独立 `lighting_estimate(bg_png)->sh_coeff(9), light_dir(θφ), color_temp, intensity`；或 A 测出"可从 relight 的 direction 换算"，降级约定 |
| **T04 relight** | ✅ 有 | 参数名与规范形近：规范 `light_dir` vs 实机 `direction:[dx,dy]` | 映射：`light_dir`→`direction`(归一化2D)；`color_temp`/`intensity` 已支持；`θ,φ` 需 B 给换算公式 |
| **T05 shadow_generate** | ✅ 有 | 程序化版已满足降级基线；参数全有 | `composite_png`→`bg_path`+`alpha_path`；`light_dir`→`direction`；`geometry`→`offset/blur_radius`（A 需定几何如何落到现参） |
| **T06 harmonize** | ✅ 有（v2 防压黑为默认，优于旧 mkl） | ⚠️ B 现接口传 `fg`+`bg`+`alpha?`，而规范输入是 `composite_png`+`mask_png` | 明确 A 传已合成 composite + mask，还是传 fg+alpha（B 现行为）。建议后续 B 适配层吸收规范入参 |
| **T07 enhance** | ❌ 无（仅 fx 特效可用作风格化） | 无景深/降噪增强；fx 的 depth_blur 只能局部景深 | A 决策：T07 用 fx 兜底(强度弱) or B 补一个 enhance(去噪/锐化/景深)；`[depth_map]` B 无 -->
| **T08 export** | section无 | A 组负责 zip/尺寸/格式落地 | A 侧实现，B 只管给最终 `final.png` |

**关键红灯**：T03 lighting_estimate 与 T07 enhance 这两项，规范明确 T03"优先用现成光照估计器输出 SH"——**在对接上这是唯一真正需要 B 补代码的点**，其余均为参数映射/适配层。建议 09-10 站会与 B 当场确认 T03 实现方式与 DF 时间。

---

## 3. A 组接入 B 组必须向 B 拿到的信息清单（逐条）

1. **8 工具统一信封确认**：每工具 `input / output / error{code,retryable} / latency_ms / model_version`。现 B 组 `pipeline.TOOL_SCHEMAS` 仅有 5 个且缺 error 与 latency 结构 → 需按规范四件套补齐（A 出模板，B 填）。
2. **唯一入参/出参字典**：见 §1 表"关键参数/产物"，已从代码抄录，A 应据此写 Schema，并与 B 逐字段确认（尤其 relight 的 `direction`、shadow 的 `offset/blur`、harmonize 的 `fg+alpha` 还是 `composite+mask`）。
3. **编号统一映射表**：规范 T0N → B 现函数。A 交付此表，B 在其侧只加适配层不改算法（避免 B 大改返工）。
4. **质量档位契约**：每个工具 `quality∈{draft,normal,fine}` → 对应内部档位参数（matting 的推理尺寸 512/1024、refiner 开关；relight/shadow/harmonize 的 strength 档）。A 的 ≤10s/≤30s 预算依赖它。此表 B 需在 09-11 前给出。
5. **延迟压测表**：8 工具（各档位）× 实测 `latency_ms` + 峰值显存。A 用 `total_time_ms` 与 stages 做草稿/精修两条预算链。
6. **错误/降级语义**：哪些异常可重试（retryable）→ A 接入 `executor` 重试逻辑；哪些不可（如权缺失/输入缺失）→ 走 L1 降级。
7. **权重与热更新**：matting 权重 `birefnet_official/model.safetensors`、refiner `checkpoints/refiner_studio/best.pt` 的路径契约；B 是否支持"运行中热更权重"（训练—推理解耦，创新四）→ A 是否需要缓存失效策略。
8. **素材/背景来源约定**：A 组 `assets.py` 素材库 与 B 组 `make_demo_assets` 程序化背景各自覆盖范围，避免 T02 双向重复。
9. **B5 生死线(09-20)对齐**：`run_pipeline` 已串通 T03→T06，A 只需把 `lighting_estimate`(T03)接上后即可整链贯通；确认 `precomputed_alpha` 复用机制对 A 多轮 DAG 节点复用的意义。

---

## 4. 现状佐证与风险

- **已实锤的编号冲突**：`pipeline.py:161` `"T03_matting"`（原文）对规范 `T01 matting`；`pipeline.py:189` `"T07_composite"` 对规范 `T07 enhance`。若 Agent 直接用 B 的工具名，DAG 会错工具。→ 该冲突在 §3.3 由映射表解决。
- **T03 lighting_estimate 是最可能拖 09-18 Planner / 09-20 生死线 2 的点**：它不仅是独立工具，还是 DAG 首节点。若 B 无法在 09-12 前给出，A 应先用"relight 方向自动估"降级接入并在 Schema 标注版本 `v0.5(降级)`。
- **latency 表缺失** 影响 ≤10s 的预算审计；在 B 补齐前，A 以 1024matting≈1.5s(CPU)/估、程序化 shadow/harmonize 各 <0.5s 的粗预算先跑通。

---

## 5. 下一步（请 A 与 B 09-10 站会当场决定）

| # | 事项 | 责任 | 时限 |
|---|---|---|---|
| 1 | 确认 T03 lighting_estimate 实现方式（B 补 SH/独立 or A 用 relight 降级） | A+B | 09-10 |
| 2 | B 给出 8 工具各档位延迟压测表 | B | 09-11 |
| 3 | A 交付"A 侧 Schema + 规范↔B 函数映射表"草稿 | A | 09-10 |
| 4 | B 增加轻量适配层（规范名→现函数），不动算法 | B | 09-12 |
| 5 | A 在 Schema 中统一 error/retryable 信封并接 A1.3 校验器 | A | 09-10 |
| 6 | T07 enhance 方案定案（fx 兜底 or B 补） | A+B | 09-11 |

> 结论一句话：B 组光影主链已可跑且优于基线；**A 组对接 B 的硬前提是"编号映射 + T03 独立化 + 四件套补全 + 延迟表"四件事**，全部可在 1 天内敲定，不阻塞 09-20 生死线 2。