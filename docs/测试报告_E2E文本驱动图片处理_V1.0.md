# 测试报告：文本指令端到端驱动图片处理（E2E）V1.0

> 日期：2026-09-10 · 被测：`agent/run.py` AgentSession 全链路（Planner→Executor→Critic 自动重规划）
> 输入：文本提示词 8 条（`agent/eval/e2e_prompts.jsonl`）+ 演播室实拍图（主播台）
> 评价：Critic（agnes VLM 5 维：Lighting/Shadow/Color/Edge/Overall，0-100）

---

## 1. 结论摘要

**文本理解层（Planner）不是瓶颈**：8 条指令规划全部正确（plan_source 全为 llm，DAG 与语义一致），指令→工具链的映射在实图上成立。

**图片效果层质量分化**（7/8 出片，Overall 均分 ≈ 58）：

| 用例 | 指令 | Overall | 判定 |
|---|---|---|---|
| P1 | 纯抠图 | （无成片→已修） | ✅修复 |
| P2 | 换新闻LED背景+打光+投影+导出 | 58 | ⚠️ 桌子被染蓝+悬浮 |
| P3 | 换访谈背景 | 70 | ⚠️ 观感尚可但**双人+双桌语义冲突** |
| P4 | 搬到全景城市+影子 | **78** | ✅ 最佳 |
| P5 | 景深虚化 | 45 | ❌ 全图模糊（非深度感知） |
| P6 | 锐化+导出 | 75 | ✅ |
| P7a | 换播客背景 | **20** | ❌ 桌上桌堆叠，画面混乱 |
| P7b | 回滚换综艺背景保留光照 | — | ⚠️ 回滚机制工作正常，但综艺素材同病 |

## 2. 根因诊断（效果差的 4 个来源，全部与"文本理解"无关）

| # | 根因 | 影响用例 | 性质 | 处置 |
|---|---|---|---|---|
| 1 | **T02 draft 素材是"整景"**：自带桌椅/人物/话筒（bg_02 有双人双桌、bg_06 是完整播客间），前景桌叠上去必然冲突 | P3 P7a P7b | 素材层 | 见 §4 建议 |
| 2 | **T07 fx 是全局滤镜**：depth_blur 对整图模糊，无 alpha/深度引导 | P5 | 代码层 | P1 待办：alpha 引导景深 |
| 3 | **engine 无成片兜底缺失**：纯抠图 DAG 产物 key（fg_path/alpha_path）不在 final 查找链 | P1 | 代码层 | ✅ **已修+验证**（`engine.py` _FINAL_KEYS） |
| 4 | **T04 relight 强度无上限**：跨光照（冷→暖）把主体染蓝/压暗 | P2 | 参数层 | P1 待办：intensity clamp |

## 3. "继续训练模型"的执行与结果

按"效果差→继续训练"的指示，训练针对的是**已确认的模型级缺陷**：fine 档 Refiner（refiner_studio）对"物体主体"OOD——精修会把绿幕墙判为前景毁掉抠图（两轮测试均复现）。

**训练过程（两轮，数据：`data/mixed/` = 演播室 960 + 通用域 496，硬链接零拷贝）**：

| 版本 | 方案 | 主播台 MAD vs 粗α | 主播台 fg_ratio | 人像域 val_l1 | 判定 |
|---|---|---|---|---|---|
| refiner_studio（旧） | 演播室人像单域 | 0.3816 | 46.6%（吞绿幕墙） | **0.96%** | OOD 毁图 |
| refiner_mixed v1 | 混合域热启动，无构图增广 | 0.3730 | 47.6%（仍毁） | 1.61% | ❌ 无效 |
| **refiner_mixed_v2** | **混合域从头训 + 构图增广(缩放0.35-1.0+平移+边沿填充)** | **0.0301** | **8.4%** ✅ | 3.16% | ✅ **OOD 解决** |

**关键发现**：
1. v1 证明"混数据但不变构图"无效——所有训练图主体都是"大且居中"，模型学到的是**构图先验**而非抠图语义；
2. v2 的构图增广（缩小+平移，画布边沿复制背景、GT 置 0）直接教会模型"绿幕墙在任意构图下都是背景"，OOD 一次修复（MAD 改善 12.7×）；
3. 代价：人像域 0.96%→3.16%（通才 vs 专才权衡）。

**部署建议**：fine 档默认用 `refiner_mixed_v2/best.pt`（双域安全）；人像专用流程可继续用 `refiner_studio/best.pt`（更锐）。另可加 MAD 门控：精修偏离粗 α 过大时自动回退粗 α，双保险。

复验脚本：`training/scripts/eval_mixed_refiner.py --w1 <权重> --tag <名>`；评测产物：`experiments/studio/mixed_refiner_eval/`。

## 4. 建议（按 ROI 排序）

1. **T02 素材重建**（收益最大）：现有 6 张整景素材改为"空场景无桌无人"版本，或对含人素材建立 `contains_people/contains_desk` 元数据，planner 语义匹配时避开冲突（如"换访谈背景"→ 空访谈间）
2. **T04 relight intensity clamp**（一行改动级）：限制色温偏移幅度，防染蓝压暗
3. **T07 alpha 引导景深**：用 T01 alpha 做距离变换当深度图，只糊背景
4. fine 档默认切换 `refiner_mixed`（训练复验通过后）

## 5. 复现

```bash
cd agent/eval && python e2e_image_test.py          # 8 条提示词 E2E（约 6 分钟）
cd ../../training/scripts && python eval_mixed_refiner.py   # 训练后复验
```

产物：`experiments/studio/e2e_text2img/`（e2e_results.json + 联络表 + 各用例成片）
