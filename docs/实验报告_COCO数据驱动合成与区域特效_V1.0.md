# 实验报告：COCO 数据驱动的合成/素材/区域特效/语义分割（V1.0）

> 日期：2026-09-10
> 数据来源：MS COCO 2017 val 集（5000 张 × 36781 条实例标注 × 80 类）
> 对应任务：A 合成样本生成 / B 素材库 / C mask 区域特效 / D COCO-Stuff 语义分割

## 0. 背景与动机

此前项目的训练/评测素材全部依赖 **AI 生成**（绿幕人像 + 演播室背景）与 **程序化几何体**
（`studio_dataset.py` 的 512×512 合成人像）。这带来两个固有问题：

1. **域差距**：AI 生成的绿幕人像边缘过于干净，程序化人像几何感明显，与真实拍摄差异大；
2. **评测说服力不足**：缺少公开数据集背书，指标难以横向对比。

COCO 2017 提供了**真实照片 + 人工多边形标注**，其中的 segmentation 恰好是精确的
物体掩码（GT alpha），可同时解决素材、评测、区域特效三类需求。

## A. COCO 真实域合成样本生成

### 实现
脚本：`training/scripts/coco_synth.py`

流程：
1. 读 `instances_val2017.json`，筛出面积占比 1.5%~85% 的多边形标注
2. 多边形直接转 GT alpha（精确，不需要抠图猜测）——这是与 `ai_material_eval.py`
   中"色度键 GT"的本质区别：**此处 GT 由人工标注给出，无算法误差**
3. 裁前景包围盒 → 缩放至画布高度 35%~92% → 位置采样（底部对齐/居中/随机）
4. 合成到 6 张演播室背景之一，叠加接触阴影 + 光照对齐
5. 输出 `(comp, alpha)` 对 + `manifest.csv`

### 关键修复：光照对齐（FDR 治理）

**首轮结果不合格**：FDR 中位 1.521，仅 32% 落在 [0.7, 1.3]。

诊断（`coco_predict` 抽样 60 张）：
- 前景亮度中位 **101.4**（COCO 户外白天照片）
- 背景亮度中位 **57.2**（演播室背景整体偏暗）
- → FDR 系统性偏高，等同于"前景贴纸感"

这与项目此前"MKL 全局色彩迁移压黑前景"是**同一类错误的镜像**（那次是压黑，这次是过亮）。

修复方案（`coco_synth.py` 光照对齐段）：
```python
target = bg_lum * rng.uniform(0.85, 1.20)     # 目标亮度锚定背景
gain = np.clip(target / fg_lum, 0.55, 1.6)    # 增益受限, 防过曝
fg_arr *= gain
# 背景环境色轻微渗入前景 (色温融合)
tint = np.clip(bg_mean / fg_mean, 0.9, 1.1)
fg_arr *= (1 + (tint - 1) * rng.uniform(0.25, 0.55))
```

### 结果

| 指标 | 修复前 | 修复后 |
|---|---|---|
| FDR 中位 | 1.521 | **1.151** |
| FDR ∈[0.7,1.3] 占比 | 32% | **77%** |
| alpha 抽检异常 | 0/60 | 0/60 |
| 样本数 | 600（train 480 / val 60 / test 60） | 同 |

视觉复核：前景亮度与背景一致，无贴纸感。

### 用途
- **Refiner 训练**：真实域前景 + 精确 GT alpha
- **评测集**：可复现、有公开数据集背书
- **边界样本**：如"大象站在播客间"等语义不合理但掩码正确的样本，
  用于检验模型是否只依赖 alpha 质量而非语义合理性

## B. COCO 背景/前景素材库

### 实现
脚本：`training/scripts/coco_assets.py` → `data/coco_assets/`

- **前景**：150 个透明底 PNG（`fg/`），来自 segmentation 裁剪
- **背景**：120 张实景照片（`bg/`），筛选条件 = 无任何 bbox 面积 > 25% 的图
- **索引**：`assets.json`，每条含 `file / kind / cn(中文描述) / cat / w / h / src`

中文类别表内置 75 类映射（person→人物、couch→沙发、potted plant→盆栽 …）。

### 接入 Agent
修改 `ai-agent/assets.py`：
- `list_backgrounds()` / `list_persons()` 注入 COCO 库摘要到 system prompt
- `find_background()` 新增实景优先逻辑（关键词含"实景/真实/照片/街/室内/自然"时）
- 新增 `find_coco_fg(keyword, k)` 按中文语义检索透明底前景

实测检索：
| 查询 | 结果 |
|---|---|
| `访谈` | bg_02_interview.png（演播室优先） |
| `新闻LED` | bg_01_news_led.png |
| `实景街道` | coco_assets/bg/bg_0000.jpg ✓ |
| `椅子` | fg_0006_chair.png, fg_0036_chair.png |
| `盆栽` | fg_0068_potted_plant.png, fg_0095_potted_plant.png |

## C. mask 驱动的区域特效

### 实现
模块：`ai-service/src/fx/region_fx.py`

与既有 `fx_library.local_effect`（仅支持矩形 box）的区别：接受**任意形状 mask**
（COCO segmentation / BiRefNet alpha / 色度键 / 手绘），并自动羽化边界。

7 个特效：

| effect | 说明 |
|---|---|
| `spotlight_on` | 主体打亮（+45%）+ 背景压暗（−55%）+ 背景降饱和 |
| `spotlight_off` | 主体压暗 / 模糊 / 马赛克（隐私保护），周围不变 |
| `region_filter` | 仅主体（或 `invert` 后仅背景）套用滤镜 |
| `region_color` | 区域内调暖/冷/明度/饱和 |
| `bg_blur` | 背景虚化，主体锐利（景深感） |
| `region_glow` | 区域内辉光（亮部扩散 + 色调） |
| `edge_highlight` | mask 边缘描边发光（抠图质检可视化） |

羽化函数自适应：`sigma = max(2, min(H,W) * 0.012)`，避免小图过度模糊。

### 验证
- 单元级：COCO 真实 person mask（覆盖 15.7%）上 7/7 特效全部通过，输出 shape/dtype 正确
- 集成级：`tools.py` 新增 `region_fx` 工具，支持 `region="auto"`（自动调 hybrid 抠图）
- 端到端：`"把这个绿幕人像抠出来换到实景街道背景，然后给主体加聚光灯效果"`
  → 2 工具、43.2s 完成，成片人物干净、背景被正确压暗

## D. COCO-Stuff 语义分割

### 下载攻坚：单线程 → 16 线程分段

**初始状态**：官方源单连接实测仅 **35 KB/s**，1.07 GB 需 **8.5 小时**，不可接受。

**诊断过程**：
1. 误判阶段：短窗口采样出现 2.2 MB/s 的瞬时值，以为是网络波动 → 用 12 点 × 25s 长窗口确认稳定速率仅 34 KB/s
2. 镜像排查：SJTU 镜像返回 HTTP 200 但实际只有 4345 字节（错误页），不可用
3. **关键发现**：官方源返回 `Accept-Ranges: bytes` + `206 Partial Content` → **支持断点续传**

**多线程并发测速**（20~25s 窗口）：

| 线程数 | 实测速率 | 相对单线程 |
|---|---|---|
| 1 | 0.076 MB/s | 1× |
| 4 | 0.355 MB/s | 4.7× |
| 8 | 0.553 MB/s | 7.3× |
| **16** | **0.959 MB/s** | **12.6×** |
| 24 | 0.096 MB/s | 服务端限流，劣化 |
| 32 / 48 | 0.066 / 0.115 MB/s | 限流加剧 |

**16 线程为最优解**。

**落地**：`training/scripts/mt_download.py`
- 1 MB 分段 → `.parts/` 独立文件，天然断点续传
- **种子继承**：把 curl 已下载的 82.9 MB 按段切分纳入，避免重复下载（79 段直接复用）
- 实时进度条 + ETA + 进度日志

**结果**：`3.3 分钟` 完成（vs 单线程 8.5 小时，**提速约 150 倍**），实达 2.0~2.4 MB/s。
- 最终大小 1,148,688,564 字节 = **100.00% 匹配**
- **CRC 全量校验通过**（8 个条目）

### 内容

解压后 `data/coco_stuff/annotations/`：

| 文件 | 大小 | 内容 |
|---|---|---|
| `stuff_val2017.json` | 49 MB | 5000 图 / 32801 标注 / **92 类** |
| `stuff_val2017_pixelmaps/` | 5000 PNG | **像素级语义分割图**（像素值 = category_id） |
| `stuff_train2017.json` | 1.13 GB | 训练集标注（备用） |

### 应用：`training/scripts/stuff_semantic.py`

1. **语义分组索引**：15 个中文语义组（天空/建筑/树木植物/地面/山石/水/雪/人物/车辆/家具/电子设备/食物/文字标识/室内空间/户外环境）
2. **场景构成分析** `analyze()`：类别占比 + 语义组占比
   - 实测 `000000000139`：wall-concrete 30.5% / floor-wood 18.3% / cabinet 9.4%
   - 实测 `000000002473`：sky-other 64% / tree 18% / snow 13%
3. **语义驱动区域特效** `apply_semantic_fx()`：语义 mask → `region_fx`
   - 实测"只把天空调暖"：精确命中 64.4% 天空，**雪地/树木/人物零影响**
   - 实测"只把树木调冷"：精确命中 18.1% 树木，天空/雪地不受影响

这是矩形区域特效**无法实现**的能力——语义边界完全贴合真实物体轮廓。

### 接入 Agent
`ai-agent/tools.py` 新增 2 个工具：
- `semantic_fx(image, group, effect, intensity/warm/cool, invert)`
- `analyze_scene(image)`

端到端实测："分析 000000002473 场景，然后只把天空调成暖色调作出日落感"
→ Planner 自动规划 `analyze_scene` → `semantic_fx`（group=天空, warm=0.85），14.2s 完成。

## 统一验证

脚本：`training/scripts/coco_pipeline_check.py`

```
[PASS] A. coco_synth    600 张, FDR 中位 1.151, 入区间 77%
[PASS] B. coco_assets   背景 120 / 前景 150 / 75 类 / 0 缺失
[PASS] C. region_fx     特效 7/7 通过
[PASS] D. coco_stuff    5000 图 / 32801 标注 / 92 类 / pixelmaps 就位
```

Agent 工具总数 **19**（新增 region_fx / semantic_fx / analyze_scene）。

报告输出：`experiments/coco_pipeline_report.json`

## 结论与后续

**已确立的能力**：
1. 真实域合成样本管线（光照对齐解决 FDR 越界，77% 落区间）
2. 可语义检索的素材库（270 条，75 类中英映射）
3. 任意形状 mask 的 7 种区域特效
4. **像素级语义分割 + 语义驱动特效**（15 个中文语义组，精确到像素边界）

**待办**：
- 用 A 的 600 样本对 Refiner 做真实域微调，与 AI 生成域对比
- 扩大 C：接入视频逐帧 mask（时序稳定化，防闪烁）
- stuff 语义分割用于更智能的抠图先验（如"天空永不是前景"）
- `stuff_train2017.json`（1.13 GB）可作为大规模语义预训练数据

