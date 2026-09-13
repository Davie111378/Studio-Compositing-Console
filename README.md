# 演播室图像合成技术（B 组）

> 状态：**生死线 1 (09-15) 已提前 6 天完成**
> 时间：2026-09-09 通宵落地
> 团队：B 组（算法/训练/实验）

## 一句话总结

设计并落地"**BiRefNet 冻结 + AlphaRefiner 精修**"两阶段抠图方案，
在合成 matting 数据集上实现 **SAD 相对下降 56.8%**（验收门槛 10%），
四组光影对比实验证明全流程 D 相对直接合成 A 显著自然（dL -68%, dAB -75%, BNR -38%）。

## 关键交付物

| 类别 | 路径 |
|---|---|
| 算法工作文档 | `docs/工作文档_演播室图像合成技术_V1.0.md` |
| B4 抠图对比报告 | `docs/实验报告_B4_抠图对比_V1.0.md` |
| B6 光影对比报告 | `docs/实验报告_B6_光影对比_V1.0.md` |
| B8 消融与失败案例 | `docs/实验报告_B8_消融与失败案例_V1.0.md` |
| Baseline 指标 CSV | `experiments/baseline/eval/eval_birefnet512.csv` |
| Refined 指标 CSV | `experiments/fine_tuned/eval/eval_refined.csv` |
| 视觉对比图（24 组） | `experiments/fine_tuned/comparison/` |
| 四组光影结果（24×4） | `experiments/relighting/{A,B,C,D}/` + `_grid_ABCD.png` |
| 消融与失败案例 | `experiments/failure/` + `ablation_grid.png` + `failure_grid.png` |
| Refiner 模型权重 | `training/checkpoints/refiner/best.pt`（40 MB） |
| BiRefNet 权重 | `ai-service/src/matting/birefnet_official/model.safetensors`（444 MB） |
| 训练三件套 | `training/checkpoints/refiner/{config.json, train.log, metrics.csv}` |

## 核心数字（test 集 n=96）

### B4 抠图对比
| 指标 | BiRefNet@512 baseline | BiRefNet + AlphaRefiner | 相对下降 |
|---|---:|---:|---:|
| **SAD** | 12 999.31 | **5 615.47** | **−56.80 %** |
| **MSE** | 0.0573 | **0.0230** | **−59.81 %** |
| **Grad** | 5 860.86 | **2 765.58** | **−52.81 %** |
| **Conn** | 0.7899 | **0.3479** | **−55.95 %** |

难例胜利点：hair SAD ↓99.2% / complex_bg ↓97.9% / fine_edge ↓71.3%。

### B6 光影对比（n=24）
| 组 | |dL| | dAB | BNR | 秒/张 |
|---|---:|---:|---:|---:|
| A 直接合成 | 43.07 | 41.96 | 19.742 | 0.10 |
| D 全流程 Ours | **13.79** | **10.63** | **12.240** | 0.23 |
| D vs A | **−68.0 %** | **−74.7 %** | **−38.0 %** | — |

## 目录结构

```
生产实习/
├── agent/                  # 新版 DAG Agent（schema / adapter / engine / planner）
├── agent_training/         # Agent SFT / 语料 / 评测
├── ai-agent/               # 旧版文本 Agent
├── ai-service/             # Agent 可调用的图像合成服务
│   ├── pipeline.py         # 统一入口 run_pipeline(...) + TOOL_SCHEMAS
│   ├── src/
│   │   ├── matting/        # T03 抠图（BiRefNet 真实 + AlphaRefiner 精修）
│   │   ├── relighting/     # T04 经典 MKL 色彩/光照迁移
│   │   ├── shadow/         # T05 程序化软阴影
│   │   ├── harmonization/  # T06 MKL 色彩一致性迁移
│   │   └── composite/      # T07 alpha over + 泊松近似
│   ├── scripts/            # 演示素材生成、carvekit 测试
│   └── api/                # Agent tool schema 存放位（待 A 组统一格式）
├── app/                    # 用户入口（CLI / Web / GUI / 启动脚本）
│   ├── studio_cli.py       # 命令行任务入口
│   ├── studio_web.py       # Web 交互工作台（端口 8765）
│   ├── studio_gui.py       # tkinter GUI
│   └── start_agent.bat     # 一键启动 Web 服务
├── data/
│   └── matting/            # 合成数据集 + manifest.csv（split 权威）
│       ├── manifest.csv    # 唯一权威切分
│       ├── test_names.txt  # 评估用清单
│       └── coarse_cache/   # BiRefNet 粗 alpha 缓存
├── docs/                   # 文档与实验报告
├── experiments/            # 实验产物
├── logs/                   # 运行日志（不纳入 git）
├── outputs/                # 产物输出（不纳入 git）
├── presentations/          # PPT 等演示材料（不纳入 git）
├── scripts/                # 一次性/辅助脚本（SVG/PDF 等）
├── third_party/            # 第三方模型与依赖
├── tmp/                    # 临时 dump（不纳入 git）
├── training/               # 训练代码、配置与 checkpoints
│   ├── scripts/            # 训练 / 评估 / 实验脚本
│   │   ├── synthetic_dataset.py
│   │   ├── train_refiner.py     # 训练入口（含 --prepare）
│   │   ├── predict_alpha.py     # 推理入口（base / refined）
│   │   ├── evaluate_matting.py  # SAD/MSE/Grad/Conn
│   │   ├── visualize_comparison.py
│   │   ├── experiment_relighting.py  # B6
│   │   └── experiment_failure.py     # B8
│   ├── checkpoints/
│   │   └── refiner/{best.pt, last.pt, train.log, metrics.csv, config.json}
│   └── configs/            # 待放各模块配置
├── web/                    # Web 前端 HTML 页面
│   ├── studio_web.html
│   └── studio_agent.html
├── 任务文档/               # 任务说明、看板、接口文档
├── 数据集/                 # 原始/下载数据集
├── 测试图片/               # 测试用例图片
├── 绿幕图片/               # 绿幕样例
├── 绿幕抠图对/             # 成对评测数据
└── 训练提示词/             # 训练用提示词
```

## 快速启动

```bash
# Web 工作台（推荐）
app\start_agent.bat
# 或命令行
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" app/studio_web.py

# 命令行批量处理
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" app/studio_cli.py

# tkinter GUI
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" app/studio_gui.py
```

## 快速复现

```bash
# 0) 切到 venv
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" -V  # 应为 3.13.12

# 1) 数据集（若已生成可跳）
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/synthetic_dataset.py \
  --root data/matting --n_per_case 120 --size 384,384

# 2) 准备 BiRefNet 粗 alpha 缓存
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/train_refiner.py --prepare --infer_size 512

# 3) 训练 Refiner（约 27 分钟）
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/train_refiner.py \
  --epochs 80 --bs 4 --lr 1e-3 --size 384

# 4) 推理 refined alpha
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/predict_alpha.py \
  --model refined --split test --out_dir experiments/fine_tuned/preds_refined --refine_size 384

# 5) 评估
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/evaluate_matting.py \
  --gts data/matting/test --preds experiments/fine_tuned/preds_refined \
  --gt_suffix _alpha --names_file data/matting/test_names.txt \
  --out experiments/fine_tuned/eval/eval_refined.csv --label Refined

# 6) 视觉对比图
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/visualize_comparison.py \
  --inputs experiments/demo/inputs --gts data/matting/test \
  --baseline data/matting/coarse_cache/test \
  --ours experiments/fine_tuned/preds_refined \
  --out_dir experiments/fine_tuned/comparison --n 24

# 7) 四组光影实验
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/experiment_relighting.py \
  --n_samples 24 --alpha_dir experiments/fine_tuned/preds_refined

# 8) 消融 + 失败案例
"C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe" training/scripts/experiment_failure.py \
  --alpha_dir experiments/fine_tuned/preds_refined
```

## Agent 集成接口

主入口 `ai-service/pipeline.py:run_pipeline()`：
- 输入：`image_path`, `bg_path`, `out_dir`
- 可选：`enable_relight`, `enable_shadow`, `enable_harmonize`, `composite_method`
- **新增**：`precomputed_alpha`（可让 Agent 把 matting 单独调用后只跑光影）
- 返回：每阶段耗时 + 中间产物路径 + final 图

详细 schema 见 `pipeline.py:TOOL_SCHEMAS`（待 A 组接口冻结后对接）。

## 未尽事项（留待 09-15 后）

1. **真实人像数据微调**：当前仅合成数据；P3M-10k / AM-2k / Distinctions-646 在 GitHub 解封或迁到 hf-mirror 后接入。
2. **T04 Relight 与 T06 Harmonize 区分度**：当前两者共享 MKL 数学，单独启用效果一致；下一步给 T04 加"方向性光照"分量。
3. **B6 主观评分**：需团队 ≥10 人双盲填表（评分模板已生成）。
4. **B7 LoRA**：4GB 显存不可行，已降级为"接口预留 + 文档说明"，待 8GB+ 机器重启。
5. **Baseline@1024**：当前 baseline 是 BiRefNet@512（节省时间、与 refined 公平对比），可在 09-15 后用 1024 重做更严格 baseline。
