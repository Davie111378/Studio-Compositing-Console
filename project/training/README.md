# training（B 组：训练脚本与配置）

- 子目录：`matting/`（B3 抠取微调）、`lora/`（B7）、`lighting/`
- 硬性要求（规范 N4）：每个训练目录必须 checkpoint（20–30 分钟一存）+ resume + train.log + config.yaml 四件套；单次租卡 ≤24h。
- 租卡前检查清单与流程见总规范 §1.4；训练完成把权重路径/指标登记进仓库根的 `model_registry.yaml`。

## sft_v1 —— Qwen2.5-0.5B-Instruct Planner LoRA 微调（2026-09-10 迭代，已归档）

- **目标**：把 Qwen2.5-0.5B-Instruct 微调为演播室合成 Planner（instruction → DAG JSON），离线运行不依赖云端 LLM。
- **方法**：LoRA r=16 / alpha=32 / dropout=0.05，target 全线性层，bf16 + gradient checkpointing，4GB 显存可训（batch 2 × accum 8，3 epochs / 338 steps）。
- **数据**：四域提示词库（绿幕换景/键控/抠图/特效/滤镜/多步/拒识/删主体，≥10 万字）构建的 SFT messages 对（`sft_train.jsonl` + `sft_val.jsonl`）。
- **记录文件**（本目录 `sft_v1_record/`）：
  - `train_log.txt` —— 完整训练日志（loss 从 1.43 收敛到 <0.1 量级）
  - `adapter_config.json` —— LoRA 配置快照（PEFT 0.20.0）
  - `eval_report.json` —— 泛化评测（19 条训练集外改写指令）：Tool Selection 36.8% / DAG Order 36.8% / Parameter 47.4% / 拒识 0%
- **结论（如实归档）**：0.5B 本地 SFT 档泛化不达标（规范 §2.2A 要求 ≥95%），未合入主线；主线 Planner 采用 RulePlanner（确定性，评测见 `agent/eval/`）+ LLM Planner（千问 API，OpenAI 兼容模式）。此记录保留作为 B 组训练链路的可复现证据。

## 历史代码（git 历史取回）

旧版布局的训练脚本在仓库提交 `b3107bc` 起的 `agent_training/` 路径下，与本目录记录一一对应：

```bash
git show b3107bc:agent_training/train_sft.py           # LoRA 训练入口
git show b3107bc:agent_training/build_sft_data.py      # SFT 数据构建（SFT_SYSTEM 单一事实来源）
git show b3107bc:agent_training/build_prompt_corpus.py # 四域提示词库生成（≥10 万字）
git show b3107bc:agent_training/build_domain_corpus.py # 领域语料（术语/契约/QA）
git show b3107bc:agent_training/eval_sft.py            # 泛化评测（产出 eval_report.json）
git show b3107bc:agent_training/eval_corpus.py         # 语料内 held-out 评测
git show b3107bc:agent_training/local_planner.py       # 本地 SFT Planner 推理封装
```

> 注意：历史脚本引用旧版布局路径（`agent_training/`、旧 `agent/planner.py` 等），在当前 `project/` 布局下需改路径后才可运行。权重与数据集按规范 N6 不入库（adapter 权重 `.safetensors` 仅本地保留）。

## 后续训练（B3 抠取微调）

- 基线 BiRefNet：SAD / MSE / Grad / Conn 四项指标先行（规范 E2/E3，口径见总规范 §1.3）；
- 微调目标：整体 SAD 相对 baseline 下降 ≥10%，难例集（透明/半透明）下降 ≥15%；
- 权重就绪后设 `BIREFNET_WEIGHT_PATH` 即自动接入 T01 四级引擎链，无需改代码。
