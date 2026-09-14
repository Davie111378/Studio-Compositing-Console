# training（B 组：训练脚本与配置）

- 子目录：`matting/`（B3 抠取微调）、`lora/`（B7）、`lighting/`
- 硬性要求（规范 N4）：每个训练目录必须 checkpoint（20–30 分钟一存）+ resume + train.log + config.yaml 四件套；单次租卡 ≤24h。
- 租卡前检查清单与流程见总规范 §1.4；训练完成把权重路径/指标登记进仓库根的 `model_registry.yaml`。
