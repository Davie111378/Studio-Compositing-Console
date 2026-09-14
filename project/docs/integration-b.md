# B 组算法接入指南（mock -> 真模型）

> 原则（规范 N2/N3）：**换模型只改工具内部实现，Agent 代码零改动**。
> 你唯一的对接面是 `ai-service/aiservice/` 下的 8 个 impl 函数 + `agent/schema/T01–T08.json` 契约。

## 1. 你要改什么

每个工具就是一个纯函数，签名固定：

```python
# ai-service/aiservice/<模块>/impl.py
def run(inputs: dict, options: dict, out_dir: Path, root: Path) -> dict:
    """inputs 里的 URI 用 common.resolve_uri() 解析为本地路径；
    产物保存到 out_dir 并返回 artifact:// URI（用 common.save_png()）。"""
```

| 工具 | 文件 | 当前 mock 算法 | 你要替换成 |
|---|---|---|---|
| T01 matting | `matting/impl.py` | 背景色距 alpha | SAM2(点/框) -> trimap -> BiRefNet |
| T02 background_generate | `diffusion/impl.py` | 关键词渐变+光晕 | SDXL/FLUX + ControlNet(depth)，**必须同时产出 depth_png** |
| T03 lighting_estimate | `lighting/impl.py::run_estimate` | 亮度质心/色温估计 | 现成光照估计器（SH9 + 方向光） |
| T04 relight | `lighting/impl.py::run_relight` | 方向渐变+色温染色 | IC-Light（背景条件模式） |
| T05 shadow_generate | `shadow/impl.py` | 程序化椭圆接触阴影 | 物理推理阴影扩散版（mock 版保留作 L1 降级与 B6-A 基线） |
| T06 harmonize | `harmonization/impl.py::run_harmonize` | 均值/方差颜色迁移 | DoveNet / Harmonizer |
| T07 enhance | `harmonization/impl.py::run_enhance` | 深度加权虚化+锐化 | depth-aware enhancement |
| T08 export | `export_impl.py` | 已是真实 IO | 不用动 |

## 2. 接入步骤（以 T01 为例）

1. 在 `ai-service/` 里加你的依赖与模型加载（torch 等装在 ai-service 的环境/镜像里，别污染 agent）。
2. 改写 `matting/impl.py::run` 内部：加载权重（启动时懒加载一次，模块级缓存）-> 推理 -> 写三个产物。
3. **不改函数签名、不改输出键名**（`rgba_png/alpha_png/mask_png`，与 T01.json 的 output schema 一致）。
4. 失败时抛 `ToolFailure(code, message, retryable)`，错误码从 T01.json 的 errors 表里选。
5. 验证：
   ```bash
   python ai-service/run.py                       # 独立起服务
   curl -X POST http://127.0.0.1:8100/invoke/matting -H "Content-Type: application/json" \
        -d '{"tool":"matting","version":"1.0","request_id":"t1","inputs":{"image":"asset://uploads/..."},"options":{"quality":"normal"},"out_dir":"runs/test/t1"}'
   python scripts/smoke_test.py                    # 全链路回归
   ```
6. 权重落 Model Registry：更新 `model_registry.yaml`（name/version/path/metrics/status/fallback），提 PR 给 A。

## 3. 约束与提醒

- **超时与延迟**：每个工具的 `latency.timeout_ms` 是 Agent 侧强杀上限（T01=15s、T02=60s、T04=60s…）。首次推理要 warmup，避免第一发超时。
- **显存**：T02/T04 与 T01 同时被调度可能同进程抢显存（规范 6.3）；docker-compose 已按工具分容器部署，本地调试注意 `CUDA_VISIBLE_DEVICES`。
- **降级链（规范 5.7）**：真实模型抛 `retryable` 错误时 Agent 自动重试并降精度档位（fine→normal→draft）；你可以在 impl 里按 `options["quality"]` 选步数/分辨率。权重加载失败请抛**不可重试**错误并让 registry fallback 到 Base Model。
- **确定性**：mock 用 seeded_random 保证同输入同输出（测试/回放依赖）。真实模型天然随机没关系，但 eval 指标要可复现（固定 seed/步数）。
- **产物对齐**：T02 的 size 默认 1024；T05 会把背景 cover 到前景画布尺寸（mock 策略），真实 pipeline 的对齐策略由你实现，但输出必须是 `composited_png` 单张成品。

## 4. 训练侧衔接（B3/B4）

- 训练产物放 `training/<方向>/`，实验记录按规范 N5 五件套进 `experiments/<exp_id>/`。
- 微调完成后只需改 `model_registry.yaml` 中对应工具的 prod 指向 + fallback，Agent 无感知切换。
