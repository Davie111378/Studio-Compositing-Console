# ImageCompose 接入手册 — 算法模型 API（B 组）

> 结论：前端 ↔ agent 的绑定已全量验证；agent ↔ ai-service 的 HTTP 分离拓扑已完整演练。
> 接入真模型当天只需要做「接入步骤」两件事，页面零改动。
>
> **2026-09-13 更新：B 组算法引擎（Studio-Compositing-Console）已合并进 ai-service**，
> 「接入真模型」的前半部分已经完成——T03/T04/T05/T06/T07 现在跑的就是 B 组真实算法
> （Lambert 半球光照估计 / 方向性重打光 / 程序化接触阴影 / harmonize-v2 和谐化 / 特效链），
> T01 抠图为四级引擎链：真实 BiRefNet（torch+权重就绪自动启用）→ rembg 离线神经网络
> （onnxruntime，已装机）→ GrabCut 双候选 → 色彩距离兜底。
> 详见 `project/ai-service/bsrc/README.md` 与 `project/ai-service/aiservice/bsrc_loader.py`。

## B 组算法引擎（已合并）

- 引擎选择：环境变量 `IMC_ENGINE=auto|bsrc|mock`（默认 auto：有 opencv 即用 B 组算法，缺依赖自动回落原 mock，信封契约不变）。
- 算法来源：`project/ai-service/bsrc/`（上游 b3107bc，与 B 组仓库逐字一致，便于后续 PR 合并）。
- 效果验收：`python project/scripts/bsrc_engine_check.py`（T01→T08 全链出图 + 引擎核对）。
- 真实 BiRefNet：`pip install torch torchvision` 后设 `BIREFNET_WEIGHT_PATH` 指向权重即自动启用，无需改任何其他代码。

## 接入步骤（当天操作）

```bash
# 1. B 组把真模型实现进 ai-service（保持信封协议与 artifact 命名不变）
# 2. 独立启动 ai-service（默认 :8100）
python ai-service/run.py

# 3. agent 切到 HTTP 模式启动
set AI_SERVICE_URL=http://127.0.0.1:8100
python scripts/run_agent.py        # :8000

# 4. 前端开发服务器（在 project/frontend 目录下）（:8899，静态 + 同源反代 /api /healthz /artifacts -> :8000）
python project/frontend/tools/dev_server.py
```

健康自检（三个 200）：

```bash
curl http://127.0.0.1:8100/healthz   # ai-service
curl http://127.0.0.1:8000/healthz   # 应显示 "provider":"HttpProvider"、"ai_service_url" 非空
curl http://127.0.0.1:8899/healthz   # 经 dev_server 代理，应返回与上相同内容
```

## 契约要点（B 组必须遵守）

- **信封协议**：`POST /invoke/{tool}`，请求 `{tool, version, request_id, inputs, options, out_dir}`；
  响应 `{status:"success", ...}` 或 `{status:"error", error:{code, message, retryable}}`。
- **artifact 落盘**：ai-service 写入 `ARTIFACTS_DIR`（与 agent 同根，同机部署共享文件系统），
  返回 `artifact://相对路径`；agent 经 `public_artifact_url` 映射为同源 `/artifacts/...`。
- **产物命名**（前端按前缀识别，命名错误时前端有 urls[0] 兜底，但请保持规范）：
  T01 `rgba.png`、T02 `bg.png`、T04 `relit.png`、T05 `composited.png`、T06 `harmonized.png`、T07 `enhanced.png`、T08 `final.png`。
- **T03 无图**：仅数值输出 `outputs.light_dir.azimuth`（度）与 `outputs.color_temp`（K）；
  字段缺失时前端徽章保持「—」占位，不报错。
- **图片尺寸自由**：前端运行时读取 naturalWidth/Height，任意分辨率都正确摆位（不依赖 1024×1536）。

## 前端容错行为（已自动化验证：tools/drift_test.js）

| 异常形态 | 前端行为 |
|---|---|
| 轮询断连/超时 | 8s 后 Agent 提示「连接不稳定，正在重试…」；持续 60s 无成功才终止；恢复后自动继续 |
| 模型输出缺字段 | 光照徽章占位「—」，无 NaN，不中断 |
| run status=failed | SUMMARY 如实显示「执行中断」+ 中断说明；成片回退到最后完成帧；回放可见中断前过程 |
| 节点 artifact 缺失 | 回放层留空不裂图；NODE 盒显示「输出—」 |
| 跨域 artifact（?api= 指独立后端） | 画布合成自动 crossOrigin=anonymous（agent CORS 已放行 *） |
| 后端完全离线 | CREATE 面板显示「离线 · 演示模式」；提交走 SimulatedJob，无假 LIVE |

## 已验证的拓扑（2026-09-13 演练记录）

1. **前端 → dev_server(:8899) → agent(:8000, EmbeddedProvider)**：real_flow_test 全绿（21.7s，7 节点，2 次真实 critic 重跑，final v3）。
2. **前端 → dev_server(:8899, IMC_AGENT_URL=:8001) → agent(:8001, HttpProvider) → ai-service(:8100)**：
   后端流程 25.6s 全 done、critic 3 轮、产物全部落盘；前端 real_flow_test 同样全绿。
   ——这就是接入真模型当天的完整拓扑，已提前彩排通过。

## 回归命令（改完任何一侧后跑）

```bash
cd project && python -m pytest tests/test_api.py -q          # 后端 API 契约
python project/scripts/bsrc_engine_check.py                  # B 组引擎全链出图验收（T01→T08）
python project/frontend/tools/backend_flow_test.py          # 后端全链路（agent :8000 需在跑）
D:\codex-tools\node-v22.17.0-win-x64\node.exe project/frontend/tools/real_flow_test.js   # 前端端到端（真实+离线）
D:\codex-tools\node-v22.17.0-win-x64\node.exe project/frontend/tools/smoke_test.js       # 13 章滚动回归
D:\codex-tools\node-v22.17.0-win-x64\node.exe project/frontend/tools/drift_test.js       # 异常形态容错
```
