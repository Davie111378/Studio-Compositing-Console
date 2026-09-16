# 部署文档（A9）

## 方式一：单机直跑（最快，演示默认）

```bash
pip install -r requirements.txt
python scripts/run_agent.py          # http://127.0.0.1:8000
python scripts/smoke_test.py         # 冒烟自检，9 项全 PASS 即健康
```

## 方式二：进程分离（B 组接真模型后用这种）

```bash
# 终端 1
python ai-service/run.py                                   # :8100
# 终端 2（Windows PowerShell: $env:AI_SERVICE_URL="http://127.0.0.1:8100"）
AI_SERVICE_URL=http://127.0.0.1:8100 python scripts/run_agent.py   # :8000
```

ai-service 挂掉时 Agent 自动回落内嵌 mock（`AI_FALLBACK_TO_EMBEDDED=0` 可关）。

## 方式三：Docker Compose

```bash
docker compose up -d --build
docker compose ps                    # agent:8000 / ai-service:8100
python scripts/smoke_test.py         # 对 8000 冒烟
```

两个服务共享 `./data/artifacts` 卷；Agent 侧 curl 健康检查通过才算启动完成。

> **网络注记**（2026-09-10 实测）：本机直连 docker.io 拉取 `python:3.12-slim` 会超时，需走镜像源：
> `docker pull docker.m.daocloud.io/library/python:3.12-slim && docker tag docker.m.daocloud.io/library/python:3.12-slim python:3.12-slim`
> 之后 `docker compose build` 正常。已在本机完整验证：构建 → up → 双容器 healthy → 冒烟 9/9 PASS。

## 方式四：完整演示拓扑（前端 + 语音，2026-09-15 起推荐）

```bash
# 终端 1（可选，独立 ai-service）
python ai-service/run.py                       # :8100
# 终端 2：Agent（自动加载项目根 .env，千问 key 填那里）
python scripts/run_agent.py                    # :8000
# 终端 3：语音服务（可选；不启动则前端语音入口显示未启动，文本输入兜底）
python speech/run.py                           # :8200
# 终端 4：前端（静态 + 同源反代 /api /healthz /artifacts -> :8000，/speech -> :8200）
python frontend/tools/dev_server.py            # :8899
```

健康自检：`curl 127.0.0.1:8899/speech/healthz`、`curl 127.0.0.1:8899/healthz` 均应 200。
浏览器打开 http://127.0.0.1:8899 → 开始创作。

## 环境变量速查（全部有默认值，可不配）

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_HOST/PORT` | 127.0.0.1:8000 | Agent 监听 |
| `AI_SERVICE_URL` | (空=内嵌) | 独立 ai-service 地址 |
| `AI_FALLBACK_TO_EMBEDDED` | 1 | HTTP 失败回落内嵌 mock |
| `ARTIFACTS_DIR / RUNS_DIR` | `<repo>/data/...` | 产物与运行记录（两服务必须一致） |
| `MOCK_DELAY_MS` | 120 | mock 工具模拟延迟（测试置 0） |
| `NODE_TIMEOUT_S` | 120 | 单节点强杀上限 |
| `MAX_NODE_RETRIES` | 1 | 可重试错误的重试次数（L1 降级） |
| `CRITIC_THRESHOLD / MAX_REPLANS` | 85 / 2 | Critic 阈值与重规划上限 |
| `PLANNER_MODE` | auto | auto / rule / llm |
| `LLM_API_BASE / LLM_API_KEY / LLM_MODEL` | 空 | 千问 DashScope 兼容端点，配好即启用 LLM Planner |
| `VLM_API_BASE / VLM_API_KEY / VLM_MODEL` | 回落 LLM_* | VLM Critic（建议 qwen-vl-max） |
| `ASR_PROVIDER / ASR_MODEL` | auto / qwen3-asr-flash | 语音输入档位（qwen/local/off） |
| `TTS_PROVIDER` | browser | browser（前端本地引擎）/ pyttsx3 |
| `IMC_AGENT_URL / IMC_SPEECH_URL` | :8000 / :8200 | dev_server 反代上游 |
| `LOG_LEVEL` | INFO | 日志级别 |

## 新机器 15 分钟启动清单（规范 A9 验收）

1. 装 Python 3.11+ 或 Docker；
2. `pip install -r requirements.txt`（或 `docker compose up -d --build`）；
3. `python scripts/run_agent.py`；
4. `python scripts/smoke_test.py` → 9 项 PASS；
5. 前端把 Base URL 指到 `http://<host>:8000`，契约见 `docs/api-spec.md`。

> 中文路径：仓库本身不要放进含中文/空格的目录运行 Docker 挂载（规范 6.3 的坑）；本机 Python 直跑对中文路径兼容。
