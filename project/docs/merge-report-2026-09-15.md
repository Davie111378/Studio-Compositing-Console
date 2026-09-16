# 系统合并报告 —— 2026-09-15

> 范围：`imagecompose-site` 前端并入工程、语音链路实装、千问模型接入层、
> 多轮/回滚 UI、验收核销。对照《项目总规范 V1.0》与《生产实习报告》所述效果。

## 1. 合并内容

| 变更 | 说明 |
|---|---|
| 前端并入 | `imagecompose-site/*` → `frontend/`（index.html / app.js / styles.css / media / tools / design_v2 / legacy-v1-canvas；INTEGRATION.md → docs/integration-site.md）。前端为纯静态站点，无构建依赖；`tools/dev_server.py`(:8899) 同源反代 `/api /healthz /artifacts /speech` |
| 旧布局退役 | 根目录旧版迭代（agent/agent_training/ai-agent/ai-service/app/web/third_party 等 235 文件）删除入库；评测语料与训练记录迁移至 `agent/eval/`、`training/sft_v1_record/`（training/README.md 有取回方式） |
| 语音链路 | 新增 `speech/`（:8200）：ASR 双档（千问 qwen3-asr-flash / 本地 faster-whisper+Silero VAD）+ TTS（浏览器 speechSynthesis 主档 / pyttsx3 兜底）；前端 CREATE 面板麦克风（浏览器原生 WAV 采集）、识别文本回填可改（二次确认降级）、AI 字幕自动语音跟读、完成评分播报、TTS 开关 |
| 千问接入 | `.env.example` 全量模板 + agent 启动自动加载 `.env`；Planner 走 DashScope OpenAI 兼容模式（LLM_*）；VLM Critic 独立配置（VLM_*，默认 qwen-vl-max）；T02 千问文生图档设计就绪（见 §3 阻塞） |
| 多轮/回滚 UI | SUMMARY 面板"继续编辑"（同 session 多轮）+ HISTORY 版本树（/versions 渲染，一键"回滚到此版"→ /rollback）；自然语言回滚语义（"换回背景上一版，保留现在的光"）同样可用 |
| 验收核销 | 107 条指令集 + scripts/planner_eval.py（§2.2A 四项全达标，含 5 例诚实失败例）；README 核销表；demo/screenshots 4 剧本归档（tools/demo_captures.js） |

## 2. 安全加固（合并过程中发现并修复）

1. **前端 API 基址 SSRF 滥用面**：原 `?api=` / meta 可把整页 API 流量指向任意主机。已改为**恒同源**（全部经 dev_server 反代），`assertFetchUrl` 拒绝一切跨域目标。
2. **服务端 T2I 下载校验**（设计稿，待落盘）：结果 URL 出自 API 响应（不可信），下载前强制 https + 域名后缀白名单 + DNS 解析阻断私网/环回/链路本地。
3. 训练/评测等历史脚本不入库（内容经 git 历史可溯），避免未审代码入库。

## 3. 已知阻塞（工具链冲突，非工程问题）

| 项 | 现象 | 处置 |
|---|---|---|
| T02 千问文生图档 | 安全插件（Mimosa）Write-scan 连续拦截"服务端下载 API 返回 URL"的代码（4 次，含 https+白名单+DNS 公网校验的版本） | 保留 mock 渐变档（规范 L3 降级线允许）；SDK 版实现已设计好，插件策略放开后即可落盘 |
| git 提交 | commit 前扫描将前端 app.js 的 `callApi`（同源 REST 轮询客户端）判为 SSRF 入口（高危）。经多轮加固后仍按模式拦截，基线原版同样被拦 | 需仓库所有者裁决插件策略（白名单/调整规则/临时停用）后分阶段提交 |

## 4. 回归记录（合并后全绿）

- `pytest tests/ -q`：37 passed
- `bsrc_engine_check.py`：B 组引擎生效（T03-T07），成片落盘
- 前端 smoke / drift / interact：全绿，零 pageerror
- voice_flow_test：UI/探针/降级/兜底 通过（ASR 实转写待 key）
- multiround_test：R1-R5 全 done（20.2s/15.9s/5.3s/6.4s/17.7s），版本树 7 角色 42 个可回滚按钮
- demo_captures：4 剧本截图/JSON 归档 demo/screenshots/
- planner_eval（107 条）：Tool 95.33% / Order 98.13% / Param 96.26% / Spatial 100%，延迟 <1ms
