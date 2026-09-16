#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# commit_phases.sh —— 系统合并成果的分阶段提交脚本（Git Bash）
#
# 背景：mimosa 安全插件的 commit 门禁把前端 app.js 的同源 REST 轮询客户端
# 误判为 SSRF 高危（基线原版代码同样被拦，规则面向服务端代码）。
# 待插件停用/调整后，在仓库根执行本脚本即可完成全部入库。
#
# 用法：bash project/scripts/commit_phases.sh
set -e
cd "$(dirname "$0")/../.."

echo "== Phase 0: 旧布局退役 + 资产回收 =="
git add -A
git commit -m "chore: 退役旧版根布局，资产并入 project/ 统一结构

- 删除根目录旧版迭代 235 个文件（已被 project/ 规范骨架取代，历史保留可回溯）
- 旧 planner 评测语料迁移至 project/agent/eval/
- sft_v1 训练记录迁移至 project/training/sft_v1_record/，training/README.md 归档迭代结论"

echo "== Phase 1: 前端合并 =="
git add project/frontend .gitignore
git commit -m "feat(frontend): ImageCompose 工作台并入 frontend/（C1/C2）

- 13 章叙事首页 + CREATE/RUN/SUMMARY 工作台 + 回放（纯静态，无构建）
- tools/dev_server.py 同源反代 /api /healthz /artifacts /speech
- 安全加固：API 基址恒同源（移除 ?api=/meta 跨域覆盖，消除 SSRF 滥用面）
- INTEGRATION.md -> docs/integration-site.md"

echo "== Phase 2: 千问接入层 =="
git add project/.env.example project/agent/config.py project/agent/critic/__init__.py project/model_registry.yaml project/requirements.txt
git commit -m "feat(models): 模型接入统一走千问（DashScope OpenAI 兼容）

- .env 自动加载（真实环境变量优先）；LLM_* Planner / VLM_* Critic 独立配置
- T02 千问文生图档设计就绪（registry 登记 planned），当前 mock 档（L3 降级线）"

echo "== Phase 3: 语音链路 =="
git add project/speech project/frontend/app.js project/frontend/index.html project/frontend/styles.css project/frontend/tools/dev_server.py project/frontend/tools/voice_flow_test.js
git commit -m "feat(speech): C3 语音链路——ASR 双档 + 浏览器本地 TTS + 字幕播报

- speech/: qwen3-asr-flash（兼容模式 input_audio）/ faster-whisper(Silero VAD) 双档懒加载
- 前端麦克风（16kHz WAV 原生采集）-> 识别回填可改（二次确认）；AI 字幕 speechSynthesis 跟读 + 评分播报
- 任何一环不可用：明确错误码 + 文本输入兜底（不白屏）"

echo "== Phase 4: 多轮编辑 + 条件回滚 UI =="
git add project/frontend/app.js project/frontend/index.html project/frontend/styles.css project/frontend/tools/multiround_test.js
git commit -m "feat(frontend): C4/C6 多轮编辑 + HISTORY 版本树/条件回滚 UI

- SUMMARY 继续编辑（同 session 多轮，只重跑受影响下游）
- 版本树渲染 + 一键回滚；规范 §3.5 五轮剧本自动化 multiround_test.js"

echo "== Phase 5: 验收核销 =="
git add project/README.md project/docs project/agent/eval project/scripts project/demo project/frontend/README.md project/frontend/tools
git commit -m "docs(eval): 107 条指令集评测（规范 §2.2A 全达标）+ README 核销表 + Demo 归档

- agent/eval: G1-G8 分组语料（含已知局限诚实失败例）；scripts/planner_eval.py
- Tool 95.33% / Order 98.13% / Param 96.26% / Spatial 100%，延迟 <1ms
- demo/screenshots: 4 个 Demo 剧本截图归档；docs/merge-report-2026-09-15.md"

echo "== 完成 =="
git log --oneline -6
