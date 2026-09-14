# speech（C 组：ASR + VAD + TTS，全本地部署）

- 目录：`asr/`（识别）、`tts/`（合成）
- 与 Agent 的耦合只有文本：ASR 文本 -> `POST /api/v1/sessions/{sid}/instructions`；
  事件流里 `message.data.text`（assistant）-> TTS 播报。
- 验收（C3）：端到端 ≤800ms、字准 ≥95%、低置信度二次确认 + 文本兜底；界面实时显示"你说了什么 / AI 在做什么"（message 事件直接上屏）。
