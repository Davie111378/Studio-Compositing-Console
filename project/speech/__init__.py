# -*- coding: utf-8 -*-
"""speech —— C3 语音链路（ASR + VAD + TTS），规范 §5.5。

与 Agent 只耦合文本：录音 -> /api/v1/speech/transcribe -> 文本 -> instructions 接口；
Agent message 事件 -> 浏览器 speechSynthesis（Windows 本地引擎）播报。
全链路可离线降级：无 key 走本地 faster-whisper；语音不可用时文本输入兜底。
"""
