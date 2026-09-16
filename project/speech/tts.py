# -*- coding: utf-8 -*-
"""TTS：主档浏览器 speechSynthesis（Windows 本地语音引擎，零依赖零流量，前端实现）。

服务端仅提供 pyttsx3 档（可选安装）作为无浏览器场景兜底：文本 -> WAV 文件。
不接入云端 TTS（合成音频需从 API 返回的 URL 下载，无浏览器场景才需要）。
"""

from __future__ import annotations

import logging
import os
import tempfile
import time

logger = logging.getLogger("speech.tts")


class TtsUnavailable(RuntimeError):
    """TTS 档位不可用。"""


def provider() -> str:
    return os.environ.get("TTS_PROVIDER", "browser").strip().lower()


def synth_to_wav(text: str) -> bytes:
    """pyttsx3（Windows SAPI5 本地引擎）合成 WAV；未安装或非 pyttsx3 档抛 TtsUnavailable。"""
    if provider() not in ("pyttsx3", "auto"):
        raise TtsUnavailable("TTS_PROVIDER 未启用服务端合成（browser 档由前端 speechSynthesis 完成）")
    try:
        import pyttsx3  # noqa: PLC0415
    except Exception as e:
        raise TtsUnavailable(f"pyttsx3 未安装（pip install pyttsx3）：{e}") from e
    engine = pyttsx3.init()
    for v in engine.getProperty("voices"):
        if any(k in (v.name or "").lower() for k in ("huihui", "yaoyao", "chinese", "zh")):
            engine.setProperty("voice", v.id)
            break
    engine.setProperty("rate", 180)
    path = os.path.join(tempfile.gettempdir(), f"imc_tts_{int(time.time() * 1000)}.wav")
    engine.save_to_file(text, path)
    engine.runAndWait()
    try:
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
