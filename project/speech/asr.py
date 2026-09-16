# -*- coding: utf-8 -*-
"""ASR 双档实现：千问 qwen3-asr-flash（OpenAI 兼容 input_audio）+ 本地 faster-whisper。

- qwen 档：ASR_BASE（默认 DashScope 兼容模式）+ ASR_MODEL；仅文本出域（规范"数据不出域"
  在本课题以用户填 key 的方式显式选择云端档时让位，本地档始终可用）。
- local 档：faster-whisper（CTranslate2 CPU int8 + Silero VAD，vad_filter=True），
  模型规模 ASR_LOCAL_MODEL（默认 small），首次使用自动下载后全离线。
- 懒加载：模型/网络只在首次转写时初始化，失败抛 SpeechUnavailable。
"""

from __future__ import annotations

import base64
import logging
import os
import time

import httpx

logger = logging.getLogger("speech.asr")


class SpeechUnavailable(RuntimeError):
    """语音档位不可用（未配置/未安装/调用失败且无回落）。"""


def _provider() -> str:
    return os.environ.get("ASR_PROVIDER", "auto").strip().lower()


def _api_base() -> str:
    return (os.environ.get("ASR_BASE", "")
            or os.environ.get("LLM_API_BASE", "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")


def _api_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("ASR_API_KEY", "")


def _local_model() -> str:
    return os.environ.get("ASR_LOCAL_MODEL", "small")


def effective_provider() -> str:
    """当前实际会采用的档位（供 /healthz 展示）。"""
    p = _provider()
    if p == "off":
        return "off"
    if p == "local":
        return "local"
    if p == "qwen":
        return "qwen"
    # auto：有 key 且装了 httpx（必有）-> qwen，否则 local
    if _api_key():
        return "qwen"
    try:
        import faster_whisper  # noqa: F401
        return "local"
    except Exception:
        return "off"


_whisper_model = None  # 懒加载单例


def _transcribe_local(audio: bytes, content_type: str) -> tuple[str, float]:
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise SpeechUnavailable(f"faster-whisper 未安装（pip install faster-whisper）：{e}") from e
    if _whisper_model is None:
        logger.info("加载 faster-whisper 模型 %s（CPU int8）", _local_model())
        _whisper_model = WhisperModel(_local_model(), device="cpu", compute_type="int8")
    import io
    import tempfile
    suffix = ".wav" if "wav" in content_type else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        tmp = f.name
    try:
        segments, info = _whisper_model.transcribe(tmp, language="zh", vad_filter=True)
        texts = [s.text.strip() for s in segments]
        text = "".join(texts)
        conf = float(info.language_probability) if getattr(info, "language_probability", None) else 0.0
        return text, conf
    finally:
        os.unlink(tmp)


def _transcribe_qwen(audio: bytes, content_type: str) -> tuple[str, float]:
    key = _api_key()
    if not key:
        raise SpeechUnavailable("DASHSCOPE_API_KEY 未配置")
    fmt = "wav" if "wav" in content_type else "mp3"
    b64 = base64.b64encode(audio).decode()
    payload = {
        "model": os.environ.get("ASR_MODEL", "qwen3-asr-flash"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": f"data:audio/{fmt};base64,{b64}",
                                                        "format": fmt}},
            ],
        }],
    }
    try:
        resp = httpx.post(
            f"{_api_base()}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=float(os.environ.get("ASR_TIMEOUT_S", "15")),
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise SpeechUnavailable(f"千问 ASR 调用失败：{e}") from e
    return text, 0.0  # 兼容模式不回传置信度，前端按文本非空判断


def transcribe(audio: bytes, content_type: str = "audio/wav") -> dict:
    """主入口：按档位转写。返回 {text, confidence, provider, ms}；全档失败抛 SpeechUnavailable。"""
    t0 = time.perf_counter()
    provider = effective_provider()
    errors: list[str] = []
    if provider == "off":
        raise SpeechUnavailable("ASR_PROVIDER=off 或无可用档位")
    order = ["qwen", "local"] if provider == "qwen" else \
            ["local", "qwen"] if provider == "local" else ["qwen", "local"]
    last: Exception | None = None
    for p in order:
        if p not in order:
            continue
        try:
            if p == "qwen" and (_provider() in ("qwen", "auto")) and _api_key():
                text, conf = _transcribe_qwen(audio, content_type)
            elif p == "local":
                text, conf = _transcribe_local(audio, content_type)
            else:
                continue
            ms = round((time.perf_counter() - t0) * 1000)
            return {"text": text, "confidence": conf, "provider": p, "ms": ms}
        except SpeechUnavailable as e:
            last = e
            errors.append(f"{p}: {e}")
            if _provider() in ("qwen", "local"):
                break  # 显式指定单档时不静默回落
    raise SpeechUnavailable("；".join(errors) or "无可用 ASR 档位")
