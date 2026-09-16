# -*- coding: utf-8 -*-
"""speech FastAPI 服务（默认 :8200）——C3 语音链路。

路由：
  GET  /healthz                      -> 档位自检
  POST /api/v1/speech/transcribe     -> multipart 音频 -> {text, confidence, provider, ms}
  POST /api/v1/speech/tts            -> {text} -> audio/wav（pyttsx3 档；browser 档返回 501）

降级原则（规范 5.7）：语音任何一环不可用都返回明确错误码，前端回退文本输入，绝不白屏。
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from speech.asr import SpeechUnavailable, effective_provider, transcribe
from speech.tts import TtsUnavailable, provider as tts_provider, synth_to_wav

logger = logging.getLogger("speech")

app = FastAPI(title="ImageCompose Speech Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class TtsIn(BaseModel):
    text: str


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "service": "speech",
        "asr_provider": effective_provider(),
        "tts_provider": tts_provider(),
    }


@app.post("/api/v1/speech/transcribe")
async def transcribe_route(request: Request) -> dict:
    """裸字节音频体（audio/wav 等），与前端 MediaRecorder 采集直传对应。"""
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="空音频")
    if len(audio) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="音频超过 8MB")
    try:
        return transcribe(audio, request.headers.get("content-type", "audio/wav"))
    except SpeechUnavailable as e:
        return JSONResponse(status_code=503, content={
            "error": {"code": "E_SPEECH_UNAVAILABLE", "message": str(e), "retryable": False},
        })


@app.post("/api/v1/speech/tts")
def tts_route(body: TtsIn) -> Response:
    text = body.text.strip()[:500]
    if not text:
        raise HTTPException(status_code=400, detail="空文本")
    try:
        wav = synth_to_wav(text)
    except TtsUnavailable as e:
        return JSONResponse(status_code=501, content={
            "error": {"code": "E_TTS_UNAVAILABLE", "message": str(e), "retryable": False},
        })
    return Response(content=wav, media_type="audio/wav")


def create_app() -> FastAPI:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    return app


if __name__ == "__main__":
    import uvicorn
    create_app()
    uvicorn.run(app, host=os.environ.get("SPEECH_HOST", "127.0.0.1"),
                port=int(os.environ.get("SPEECH_PORT", "8200")))
