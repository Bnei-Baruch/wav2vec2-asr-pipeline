import asyncio
import logging
import os
import tempfile
import urllib.request
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from wisper.inference_wx import transcribe as wx_transcribe
from api.config import (
    MODELS,
    DEFAULT_MODEL,
    DEVICE,
    CT2_CACHE_DIR,
    CT2_QUANTIZATION,
    ALIGN_MODEL,
    WHISPERX_BATCH_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("api")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="STT API")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# one lock per model — serializes requests, prevents concurrent GPU use
_locks: dict[str, asyncio.Lock] = {alias: asyncio.Lock() for alias in MODELS}

_device = DEVICE if DEVICE != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")


def _fmt_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(c['start'])} --> {_fmt_srt_time(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


def _infer(model_path: str, wav_path: str) -> dict:
    return wx_transcribe(
        wav_path,
        model_path,
        _device,
        ct2_cache_dir=CT2_CACHE_DIR,
        align_model=ALIGN_MODEL,
        batch_size=WHISPERX_BATCH_SIZE,
        quantization=CT2_QUANTIZATION,
    )


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


# POST /stt?model=finetuned|ivrit-ai|base  (file upload or url=...)
# Response: {"text": "...", "chunks": [{"start": 0.0, "end": 3.5, "text": "..."}], "srt": "..."}
@app.post("/stt")
async def speech_to_text(
    file: Optional[UploadFile] = File(default=None),
    model: str = DEFAULT_MODEL,
    url: Optional[str] = None,
):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Available: {list(MODELS)}")
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide file or url")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if url:
                ext = os.path.splitext(url.split("?")[0])[1].lower() or ".mp3"
                src_path = os.path.join(tmpdir, f"input{ext}")
                urllib.request.urlretrieve(url, src_path)
            else:
                ext = os.path.splitext(file.filename)[1].lower() or ".mp3"
                src_path = os.path.join(tmpdir, f"input{ext}")
                content = await file.read()
                with open(src_path, "wb") as f:
                    f.write(content)

            wav_path = os.path.join(tmpdir, "audio.wav")
            audio = AudioSegment.from_file(src_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")

            loop = asyncio.get_event_loop()
            async with _locks[model]:
                result = await loop.run_in_executor(None, _infer, MODELS[model], wav_path)
    except HTTPException:
        raise
    except Exception:
        logger.exception("STT failed: model=%s file=%s url=%s", model, file and file.filename, url)
        raise HTTPException(status_code=500, detail="Internal transcription error")

    # WhisperX returns chunks already shaped as {start, end, text, words}.
    chunks = [
        {
            "start": c.get("start") or 0.0,
            "end": c.get("end") or 0.0,
            "text": (c.get("text") or "").strip(),
            "words": c.get("words", []),
        }
        for c in result.get("chunks", [])
    ]

    return JSONResponse({
        "text": result.get("text", "").strip(),
        "chunks": chunks,
        "srt": _build_srt(chunks),
    })
