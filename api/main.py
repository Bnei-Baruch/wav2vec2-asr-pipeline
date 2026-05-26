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

from wisper.constants import LANGUAGE, TASK
from wisper.inference import load_pipeline
from api.config import MODELS, DEFAULT_MODEL, DEVICE

logger = logging.getLogger("api")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="STT API")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# one lock per model — serializes requests, prevents concurrent GPU use
_locks: dict[str, asyncio.Lock] = {alias: asyncio.Lock() for alias in MODELS}

_device = DEVICE if DEVICE != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")


def _infer(model_path: str, wav_path: str) -> dict:
    pipe = load_pipeline(model_path=model_path, device=_device)
    try:
        return pipe(
            wav_path,
            generate_kwargs={"language": LANGUAGE, "task": TASK, "num_beams": 1, "do_sample": False},
            return_timestamps=True,
        )
    finally:
        del pipe
        if "cuda" in _device:
            torch.cuda.empty_cache()


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


# POST /stt?model=finetuned|ivrit-ai|base  (file upload or url=...)
# Response: {"text": "...", "chunks": [{"start": 0.0, "end": 3.5, "text": "..."}]}
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

    chunks = [
        {"start": ts[0], "end": ts[1], "text": chunk["text"].strip()}
        for chunk in result.get("chunks", [])
        if (ts := chunk.get("timestamp", (0.0, 0.0)))
    ]

    return JSONResponse({"text": result.get("text", "").strip(), "chunks": chunks})
