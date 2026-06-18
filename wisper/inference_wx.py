"""WhisperX inference: faster-whisper (CT2) transcription + wav2vec2 forced alignment.

HuggingFace Whisper checkpoints are converted to CTranslate2 format on first use and
cached on disk, since WhisperX/faster-whisper cannot load raw HF checkpoints directly.
"""

import gc
import os

import torch
import whisperx

from .constants import LANGUAGE
from .convert_ct2 import convert as convert_to_ct2


def _ensure_ct2(model_path: str, cache_dir: str, quantization: str = "float16") -> str:
    """Return a CT2 model dir for an HF checkpoint / hub id, converting + caching on first use."""
    # Already a CT2 model directory?
    if os.path.isfile(os.path.join(model_path, "model.bin")):
        return model_path

    safe = model_path.strip("./").replace("/", "__").replace("\\", "__")
    out_dir = os.path.join(cache_dir, safe)
    if os.path.isfile(os.path.join(out_dir, "model.bin")):
        return out_dir

    os.makedirs(cache_dir, exist_ok=True)
    convert_to_ct2(model_path=model_path, output_dir=out_dir, quantization=quantization)
    return out_dir


def _split_device(device: str):
    """Split 'cuda:0' -> ('cuda', 0); 'cpu' -> ('cpu', 0)."""
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        return "cuda", idx
    return "cpu", 0


def _free(*objs, cuda: bool):
    for obj in objs:
        del obj
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


def transcribe(
    audio_path: str,
    model_path: str,
    device: str,
    *,
    ct2_cache_dir: str,
    align_model: str,
    batch_size: int = 8,
    quantization: str = "float16",
) -> dict:
    """Transcribe + force-align an audio file.

    Returns {"text": str, "chunks": [{"start", "end", "text", "words": [...]}]}.
    """
    dev, dev_index = _split_device(device)
    compute_type = "float16" if dev == "cuda" else "int8"

    ct2_path = _ensure_ct2(model_path, ct2_cache_dir, quantization)
    audio = whisperx.load_audio(audio_path)

    # 1. Transcription (faster-whisper / CT2 backend with VAD chunking).
    model = whisperx.load_model(
        ct2_path,
        device=dev,
        device_index=dev_index,
        compute_type=compute_type,
        language=LANGUAGE,
    )
    try:
        result = model.transcribe(audio, batch_size=batch_size, language=LANGUAGE)
    finally:
        _free(model, cuda=(dev == "cuda"))

    segments = result.get("segments", [])

    # 2. Forced alignment for accurate word-level timestamps.
    if segments:
        align_obj, metadata = whisperx.load_align_model(
            language_code=LANGUAGE, device=dev, model_name=align_model,
        )
        try:
            aligned = whisperx.align(
                segments, align_obj, metadata, audio, dev,
                return_char_alignments=False,
            )
            segments = aligned.get("segments", segments)
        finally:
            _free(align_obj, cuda=(dev == "cuda"))

    chunks = []
    for seg in segments:
        words = [
            {
                "word": w.get("word"),
                "start": w.get("start"),
                "end": w.get("end"),
                "score": w.get("score"),
            }
            for w in seg.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        chunks.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": (seg.get("text") or "").strip(),
            "words": words,
        })

    full_text = " ".join(c["text"] for c in chunks).strip()
    return {"text": full_text, "chunks": chunks}
