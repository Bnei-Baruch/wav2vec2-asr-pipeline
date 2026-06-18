"""WhisperX inference: faster-whisper (CT2) transcription + wav2vec2 forced alignment.

HuggingFace Whisper checkpoints are converted to CTranslate2 format on first use and
cached on disk, since WhisperX/faster-whisper cannot load raw HF checkpoints directly.
"""

import gc
import os
import shutil

import torch

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the omegaconf
# objects inside WhisperX's Pyannote VAD checkpoint (UnpicklingError). These models
# come from HuggingFace and are trusted, so restore the legacy full-pickle load.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    # Force off even when callers (e.g. lightning_fabric) pass weights_only=True.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

import whisperx  # noqa: E402  (import after the torch.load patch)

from .constants import LANGUAGE
from .convert_ct2 import convert as convert_to_ct2

# Training-state files that aren't needed for inference / conversion.
_TRAINING_FILES = {
    "optimizer.pt", "scheduler.pt", "rng_state.pth",
    "trainer_state.json", "training_args.bin", "scaler.pt",
}


def _has_processor_files(d: str) -> bool:
    """True if the dir already has a tokenizer and feature-extractor config."""
    has_tokenizer = os.path.isfile(os.path.join(d, "tokenizer.json")) or \
        os.path.isfile(os.path.join(d, "vocab.json"))
    has_preprocessor = os.path.isfile(os.path.join(d, "preprocessor_config.json"))
    return has_tokenizer and has_preprocessor


def _stage_with_processor(model_path: str, base_model: str, staging_dir: str) -> str:
    """Symlink the model's weights/config into staging_dir and add the tokenizer +
    feature extractor from base_model (raw Trainer checkpoints usually omit these)."""
    from transformers import WhisperProcessor

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    for name in os.listdir(model_path):
        if name in _TRAINING_FILES:
            continue
        src = os.path.join(model_path, name)
        if os.path.isfile(src):
            os.symlink(os.path.abspath(src), os.path.join(staging_dir, name))

    WhisperProcessor.from_pretrained(base_model).save_pretrained(staging_dir)
    return staging_dir


def _ensure_ct2(
    model_path: str,
    cache_dir: str,
    quantization: str = "float16",
    base_model: str = None,
) -> str:
    """Return a CT2 model dir for an HF checkpoint / hub id, converting + caching on first use."""
    # Already a CT2 model directory?
    if os.path.isfile(os.path.join(model_path, "model.bin")):
        return model_path

    safe = model_path.strip("./").replace("/", "__").replace("\\", "__")
    out_dir = os.path.join(cache_dir, safe)
    if os.path.isfile(os.path.join(out_dir, "model.bin")):
        return out_dir

    os.makedirs(cache_dir, exist_ok=True)

    # Local checkpoints may lack tokenizer/preprocessor files; stage them from base_model.
    src = model_path
    staging = None
    if os.path.isdir(model_path) and not _has_processor_files(model_path):
        if not base_model:
            raise RuntimeError(
                f"'{model_path}' has no tokenizer/preprocessor and no base_model was given "
                f"to supply them. Set BASE_MODEL in api/config.py."
            )
        staging = os.path.join(cache_dir, safe + "__staging")
        src = _stage_with_processor(model_path, base_model, staging)

    try:
        convert_to_ct2(model_path=src, output_dir=out_dir, quantization=quantization)
    finally:
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
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
    base_model: str = None,
) -> dict:
    """Transcribe + force-align an audio file.

    Returns {"text": str, "chunks": [{"start", "end", "text", "words": [...]}]}.
    """
    dev, dev_index = _split_device(device)
    compute_type = "float16" if dev == "cuda" else "int8"

    ct2_path = _ensure_ct2(model_path, ct2_cache_dir, quantization, base_model=base_model)
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
