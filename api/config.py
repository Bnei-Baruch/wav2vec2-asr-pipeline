DEVICE = "auto"  # auto | cuda | cpu

DEFAULT_MODEL = "whisper-large-v3-he-lr-1e5"

# Model registry: alias -> HuggingFace model ID or local path
MODELS: dict[str, str] = {
    "whisper-large-v3-he-lr-1e5": "./models/whisper-large-v3-he-lr-1e5/checkpoint-26000",
    "ivrit-ai":                   "ivrit-ai/whisper-large-v3",
    "whisper-v3-audiofolder":     "./models/whisper-v3-audiofolder",
}

# WhisperX settings
# HF checkpoints are converted to CTranslate2 on first use and cached here.
CT2_CACHE_DIR = "./models/_ct2_cache"
CT2_QUANTIZATION = "float16"  # float16 | float32 | int8 | int8_float16
# wav2vec2 model used for word-level forced alignment (Hebrew).
ALIGN_MODEL = "imvladikon/wav2vec2-xls-r-300m-hebrew"
WHISPERX_BATCH_SIZE = 8
