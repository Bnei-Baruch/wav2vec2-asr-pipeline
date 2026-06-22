import os

from dotenv import load_dotenv

load_dotenv()  # read CLAUDE_API_KEY and friends from .env

DEVICE = "auto"  # auto | cuda | cpu

DEFAULT_MODEL = "whisper-large-v3-he-lr-1e5"

# Model registry: alias -> HuggingFace model ID or local path
MODELS: dict[str, str] = {
    "whisper-large-v3-he-lr-1e5": "./models/whisper-large-v3-he-lr-1e5/checkpoint-26000",
    "whisper-v3-audiofolder":     "./models/whisper-v3-audiofolder",
}

# WhisperX settings
# HF checkpoints are converted to CTranslate2 on first use and cached here.
CT2_CACHE_DIR = "./models/_ct2_cache"
CT2_QUANTIZATION = "float16"  # float16 | float32 | int8 | int8_float16
# Source of tokenizer/feature-extractor when a local checkpoint omits them
# (raw Trainer checkpoints often do). These fine-tunes share the base's tokenizer.
BASE_MODEL = "ivrit-ai/whisper-large-v3"
# wav2vec2 model used for word-level forced alignment (Hebrew).
ALIGN_MODEL = "imvladikon/wav2vec2-xls-r-300m-hebrew"
WHISPERX_BATCH_SIZE = 8

# Claude API — used to review ASR output for likely recognition errors.
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
