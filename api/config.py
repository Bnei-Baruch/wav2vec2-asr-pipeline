import os

from dotenv import load_dotenv

load_dotenv()  # read CLAUDE_API_KEY and friends from .env

DEVICE = "auto"  # auto | cuda | cpu

DEFAULT_MODEL = "whisper-prev"

# Model registry: alias -> HuggingFace model ID or local path
MODELS: dict[str, str] = {
    "whisper-prev":   "./models/whisper-prev",
    "whisper-noprev": "./models/whisper-noprev",
}

# Domain-term glossary (Kabbalah terms/abbreviations), one term per line.
# Fed verbatim to faster-whisper `hotwords` — the file itself is the curation
# (no code-side filtering). NOTE: faster-whisper truncates hotwords to ~223
# tokens, so terms are biased in FILE ORDER — put the most important first.
TERMS_FILE = os.path.join(os.path.dirname(__file__), "terms_he.txt")


def load_terms(path: str = TERMS_FILE) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [t.strip() for t in f if t.strip()]
    except FileNotFoundError:
        return []


def build_hotwords() -> str | None:
    terms = load_terms()
    return ", ".join(terms) if terms else None


HOTWORDS = build_hotwords()


# WhisperX settings
# HF checkpoints are converted to CTranslate2 on first use and cached here.
CT2_CACHE_DIR = "./models/_ct2_cache"
CT2_QUANTIZATION = "float16"  # float16 | float32 | int8 | int8_float16
# Source of tokenizer/feature-extractor when a local checkpoint omits them
# (raw Trainer checkpoints often do). These fine-tunes share the base's tokenizer.
BASE_MODEL = "ivrit-ai/whisper-large-v3"
# wav2vec2 model used for word-level forced alignment (Hebrew).
ALIGN_MODEL = "imvladikon/wav2vec2-xls-r-300m-hebrew"
# RTX 3070 has only 8 GB VRAM; large-v3 fp16 + batched decode OOMs at 8.
# Keep small (raise only on a bigger GPU).
WHISPERX_BATCH_SIZE = 4

# Claude API — used to review ASR output for likely recognition errors.
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
