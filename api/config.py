import os
from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "auto")  # auto | cuda | cpu

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "whisper-large-v3-he-lr-1e5")

# Model registry: alias -> HuggingFace model ID or local path
MODELS: dict[str, str] = {
    "whisper-large-v3-he-lr-1e5":    os.getenv("FINETUNED_MODEL_DIR", "./models/whisper-large-v3-he-lr-1e5/checkpoint-26000"),
    "ivrit-ai":                       os.getenv("IVRIT_AI_MODEL",      "ivrit-ai/whisper-large-v3"),
    "whisper-large-v3-he-base":       os.getenv("BASE_MODEL",          "./models/whisper-large-v3-he-base"),
}
