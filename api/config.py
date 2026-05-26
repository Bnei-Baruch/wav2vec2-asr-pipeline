import os
from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "auto")  # auto | cuda | cpu

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "finetuned")

# Model registry: alias -> HuggingFace model ID or local path
MODELS: dict[str, str] = {
    "finetuned": os.getenv("FINETUNED_MODEL_DIR", "./models/whisper-large-v3-he/final"),
    "ivrit-ai":  os.getenv("IVRIT_AI_MODEL",      "ivrit-ai/whisper-large-v3"),
    "base":      os.getenv("BASE_MODEL",           "openai/whisper-large-v3"),
}
