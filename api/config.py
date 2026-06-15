DEVICE = "auto"  # auto | cuda | cpu

DEFAULT_MODEL = "whisper-large-v3-he-lr-1e5"

# Model registry: alias -> HuggingFace model ID or local path
MODELS: dict[str, str] = {
    "whisper-large-v3-he-lr-1e5": "./models/whisper-large-v3-he-lr-1e5/checkpoint-26000",
    "ivrit-ai":                   "ivrit-ai/whisper-large-v3",
    "whisper-v3-audiofolder":     "./models/whisper-v3-audiofolder",
}
