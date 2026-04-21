import argparse
import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from .constants import MODEL_DIR, BASE_MODEL_ID, LANGUAGE, TASK


def load_pipeline(model_path: str = None, device: str = None):
    model_path = model_path or f"{MODEL_DIR}/final"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    )
    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device=device,
    )

    return pipe


def transcribe(audio_path: str, model_path: str = None, device: str = None):
    print(f"[transcribe] audio_path='{audio_path}' abs='{os.path.abspath(audio_path)}' exists={os.path.isfile(audio_path)}")
    if not os.path.isfile(audio_path):
        raise ValueError(f"Audio path '{audio_path}' does not exist or is not a file")
    pipe = load_pipeline(model_path, device)
    result = pipe(
        audio_path,
        generate_kwargs={"language": LANGUAGE, "task": TASK},
        return_timestamps=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Whisper inference")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default=None, help=f"Model path (default: {MODEL_DIR}/final)")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--timestamps", action="store_true", help="Show timestamps")
    args = parser.parse_args()

    result = transcribe(args.audio_path, args.model, args.device)

    if args.timestamps and "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (None, None))
            print(f"[{ts[0]:.1f}s - {ts[1]:.1f}s] {chunk['text']}")
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
