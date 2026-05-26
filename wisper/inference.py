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

    if "cuda" in device:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    # Newer transformers may store eos_token_id as a list, which breaks
    # SuppressTokensLogitsProcessor slice indexing. Normalize to int.
    for cfg in (model.config, getattr(model, "generation_config", None)):
        if cfg is not None and isinstance(getattr(cfg, "eos_token_id", None), list):
            cfg.eos_token_id = cfg.eos_token_id[0]

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=1,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device=device,
    )

    return pipe


def transcribe(audio_path: str, model_path: str = None, device: str = None):
    print(f"[transcribe] audio_path='{audio_path}' abs='{os.path.abspath(audio_path)}' exists={os.path.isfile(audio_path)}")
    if not os.path.isfile(audio_path):
        raise ValueError(f"Audio path '{audio_path}' does not exist or is not a file")
    pipe = load_pipeline(model_path, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = pipe(
        audio_path,
        generate_kwargs={"language": LANGUAGE, "task": TASK, "num_beams": 1, "do_sample": False},
        return_timestamps=True,
        #return_timestamps="word",
    )
    return result


def _format_srt_time(seconds: float) -> str:
    seconds = seconds or 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_results(result: dict, base_path: str):
    txt_path = base_path + ".txt"
    srt_path = base_path + ".srt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.get("text", "").strip() + "\n")
    print(f"Saved text: {txt_path}")

    chunks = result.get("chunks", [])
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            ts = chunk.get("timestamp", (0.0, 0.0))
            start = _format_srt_time(ts[0])
            end = _format_srt_time(ts[1])
            f.write(f"{i}\n{start} --> {end}\n{chunk['text'].strip()}\n\n")
    print(f"Saved SRT:  {srt_path}")


def main():
    parser = argparse.ArgumentParser(description="Whisper inference")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default=BASE_MODEL_ID, help=f"Model path (default: {BASE_MODEL_ID})")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--timestamps", action="store_true", help="Show timestamps")
    parser.add_argument("--out", default="text_whisper", help="Output base path (default: text_whisper)")
    args = parser.parse_args()

    result = transcribe(args.audio_path, args.model, args.device)
    save_results(result, args.out)

    if args.timestamps and "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (None, None))
            print(f"[{ts[0]:.1f}s - {ts[1]:.1f}s] {chunk['text']}")
    else:
        print(result["text"])


# Example: python -m wisper.inference audio.wav --model openai/whisper-large-v3 --timestamps
if __name__ == "__main__":
    main()
