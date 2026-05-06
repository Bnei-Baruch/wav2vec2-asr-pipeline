import argparse
import os
import torch
from faster_whisper import WhisperModel

from .constants import MODEL_DIR, BASE_MODEL_ID, LANGUAGE, TASK
from .inference import _format_srt_time, save_results

WORDS_PER_CHUNK = 10


def load_model(model_path: str = None, device: str = None):
    model_path = model_path or f"{MODEL_DIR}/ct2"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "float32"
    return WhisperModel(model_path, device=device, compute_type=compute_type)


def _group_words(segments, words_per_chunk: int = WORDS_PER_CHUNK):
    chunks = []
    current_words = []
    current_start = None
    current_end = None

    for segment in segments:
        for word in segment.words:
            if current_start is None:
                current_start = word.start
            current_words.append(word.word)
            current_end = word.end

            if len(current_words) >= words_per_chunk:
                chunks.append({
                    "text": "".join(current_words),
                    "timestamp": (current_start, current_end),
                })
                current_words = []
                current_start = None

    if current_words:
        chunks.append({
            "text": "".join(current_words),
            "timestamp": (current_start, current_end),
        })

    return chunks


def transcribe(audio_path: str, model_path: str = None, device: str = None, words_per_chunk: int = WORDS_PER_CHUNK):
    print(f"[transcribe] audio_path='{audio_path}' abs='{os.path.abspath(audio_path)}' exists={os.path.isfile(audio_path)}")
    if not os.path.isfile(audio_path):
        raise ValueError(f"Audio path '{audio_path}' does not exist or is not a file")

    model = load_model(model_path, device)
    segments, _ = model.transcribe(
        audio_path,
        language=LANGUAGE,
        task=TASK,
        word_timestamps=True,
    )
    segments = list(segments)

    chunks = _group_words(segments, words_per_chunk)
    full_text = "".join(s.text for s in segments)
    return {"text": full_text, "chunks": chunks}


def main():
    parser = argparse.ArgumentParser(description="Faster-Whisper inference")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default=None, help=f"CT2 model path (default: {MODEL_DIR}/ct2)")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--words", type=int, default=WORDS_PER_CHUNK, help=f"Words per SRT chunk (default: {WORDS_PER_CHUNK})")
    parser.add_argument("--timestamps", action="store_true", help="Show timestamps")
    parser.add_argument("--out", default="text_whisper", help="Output base path (default: text_whisper)")
    args = parser.parse_args()

    result = transcribe(args.audio_path, args.model, args.device, args.words)
    save_results(result, args.out)

    if args.timestamps and "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (None, None))
            print(f"[{ts[0]:.1f}s - {ts[1]:.1f}s] {chunk['text']}")
    else:
        print(result["text"])


# Example: python -m wisper.inference_fw audio.wav --timestamps --out result
if __name__ == "__main__":
    main()
