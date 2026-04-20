import argparse
import os
import torch
import librosa
from transformers import (
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    pipeline,
)
from pyctcdecode import build_ctcdecoder
from .constants import MODEL_DIR, VOCAB_PATH, KENLM_MODEL_PATH

MODEL_NAME = f"{MODEL_DIR}/checkpoint-329670"


def main(audio_path):
    print(f"Running pipeline with model {MODEL_NAME}")
    result = run_pipeline(audio_path)
    print(f"\n=== RAW CHUNKS ({len(result.get('chunks', []))} total) ===")
    for i, chunk in enumerate(result.get("chunks", [])):
        print(f"  [{i}] text={repr(chunk['text'])} ts={chunk['timestamp']}")
    make_srt(result)
    save_as_txt(result)

def save_as_txt(result):
    with open("text.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Transcription saved to text.txt")

def run_pipeline_no_lm(audio_path):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    audio, rate = librosa.load(audio_path, sr=16000)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids[0])
    print(f"Transcription: {transcription}")
    return transcription


def run_pipeline(audio_path):
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    vocab_dict = tokenizer.get_vocab()
    vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]

    if "|" in vocab_dict:
        vocab[vocab_dict["|"]] = " "
    elif " " in vocab_dict:
        vocab[vocab_dict[" "]] = " "
    else:
        print("Warning: vocab has no word delimiter token ('|' or ' ')")

    print(f"Vocab: {vocab}")

    if os.path.exists(KENLM_MODEL_PATH):
        decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=KENLM_MODEL_PATH,
        )
    else:
        print(f"Warning: kenlm model not found at {KENLM_MODEL_PATH}, using decoder without LM")
        decoder = build_ctcdecoder(labels=vocab)

    print(f"Decoder: {decoder}")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder=decoder,
    )

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor_with_lm.tokenizer,
        feature_extractor=processor_with_lm.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )

    result = asr_pipeline(audio_path, return_timestamps="word", batch_size=16, chunk_length_s=30, stride_length_s=5)

    print(f"\n\n\nResult text: \n{result['text']}\n\n\n")
    return result


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def make_srt(result):
    srt_content = ""
    for i, chunk in enumerate(result["chunks"], start=1):
        start_time = format_timestamp(chunk["timestamp"][0])
        end_time = format_timestamp(chunk["timestamp"][1])
        text = chunk["text"].strip()

        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

    with open("subtitles.srt", "w", encoding="utf-8") as f:
        f.write(srt_content)


def check_is_ready():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Number of devices: {torch.cuda.device_count()}")
    else:
        print("GPU not found, using CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument("--audio_path", type=str, required=False)
    parser.add_argument("--check", action="store_true", required=False)
    args = parser.parse_args()
    print(f"Args: {args}")
    if args.check:
        print("Checking if environment is ready...")
        check_is_ready()
        print("Environment is ready")
        exit(0)
    else:
        audio_path = args.audio_path
        if not audio_path or not os.path.exists(audio_path):
            audio_path = "audio.mp3"
        main(audio_path)
