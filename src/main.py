import argparse
import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, pipeline
from pyctcdecode import build_ctcdecoder
import kenlm

MODEL_NAME = "facebook/wav2vec2-base-960h"
KENLM_MODEL_PATH = "../kenlm/model.arpa"

def main(audio_path):
    print(f"Running pipeline with model {MODEL_NAME}")
    result = run_pipeline(audio_path)
    make_srt(result)


def run_pipeline(audio_path):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    # 2. Подготовка словаря (нужен список токенов, отсортированный по ID)
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = sorted((v, k) for k, v in vocab_dict.items())
    vocab = [k for _, k in sorted_vocab_dict]

    # 3. Создание декодера (укажите путь к вашему .arpa или .bin файлу)
    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=KENLM_MODEL_PATH,
    )

    # 4. Сборка Wav2Vec2ProcessorWithLM
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )

    # 5. Использование в пайплайне
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        tokenizer=processor_with_lm,  # Передаем процессор с LM как токенизатор
    )

    # Результат
    result = asr_pipeline(audio_path, return_timestamps="word", batch_size=16)
    print(f"Result: {result}")
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


def check_environment():
    print(f"Torch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    try:
        print("Transformers: Wav2Vec2 available")
        model = kenlm.Model
        print("KenLM: available")
        print("Pyannote.audio: available")
    except ImportError as e:
        print(f"Missing dependency: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()
    audio_path = args.audio_path
    if not audio_path or not os.path.exists(audio_path):
        audio_path = "audio.mp3"
    main(audio_path)
