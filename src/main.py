import argparse
import os
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC,
    pipeline,
)
from pyctcdecode import build_ctcdecoder

MODEL_NAME = "facebook/wav2vec2-base-960h"
KENLM_MODEL_PATH = "./kenlm.arpa"


def main(audio_path):
    print(f"Running pipeline with model {MODEL_NAME}")
    result = run_pipeline(audio_path)
    make_srt(result)


def run_pipeline(audio_path):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = sorted((v, k) for k, v in vocab_dict.items())
    vocab = [k for _, k in sorted_vocab_dict]

    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=KENLM_MODEL_PATH,
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )
    #speech, rate = torchaudio.load(audio_path)
    #if rate != 16000:
    #    resampler = torchaudio.transforms.Resample(rate, 16000)
    #    speech = resampler(speech)
    #speech = speech.squeeze().numpy()
    #inputs = processor_with_lm(speech, sampling_rate=16000, return_tensors="pt")
    #with torch.no_grad():
    #    logits = model(inputs.input_values).logits
    #result = processor_with_lm.batch_decode(logits.numpy())

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        feature_extractor=processor_with_lm.feature_extractor,
        tokenizer=processor_with_lm.tokenizer,
        decoder=decoder,
        device=0 if torch.cuda.is_available() else -1,
    )

    result = asr_pipeline(audio_path, return_timestamps="word", batch_size=16)

    print(f"\n\n\nResult text: \n{result}\n\n\n")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument("--audio_path", type=str, required=False)
    args = parser.parse_args()
    print(f"Args: {args}")
    audio_path = args.audio_path
    if not audio_path or not os.path.exists(audio_path):
        audio_path = "audio.mp3"
    main(audio_path)
