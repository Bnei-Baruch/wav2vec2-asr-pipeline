import argparse
import os
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    pipeline,
)
from pyctcdecode import build_ctcdecoder

# MODEL_NAME = "facebook/wav2vec2-base-960h"
MODEL_NAME = "./models/wav2vec2-large-xlsr-custom"
VOCAB_PATH = "./vocab.json"
KENLM_MODEL_PATH = "./kenlm.arpa"


def main(audio_path):
    print(f"Running pipeline with model {MODEL_NAME}")
    result = run_pipeline(audio_path)
    make_srt(result)


def run_pipeline(audio_path):
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

    vocab_dict = tokenizer.get_vocab()
    vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]

    if "|" in vocab_dict:
        vocab[vocab_dict["|"]] = " "

    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=KENLM_MODEL_PATH,
    )

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
    # speech, rate = torchaudio.load(audio_path)
    # if rate != 16000:
    #    resampler = torchaudio.transforms.Resample(rate, 16000)
    #    speech = resampler(speech)
    # speech = speech.squeeze().numpy()
    # inputs = processor_with_lm(speech, sampling_rate=16000, return_tensors="pt")
    # with torch.no_grad():
    #    logits = model(inputs.input_values).logits
    # result = processor_with_lm.batch_decode(logits.numpy())

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
