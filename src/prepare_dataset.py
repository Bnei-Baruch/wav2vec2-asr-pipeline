import os
import csv
from os.path import isfile, join, exists
import time
import urllib.request
import pysrt
from pydub import AudioSegment
from tqdm import tqdm
import argparse
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)
from .constants import ROW_DATA_DIR, DATASET_DIR, VOCAB_PATH
from .utils import load_dataset_from_dir


def prepare_data(audio_path, srt_path, output_dir):
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    print(f"Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    # Load SRT
    print(f"Loading subtitles: {srt_path}")
    subs = pysrt.open(srt_path)

    metadata = []

    print("Processing segments...")
    for i, sub in enumerate(tqdm(subs)):
        start_ms = msBySubEdge(sub.start)
        end_ms = msBySubEdge(sub.end)
        text = sub.text.replace("\n", " ").strip()
        if not text:
            print(f"empty text")
            continue

        if end_ms - start_ms > 15000:
            print(f"too long: {sub.text} duration: {end_ms - start_ms}ms")
            continue

        if end_ms - start_ms < 1000:
            print(f"too short: {sub.text} duration: {end_ms - start_ms}ms")
            continue

        chunk = audio[start_ms:end_ms]
        chunk = chunk.set_frame_rate(16000).set_channels(1)

        clip_name = f"clip_{i:06d}.wav"
        clip_path = os.path.join(clips_dir, clip_name)
        chunk.export(clip_path, format="wav")

        metadata.append({"sentence": text, "file_name": f"clips/{clip_name}"})

    print(f"Writing metadata to {os.path.join(output_dir, 'metadata.csv')}")
    with open(
        os.path.join(output_dir, "metadata.csv"), "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "file_name"])
        writer.writeheader()
        writer.writerows(metadata)

    print("Done! Dataset is ready.")


def msBySubEdge(subEdge):
    return (
        subEdge.hours * 3600 + subEdge.minutes * 60 + subEdge.seconds
    ) * 1000 + subEdge.milliseconds


def prepare_data_by_uid(uid: str):
    dir = join(ROW_DATA_DIR, uid)
    if not exists(dir):
        return None
    files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    audio_path = None
    srt_path = None
    for file in files:
        if file.endswith(".csv"):
            url = url_from_csv(join(dir, file))
            a_file = url.split("/")[-1]
            audio_path = join(dir, a_file)
            if not isfile(audio_path):
                download_audio(url, dir)
                time.sleep(10)
        elif file.endswith(".srt"):
            srt_path = join(dir, file)
        else:
            continue
    if audio_path is None or srt_path is None:
        print(f"Audio or SRT file not found for {uid}")
        return None
    output_dir = join(DATASET_DIR, f"{uid}")
    os.makedirs(output_dir, exist_ok=True)
    prepare_data(audio_path, srt_path, output_dir)


def url_from_csv(path: str):
    print(f"Reading CSV: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        url = list(reader)[1][2]
        print(f"URL: {url}")
        return url


def download_audio(url: str, dir: str):
    os.makedirs(dir, exist_ok=True)
    filename = url.split("/")[-1]
    print(f"Downloading audio: {url}")
    audio_path = os.path.join(dir, filename)
    urllib.request.urlretrieve(url, audio_path)

    return audio_path


def data_to_dataset():
    ds = load_dataset_from_dir()

    if not os.path.exists(VOCAB_PATH):
        exit(f"Vocab file not found: {VOCAB_PATH}")

    print("Create Processor")
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    print("Prepare Audio")
    split = ds.train_test_split(test_size=0.05, seed=42)
    eval_ds = split["test"]
    train_ds = split["train"]

    print(f"Eval dataset size: {len(eval_ds)}")
    print(f"Train dataset size: {len(train_ds)}")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = [
            processor(a["array"], sampling_rate=a["sampling_rate"]).input_values[0]
            for a in audio
        ]
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    print("Prepare Eval Dataset")
    eval_ds = eval_ds.map(
        prepare_dataset,
        remove_columns=["audio", "sentence"],
        keep_in_memory=False,
        batch_size=100,
        batched=True,
        num_proc=1,
    )
    eval_ds.save_to_disk("./eval")

    print("Prepare Train Dataset")
    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=["audio", "sentence"],
        keep_in_memory=False,
        batch_size=100,
        batched=True,
        num_proc=1,
    )
    train_ds.save_to_disk("./train")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Audio+SRT to HuggingFace AudioFolder dataset"
    )
    parser.add_argument("--uid", required=False, help="Content unit uid")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare data")
    args = parser.parse_args()
    print(f"Args: {args}")

    if args.skip_prepare:
        print("Skipping prepare data")
        data_to_dataset()
        exit()

    os.makedirs(DATASET_DIR, exist_ok=True)
    dirs = [
        d
        for d in os.listdir(ROW_DATA_DIR)
        if os.path.isdir(os.path.join(ROW_DATA_DIR, d))
    ]
    for dir in dirs:
        print(f"Preparing dataset for {dir}")
        if os.path.exists(os.path.join(DATASET_DIR, dir)):
            print(f"Dataset already exists for {dir}")
            continue
        prepare_data_by_uid(dir)
    data_to_dataset()
