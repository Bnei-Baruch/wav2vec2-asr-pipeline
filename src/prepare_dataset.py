import os
import csv
from os.path import isfile, join, exists
import time
import urllib.request
import pysrt
from pydub import AudioSegment
from tqdm import tqdm
import argparse
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from .constants import ROW_DATA_DIR, DATASET_DIR


def prepare_dataset(audio_path, srt_path, csv_path):
    print(f"Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    # Load SRT
    print(f"Loading subtitles: {srt_path}")
    subs = pysrt.open(srt_path)

    metadata = []

    fe = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    print("Processing segments...")
    for i, sub in enumerate(tqdm(subs)):
        start_ms = (
            sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds
        ) * 1000 + sub.start.milliseconds
        end_ms = (
            sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds
        ) * 1000 + sub.end.milliseconds

        if end_ms - start_ms > 15000:
            print(f"too long: {sub.text} duration: {end_ms - start_ms}ms")
            continue

        if end_ms - start_ms < 500 or sub.text.strip() == "":
            print(f"too short or empty: {sub.text} duration: {end_ms - start_ms}ms")
            continue

        chunk = audio[start_ms:end_ms]
        chunk = chunk.set_frame_rate(16000).set_channels(1)

        samples = np.array(chunk.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (8 * chunk.sample_width - 1)
        out = fe(samples, sampling_rate=16000, return_tensors="np")

        text = sub.text.replace("\n", " ").strip()

        if text:
            metadata.append({"sentence": text, "input_values": out["input_values"][0]})

    print(f"Writing metadata to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "input_values"])
        writer.writeheader()
        writer.writerows(metadata)

    print("Done! Dataset is ready.")


def prepare_dataset_by_uid(uid: str):
    dir = join(ROW_DATA_DIR, uid)
    if not exists(dir):
        return None
    files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    csv_path = join(DATASET_DIR, f"{uid}.csv")
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
    return prepare_dataset(audio_path, srt_path, csv_path)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Audio+SRT to HuggingFace AudioFolder dataset"
    )
    parser.add_argument("--uid", required=False, help="Content unit uid")
    args = parser.parse_args()
    os.makedirs(DATASET_DIR, exist_ok=True)
    # prepare_dataset_by_uid(args.uid)
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
        prepare_dataset_by_uid(dir)
