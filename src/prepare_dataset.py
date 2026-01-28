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



def prepare_dataset(audio_path, srt_path, output_dir):
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

        metadata.append({"sentence": text, "path": clip_path})
        
    print(f"Writing metadata to {os.path.join(output_dir, 'metadata.csv')}")
    with open(os.path.join(output_dir, 'metadata.csv'), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "path"])
        writer.writeheader()
        writer.writerows(metadata)

    print("Done! Dataset is ready.")

def msBySubEdge(subEdge):
    return (
        subEdge.hours * 3600 + subEdge.minutes * 60 + subEdge.seconds
    ) * 1000 + subEdge.milliseconds

def prepare_dataset_by_uid(uid: str):
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
    prepare_dataset(audio_path, srt_path, output_dir)


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
