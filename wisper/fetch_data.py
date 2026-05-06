"""Download audio from CSV URLs and build audiofolder datasets under DATASET_DIR."""

import argparse
import csv
import os
import re
import time
import urllib.request
from os.path import exists, isfile, join

import pysrt
from pydub import AudioSegment
from tqdm import tqdm

from .constants import DATASET_DIR, ROW_DATA_DIR


def ms_by_sub_edge(sub_edge):
    return (
        sub_edge.hours * 3600 + sub_edge.minutes * 60 + sub_edge.seconds
    ) * 1000 + sub_edge.milliseconds


def prepare_data(audio_path, srt_path, output_dir):
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    print(f"Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    print(f"Loading subtitles: {srt_path}")
    subs = pysrt.open(srt_path)

    metadata = []

    print("Processing segments...")
    for i, sub in enumerate(tqdm(subs)):
        start_ms = ms_by_sub_edge(sub.start)
        end_ms = ms_by_sub_edge(sub.end)

        text = sub.text.replace("\n", " ")
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        duration_ms = end_ms - start_ms
        if duration_ms > 30000:
            print(f"too long ({duration_ms}ms), skipping: {text[:40]}")
            continue
        if duration_ms < 300:
            print(f"too short ({duration_ms}ms), skipping: {text[:40]}")
            continue

        chunk = audio[start_ms:end_ms]
        chunk = chunk.set_frame_rate(16000).set_channels(1)

        clip_name = f"clip_{i:06d}.wav"
        clip_path = os.path.join(clips_dir, clip_name)
        chunk.export(clip_path, format="wav")

        metadata.append({"sentence": text, "file_name": f"clips/{clip_name}"})

    csv_path = os.path.join(output_dir, "metadata.csv")
    print(f"Writing metadata: {csv_path} ({len(metadata)} segments)")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "file_name"])
        writer.writeheader()
        writer.writerows(metadata)


def prepare_data_by_uid(uid: str):
    dir_path = join(ROW_DATA_DIR, uid)
    if not exists(dir_path):
        return None
    files = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f))]
    audio_path = None
    srt_path = None
    for file in files:
        if file.endswith(".csv"):
            url = url_from_csv(join(dir_path, file))
            a_file = url.split("/")[-1]
            audio_path = join(dir_path, a_file)
            if not isfile(audio_path):
                download_audio(url, dir_path)
                time.sleep(10)
        elif file.endswith(".srt"):
            srt_path = join(dir_path, file)
    if audio_path is None or srt_path is None:
        print(f"Audio or SRT file not found for {uid}")
        return None
    output_dir = join(DATASET_DIR, uid)
    os.makedirs(output_dir, exist_ok=True)
    prepare_data(audio_path, srt_path, output_dir)
    return srt_path


def url_from_csv(path: str):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)[1][2]


def download_audio(url: str, dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    filename = url.split("/")[-1]
    audio_path = os.path.join(dir_path, filename)
    urllib.request.urlretrieve(url, audio_path)
    return audio_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch audio from row_data (CSV URL + SRT) and write audiofolder under dataset/"
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    os.makedirs(DATASET_DIR, exist_ok=True)

    dirs = os.listdir(ROW_DATA_DIR)
    processed = []
    for d in dirs:
        if os.path.exists(os.path.join(DATASET_DIR, d)):
            print(f"Skipping {d} (already exists)")
            continue
        print(f"Preparing {d}")
        srt_path = prepare_data_by_uid(d)
        processed.append(( d, srt_path))

    if processed:
        with open("whisper_data.txt", "w") as f:
            f.write(f"Processed: {len(processed)}\n")
            f.write(f"First: {processed[0]}\n")
            f.write(f"Last: {processed[-1]}\n")
            f.write("Full list:\n")
            for d in processed:
                f.write(f"  {d}\n")


# Example: python -m wisper.fetch_data
if __name__ == "__main__":
    main()
