import os
import csv
from os.path import isfile, join, exists
import time
import urllib.request
import pysrt
from pydub import AudioSegment
from tqdm import tqdm
import argparse

ROW_DATA_DIR = "./row_data"
DATASET_DIR = "./dataset"


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
        start_ms = (
            sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds
        ) * 1000 + sub.start.milliseconds
        end_ms = (
            sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds
        ) * 1000 + sub.end.milliseconds

        if end_ms - start_ms > 10000:
            print(f"too long: {sub.text} {start_ms}ms - {end_ms}ms {end_ms - start_ms}ms")
            continue

        chunk = audio[start_ms:end_ms]

        clip_name = f"clip_{i:06d}.wav"
        clip_path = os.path.join(clips_dir, clip_name)

        chunk = chunk.set_frame_rate(16000).set_channels(1)
        chunk.export(clip_path, format="wav")
        text = sub.text.replace("\n", " ").strip()

        if text:  # skip empty
            metadata.append({"file_name": f"clips/{clip_name}", "sentence": text})

    csv_path = os.path.join(output_dir, "metadata.csv")
    print(f"Writing metadata to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "sentence"])
        writer.writeheader()
        writer.writerows(metadata)

    print("Done! Dataset is ready.")


def prepare_dataset_by_uid(uid: str):
    dir = join(ROW_DATA_DIR, uid)
    if not exists(dir):
        return None
    files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    output_dir = join(DATASET_DIR, uid)
    os.makedirs(output_dir, exist_ok=True)
    for file in files:
        if file.endswith(".csv"):
            url = url_from_csv(join(dir, file))
            audio_path = download_audio(url, output_dir)
        else:
            srt_path = join(dir, file)
    return prepare_dataset(audio_path, srt_path, output_dir)


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
    # prepare_dataset_by_uid(args.uid)
    dirs = [d for d in os.listdir(ROW_DATA_DIR) if os.path.isdir(os.path.join(ROW_DATA_DIR,d))]
    for dir in dirs:
        print(f"Preparing dataset for {dir}")
        time.sleep(10)
        prepare_dataset_by_uid(dir)
