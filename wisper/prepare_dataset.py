import os
import csv
import re
import time
import urllib.request
from os.path import isfile, join, exists

import pysrt
from pydub import AudioSegment
from tqdm import tqdm
import argparse
from datasets import load_dataset, Audio

from transformers import WhisperProcessor

from .constants import ROW_DATA_DIR, DATASET_DIR, BASE_MODEL_ID, LANGUAGE, TASK


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
        text = re.sub(r"[^א-ת\s]", "", text)
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


def load_dataset_from_dir():
    """Load all audiofolder sub-datasets from DATASET_DIR and concatenate."""
    from datasets import concatenate_datasets

    all_ds = []
    for name in sorted(os.listdir(DATASET_DIR)):
        sub_dir = os.path.join(DATASET_DIR, name)
        meta = os.path.join(sub_dir, "metadata.csv")
        if not os.path.isdir(sub_dir) or not os.path.isfile(meta):
            continue
        ds = load_dataset("audiofolder", data_dir=sub_dir, split="train")
        all_ds.append(ds)
    if not all_ds:
        raise RuntimeError(f"No datasets found in {DATASET_DIR}")
    combined = concatenate_datasets(all_ds)
    print(f"Loaded {len(combined)} samples from {len(all_ds)} sub-datasets")
    return combined


def _cleanup_hf_cache():
    import shutil
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    if os.path.isdir(cache_dir):
        before = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, filenames in os.walk(cache_dir)
            for f in filenames
        ) / (1024 ** 3)
        shutil.rmtree(cache_dir)
        print(f"Cleaned HF datasets cache ({before:.1f} GB): {cache_dir}")


def data_to_dataset():
    """Convert audiofolder data into Whisper-ready train/eval arrow datasets."""
    ds = load_dataset_from_dir()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    split = ds.train_test_split(test_size=0.05, seed=42)
    eval_ds = split["test"]
    train_ds = split["train"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    def prepare_batch(batch):
        audio = batch["audio"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        labels = processor.tokenizer(batch["sentence"]).input_ids

        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch

    print("Preparing eval dataset...")
    eval_ds = eval_ds.map(
        prepare_batch,
        remove_columns=eval_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
    )
    eval_ds.save_to_disk("./whisper_eval")
    eval_ds.cleanup_cache_files()
    _cleanup_hf_cache()
    print(f"Saved eval: {len(eval_ds)} samples -> ./whisper_eval")

    cols = train_ds.column_names
    total = len(train_ds)
    n_parts = 5
    part_size = total // n_parts

    print(f"Preparing train dataset in {n_parts} parts ({part_size}+ samples each)...")
    for i in range(n_parts):
        start = i * part_size
        end = total if i == n_parts - 1 else (i + 1) * part_size
        part = train_ds.select(range(start, end))
        part = part.map(prepare_batch, remove_columns=cols, num_proc=1, load_from_cache_file=False)
        part.save_to_disk(f"./whisper_train/part_{i}")
        part.cleanup_cache_files()
        del part
        _cleanup_hf_cache()
        print(f"  Part {i}: {start}-{end} saved, cache cleaned")

    print(f"Saved train: {total} samples in {n_parts} parts -> ./whisper_train/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Whisper fine-tuning")
    parser.add_argument("--uid", required=False, help="Content unit uid")
    parser.add_argument(
        "--skip-prepare",
        default=True,
        action="store_true",
        help="Skip SRT slicing, go straight to dataset encoding",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    print(f"Args: {args}")

    if args.skip_prepare:
        data_to_dataset()
        exit()

    os.makedirs(DATASET_DIR, exist_ok=True)

    if args.uid:
        prepare_data_by_uid(args.uid)
    else:
        dirs = sorted(
            d for d in os.listdir(ROW_DATA_DIR)
            if os.path.isdir(os.path.join(ROW_DATA_DIR, d))
        )
        for d in dirs[args.start:args.end]:
            if os.path.exists(os.path.join(DATASET_DIR, d)):
                print(f"Skipping {d} (already exists)")
                continue
            print(f"Preparing {d}")
            prepare_data_by_uid(d)

    data_to_dataset()
