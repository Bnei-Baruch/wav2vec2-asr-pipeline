import os
from datasets import load_dataset, Audio, concatenate_datasets

from transformers import WhisperProcessor

from .constants import DATASET_DIR, BASE_MODEL_ID

CHUNK_SIZE = 200


def load_dataset_from_dir(chunk: int = 0):
    """Load audiofolders from DATASET_DIR, concatenate, then take rows [chunk*size : (chunk+1)*size)."""
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
    n = len(combined)
    start = chunk * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n)
    if start >= n:
        raise IndexError(
            f"chunk={chunk} (rows {start}–…) is past dataset end (n={n})"
        )
    combined = combined.select(range(start, end))
    print(
        f"Loaded {n} samples from {len(all_ds)} sub-datasets; using chunk {chunk}: rows [{start}, {end}) ({len(combined)} rows)"
    )
    return combined


def data_to_dataset(chunk: int = 0):
    """Convert one chunk of audiofolder data into Whisper-ready train/eval arrow datasets."""
    ds = load_dataset_from_dir(chunk=chunk)
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
    print(f"Saved eval: {len(eval_ds)} samples -> ./whisper_eval")

    print("Preparing train dataset...")
    train_ds = train_ds.map(
        prepare_batch,
        remove_columns=train_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
    )
    train_ds.save_to_disk("./whisper_train")
    train_ds.cleanup_cache_files()
    print(f"Saved train: {len(train_ds)} samples -> ./whisper_train/")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ch", type=int, default=0, help=f"Which slice of {CHUNK_SIZE} rows (0-based).")
    args = p.parse_args()
    data_to_dataset(chunk=args.ch)
