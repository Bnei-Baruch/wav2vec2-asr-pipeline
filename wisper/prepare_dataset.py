import csv
import os
from datasets import Audio, Dataset

from .constants import DATASET_DIR

TRAIN_OUT = "./train"
EVAL_OUT = "./eval"
EVAL_SIZE = 0.1


def load_dataset_from_dir():
    records = []
    for name in sorted(os.listdir(DATASET_DIR)):
        sub_dir = os.path.join(DATASET_DIR, name)
        meta = os.path.join(sub_dir, "metadata.csv")
        if not os.path.isdir(sub_dir) or not os.path.isfile(meta):
            continue
        with open(meta, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records.append({
                    "audio": os.path.join(sub_dir, row["file_name"]),
                    "sentence": row["sentence"],
                })
    if not records:
        raise RuntimeError(f"No datasets found in {DATASET_DIR}")
    return Dataset.from_list(records).cast_column("audio", Audio(sampling_rate=16000))


def data_to_dataset():
    ds = load_dataset_from_dir()
    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)

    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    train_ds.save_to_disk(TRAIN_OUT)
    print(f"Saved train: {len(train_ds)} samples -> {TRAIN_OUT}")

    eval_ds.save_to_disk(EVAL_OUT)
    print(f"Saved eval: {len(eval_ds)} samples -> {EVAL_OUT}")


if __name__ == "__main__":
    data_to_dataset()
