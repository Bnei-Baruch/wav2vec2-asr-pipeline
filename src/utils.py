import os
from datasets import concatenate_datasets, load_dataset
import re
from .constants import DATASET_DIR

CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:"\“\%\‘\”\]]'


def remove_special_characters(batch):
    batch["sentence"] = [
        re.sub(CHARS_TO_IGNORE_REGEX, "", s).lower() if s is not None else ""
        for s in batch["sentence"]
    ]
    return batch


def load_dataset_from_dir():
    print(f"Loading local dataset from {DATASET_DIR}...")
    datasets = []
    for dir in os.listdir(DATASET_DIR):
        if not os.path.isdir(os.path.join(DATASET_DIR, dir)):
            print(f"Not a directory: {dir}")
            continue
        if not os.path.exists(os.path.join(DATASET_DIR, dir, "metadata.csv")):
            print(f"No metadata.csv file found for {dir}")
            continue
        ds = load_dataset("audiofolder", data_dir=os.path.join(DATASET_DIR, dir))
        datasets.append(ds["train"])
    dataset = concatenate_datasets(datasets)
    print(f"Dataset size: {len(dataset)}")
    dataset = dataset.map(
        remove_special_characters,
        batched=True,
        batch_size=1000,
        keep_in_memory=False,
        num_proc=1,
    )
    return dataset
