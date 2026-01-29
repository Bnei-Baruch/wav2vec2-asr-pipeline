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

    dirs = [os.path.join(DATASET_DIR, uid) for uid in os.listdir(DATASET_DIR)]
    all_datasets = [load_dataset("audiofolder", data_dir=d)["train"] for d in dirs]
    dataset = concatenate_datasets(all_datasets)
    print(f"Dataset size: {len(dataset)}")

    print("Text Preprocessing")
    dataset = dataset.map(
        remove_special_characters, batched=True, batch_size=1000, keep_in_memory=False
    )
    return dataset
