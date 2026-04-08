import os
from datasets import concatenate_datasets, load_dataset
from .constants import DATASET_DIR


def load_dataset_from_dir():
    print(f"Loading local dataset from {DATASET_DIR}...")
    datasets = []
    it = 0
    for dir in os.listdir(DATASET_DIR):
        if not os.path.isdir(os.path.join(DATASET_DIR, dir)):
            print(f"Not a directory: {dir}")
            continue
        if not os.path.exists(os.path.join(DATASET_DIR, dir, "metadata.csv")):
            print(f"No metadata.csv file found for {dir}")
            continue
        ds = load_dataset("audiofolder", data_dir=os.path.join(DATASET_DIR, dir))
        datasets.append(ds["train"])
        it += 1
        print(f"Loaded {it} datasets")
    dataset = concatenate_datasets(datasets)
    print(f"Dataset size: {len(dataset)}")
    return dataset
