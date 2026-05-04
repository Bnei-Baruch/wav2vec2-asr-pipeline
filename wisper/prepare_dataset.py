import argparse
import os
from datasets import load_dataset, Audio, concatenate_datasets

from transformers import WhisperProcessor

from .constants import DATASET_DIR, BASE_MODEL_ID

CHUNK_SIZE = 200


def load_dataset_from_dir():
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
    return concatenate_datasets(all_ds)


def data_to_dataset(out: str = "./train"):
    ds = load_dataset_from_dir()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    train_ds = ds
    print(f"Train: {len(train_ds)}")

    def prepare_batch(batch):
        audio = batch["audio"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        labels = processor.tokenizer(batch["sentence"]).input_ids

        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch

    print("Preparing train dataset...")
    train_ds = train_ds.map(
        prepare_batch,
        remove_columns=train_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
    )
    train_ds.save_to_disk(out)
    train_ds.cleanup_cache_files()
    print(f"Saved train: {len(train_ds)} samples -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./train")
    args = parser.parse_args()
    data_to_dataset(out=args.out)
