import csv
import os
from datasets import Dataset, Audio
from transformers import WhisperProcessor

from .constants import DATASET_DIR, BASE_MODEL_ID, LANGUAGE, TASK

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
    return Dataset.from_list(records)


def data_to_dataset():
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    ds = load_dataset_from_dir()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def extract_features(batch):
        arrays = [a["array"] for a in batch["audio"]]
        feats = processor.feature_extractor(arrays, sampling_rate=16000)
        batch["input_features"] = feats.input_features
        batch["labels"] = [processor.tokenizer(s).input_ids for s in batch["sentence"]]
        return batch

    ds = ds.map(
        extract_features,
        batched=True,
        batch_size=32,
        remove_columns=["audio", "sentence"],
        desc="Extracting features",
    )

    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    train_ds.save_to_disk(TRAIN_OUT)
    print(f"Saved train: {len(train_ds)} samples -> {TRAIN_OUT}")

    eval_ds.save_to_disk(EVAL_OUT)
    print(f"Saved eval: {len(eval_ds)} samples -> {EVAL_OUT}")


#Example: python -m wisper.prepare_dataset
if __name__ == "__main__":
    data_to_dataset()
