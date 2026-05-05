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


TRAIN_OUT = "./train"
EVAL_OUT = "./eval"
EVAL_SIZE = 0.1


def data_to_dataset():
    ds = load_dataset_from_dir()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
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

    map_kwargs = dict(remove_columns=train_ds.column_names, num_proc=1, load_from_cache_file=False)

    print("Preparing train dataset...")
    train_ds = train_ds.map(prepare_batch, **map_kwargs)
    train_ds.save_to_disk(TRAIN_OUT)
    train_ds.cleanup_cache_files()
    print(f"Saved train: {len(train_ds)} samples -> {TRAIN_OUT}")

    print("Preparing eval dataset...")
    eval_ds = eval_ds.map(prepare_batch, **map_kwargs)
    eval_ds.save_to_disk(EVAL_OUT)
    eval_ds.cleanup_cache_files()
    print(f"Saved eval: {len(eval_ds)} samples -> {EVAL_OUT}")


if __name__ == "__main__":
    data_to_dataset()
