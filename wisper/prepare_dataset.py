import csv
import os
from datasets import Dataset, Audio, Features, Array2D, Sequence, Value
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

    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    features = Features({
        "input_features": Array2D(shape=(128, 3000), dtype="float32"),
        "labels": Sequence(Value("int64")),
    })

    BATCH_SIZE = 128

    def make_generator(source_ds):
        batch_audio, batch_sentences = [], []
        for sample in source_ds:
            batch_audio.append(sample["audio"]["array"])
            batch_sentences.append(sample["sentence"])
            if len(batch_audio) >= BATCH_SIZE:
                feats = processor.feature_extractor(batch_audio, sampling_rate=16000)
                for feat, sent in zip(feats.input_features, batch_sentences):
                    labels = processor.tokenizer(sent).input_ids
                    if labels[0] == processor.tokenizer.bos_token_id:
                        labels = labels[1:]
                    yield {"input_features": feat, "labels": labels}
                batch_audio, batch_sentences = [], []
        if batch_audio:
            feats = processor.feature_extractor(batch_audio, sampling_rate=16000)
            for feat, sent in zip(feats.input_features, batch_sentences):
                labels = processor.tokenizer(sent).input_ids
                if labels[0] == processor.tokenizer.bos_token_id:
                    labels = labels[1:]
                yield {"input_features": feat, "labels": labels}

    print("Preparing eval dataset...")
    eval_result = Dataset.from_generator(
        make_generator,
        gen_kwargs={"source_ds": eval_ds},
        features=features,
        writer_batch_size=2000,
    )
    eval_result.save_to_disk(EVAL_OUT)
    eval_result.cleanup_cache_files()
    print(f"Saved eval: {len(eval_result)} samples -> {EVAL_OUT}")

    print("Preparing train dataset...")
    train_result = Dataset.from_generator(
        make_generator,
        gen_kwargs={"source_ds": train_ds},
        features=features,
        writer_batch_size=2000,
    )
    train_result.save_to_disk(TRAIN_OUT)
    train_result.cleanup_cache_files()
    print(f"Saved train: {len(train_result)} samples -> {TRAIN_OUT}")


#Example: python -m wisper.prepare_dataset
if __name__ == "__main__":
    data_to_dataset()
