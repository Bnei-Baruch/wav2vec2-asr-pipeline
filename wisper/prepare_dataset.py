from datasets import load_dataset, Dataset, Audio, Features, Array2D, Sequence, Value
from transformers import WhisperProcessor

from .constants import DATASET_DIR, BASE_MODEL_ID

TRAIN_OUT = "./train"
EVAL_OUT = "./eval"
EVAL_SIZE = 0.1
NUM_TRAIN_SHARDS = 10


def load_dataset_from_dir():
    ds = load_dataset("audiofolder", data_dir=DATASET_DIR, split="train")
    if len(ds) == 0:
        raise RuntimeError(f"No datasets found in {DATASET_DIR}")
    return ds


def data_to_dataset():
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

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
                    yield {"input_features": feat, "labels": processor.tokenizer(sent).input_ids}
                batch_audio, batch_sentences = [], []
        if batch_audio:
            feats = processor.feature_extractor(batch_audio, sampling_rate=16000)
            for feat, sent in zip(feats.input_features, batch_sentences):
                yield {"input_features": feat, "labels": processor.tokenizer(sent).input_ids}

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

    total_train = 0
    for shard_idx in range(NUM_TRAIN_SHARDS):
        shard = train_ds.shard(num_shards=NUM_TRAIN_SHARDS, index=shard_idx)
        out_path = f"{TRAIN_OUT}_{shard_idx}"
        print(f"Preparing train shard {shard_idx + 1}/{NUM_TRAIN_SHARDS} ({len(shard)} samples)...")
        shard_result = Dataset.from_generator(
            make_generator,
            gen_kwargs={"source_ds": shard},
            features=features,
            writer_batch_size=2000,
        )
        shard_result.save_to_disk(out_path)
        shard_result.cleanup_cache_files()
        total_train += len(shard_result)
        print(f"Saved shard {shard_idx}: {len(shard_result)} samples -> {out_path}")
    print(f"Done. Total train samples: {total_train}")


#Example: python -m wisper.prepare_dataset
if __name__ == "__main__":
    data_to_dataset()
