import os
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import evaluate
from datasets import load_from_disk, concatenate_datasets, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from .constants import BASE_MODEL_ID, MODEL_DIR, LANGUAGE, TASK, TRAINING_ARGS
from .prepare_dataset import load_dataset_from_dir


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    raw_audio: bool = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if self.raw_audio:
            input_features = [
                {
                    "input_features": self.processor.feature_extractor(
                        f["audio"]["array"], sampling_rate=f["audio"]["sampling_rate"]
                    ).input_features[0]
                }
                for f in features
            ]
            label_features = [
                {"input_ids": self.processor.tokenizer(f["sentence"]).input_ids}
                for f in features
            ]
        else:
            input_features = [{"input_features": f["input_features"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _load_precomputed():
    """Load precomputed Arrow datasets (with input_features/labels)."""
    eval_ds = load_from_disk("./whisper_eval")
    print(f"Eval dataset size: {len(eval_ds)}")

    train_parts_dir = "./whisper_train"
    parts = sorted(
        d
        for d in os.listdir(train_parts_dir)
        if os.path.isdir(os.path.join(train_parts_dir, d)) and d.startswith("part_")
    )
    print(f"Parts: {parts}")
    if parts:
        train_ds = concatenate_datasets(
            [load_from_disk(os.path.join(train_parts_dir, p)) for p in parts]
        )
    else:
        train_ds = load_from_disk(train_parts_dir)
    print(f"Train dataset size: {len(train_ds)}")
    return train_ds, eval_ds


def _load_raw():
    """Load raw audiofolder — feature extraction happens in DataCollator."""
    ds = load_dataset_from_dir()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    split = ds.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def train(raw_audio: bool = False):
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Output dir: {MODEL_DIR}")
    print(
        f"Mode: {'raw audio (on-the-fly features)' if raw_audio else 'precomputed features'}"
    )

    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    print(f"Processor loaded: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    if raw_audio:
        train_ds, eval_ds = _load_raw()
    else:
        train_ds, eval_ds = _load_precomputed()
    print(
        f"Datasets loaded: train={len(train_ds)}, eval={len(eval_ds)} ({time.perf_counter() - t0:.1f}s)"
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        ),
        raw_audio=raw_audio,
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    t0 = time.perf_counter()
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    print(f"Model loaded: {time.perf_counter() - t0:.1f}s")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_DIR,
        **TRAINING_ARGS,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(MODEL_DIR, "final"))
    processor.save_pretrained(os.path.join(MODEL_DIR, "final"))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-audio",
        action="store_true",
        help="Load raw audiofolder",
    )
    args = parser.parse_args()
    train(raw_audio=args.raw_audio)
