import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import torch
import evaluate
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from .utils import load_dataset_from_dir
from .constants import VOCAB_PATH, BASE_MODEL_ID, MODEL_DIR


def train():
    print("Text Preprocessing")
    dataset = load_dataset_from_dir()

    if not os.path.exists(VOCAB_PATH):
        exit(f"Vocab file not found: {VOCAB_PATH}")

    print("Create Processor")
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    print("Prepare Audio")
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Eval dataset size: {len(eval_ds)}")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = [
            processor(a["array"], sampling_rate=a["sampling_rate"]).input_values[0]
            for a in audio
        ]
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    print(f"Dataset column names: {dataset.column_names}")
    drop_columns = [c for c in train_ds.column_names if c not in ["audio", "sentence"]]
    print(f"Drop columns: {drop_columns}")
    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=drop_columns,
        keep_in_memory=False,
        batch_size=100,
        batched=True,
        num_proc=1,
    )
    eval_ds = eval_ds.map(
        prepare_dataset,
        remove_columns=drop_columns,
        keep_in_memory=False,
        batch_size=100,
        batched=True,
        num_proc=1,
    )

    print("Create Data Collator")

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [
                {"input_values": feature["input_values"]} for feature in features
            ]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print("Create Metric")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    print("Create Model")
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_ID,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    print("Create Trainer")
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=30,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=True,
        logging_steps=50,
        learning_rate=2e-4,
        warmup_steps=100,
        save_total_limit=2,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
    )

    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    train()
