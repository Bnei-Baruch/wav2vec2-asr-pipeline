import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from .constants import VOCAB_PATH, BASE_MODEL_ID, MODEL_DIR


def train():
    print("Text Preprocessing")

    if not os.path.exists(VOCAB_PATH):
        exit(f"Vocab file not found: {VOCAB_PATH}")

    print("Create Processor")
    t0 = time.perf_counter()
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
    print(f"Create Processor: {time.perf_counter() - t0:.2f}s")

    print("Prepare Audio")
    t0 = time.perf_counter()
    eval_ds = load_from_disk("./eval")
    print(f"load_from_disk('./eval'): {time.perf_counter() - t0:.2f}s")
    print(f"Eval dataset size: {len(eval_ds)}")
    print(eval_ds.column_names)

    t0 = time.perf_counter()
    train_ds = load_from_disk("./train")
    print(f"load_from_disk('./train'): {time.perf_counter() - t0:.2f}s")
    print(f"Train dataset size: {len(train_ds)}")
    print(train_ds.column_names)

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
    t0 = time.perf_counter()
    wer_metric = evaluate.load("wer")
    print(f"evaluate.load('wer'): {time.perf_counter() - t0:.2f}s")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    print("Create Model")
    t0 = time.perf_counter()
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
    model.freeze_feature_encoder()
    print(f"from_pretrained(model): {time.perf_counter() - t0:.2f}s")

    print("Create Trainer")
    t0 = time.perf_counter()
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=30,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        learning_rate=3e-5,
        warmup_steps=500,
        save_total_limit=3,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False
    )
    print(f"TrainingArguments(...): {time.perf_counter() - t0:.2f}s")

    class StartupTimingCallback(TrainerCallback):
        def __init__(self, t_train_called: float):
            self.t_train_called = t_train_called
            self._printed_first_step = False

        def on_step_begin(self, args, state, control, **kwargs):
            if not self._printed_first_step and state.global_step == 0:
                self._printed_first_step = True
                print(
                    f"Time to first train step: {time.perf_counter() - self.t_train_called:.2f}s"
                )

    t0 = time.perf_counter()
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
    )
    print(f"Trainer(...): {time.perf_counter() - t0:.2f}s")

    print("Starting training...")
    t_train_called = time.perf_counter()
    trainer.add_callback(StartupTimingCallback(t_train_called))
    trainer.train()


if __name__ == "__main__":
    train()
