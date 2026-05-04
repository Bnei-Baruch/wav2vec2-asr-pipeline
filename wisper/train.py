import json
import os
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from .constants import BASE_MODEL_ID, MODEL_DIR, LANGUAGE, TASK, TRAINING_ARGS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class ResetBestMetricCallback(TrainerCallback):
    def on_train_begin(self, _args, state, _control, **_kwargs):
        state.best_metric = None
        state.best_model_checkpoint = None


class BF16Seq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys=ignore_keys
                )
        except Exception as e:
            print(f"[eval] FAILED: {type(e).__name__}: {e}")
            raise


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
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

def _checkpoint_epoch(checkpoint_dir: str) -> float:
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(state_file) as f:
        return json.load(f)["epoch"]


def train(resume_from_checkpoint: Optional[str] = None):
    model_source = (
        resume_from_checkpoint if resume_from_checkpoint else BASE_MODEL_ID
    )
    print(f"Load weights from: {model_source}")
    print(f"Output dir: {MODEL_DIR}")

    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    print(f"Processor loaded: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()

    eval_ds = load_from_disk("./eval")
    print(f"Eval dataset size: {len(eval_ds)}")

    train_ds = load_from_disk("./train")
    print(f"Train dataset size: {len(train_ds)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        ),
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
        model_source,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.freeze_encoder()
    print(f"Model loaded: {time.perf_counter() - t0:.1f}s")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    base_epoch = int(_checkpoint_epoch(resume_from_checkpoint)) if resume_from_checkpoint else 0
    num_train_epochs = base_epoch + TRAINING_ARGS["num_train_epochs"]
    if resume_from_checkpoint:
        print(f"Resuming from epoch {base_epoch}, training until epoch {num_train_epochs}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_DIR,
        **{**TRAINING_ARGS, "num_train_epochs": num_train_epochs},
    )
    
    callbacks = [ResetBestMetricCallback()] if resume_from_checkpoint else []
    trainer = BF16Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("Saving final model...")
    trainer.save_model(os.path.join(MODEL_DIR, "final"))
    processor.save_pretrained(os.path.join(MODEL_DIR, "final"))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=None,
        metavar="DIR",
        help="Example: ./models/whisper-large-v3-he/checkpoint-400",
    )
    args = parser.parse_args()
    train(resume_from_checkpoint=args.resume)
