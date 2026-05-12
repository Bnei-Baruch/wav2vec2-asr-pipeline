import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from .constants import BASE_MODEL_ID, MODEL_DIR, LANGUAGE, TASK, TRAINING_ARGS, DATASET_DIR

EVAL_SIZE = 0.1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = self.processor.feature_extractor(
            [f["audio"]["array"] for f in features],
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features

        labels_batch = self.processor.tokenizer(
            [f["sentence"] for f in features],
            return_tensors="pt",
            padding=True,
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


def train():
    print(f"Load weights from: {BASE_MODEL_ID}")
    print(f"Output dir: {MODEL_DIR}")

    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    print(f"Processor loaded: {time.perf_counter() - t0:.1f}s")

    ds = load_dataset("audiofolder", data_dir=DATASET_DIR, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    t0 = time.perf_counter()
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
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

    training_args = Seq2SeqTrainingArguments(output_dir=MODEL_DIR, **TRAINING_ARGS)

    trainer = BF16Seq2SeqTrainer(
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


## Example: torchrun --nproc_per_node=2 -m wisper.train > logs/train_1.log 2> logs/train_2.log
if __name__ == "__main__":
    train()
