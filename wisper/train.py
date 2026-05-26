import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import evaluate
from datasets import load_dataset, Audio

logger = logging.getLogger(__name__)
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
            logger.error("[eval] FAILED: %s: %s", type(e).__name__, e)
            raise


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    _logged: bool = field(default=False, init=False, repr=False)

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
        sot_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == sot_id).all().cpu().item():
            labels = labels[:, 1:]

        if not self._logged:
            self._logged = True
            f0 = features[0]
            sentence = f0.get("sentence", "MISSING")
            decoded = self.processor.tokenizer.decode(
                [t for t in labels[0].tolist() if t != -100],
                skip_special_tokens=True,
            )
            first_ids = [t for t in labels[0].tolist() if t != -100]
            logger.info(
                "[collator] keys=%s  sentence=%r  audio_sr=%s  audio_len=%s  "
                "feat_shape=%s  feat_min=%.3f  feat_max=%.3f  "
                "labels_shape=%s  labels_ids=[%s..%s]  labels_first8=%s  labels_decoded=%r",
                list(f0.keys()),
                sentence[:80],
                f0["audio"]["sampling_rate"],
                len(f0["audio"]["array"]),
                tuple(input_features.shape),
                input_features.min().item(),
                input_features.max().item(),
                tuple(labels.shape),
                labels[labels != -100].min().item(),
                labels[labels != -100].max().item(),
                first_ids[:8],
                decoded[:80],
            )

        return {"input_features": input_features, "labels": labels}


def train(model_id: str = BASE_MODEL_ID):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Load weights from: %s", model_id)
    logger.info("Output dir: %s", MODEL_DIR)

    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.clean_up_tokenization_spaces = False
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK, predict_timestamps=False)
    logger.info("Processor loaded: %.1fs", time.perf_counter() - t0)

    ds = load_dataset("audiofolder", data_dir=DATASET_DIR, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    if len(eval_ds) > 2000:
        eval_ds = eval_ds.shuffle(seed=42).select(range(2000))
    logger.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))

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
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.freeze_encoder()
    logger.info("Model loaded: %.1fs", time.perf_counter() - t0)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s trainable / %s total", f"{trainable:,}", f"{total:,}")

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

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=False)

    logger.info("Saving final model...")
    trainer.save_model(os.path.join(MODEL_DIR, "final"))
    processor.save_pretrained(os.path.join(MODEL_DIR, "final"))
    logger.info("Done.")


## Example: torchrun --nproc_per_node=2 -m wisper.train > logs/train_1.log 2> logs/train_2.log
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL_ID)
    args = parser.parse_args()
    train(model_id=args.model)
