import argparse
import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
)
from .constants import BASE_MODEL_ID, LANGUAGE, TASK
from .prepare_dataset import EVAL_OUT
from .train import DataCollatorSpeechSeq2SeqWithPadding, BF16Seq2SeqTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evaluate_baseline(model_id: str = None):
    model_id = model_id or BASE_MODEL_ID
    print(f"Model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    eval_ds = load_from_disk(EVAL_OUT)
    print(f"Eval dataset size: {len(eval_ds)}")

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
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    training_args = Seq2SeqTrainingArguments(
        output_dir="./baseline_eval_tmp",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        bf16=True,
        eval_accumulation_steps=4,
        remove_unused_columns=False,
    )

    trainer = BF16Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    print("Evaluating baseline model...")
    results = trainer.evaluate()
    if trainer.is_world_process_zero():
        print(f"\nBaseline WER: {results['eval_wer']:.4f} ({results['eval_wer']*100:.2f}%)")
    return results


# Example: torchrun --nproc_per_node=2 -m wisper.evaluate_baseline --model openai/whisper-large-v3 > logs/evw_1.log 2> logs/evw_2.log
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model ID or path (default: BASE_MODEL_ID from constants)")
    args = parser.parse_args()
    evaluate_baseline(args.model)
