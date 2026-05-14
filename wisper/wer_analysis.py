import argparse
import os
from collections import Counter, defaultdict

import torch
import evaluate
from datasets import load_dataset, Audio
from jiwer import process_words
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
)

from .constants import BASE_MODEL_ID, LANGUAGE, TASK, DATASET_DIR
from .train import DataCollatorSpeechSeq2SeqWithPadding, BF16Seq2SeqTrainer, EVAL_SIZE
from ch_ds.punct import check_punct

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def analyze_errors(pred_strs: list[str], ref_strs: list[str]) -> None:
    total_sub = total_del = total_ins = total_hits = 0
    substitution_pairs: Counter = Counter()
    deleted_words: Counter = Counter()
    inserted_words: Counter = Counter()
    flag_wer: dict[str, list[float]] = defaultdict(list)
    clean_wer: list[float] = []
    per_sample: list[tuple[float, str, str]] = []

    for pred, ref in zip(pred_strs, ref_strs):
        try:
            r = process_words(ref, pred)
        except Exception:
            continue

        total_sub += r.substitutions
        total_del += r.deletions
        total_ins += r.insertions
        total_hits += r.hits

        ref_words = ref.split()
        hyp_words = pred.split()
        for alignment in r.alignments:
            for chunk in alignment:
                if chunk.type == 'substitute':
                    rw = ' '.join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                    hw = ' '.join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                    substitution_pairs[(rw, hw)] += 1
                elif chunk.type == 'delete':
                    for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                        deleted_words[w] += 1
                elif chunk.type == 'insert':
                    for w in hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                        inserted_words[w] += 1

        flags = check_punct(ref)
        if flags:
            for f in flags:
                flag_wer[f].append(r.wer)
        else:
            clean_wer.append(r.wer)

        per_sample.append((r.wer, ref, pred))

    total_errors = total_sub + total_del + total_ins

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print(f"\n--- Error type breakdown ({total_errors} total errors) ---")
    if total_errors:
        print(f"  Substitutions: {total_sub:5d}  ({100 * total_sub / total_errors:.1f}%)")
        print(f"  Deletions:     {total_del:5d}  ({100 * total_del / total_errors:.1f}%)")
        print(f"  Insertions:    {total_ins:5d}  ({100 * total_ins / total_errors:.1f}%)")

    print(f"\n--- Average WER by punctuation flag ---")
    clean_avg = sum(clean_wer) / len(clean_wer) if clean_wer else 0.0
    print(f"  {'clean (no flags)':<32s}: {clean_avg:.4f}  ({len(clean_wer)} samples)")
    for flag, wers in sorted(flag_wer.items(), key=lambda x: -(sum(x[1]) / len(x[1]))):
        avg = sum(wers) / len(wers)
        print(f"  {flag:<32s}: {avg:.4f}  ({len(wers)} samples)")

    print(f"\n--- Top 20 substitutions (reference → hypothesis) ---")
    for (rw, hw), count in substitution_pairs.most_common(20):
        print(f"  {count:4d}x  {rw!r:30s} → {hw!r}")

    print(f"\n--- Top 20 deleted words (in reference, missed by model) ---")
    for w, count in deleted_words.most_common(20):
        print(f"  {count:4d}x  {w!r}")

    print(f"\n--- Top 20 inserted words (hallucinated by model) ---")
    for w, count in inserted_words.most_common(20):
        print(f"  {count:4d}x  {w!r}")

    print(f"\n--- Worst 10 samples ---")
    for wer_val, ref, hyp in sorted(per_sample, reverse=True)[:10]:
        print(f"  WER={wer_val:.3f}")
        print(f"    REF: {ref[:120]}")
        print(f"    HYP: {hyp[:120]}")


def run_wer_analysis(model_id: str = None, eval_size: int = None):
    model_id = model_id or BASE_MODEL_ID
    print(f"Model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    ds = load_dataset("audiofolder", data_dir=DATASET_DIR, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    split = ds.train_test_split(test_size=EVAL_SIZE, seed=42)
    eval_ds = split["test"]
    if eval_size is not None and eval_size < len(eval_ds):
        eval_ds = eval_ds.shuffle(seed=42).select(range(eval_size))
    print(f"Eval dataset size: {len(eval_ds)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    training_args = Seq2SeqTrainingArguments(
        output_dir="./analysis_eval_tmp",
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
        processing_class=processor,
    )

    print("Running inference...")
    output = trainer.predict(eval_ds)

    label_ids = output.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_strs = processor.tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
    ref_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    overall_wer = wer_metric.compute(predictions=pred_strs, references=ref_strs)

    if trainer.is_world_process_zero():
        print(f"\nOverall WER: {overall_wer:.4f} ({overall_wer * 100:.2f}%)")
        analyze_errors(pred_strs, ref_strs)

    return {"eval_wer": overall_wer}


# Example: torchrun --nproc_per_node=2 -m wisper.wer_analysis --model ./models/whisper-large-v3-he/final > logs/wer_1.log 2> logs/wer_2.log
# Example (2000 samples): torchrun --nproc_per_node=2 -m wisper.wer_analysis --model ./models/whisper-large-v3-he/final --eval_size 2000 > logs/wer_1.log 2> logs/wer_2.log
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model ID or path (default: BASE_MODEL_ID)")
    parser.add_argument("--eval_size", type=int, default=None, help="Number of eval samples (default: all test split)")
    args = parser.parse_args()
    run_wer_analysis(args.model, args.eval_size)
