"""
Check round-trip tokenization quality: sentence → tokenize → decode → compare.
Mismatches indicate noisy labels that confuse the model during training.

Usage:
    python -m ch_ds.check_tokenization
    python -m ch_ds.check_tokenization --data_dir ./dataset --limit 50
"""
from __future__ import annotations

import argparse
import csv
import os

from transformers import WhisperProcessor

from .normalize import _find_metadata_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--model_id", default="ivrit-ai/whisper-large-v3")
    parser.add_argument("--limit", type=int, default=20, help="Max mismatches to print")
    args = parser.parse_args()

    print(f"Loading processor from {args.model_id} ...")
    processor = WhisperProcessor.from_pretrained(args.model_id)
    processor.tokenizer.clean_up_tokenization_spaces = False
    processor.tokenizer.set_prefix_tokens(language="he", task="transcribe", predict_timestamps=False)

    csv_files = _find_metadata_files(args.data_dir)
    if not csv_files:
        print(f"No metadata.csv found in {args.data_dir}")
        return

    total = mismatches = 0
    shown = 0

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            sentence = row.get("sentence", "").strip()
            if not sentence:
                continue
            total += 1

            ids = processor.tokenizer(sentence).input_ids
            decoded = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()

            if decoded != sentence:
                mismatches += 1
                if shown < args.limit:
                    shown += 1
                    print(f"\n[{csv_path}]")
                    print(f"  orig:    {sentence!r}")
                    print(f"  decoded: {decoded!r}")

    pct = 100 * mismatches / total if total else 0
    print(f"\n{'=' * 60}")
    print(f"Total sentences : {total:,}")
    print(f"Mismatches      : {mismatches:,}  ({pct:.2f}%)")
    if mismatches > shown:
        print(f"(showing first {shown}, run with --limit N to see more)")


if __name__ == "__main__":
    main()
