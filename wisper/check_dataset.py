"""
Validates the output of prepare_dataset.py (./train and ./eval directories).

Usage:
    python -m wisper.check_dataset
    python -m wisper.check_dataset --train ./train --eval ./eval --samples 5
"""
import argparse
import numpy as np
from collections import Counter
from datasets import load_from_disk
from transformers import WhisperProcessor
from .constants import BASE_MODEL_ID, LANGUAGE, TASK

EMPTY_LABEL_MAX_LEN = 6  # labels shorter than this are likely empty/broken sentences


def check_split(ds, name: str, processor: WhisperProcessor, n_samples: int):
    print(f"\n{'='*60}")
    print(f"  {name.upper()}  —  {len(ds):,} samples")
    print(f"{'='*60}")

    label_lengths = [len(x) for x in ds["labels"]]
    short = sum(1 for l in label_lengths if l <= EMPTY_LABEL_MAX_LEN)

    print(f"Label length — min: {min(label_lengths)}, max: {max(label_lengths)}, "
          f"mean: {np.mean(label_lengths):.1f}, median: {np.median(label_lengths):.1f}")
    print(f"Short labels (len <= {EMPTY_LABEL_MAX_LEN}): {short} ({100*short/len(ds):.1f}%)  ← likely empty sentences")

    # Check input_features shape and NaN/inf on first 500 samples
    check_n = min(500, len(ds))
    nan_count = inf_count = wrong_shape = 0
    for i in range(check_n):
        feat = np.array(ds[i]["input_features"])
        if feat.shape != (128, 3000):
            wrong_shape += 1
        if np.isnan(feat).any():
            nan_count += 1
        if np.isinf(feat).any():
            inf_count += 1

    print(f"\nFeature check (first {check_n} samples):")
    print(f"  Wrong shape (≠128×3000): {wrong_shape}")
    print(f"  NaN:  {nan_count}")
    print(f"  Inf:  {inf_count}")

    print(f"\nSample labels (decoded):")
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)
    for i in indices:
        ids = ds[i]["labels"]
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  [{i:6d}] len={len(ids):3d}  {repr(text[:80])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./train")
    parser.add_argument("--eval", default="./eval")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample labels to decode")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    for path, name in [(args.train, "train"), (args.eval, "eval")]:
        try:
            ds = load_from_disk(path)
            check_split(ds, name, processor, args.samples)
        except Exception as e:
            print(f"\n[{name}] Failed to load from '{path}': {e}")

    print("\nDone.")


# Example: python -m wisper.check_dataset
if __name__ == "__main__":
    main()
