"""
Validates the output of prepare_dataset.py (./train and ./eval directories).

Usage:
    python -m wisper.check_dataset
    python -m wisper.check_dataset --train ./train --eval ./eval --samples 5
"""
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor
from .constants import BASE_MODEL_ID, LANGUAGE, TASK


def check_split(ds, name: str, processor: WhisperProcessor, n_samples: int):
    print(f"\n{'='*60}")
    print(f"  {name.upper()}  —  {len(ds):,} samples")
    print(f"{'='*60}")

    print(f"Total samples: {len(ds):,}")
    print(f"\nSamples:")
    indices = [int(i) for i in np.linspace(0, len(ds) - 1, n_samples)]
    for i in indices:
        feat = np.array(ds[i]["input_features"])
        ids = ds[i]["labels"]
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        shape_ok = "OK" if feat.shape == (128, 3000) else f"BAD{feat.shape}"
        feat_info = f"min={feat.min():.2f} max={feat.max():.2f}"
        has_nan = " NaN!" if np.isnan(feat).any() else ""
        has_inf = " Inf!" if np.isinf(feat).any() else ""
        ids_info = f"ids=[{min(ids)}..{max(ids)}]" if ids else "ids=EMPTY"
        print(f"  [{i:6d}] feat={shape_ok} {feat_info}{has_nan}{has_inf}  labels len={len(ids):3d} {ids_info}  {repr(text[:60])}")


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
