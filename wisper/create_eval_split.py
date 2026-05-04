"""
Randomly samples rows from each subdirectory of DATASET_DIR and merges them
into a single flat eval directory:
    <out>/clips/{subdir}__{original_basename}
    <out>/metadata.csv  (columns: sentence, file_name, source)

Usage:
    python -m wisper.create_eval_split [--fraction 0.05] [--out ./dataset_eval] [--seed 42]
"""

import argparse
import csv
import os
import random
import shutil

from .constants import DATASET_DIR


def sample_subdir(subdir_path: str, fraction: float, rng: random.Random) -> list[dict]:
    meta_path = os.path.join(subdir_path, "metadata.csv")
    with open(meta_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    k = max(1, round(len(rows) * fraction))
    return rng.sample(rows, min(k, len(rows)))


def create_eval_split(fraction: float, out_dir: str, seed: int) -> None:
    rng = random.Random(seed)

    subdirs = sorted(
        name for name in os.listdir(DATASET_DIR)
        if os.path.isfile(os.path.join(DATASET_DIR, name, "metadata.csv"))
    )
    if not subdirs:
        raise RuntimeError(f"No valid subdirectories found in {DATASET_DIR}")

    clips_dst = os.path.join(out_dir, "clips")
    os.makedirs(clips_dst, exist_ok=True)

    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "file_name", "source"])
        writer.writeheader()

        total = 0
        for name in subdirs:
            src_dir = os.path.join(DATASET_DIR, name)
            samples = sample_subdir(src_dir, fraction, rng)

            for row in samples:
                basename = os.path.basename(row["file_name"])
                new_filename = f"clips/{name}__{basename}"
                shutil.copy2(os.path.join(src_dir, row["file_name"]), os.path.join(out_dir, new_filename))
                writer.writerow({"sentence": row["sentence"], "file_name": new_filename, "source": name})

            total += len(samples)
            print(f"  {name}: {len(samples)} samples")

    print(f"\nTotal eval samples: {total}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.05, help="Fraction to sample from each subdir (default: 0.05)")
    parser.add_argument("--out", default="./dataset_eval", help="Output directory (default: ./dataset_eval)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_eval_split(fraction=args.fraction, out_dir=args.out, seed=args.seed)
