"""
Fast audio/text mismatch detector — no GPU needed.

Flags samples where audio duration vs. word count ratio looks wrong:
  - words_per_sec > MAX_WPS  → text too long for the audio (likely mismatch)
  - words_per_sec < MIN_WPS  → text too short for the audio (silence or mismatch)

Usage:
    python -m wisper.check_mismatch [--data_dir ./dataset] [--out mismatches.csv]
"""

import argparse
import csv
import os
import wave
from pathlib import Path

from .constants import DATASET_DIR

MIN_WPS = 0.3   # fewer words/sec than this → suspiciously sparse
MAX_WPS = 5.0   # more words/sec than this  → suspiciously dense
MIN_DURATION = 0.3  # seconds — clips shorter than this are almost always junk


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def scan_dir(data_dir: str) -> list[dict]:
    suspicious = []
    total = 0
    errors = 0

    root = Path(data_dir)
    metadata_files = sorted(root.rglob("metadata.csv"))
    if not metadata_files:
        print(f"No metadata.csv found under {data_dir}")
        return []

    for meta_path in metadata_files:
        subdir = meta_path.parent
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            sentence = row.get("sentence", "").strip()
            file_name = row.get("file_name", "").strip()
            audio_path = subdir / file_name

            total += 1
            if not audio_path.exists():
                errors += 1
                suspicious.append({
                    "reason": "file_missing",
                    "wps": None,
                    "duration": None,
                    "words": len(sentence.split()),
                    "sentence": sentence[:120],
                    "audio_path": str(audio_path),
                })
                continue

            try:
                dur = _wav_duration(str(audio_path))
            except Exception as e:
                errors += 1
                suspicious.append({
                    "reason": f"read_error: {e}",
                    "wps": None,
                    "duration": None,
                    "words": len(sentence.split()),
                    "sentence": sentence[:120],
                    "audio_path": str(audio_path),
                })
                continue

            words = len(sentence.split())
            wps = words / dur if dur > 0 else float("inf")

            reason = None
            if dur < MIN_DURATION:
                reason = "too_short"
            elif wps > MAX_WPS:
                reason = "text_too_dense"
            elif wps < MIN_WPS:
                reason = "text_too_sparse"

            if reason:
                suspicious.append({
                    "reason": reason,
                    "wps": round(wps, 2),
                    "duration": round(dur, 2),
                    "words": words,
                    "sentence": sentence[:120],
                    "audio_path": str(audio_path),
                })

    print(f"\nScanned {total} samples across {len(metadata_files)} metadata file(s)")
    print(f"Errors (missing/unreadable): {errors}")
    print(f"Suspicious samples: {len(suspicious)}")
    return suspicious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATASET_DIR)
    parser.add_argument("--out", default="mismatches.csv", help="Output CSV path")
    args = parser.parse_args()

    suspicious = scan_dir(args.data_dir)
    if not suspicious:
        print("No mismatches found.")
        return

    by_reason: dict[str, list] = {}
    for s in suspicious:
        by_reason.setdefault(s["reason"], []).append(s)

    print("\n--- Breakdown by reason ---")
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"  {reason:<20s}: {len(items)}")

    print("\n--- Examples (up to 5 per category) ---")
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{reason}]")
        for s in items[:5]:
            dur_str = f"{s['duration']}s" if s['duration'] is not None else "N/A"
            wps_str = f"{s['wps']} wps" if s['wps'] is not None else ""
            print(f"    {dur_str:8s} {wps_str:10s}  \"{s['sentence'][:80]}\"")
            print(f"              {s['audio_path']}")

    fieldnames = ["reason", "wps", "duration", "words", "sentence", "audio_path"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(suspicious)
    print(f"\nFull list saved to: {args.out}")


if __name__ == "__main__":
    main()
