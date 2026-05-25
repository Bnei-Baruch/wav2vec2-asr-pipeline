"""
Clip ↔ metadata integrity checker.

Two checks per dataset subset (every directory containing metadata.csv):
  - missing_clip  : metadata row whose file_name points to a non-existent WAV
  - orphan_clip   : WAV file in clips/ that has no corresponding metadata row

Usage:
    python -m wisper.check_pairs [--data_dir ./dataset] [--out pairs_issues.csv]
"""

import argparse
import csv
from pathlib import Path

from .constants import DATASET_DIR


def scan_dir(data_dir: str) -> list[dict]:
    issues = []
    total_metadata = 0
    total_clips = 0

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

        # file_name values referenced in metadata (relative to subdir)
        referenced: set[Path] = set()
        for row in rows:
            file_name = row.get("file_name", "").strip()
            if file_name:
                referenced.add((subdir / file_name).resolve())

        total_metadata += len(rows)

        # check 1: every metadata row must have a matching clip
        for row in rows:
            sentence = row.get("sentence", "").strip()
            file_name = row.get("file_name", "").strip()
            audio_path = (subdir / file_name).resolve() if file_name else None

            if not file_name:
                issues.append({
                    "kind": "missing_file_name",
                    "audio_path": "",
                    "sentence": sentence[:120],
                    "subset": subdir.name,
                })
            elif not audio_path.exists():
                issues.append({
                    "kind": "missing_clip",
                    "audio_path": str(audio_path),
                    "sentence": sentence[:120],
                    "subset": subdir.name,
                })

        # check 2: every WAV in clips/ must appear in metadata
        clips_dir = subdir / "clips"
        if clips_dir.is_dir():
            wav_files = sorted(clips_dir.glob("*.wav"))
            total_clips += len(wav_files)
            for wav in wav_files:
                if wav.resolve() not in referenced:
                    issues.append({
                        "kind": "orphan_clip",
                        "audio_path": str(wav),
                        "sentence": "",
                        "subset": subdir.name,
                    })

    print(f"\nScanned {len(metadata_files)} subset(s)")
    print(f"  Metadata rows : {total_metadata}")
    print(f"  WAV files     : {total_clips}")
    print(f"  Issues found  : {len(issues)}")
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATASET_DIR)
    parser.add_argument("--out", default="pairs_issues.csv")
    args = parser.parse_args()

    issues = scan_dir(args.data_dir)
    if not issues:
        print("All clips and metadata are in sync.")
        return

    by_kind: dict[str, list] = {}
    for issue in issues:
        by_kind.setdefault(issue["kind"], []).append(issue)

    print("\n--- Breakdown by kind ---")
    for kind, items in sorted(by_kind.items(), key=lambda x: -len(x[1])):
        print(f"  {kind:<20s}: {len(items)}")

    print("\n--- Examples (up to 5 per kind) ---")
    for kind, items in sorted(by_kind.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{kind}]")
        for item in items[:5]:
            line = f"    {item['audio_path']}"
            if item["sentence"]:
                line += f'  — "{item["sentence"][:80]}"'
            print(line)

    fieldnames = ["kind", "subset", "audio_path", "sentence"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issues)
    print(f"\nFull list saved to: {args.out}")


if __name__ == "__main__":
    main()
