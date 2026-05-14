from __future__ import annotations

import argparse
import csv
import os

from .punct import (
    PUNCT_DIALOGUE_DASH,
    PUNCT_DOUBLE_DASH,
    PUNCT_DOUBLE_SPACE,
    PUNCT_INVISIBLE,
    PUNCT_REPEATED,
    PUNCT_SPACE_BEFORE,
)


def normalize_text(text: str) -> str:
    """Fix punctuation inconsistencies while preserving punctuation."""
    if PUNCT_INVISIBLE.search(text):
        text = PUNCT_INVISIBLE.sub('', text)
    if PUNCT_DIALOGUE_DASH.search(text):
        text = PUNCT_DIALOGUE_DASH.sub('', text)
    if PUNCT_DOUBLE_DASH.search(text):
        text = PUNCT_DOUBLE_DASH.sub('—', text)
    if PUNCT_REPEATED.search(text):
        text = PUNCT_REPEATED.sub(r'\1', text)
    if PUNCT_DOUBLE_SPACE.search(text):
        text = PUNCT_DOUBLE_SPACE.sub(' ', text)
    if PUNCT_SPACE_BEFORE.search(text):
        text = PUNCT_SPACE_BEFORE.sub(r'\1', text)
    return text.strip()


def _find_metadata_files(data_dir: str) -> list[str]:
    result = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        if 'metadata.csv' in files:
            result.append(os.path.join(root, 'metadata.csv'))
    return sorted(result)


def _process_file(csv_path: str, apply: bool, preview_limit: int) -> tuple[int, int]:
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    changed: list[tuple[int, str, str]] = []
    new_rows = []
    for i, row in enumerate(rows):
        orig = row.get('sentence', '')
        norm = normalize_text(orig)
        new_rows.append({**row, 'sentence': norm})
        if norm != orig:
            changed.append((i, orig, norm))

    if changed and not apply:
        shown = min(len(changed), preview_limit)
        print(f"\n  {csv_path}  ({len(changed)} changes, showing {shown})")
        for idx, orig, norm in changed[:preview_limit]:
            print(f"    row {idx:5d}  {orig!r}")
            print(f"           → {norm!r}")

    if apply and changed:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)

    return len(rows), len(changed)


def main():
    parser = argparse.ArgumentParser(description="Normalize sentence text in dataset metadata.csv files")
    parser.add_argument('--data_dir', default='./dataset', help='Root dataset directory')
    parser.add_argument('--apply', action='store_true', help='Write changes to files (default: dry-run)')
    parser.add_argument('--preview', type=int, default=5, help='Number of example changes to show per file (default: 5)')
    args = parser.parse_args()

    csv_files = _find_metadata_files(args.data_dir)
    if not csv_files:
        print(f"No metadata.csv files found in {args.data_dir}")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Scanning {len(csv_files)} metadata file(s) in {args.data_dir}")

    total_rows = total_changed = 0
    file_stats: list[tuple[str, int, int]] = []
    for csv_path in csv_files:
        rows, changed = _process_file(csv_path, apply=args.apply, preview_limit=args.preview)
        total_rows += rows
        total_changed += changed
        if changed:
            file_stats.append((csv_path, rows, changed))

    print(f"\n{'=' * 60}")
    print(f"Total rows:    {total_rows:,}")
    print(f"Changed rows:  {total_changed:,}  ({100 * total_changed / total_rows:.2f}%)" if total_rows else "Changed rows:  0")
    print(f"Files with changes: {len(file_stats)}")
    if not args.apply and total_changed:
        print(f"\nRun with --apply to write changes.")


# Example: python -m ch_ds.normalize --data_dir ./dataset
# Example: python -m ch_ds.normalize --data_dir ./dataset --apply
if __name__ == '__main__':
    main()
