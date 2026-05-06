"""
Reads a result CSV (columns: source, index, wav_path, text, duration_s)
and copies every wav_path file into --out, preserving relative subdirectory structure.
Also writes a metadata.csv in the output directory with updated wav_path values.

Usage:
    python -m ch_ds.copy_result result_passed.csv --out /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_): return it  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(description='Copy wav files from a result CSV to an output dir.')
    parser.add_argument('csv', help='Path to result CSV file')
    parser.add_argument('--out', required=True, help='Destination directory')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f'ERROR: CSV file not found: {args.csv}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    with open(args.csv, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            print('ERROR: CSV is empty', file=sys.stderr)
            sys.exit(1)

        required = {'wav_path'}
        missing = required - set(reader.fieldnames)
        if missing:
            print(f'ERROR: CSV is missing columns: {missing}', file=sys.stderr)
            sys.exit(1)

        rows = list(reader)

    wav_paths = [r['wav_path'].strip() for r in rows if r.get('wav_path', '').strip()]
    common_prefix = os.path.commonpath(wav_paths) if wav_paths else ''
    if os.path.isfile(common_prefix):
        common_prefix = os.path.dirname(common_prefix)

    out_rows: list[dict] = []
    copied = skipped = 0

    for row in tqdm(rows, desc='copying', unit='file'):
        src = row.get('wav_path', '').strip()
        if not src:
            skipped += 1
            continue

        if not os.path.isfile(src):
            print(f'WARN: file not found, skipping: {src}', file=sys.stderr)
            skipped += 1
            continue

        rel = os.path.relpath(src, common_prefix)
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

        out_row = dict(row)
        out_row['_dst'] = dst
        out_rows.append(out_row)
        copied += 1

    if out_rows:
        # parse_metadata expects: file_name (relative), sentence
        fieldnames = ['file_name', 'sentence']

        by_dir: dict[str, list[dict]] = {}
        for r in out_rows:
            d = os.path.dirname(r['_dst'])
            by_dir.setdefault(d, []).append(r)

        for d, dir_rows in by_dir.items():
            meta_dir = os.path.dirname(d)
            meta_path = os.path.join(meta_dir, 'metadata.csv')
            with open(meta_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in dir_rows:
                    writer.writerow({
                        'file_name': os.path.relpath(r['_dst'], meta_dir),
                        'sentence': r.get('text', ''),
                    })
            print(f'Wrote metadata: {meta_path}')

    print(f'Done. Copied: {copied}, skipped: {skipped}')


# Example: python -m ch_ds.copy_result results/all_passed.csv --out dataset_copy/
if __name__ == '__main__':
    main()
