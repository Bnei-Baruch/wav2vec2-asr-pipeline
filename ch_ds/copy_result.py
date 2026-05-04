"""
Reads a result CSV (columns: source, index, wav_path, text, duration_s)
and copies every wav_path file flat into --out.
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Copy wav files from a result CSV to --out dir.')
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

    out_rows: list[dict] = []
    copied = skipped = 0

    for row in rows:
        src = row.get('wav_path', '').strip()
        if not src:
            skipped += 1
            continue

        if not os.path.isfile(src):
            print(f'WARN: file not found, skipping: {src}', file=sys.stderr)
            skipped += 1
            continue

        dst = os.path.join(args.out, os.path.basename(src))

        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

        out_row = dict(row)
        out_row['wav_path'] = dst
        out_rows.append(out_row)
        copied += 1

    if out_rows:
        meta_path = os.path.join(args.out, 'metadata.csv')
        with open(meta_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f'Wrote metadata: {meta_path}')

    print(f'Done. Copied: {copied}, skipped: {skipped}')


if __name__ == '__main__':
    main()
