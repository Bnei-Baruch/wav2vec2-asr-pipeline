from __future__ import annotations

import argparse
import csv
import os

import re

from .punct import (
    PUNCT_DIALOGUE_DASH,
    PUNCT_DOUBLE_DASH,
    PUNCT_DOUBLE_SPACE,
    PUNCT_INVISIBLE,
    PUNCT_REPEATED,
    PUNCT_SPACE_BEFORE,
)

_SPACE_BEFORE_SUB = re.compile(r'\s([,;.!?])(?!\Z)')
# .. or ....+ → ... (wrong dot count instead of ellipsis)
_WRONG_DOTS       = re.compile(r'(?<!\.)\.{2}(?!\.)(?=\s|$)|\.{4,}(?=\s|$)')
# ASCII " or "" between Hebrew letters → ״ (U+05F4).
# Two cases are replaced:
#   A) ≥2 Hebrew letters before the quote (e.g. צה"ל, מש"ה)
#   B) exactly 1 Hebrew letter before AND exactly 1 after — 2-letter abbrev (e.g. ז"א, ד"ר)
# Case excluded: 1 letter before + ≥2 after (e.g. ש"אבא — real speech, not abbreviation)
_ASCII_GERSHAYIM = re.compile(
    r'(?<=[א-ת]{2})\x22{1,2}(?=[א-ת])'                              # A: ≥2 before
    r'|(?<=[א-ת])(?<![א-ת]{2})\x22{1,2}(?=[א-ת])(?![א-ת]{2})'      # B: exactly 1 before, 1 after
)
# " not followed by a Hebrew letter — indicates a real closing quote in the text
_HAS_CLOSING_QUOT = re.compile(r'\x22(?![א-ת])')
# full Hebrew word containing " between letters (for word-level logging)
_ASCII_GERSHAYIM_WORD = re.compile(r'[א-ת]+(?:\x22{1,2}[א-ת]+)+')
# dash that is NOT at position 0 (mid-line dash, not a dialogue opener)
_DASH_MID_LINE = re.compile(r'(?<=.)-')

ALL_FIXES: frozenset[str] = frozenset({
    'invisible',
    'dialogue_dash',
    'double_dash',
    'wrong_dots',
    'ascii_quot',
    'punct_repeated',
    'double_space',
    'space_before',
})


def normalize_text(text: str, apply: bool, fixes: frozenset[str] = ALL_FIXES) -> str:
    if 'invisible' in fixes and PUNCT_INVISIBLE.search(text):
        text = PUNCT_INVISIBLE.sub('', text)
    if 'dialogue_dash' in fixes and PUNCT_DIALOGUE_DASH.search(text):
        text = PUNCT_DIALOGUE_DASH.sub('', text)
    if 'double_dash' in fixes and PUNCT_DOUBLE_DASH.search(text):
        text = PUNCT_DOUBLE_DASH.sub('—', text)
    if 'wrong_dots' in fixes and _WRONG_DOTS.search(text):
        text = _WRONG_DOTS.sub('...', text)
    if 'ascii_quot' in fixes and _ASCII_GERSHAYIM.search(text) and not _HAS_CLOSING_QUOT.search(text):
        text = _ASCII_GERSHAYIM.sub('״', text)
    if 'punct_repeated' in fixes and PUNCT_REPEATED.search(text) and not apply:
        text = PUNCT_REPEATED.sub(r'\1', text)
    if 'double_space' in fixes and PUNCT_DOUBLE_SPACE.search(text) and not apply:
        text = PUNCT_DOUBLE_SPACE.sub(' ', text)
    if 'space_before' in fixes and PUNCT_SPACE_BEFORE.search(text) and not apply:
        text = _SPACE_BEFORE_SUB.sub(r'\1', text)
    return text.strip()


def _find_metadata_files(data_dir: str) -> list[str]:
    result = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        if 'metadata.csv' in files:
            result.append(os.path.join(root, 'metadata.csv'))
    return sorted(result)


def _process_file(
    csv_path: str,
    apply: bool,
    preview_limit: int,
    fixes: frozenset[str],
    word_log: dict[tuple[str, str], tuple[str, str]] | None = None,
    dash_log: list[tuple[str, str]] | None = None,
) -> tuple[int, int]:
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    changed: list[tuple[int, str, str]] = []
    new_rows = []
    for i, row in enumerate(rows):
        orig = row.get('sentence', '')
        norm = normalize_text(orig, apply, fixes)
        new_rows.append({**row, 'sentence': norm})
        if norm != orig:
            changed.append((i, orig, norm))
            if word_log is not None:
                for w_orig in _ASCII_GERSHAYIM_WORD.findall(orig):
                    w_norm = _ASCII_GERSHAYIM.sub('״', w_orig)
                    if w_orig != w_norm:
                        word_log.setdefault((w_orig, w_norm), (orig, csv_path))
        if dash_log is not None and _DASH_MID_LINE.search(orig):
            dash_log.append((orig, csv_path))

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
    parser.add_argument(
        '--fixes', nargs='+', metavar='FIX',
        help=f'Fixes to apply (default: all). Available: {", ".join(sorted(ALL_FIXES))}',
    )
    args = parser.parse_args()

    if args.fixes:
        unknown = set(args.fixes) - ALL_FIXES
        if unknown:
            parser.error(f'Unknown fix(es): {", ".join(sorted(unknown))}. Available: {", ".join(sorted(ALL_FIXES))}')
        fixes = frozenset(args.fixes)
    else:
        fixes = ALL_FIXES

    csv_files = _find_metadata_files(args.data_dir)
    if not csv_files:
        print(f"No metadata.csv files found in {args.data_dir}")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] fixes={', '.join(sorted(fixes))}  files={len(csv_files)}")

    word_log: dict[tuple[str, str], tuple[str, str]] | None = {} if fixes == frozenset({'ascii_quot'}) and not args.apply else None
    dash_log: list[tuple[str, str]] | None = [] if 'dialogue_dash' in fixes else None

    total_rows = total_changed = 0
    file_stats: list[tuple[str, int, int]] = []
    for csv_path in csv_files:
        rows, changed = _process_file(csv_path, apply=args.apply, preview_limit=args.preview, fixes=fixes, word_log=word_log, dash_log=dash_log)
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

    if word_log:
        os.makedirs('./result', exist_ok=True)
        out_path = os.path.join('./result', 'ascii_quot_words.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            for (w_orig, w_norm), (sentence, csv_path) in sorted(word_log.items()):
                f.write(f"{w_orig}\t{w_norm}\t{sentence}\t{csv_path}\n")
        print(f"\nUnique ascii_quot replacements ({len(word_log)}): {out_path}")

    if dash_log is not None:
        os.makedirs('./result', exist_ok=True)
        out_path = os.path.join('./result', 'dialogue_dash_mid.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            for sentence, csv_path in dash_log:
                f.write(f"{sentence}\t{csv_path}\n")
        print(f"\nMid-line dashes ({len(dash_log)} rows): {out_path}")


# Example: python -m ch_ds.normalize --data_dir ./dataset
# Example: python -m ch_ds.normalize --data_dir ./dataset --apply
# Example: python -m ch_ds.normalize --data_dir ./dataset --fixes wrong_dots ascii_quot
# Example: python -m ch_ds.normalize --data_dir ./dataset --fixes ascii_quot dialogue_dash double_dash double_space invisible punct_repeated space_before wrong_dots
if __name__ == '__main__':
    main()
