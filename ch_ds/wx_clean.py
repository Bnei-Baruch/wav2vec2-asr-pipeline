"""
Deletes WAV clips flagged by WhisperX QC (wx.py) and removes their rows from metadata.csv.

Reads a _flagged.csv produced by wx.py.

Flags handled: wrong_language, high_wer, empty_transcription
By default all three flag types trigger deletion.

Use --flags to restrict which flag types cause deletion.
Use --min-wer to skip high_wer-only entries whose WER is below a stricter threshold
(useful when the detection threshold in config is loose but you want to delete only
the worst ones, e.g. --min-wer 0.8 keeps entries with WER 0.5-0.8).

Without --apply, runs in dry-run mode (nothing is deleted).

Usage:
    python -m ch_ds.wx_clean results/wx_flagged.csv
    python -m ch_ds.wx_clean results/wx_flagged.csv --flags high_wer empty_transcription
    python -m ch_ds.wx_clean results/wx_flagged.csv --min-wer 0.8 --apply
    python -m ch_ds.wx_clean results/wx_r0_flagged.csv results/wx_r1_flagged.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import Counter, defaultdict

from . import config
from .clean import _delete_wavs, _rewrite_metadata

ALL_FLAGS = list(config.WX_FLAG_ORDER)  # wrong_language, high_wer, empty_transcription


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.wx_clean')
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(config.LOG_PATH, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    base, ext = os.path.splitext(config.LOG_PATH)
    eh = logging.FileHandler(f'{base}_errors{ext}', encoding='utf-8')
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(eh)
    return logger


log = _setup_logger()


def flag_summary(flagged_csvs: list[str]) -> int:
    """Logs a breakdown by flag type and returns the total number of flagged rows."""
    counts: Counter = Counter()
    total = 0
    for path in flagged_csvs:
        try:
            with open(path, encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    for fl in row.get('flags', '').split('|'):
                        fl = fl.strip()
                        if fl:
                            counts[fl] += 1
                    total += 1
        except FileNotFoundError:
            log.error('Flagged CSV not found: %s', path)
        except Exception as e:
            log.error('Failed to read %s: %s', path, e)

    log.info('Flagged entries total : %d', total)
    log.info('Flag breakdown:')
    for fl, n in counts.most_common():
        pct = 100 * n / total if total else 0
        log.info('  %-25s %6d  (%.1f%%)', fl, n, pct)
    log.info('')
    return total


def load_flagged(
    flagged_csvs: list[str],
    active_flags: set[str],
    min_wer: float | None,
) -> dict[str, dict[int, str]]:
    """
    Returns {source_metadata_path: {row_index: wav_path}} for entries to delete.

    An entry is selected when at least one of its flags is in active_flags.
    Exception: if high_wer is the *only* active trigger for an entry and its WER
    is below min_wer, the entry is skipped (kept).
    """
    by_source: dict[str, dict[int, str]] = defaultdict(dict)
    skipped_wer = 0
    already_gone = 0

    for path in flagged_csvs:
        try:
            with open(path, encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    flags = {fl.strip() for fl in row.get('flags', '').split('|') if fl.strip()}
                    triggered = flags & active_flags
                    if not triggered:
                        continue

                    # If high_wer is the sole trigger, apply the stricter WER floor
                    if triggered == {'high_wer'} and min_wer is not None:
                        try:
                            wer = float(row.get('wer') or 0)
                        except ValueError:
                            wer = 0.0
                        if wer < min_wer:
                            skipped_wer += 1
                            continue

                    wav_path = row['wav_path']
                    if not os.path.exists(wav_path):
                        already_gone += 1
                        continue

                    source = row['source']
                    index = int(row['index'])
                    by_source[source][index] = wav_path
        except FileNotFoundError:
            log.error('Flagged CSV not found: %s', path)
        except Exception as e:
            log.error('Failed to read %s: %s', path, e)

    if already_gone:
        log.info('Skipped %d entries (WAV already deleted from a previous run)', already_gone)
    if skipped_wer:
        log.info('Skipped %d high_wer-only entries (WER < %.2f)', skipped_wer, min_wer)

    return dict(by_source)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Delete WAV clips flagged by WhisperX QC and clean metadata.csv. '
            'Dry-run by default; pass --apply to actually delete.'
        )
    )
    parser.add_argument(
        'flagged_csv',
        nargs='*',
        default=[f'{config.WX_EXPORT_BASE}_flagged.csv'],
        metavar='FLAGGED_CSV',
        help='Path(s) to _flagged.csv from wx.py (default: %(default)s)',
    )
    parser.add_argument(
        '--flags',
        nargs='+',
        default=ALL_FLAGS,
        choices=ALL_FLAGS,
        metavar='FLAG',
        help=(
            f'Flag types that trigger deletion (default: all — {ALL_FLAGS}). '
            'Example: --flags high_wer empty_transcription'
        ),
    )
    parser.add_argument(
        '--min-wer',
        type=float,
        default=None,
        metavar='THRESHOLD',
        help=(
            'For entries flagged only as high_wer: skip deletion if WER < THRESHOLD. '
            'Useful when you want to keep borderline entries. '
            'Example: --min-wer 0.8 keeps entries with WER in [0.5, 0.8).'
        ),
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually delete files and rewrite metadata (without this flag, dry-run only)',
    )
    args = parser.parse_args()

    dry_run = not args.apply
    active_flags = set(args.flags)

    log.info('Flagged CSV(s)  : %s', ', '.join(args.flagged_csv))
    log.info('Active flags    : %s', ', '.join(sorted(active_flags)))
    if args.min_wer is not None:
        log.info('Min WER (high_wer-only entries) : %.2f', args.min_wer)
    log.info('Mode            : %s', 'APPLY' if args.apply else 'DRY RUN')
    log.info('')

    total_flagged = flag_summary(args.flagged_csv)

    by_source = load_flagged(args.flagged_csv, active_flags, args.min_wer)
    if not by_source:
        log.info('Nothing to clean.')
        return

    total_entries = sum(len(v) for v in by_source.values())
    pct_selected = 100 * total_entries / total_flagged if total_flagged else 0
    log.info('Selected for deletion : %d  (%.1f%% of flagged)', total_entries, pct_selected)
    log.info('')

    for source, idx_wav in sorted(by_source.items()):
        log.info('  %s  →  %d row(s)', source, len(idx_wav))
        for idx in sorted(idx_wav):
            log.debug('    index=%-5d  %s', idx, idx_wav[idx])

    log.info('')

    if dry_run:
        log.info('Dry run — pass --apply to delete files and rewrite metadata.')
        return

    total_deleted = 0
    total_removed = 0

    for source, index_to_wav in sorted(by_source.items()):
        log.info('Processing: %s', source)
        total_deleted += _delete_wavs(index_to_wav, dry_run=False)
        removed = _rewrite_metadata(source, set(index_to_wav.keys()), dry_run=False)
        total_removed += removed
        log.info('  %d WAV(s) deleted, %d metadata row(s) removed', len(index_to_wav), removed)

    pct_of_flagged  = 100 * total_deleted / total_flagged  if total_flagged  else 0
    pct_of_selected = 100 * total_deleted / total_entries if total_entries else 0

    log.info('')
    log.info('=== WX CLEAN REPORT ===')
    log.info('Flagged entries total : %d', total_flagged)
    log.info('Selected for deletion : %d  (%.1f%% of flagged)', total_entries, pct_selected)
    log.info('WAV files deleted     : %d  (%.1f%% of flagged / %.1f%% of selected)',
             total_deleted, pct_of_flagged, pct_of_selected)
    log.info('Metadata rows removed : %d', total_removed)
    log.info('Done.')


# Examples:
#   python -m ch_ds.wx_clean                                          # dry-run, default path
#   python -m ch_ds.wx_clean results/wx_flagged.csv --apply           # delete all flagged
#   python -m ch_ds.wx_clean results/wx_flagged.csv --flags high_wer empty_transcription --apply
#   python -m ch_ds.wx_clean results/wx_flagged.csv --min-wer 0.8 --apply
#   python -m ch_ds.wx_clean results/wx_r0_flagged.csv results/wx_r1_flagged.csv --apply
if __name__ == '__main__':
    main()
