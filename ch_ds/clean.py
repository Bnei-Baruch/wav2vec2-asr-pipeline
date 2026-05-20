"""
Deletes WAV clips and removes their rows from metadata.csv files.

Reads a _rejected.csv produced by all.py and filters by --flag substring.
Without --apply, shows what would be deleted (dry run).

Usage:
    python -m ch_ds.clean results/all_rejected.csv --flag audio:clipping
    python -m ch_ds.clean results/all_rejected.csv --flag audio:clipping --apply
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict

from . import config


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.clean')
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


def flag_summary(rejected_csv: str) -> None:
    """Prints a count-per-flag breakdown of all entries in the rejected CSV."""
    counts: dict[str, int] = defaultdict(int)
    try:
        with open(rejected_csv, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                for r in row.get('reasons', '').split('|'):
                    r = r.strip()
                    if r:
                        counts[r] += 1
    except FileNotFoundError:
        log.error('Rejected CSV not found: %s', rejected_csv)
        return
    except Exception as e:
        log.error('Failed to read %s: %s', rejected_csv, e)
        return

    if not counts:
        log.info('No rejected entries found.')
        return

    total = sum(counts.values())
    log.info('Rejection counts by flag (%d total flag hits):', total)
    for flag, n in sorted(counts.items(), key=lambda x: -x[1]):
        log.debug('  %-40s  %d', flag, n)
    log.info('')


def load_rejected(rejected_csv: str, flag: str) -> dict[str, dict[int, str]]:
    """
    Returns {source_metadata_path: {row_index: wav_path}} for entries
    whose reasons contain the given flag substring.
    """
    by_source: dict[str, dict[int, str]] = defaultdict(dict)
    try:
        with open(rejected_csv, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                reasons_raw = row.get('reasons', '')
                reasons = [r.strip() for r in reasons_raw.split('|') if r.strip()]
                if not any(flag in r for r in reasons):
                    continue
                source   = row['source']
                index    = int(row['index'])
                wav_path = row['wav_path']
                by_source[source][index] = wav_path
    except FileNotFoundError:
        log.error('Rejected CSV not found: %s', rejected_csv)
        return {}
    except Exception as e:
        log.error('Failed to read %s: %s', rejected_csv, e)
        return {}
    return dict(by_source)


def _delete_wavs(index_to_wav: dict[int, str], dry_run: bool) -> int:
    deleted = 0
    for wav_path in index_to_wav.values():
        if dry_run:
            log.info('  [dry] would delete %s', wav_path)
            continue
        try:
            os.remove(wav_path)
            deleted += 1
            log.debug('  deleted %s', wav_path)
        except FileNotFoundError:
            log.warning('  WAV not found (skipped): %s', wav_path)
        except OSError as e:
            log.error('  cannot delete %s: %s', wav_path, e)
    return deleted


def _rewrite_metadata(source: str, indices_to_remove: set[int], dry_run: bool) -> int:
    """Rewrites metadata.csv keeping only rows not in indices_to_remove. Returns rows removed."""
    if not os.path.exists(source):
        log.warning('metadata.csv not found: %s', source)
        return 0

    try:
        with open(source, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            all_rows = list(reader)
    except Exception as e:
        log.error('Cannot read %s: %s', source, e)
        return 0

    kept = [row for i, row in enumerate(all_rows) if i not in indices_to_remove]
    removed = len(all_rows) - len(kept)

    if dry_run:
        log.info('  [dry] would rewrite %s: %d → %d rows', source, len(all_rows), len(kept))
        return removed

    try:
        with open(source, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        log.debug('  rewrote %s (%d rows kept)', source, len(kept))
    except Exception as e:
        log.error('Cannot write %s: %s', source, e)

    return removed


def main():
    parser = argparse.ArgumentParser(
        description='Delete rejected WAV clips and clean metadata, filtered by flag.'
    )
    parser.add_argument(
        'rejected_csv',
        nargs='?',
        default=f'{config.ALL_EXPORT_BASE}_rejected.csv',
        help='Path to _rejected.csv (default: %(default)s)',
    )
    parser.add_argument(
        '--flag',
        required=True,
        help='Filter entries whose reasons contain this substring (e.g. "audio:clipping", "lang:", "txt:")',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually delete files and rewrite metadata (without this flag, dry-run only)',
    )
    args = parser.parse_args()

    dry_run = not args.apply

    log.info('Rejected CSV : %s', args.rejected_csv)
    log.info('Flag filter  : %s', args.flag)
    log.info('Mode         : %s', 'APPLY' if args.apply else 'DRY RUN')
    log.info('')
    flag_summary(args.rejected_csv)

    by_source = load_rejected(args.rejected_csv, args.flag)
    if not by_source:
        log.info('No matching entries found.')
        return

    total_entries = sum(len(v) for v in by_source.values())
    log.info('Matched %d entry/entries across %d metadata file(s)', total_entries, len(by_source))
    log.info('')

    for source, idx_wav in sorted(by_source.items()):
        log.info('  %s  →  %d row(s)', source, len(idx_wav))
        for idx in sorted(idx_wav):
            log.info('    index=%-5d  %s', idx, idx_wav[idx])

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

    log.info('')
    log.info('=== CLEAN REPORT ===')
    log.info('WAV files deleted     : %d', total_deleted)
    log.info('Metadata rows removed : %d', total_removed)
    log.info('Done.')


# Example: python -m ch_ds.clean results/all_rejected.csv --flag audio:clipping
if __name__ == '__main__':
    main()
