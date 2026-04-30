from __future__ import annotations

import csv
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass

from . import config
from .punct import check_punct, punct_detail

CYRILLIC = re.compile(r'[Ѐ-ӿ]')
LATIN    = re.compile(r'[a-zA-Z]')
HTML_TAG = re.compile(r'<[^>]+>')


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.txt')
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


@dataclass
class MetadataEntry:
    source: str    # absolute path to metadata.csv
    index: int     # row index (0-based)
    wav_path: str  # absolute path to wav file
    text: str      # sentence


def find_metadata_files(data_dir: str) -> list[str]:
    """Return sorted list of metadata.csv paths found under data_dir."""
    result = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        if 'metadata.csv' in files:
            result.append(os.path.join(root, 'metadata.csv'))
    return sorted(result)


def parse_metadata(csv_path: str) -> list[MetadataEntry]:
    """Parse a metadata.csv into MetadataEntry objects."""
    base_dir = os.path.dirname(csv_path)
    entries = []
    try:
        with open(csv_path, encoding='utf-8-sig', newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                wav_path = os.path.join(base_dir, row['file_name'])
                entries.append(MetadataEntry(
                    source=csv_path,
                    index=i,
                    wav_path=wav_path,
                    text=row['sentence'],
                ))
    except Exception as e:
        log.error('Failed to parse %s: %s', csv_path, e)
    return entries


def check_entry(entry: MetadataEntry) -> list[str]:
    flags = []
    text = entry.text

    if not text.strip():
        flags.append('empty')
        return flags

    if len(text.strip()) <= config.TOO_SHORT_LEN:
        flags.append('too_short')

    if CYRILLIC.search(text):
        flags.append('cyrillic')

    if LATIN.search(text):
        flags.append('latin')

    if HTML_TAG.search(text):
        flags.append('html_tag')

    flags.extend(check_punct(text))

    return flags


def main():
    data_dir = config.DATA_DIR
    log.info('Starting text QC | data_dir=%s | log=%s', data_dir, config.LOG_PATH)

    metadata_files = find_metadata_files(data_dir)
    if not metadata_files:
        log.error('No metadata.csv files found in %s', data_dir)
        return

    log.info('Found %d metadata file(s)', len(metadata_files))

    base = config.TXT_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    n = 0
    n_flagged = 0
    flag_counter: Counter = Counter()
    text_counter: Counter = Counter()
    samples_buf: dict[str, list] = {ft: [] for ft in config.FLAG_ORDER}

    try:
        fa = open(f'{base}_all.csv',     'w', newline='', encoding='utf-8')
        fp = open(f'{base}_passed.csv',  'w', newline='', encoding='utf-8')
        ff = open(f'{base}_flagged.csv', 'w', newline='', encoding='utf-8')
        fl = open(f'{base}_latin.csv',   'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    wa, wp, wf, wl = csv.writer(fa), csv.writer(fp), csv.writer(ff), csv.writer(fl)
    wa.writerow(['source', 'index', 'wav_path', 'text', 'flags', 'passed'])
    wp.writerow(['source', 'index', 'wav_path', 'text'])
    wf.writerow(['source', 'index', 'wav_path', 'text', 'flags'])
    wl.writerow(['source', 'index', 'wav_path', 'text'])

    try:
        for csv_path in metadata_files:
            rel = os.path.relpath(csv_path, data_dir)
            entries = parse_metadata(csv_path)
            log.info('  %s  (%d entries)', rel, len(entries))

            for entry in entries:
                n += 1
                text_counter[entry.text] += 1

                try:
                    flags = check_entry(entry)
                except Exception as e:
                    log.error('check_entry failed for %s #%d: %s', rel, entry.index, e)
                    continue

                flag_counter.update(flags)
                row = [entry.source, entry.index, entry.wav_path, entry.text]

                if 'latin' in flags:
                    wl.writerow(row)

                # latin is a soft flag — entry still passes unless other flags present
                reject_flags = [f for f in flags if f != 'latin']

                if reject_flags:
                    n_flagged += 1
                    wa.writerow(row + ['|'.join(flags), 'no'])
                    wf.writerow(row + ['|'.join(flags)])
                    for f in flags:
                        log.debug('[%s] %s #%d  text="%s"', f, rel, entry.index, entry.text[:60])
                        if len(samples_buf.get(f, [])) < config.MAX_PRINT:
                            samples_buf.setdefault(f, []).append(entry)
                else:
                    wa.writerow(row + ['|'.join(flags) if flags else '', 'yes'])
                    wp.writerow(row)
    finally:
        fa.close(); fp.close(); ff.close(); fl.close()

    if n == 0:
        log.error('No entries found — check metadata.csv format')
        return

    dup_texts = sum(c - 1 for c in text_counter.values() if c > 1)

    log.info('')
    log.info('=== TEXT QC REPORT ===')
    log.info('Total entries   : %d', n)
    log.info('Flagged entries : %d (%.1f%%)', n_flagged, 100 * n_flagged / n)
    log.info('Duplicate text  : %d extra rows', dup_texts)
    log.info('')
    log.info('--- Flags breakdown ---')
    for flag, count in flag_counter.most_common():
        log.info('  %-16s %6d  (%.2f%%)', flag, count, 100 * count / n)

    for flag_type in config.FLAG_ORDER:
        samples = samples_buf.get(flag_type, [])
        if not samples:
            continue
        total = flag_counter.get(flag_type, 0)
        log.info('')
        log.info('--- %s (%d total, showing up to %d) ---', flag_type, total, config.MAX_PRINT)
        for entry in samples:
            log.warning('[%s] %s #%d  text="%s"',
                        flag_type, os.path.relpath(entry.source, data_dir),
                        entry.index, entry.text[:80])

    log.info('Saved: %s_{all,passed,flagged,latin}.csv', base)
    log.info('Done.')


if __name__ == '__main__':
    main()
