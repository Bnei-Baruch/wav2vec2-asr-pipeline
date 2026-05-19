"""
Runs all dataset checks (txt, audio, lang) on every metadata entry.
Writes entries that passed all checks to ALL_EXPORT_BASE CSV files.
"""
from __future__ import annotations

import csv
import logging
import os
import tempfile
from dataclasses import dataclass, field

import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # type: ignore[misc]

from . import config
from .txt import MetadataEntry, find_metadata_files, parse_metadata, check_entry
from .audio import check_wav
from .lang import _load_audio, _detect_language


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.all')
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

SR = 16_000


@dataclass
class EntryResult:
    source: str
    index: int
    wav_path: str
    text: str
    duration_s: float | None = None
    rejected: bool = False
    reasons: list[str] = field(default_factory=list)


def _check_txt(entry: MetadataEntry) -> tuple[bool, list[str]]:
    flags = check_entry(entry)
    fatal = [f for f in flags if f not in config.ALL_TXT_SOFT_FLAGS]
    if fatal:
        return False, [f'txt:{"|".join(fatal)}']
    return True, []


def _check_audio(wav_path: str) -> tuple[bool, list[str]]:
    result = check_wav(wav_path)
    if result.flags:
        return False, [f'audio:{"|".join(result.flags)}']
    return True, []


def _check_mismatch(text: str, duration_s: float) -> tuple[bool, list[str]]:
    words = len(text.split())
    wps = words / duration_s if duration_s > 0 else float("inf")
    if wps > config.MISMATCH_MAX_WPS:
        return False, [f'mismatch:text_too_dense']
    if wps < config.MISMATCH_MIN_WPS:
        return False, [f'mismatch:text_too_sparse']
    return True, []


def _check_lang(wav_path: str, model, tmp_dir: str) -> tuple[bool, list[str]]:
    import numpy as np

    audio = _load_audio(wav_path)
    if audio is None:
        return False, ['lang:unreadable']

    window_samples = int(config.LANG_WINDOW_S * SR)
    stride_samples = int(config.LANG_STRIDE_S * SR)
    starts = list(range(0, len(audio) - window_samples + 1, stride_samples)) or [0]

    foreign: list[str] = []
    for start in starts:
        chunk = audio[start:min(start + window_samples, len(audio))]
        if np.abs(chunk).mean() < 1e-4:
            continue
        det = _detect_language(model, chunk, tmp_dir)
        if det is None:
            continue
        lang, prob = det
        if prob >= config.LANG_MIN_PROB and lang != config.LANG_EXPECTED:
            start_s = start / SR
            end_s = (start + window_samples) / SR
            foreign.append(f'{lang}@{start_s:.0f}-{end_s:.0f}s')

    if foreign:
        return False, [f'lang:{" ".join(foreign)}']
    return True, []


def main():
    data_dir = config.DATA_DIR
    log.info('Starting full dataset check | data_dir=%s', data_dir)
    log.info('Checks: txt + audio%s', ' + lang' if config.ALL_RUN_LANG else '')
    log.info('Output base: %s', config.ALL_EXPORT_BASE)

    metadata_files = find_metadata_files(data_dir)
    if not metadata_files:
        log.error('No metadata.csv files found in %s', data_dir)
        return

    all_entries: list[MetadataEntry] = []
    for csv_path in metadata_files:
        all_entries.extend(parse_metadata(csv_path))

    if not all_entries:
        log.error('No entries found in %s', data_dir)
        return

    log.info('Found %d entries across %d metadata file(s)', len(all_entries), len(metadata_files))

    lang_model = None
    if config.ALL_RUN_LANG:
        device = config.LANG_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu')
        compute_type = config.LANG_COMPUTE_TYPE if device == 'cuda' else 'int8'
        log.info('Loading lang model: %s  device=%s', config.LANG_MODEL, device)
        try:
            from faster_whisper import WhisperModel
            lang_model = WhisperModel(config.LANG_MODEL, device=device, compute_type=compute_type)
        except Exception as e:
            log.error('Failed to load lang model: %s — lang check will be skipped', e)

    base = config.ALL_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    n = 0
    n_passed = 0
    total_good_hours = 0.0
    rejected_samples: list[EntryResult] = []

    try:
        fa = open(f'{base}_all.csv',      'w', newline='', encoding='utf-8')
        fp = open(f'{base}_passed.csv',   'w', newline='', encoding='utf-8')
        fr = open(f'{base}_rejected.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    _hdr = ['source', 'index', 'wav_path', 'text', 'duration_s']
    wa, wp, wr = csv.writer(fa), csv.writer(fp), csv.writer(fr)
    wa.writerow(_hdr + ['reasons', 'passed'])
    wp.writerow(_hdr)
    wr.writerow(_hdr + ['reasons'])

    def _f(v): return f'{v:.2f}' if v is not None else ''

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for entry in tqdm(all_entries, desc='full QC', unit='entry'):
                rel = os.path.relpath(entry.wav_path, data_dir)

                try:
                    audio_r = check_wav(entry.wav_path)
                except Exception as e:
                    log.error('Failed to read audio for %s: %s', rel, e)
                    continue

                r = EntryResult(
                    source=entry.source,
                    index=entry.index,
                    wav_path=entry.wav_path,
                    text=entry.text,
                    duration_s=audio_r.duration_s,
                )

                try:
                    ok_txt, reasons_txt = _check_txt(entry)
                    if not ok_txt:
                        r.rejected = True
                        r.reasons.extend(reasons_txt)
                        log.debug('  FAIL txt: %s', reasons_txt)
                except Exception as e:
                    log.error('_check_txt failed for %s: %s', rel, e)
                    r.rejected = True
                    r.reasons.append('txt_error')

                try:
                    ok_audio, reasons_audio = _check_audio(entry.wav_path)
                    if not ok_audio:
                        r.rejected = True
                        r.reasons.extend(reasons_audio)
                        log.debug('  FAIL audio: %s', reasons_audio)
                except Exception as e:
                    log.error('_check_audio failed for %s: %s', rel, e)
                    r.rejected = True
                    r.reasons.append('audio_error')

                if r.duration_s is not None:
                    try:
                        ok_mm, reasons_mm = _check_mismatch(entry.text, r.duration_s)
                        if not ok_mm:
                            r.rejected = True
                            r.reasons.extend(reasons_mm)
                            log.debug('  FAIL mismatch: %s', reasons_mm)
                    except Exception as e:
                        log.error('_check_mismatch failed for %s: %s', rel, e)

                if config.ALL_RUN_LANG and lang_model is not None:
                    try:
                        ok_lang, reasons_lang = _check_lang(entry.wav_path, lang_model, tmp_dir)
                        if not ok_lang:
                            r.rejected = True
                            r.reasons.extend(reasons_lang)
                            log.debug('  FAIL lang: %s', reasons_lang)
                    except Exception as e:
                        log.error('_check_lang failed for %s: %s', rel, e)
                        r.rejected = True
                        r.reasons.append('lang_error')

                n += 1
                row_base = [r.source, r.index, r.wav_path, r.text, _f(r.duration_s)]
                if r.rejected:
                    wa.writerow(row_base + ['|'.join(r.reasons), 'no'])
                    wr.writerow(row_base + ['|'.join(r.reasons)])
                    log.warning('  REJECTED  %s  reasons=%s', rel, r.reasons)
                    if len(rejected_samples) < config.MAX_PRINT:
                        rejected_samples.append(r)
                else:
                    n_passed += 1
                    if r.duration_s:
                        total_good_hours += r.duration_s / 3600
                    wa.writerow(row_base + ['', 'yes'])
                    wp.writerow(row_base)
                    log.debug('  OK  %s', rel)
    finally:
        fa.close(); fp.close(); fr.close()

    n_rejected = n - n_passed
    log.info('')
    log.info('=== FULL CHECK REPORT ===')
    log.info('Total entries: %d', n)
    log.info('Passed       : %d (%.1f%%)', n_passed,   100 * n_passed   / n if n else 0)
    log.info('Rejected     : %d (%.1f%%)', n_rejected, 100 * n_rejected / n if n else 0)
    if n_passed:
        log.info('Good audio   : %.1f hours', total_good_hours)

    if rejected_samples:
        log.info('')
        log.info('--- Rejected entries (showing up to %d) ---', config.MAX_PRINT)
        for r in rejected_samples:
            rel = os.path.relpath(r.wav_path, data_dir)
            log.warning('  %s  →  %s', rel, ' | '.join(r.reasons))

    log.info('Saved: %s_{all,passed,rejected}.csv', base)
    log.info('Done.')


# Example: python -m ch_ds.all
if __name__ == '__main__':
    main()
