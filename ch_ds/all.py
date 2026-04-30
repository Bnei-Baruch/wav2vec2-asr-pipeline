"""
Runs all dataset checks (txt, audio, lang) on every (srt, mp3) pair.
Writes pairs that passed all checks to ALL_EXPORT_BASE CSV files.
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
from .txt import find_pairs, parse_srt, check_entry
from .audio import check_mp3
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
class PairResult:
    srt_path: str
    mp3_path: str
    source_dir: str = ''
    duration_s: float | None = None
    srt_entries: int = 0
    text: str = ''
    rejected: bool = False
    reasons: list[str] = field(default_factory=list)


def _check_txt(srt_path: str) -> tuple[bool, list[str]]:
    """Returns (ok, reasons). ok=False если файл не прошёл текстовые проверки."""
    entries = parse_srt(srt_path)
    if not entries:
        return False, ['no_srt_entries']

    flagged_entries = 0
    fatal_found = []

    for entry in entries:
        flags = check_entry(entry)
        if not flags:
            continue
        flagged_entries += 1
        for f in flags:
            if f in config.ALL_TXT_FATAL_FLAGS:
                if f not in fatal_found:
                    fatal_found.append(f)

    reasons = []
    if fatal_found:
        reasons.append(f'txt_fatal:{"|".join(fatal_found)}')

    ratio = flagged_entries / len(entries)
    if ratio > config.ALL_TXT_MAX_FLAG_RATIO:
        reasons.append(f'txt_flag_ratio:{ratio:.2%}')

    return len(reasons) == 0, reasons


def _check_audio(mp3_path: str) -> tuple[bool, list[str]]:
    """Returns (ok, reasons)."""
    result = check_mp3(mp3_path)
    if result.flags:
        return False, [f'audio:{"|".join(result.flags)}']
    return True, []


def _check_lang(mp3_path: str, model, tmp_dir: str) -> tuple[bool, list[str]]:
    """Returns (ok, reasons). Sliding window language detection."""
    import numpy as np

    audio = _load_audio(mp3_path)
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

    pairs = find_pairs(data_dir)
    valid = [(srt, mp3) for srt, mp3 in pairs if mp3 is not None]
    skipped = len(pairs) - len(valid)
    if skipped:
        log.warning('Skipped %d SRT(s) with no .mp3', skipped)
    if not valid:
        log.error('No valid pairs found in %s', data_dir)
        return

    log.info('Found %d (srt, mp3) pairs', len(valid))

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
    rejected_samples: list[PairResult] = []

    try:
        fa = open(f'{base}_all.csv',      'w', newline='', encoding='utf-8')
        fp = open(f'{base}_passed.csv',   'w', newline='', encoding='utf-8')
        fr = open(f'{base}_rejected.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    _hdr_base = ['mp3_path', 'srt_path', 'source_dir', 'duration_s', 'srt_entries']
    wa, wp, wr = csv.writer(fa), csv.writer(fp), csv.writer(fr)
    wa.writerow(_hdr_base + ['reasons', 'passed'])
    wp.writerow(_hdr_base + ['text'])
    wr.writerow(_hdr_base + ['reasons'])

    def _f(v): return f'{v:.2f}' if v is not None else ''

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for srt_path, mp3_path in tqdm(valid, desc='full QC', unit='pair'):
                rel_mp3 = os.path.relpath(mp3_path, data_dir)

                try:
                    entries = parse_srt(srt_path)
                    full_text = ' '.join(e.text for e in entries).strip()
                    from .audio import check_mp3 as _check_mp3_inner
                    audio_r = _check_mp3_inner(mp3_path)
                except Exception as e:
                    log.error('Failed to process pair %s: %s', rel_mp3, e)
                    continue

                source_dir = os.path.relpath(os.path.dirname(srt_path), data_dir)
                r = PairResult(
                    srt_path=srt_path, mp3_path=mp3_path, source_dir=source_dir,
                    duration_s=audio_r.duration_s, srt_entries=len(entries), text=full_text,
                )

                try:
                    ok_txt, reasons_txt = _check_txt(srt_path)
                    if not ok_txt:
                        r.rejected = True; r.reasons.extend(reasons_txt)
                        log.debug('  FAIL txt: %s', reasons_txt)
                except Exception as e:
                    log.error('_check_txt failed for %s: %s', rel_mp3, e)
                    r.rejected = True; r.reasons.append('txt_error')

                try:
                    ok_audio, reasons_audio = _check_audio(mp3_path)
                    if not ok_audio:
                        r.rejected = True; r.reasons.extend(reasons_audio)
                        log.debug('  FAIL audio: %s', reasons_audio)
                except Exception as e:
                    log.error('_check_audio failed for %s: %s', rel_mp3, e)
                    r.rejected = True; r.reasons.append('audio_error')

                if config.ALL_RUN_LANG and lang_model is not None:
                    try:
                        ok_lang, reasons_lang = _check_lang(mp3_path, lang_model, tmp_dir)
                        if not ok_lang:
                            r.rejected = True; r.reasons.extend(reasons_lang)
                            log.debug('  FAIL lang: %s', reasons_lang)
                    except Exception as e:
                        log.error('_check_lang failed for %s: %s', rel_mp3, e)
                        r.rejected = True; r.reasons.append('lang_error')

                n += 1
                row_base = [r.mp3_path, r.srt_path, r.source_dir, _f(r.duration_s), r.srt_entries]
                if r.rejected:
                    wa.writerow(row_base + ['|'.join(r.reasons), 'no'])
                    wr.writerow(row_base + ['|'.join(r.reasons)])
                    log.warning('  REJECTED  %s  reasons=%s', rel_mp3, r.reasons)
                    if len(rejected_samples) < config.MAX_PRINT:
                        rejected_samples.append(r)
                else:
                    n_passed += 1
                    if r.duration_s:
                        total_good_hours += r.duration_s / 3600
                    wa.writerow(row_base + ['', 'yes'])
                    wp.writerow(row_base + [r.text])
                    log.debug('  OK  %s', rel_mp3)
    finally:
        fa.close(); fp.close(); fr.close()

    n_rejected = n - n_passed
    log.info('')
    log.info('=== FULL CHECK REPORT ===')
    log.info('Total pairs  : %d', n)
    log.info('Passed       : %d (%.1f%%)', n_passed,   100 * n_passed   / n if n else 0)
    log.info('Rejected     : %d (%.1f%%)', n_rejected, 100 * n_rejected / n if n else 0)
    if n_passed:
        log.info('Good audio   : %.1f hours', total_good_hours)

    if rejected_samples:
        log.info('')
        log.info('--- Rejected files (showing up to %d) ---', config.MAX_PRINT)
        for r in rejected_samples:
            rel = os.path.relpath(r.mp3_path, data_dir)
            log.warning('  %s  →  %s', rel, ' | '.join(r.reasons))

    log.info('Saved: %s_{all,passed,rejected}.csv', base)
    log.info('Done.')




if __name__ == '__main__':
    main()
