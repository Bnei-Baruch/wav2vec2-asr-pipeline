"""
Runs all dataset checks (txt, audio, lang) on every (srt, mp3) pair.
Writes pairs that passed all checks to ALL_OUTPUT_PATH CSV.
"""
import csv
import logging
import os
import tempfile
from dataclasses import dataclass, field

import torch

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
    logger.addHandler(ch)
    logger.addHandler(fh)
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
    log.info('Output: %s', config.ALL_OUTPUT_PATH)

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
        from faster_whisper import WhisperModel
        lang_model = WhisperModel(config.LANG_MODEL, device=device, compute_type=compute_type)

    results: list[PairResult] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (srt_path, mp3_path) in enumerate(valid):
            rel_mp3 = os.path.relpath(mp3_path, data_dir)
            log.info('[%d/%d] %s', i + 1, len(valid), rel_mp3)

            entries = parse_srt(srt_path)
            full_text = ' '.join(e.text for e in entries).strip()

            from .audio import check_mp3 as _check_mp3_inner
            audio_r = _check_mp3_inner(mp3_path)

            source_dir = os.path.relpath(os.path.dirname(srt_path), data_dir)

            r = PairResult(
                srt_path=srt_path,
                mp3_path=mp3_path,
                source_dir=source_dir,
                duration_s=audio_r.duration_s,
                srt_entries=len(entries),
                text=full_text,
            )

            # txt
            ok_txt, reasons_txt = _check_txt(srt_path)
            if not ok_txt:
                r.rejected = True
                r.reasons.extend(reasons_txt)
                log.debug('  FAIL txt: %s', reasons_txt)

            # audio
            ok_audio, reasons_audio = _check_audio(mp3_path)
            if not ok_audio:
                r.rejected = True
                r.reasons.extend(reasons_audio)
                log.debug('  FAIL audio: %s', reasons_audio)

            # lang
            if config.ALL_RUN_LANG and lang_model is not None:
                ok_lang, reasons_lang = _check_lang(mp3_path, lang_model, tmp_dir)
                if not ok_lang:
                    r.rejected = True
                    r.reasons.extend(reasons_lang)
                    log.debug('  FAIL lang: %s', reasons_lang)

            if r.rejected:
                log.warning('  REJECTED  %s  reasons=%s', rel_mp3, r.reasons)
            else:
                log.debug('  OK  %s', rel_mp3)

            results.append(r)

    good = [r for r in results if not r.rejected]
    bad  = [r for r in results if r.rejected]
    n = len(results)

    log.info('')
    log.info('=== FULL CHECK REPORT ===')
    log.info('Total pairs  : %d', n)
    log.info('Passed       : %d (%.1f%%)', len(good), 100 * len(good) / n if n else 0)
    log.info('Rejected     : %d (%.1f%%)', len(bad),  100 * len(bad)  / n if n else 0)

    if good:
        total_h = sum(r.duration_s for r in good if r.duration_s) / 3600
        log.info('Good audio   : %.1f hours', total_h)

    if bad:
        log.info('')
        log.info('--- Rejected files ---')
        for r in bad:
            rel = os.path.relpath(r.mp3_path, data_dir)
            log.warning('  %s  →  %s', rel, ' | '.join(r.reasons))

    # write good to CSV
    with open(config.ALL_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_dir', 'mp3_path', 'srt_path', 'duration_s', 'srt_entries', 'text'])
        for r in good:
            writer.writerow([
                r.source_dir,
                r.mp3_path,
                r.srt_path,
                f'{r.duration_s:.2f}' if r.duration_s is not None else '',
                r.srt_entries,
                r.text,
            ])

    log.info('')
    log.info('Written %d good file(s) -> %s', len(good), config.ALL_OUTPUT_PATH)
    log.info('Done.')


if __name__ == '__main__':
    main()
