from __future__ import annotations

import csv
import logging
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field

import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # type: ignore[misc]

from . import config
from .txt import find_pairs, parse_srt


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.wx')
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

_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_NIKUD = re.compile(r'[ְ-ׇ]')  # hebrew diacritics


def _normalize(text: str) -> str:
    text = _NIKUD.sub('', text)
    text = unicodedata.normalize('NFC', text)
    text = _PUNCT.sub('', text)
    return ' '.join(text.split()).strip()


def _srt_ground_truth(srt_path: str) -> str:
    entries = parse_srt(srt_path)
    return _normalize(' '.join(e.text for e in entries))


def _resolve_device() -> tuple[str, str]:
    if config.WX_DEVICE:
        device = config.WX_DEVICE
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_type = config.WX_COMPUTE_TYPE if device == 'cuda' else 'int8'
    return device, compute_type


@dataclass
class WxResult:
    mp3_path: str
    srt_path: str
    detected_lang: str | None = None
    hypothesis: str | None = None
    reference: str | None = None
    wer: float | None = None
    avg_word_score: float | None = None
    flags: list[str] = field(default_factory=list)


def _compute_wer(reference: str, hypothesis: str) -> float:
    from jiwer import wer as jiwer_wer
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0
    return jiwer_wer(reference, hypothesis)


def run(pairs: list[tuple[str, str]], device: str, compute_type: str) -> list[WxResult]:
    import whisperx

    log.info('Loading WhisperX model: %s  device=%s  compute_type=%s',
             config.WX_MODEL, device, compute_type)
    model = whisperx.load_model(
        config.WX_MODEL,
        device,
        compute_type=compute_type,
        language=config.WX_LANGUAGE,
    )

    align_model, align_meta = None, None
    if config.WX_ALIGN:
        try:
            align_model, align_meta = whisperx.load_align_model(
                language_code=config.WX_LANGUAGE, device=device
            )
            log.info('Alignment model loaded for lang=%s', config.WX_LANGUAGE)
        except Exception as e:
            log.warning('Alignment model unavailable for lang=%s: %s', config.WX_LANGUAGE, e)

    results: list[WxResult] = []

    for srt_path, mp3_path in tqdm(pairs, desc='whisperx QC', unit='file'):
        rel = os.path.relpath(mp3_path, config.DATA_DIR)

        r = WxResult(mp3_path=mp3_path, srt_path=srt_path)
        try:
            r.reference = _srt_ground_truth(srt_path)
        except Exception as e:
            log.error('Failed to read SRT ground truth for %s: %s', rel, e)
            r.flags.append('empty_transcription')
            results.append(r)
            continue

        try:
            audio = whisperx.load_audio(mp3_path)
            wx_out = model.transcribe(audio, batch_size=config.WX_BATCH_SIZE)
        except Exception as e:
            log.warning('Transcription failed for %s: %s', rel, e)
            r.flags.append('empty_transcription')
            results.append(r)
            continue

        r.detected_lang = wx_out.get('language')
        segments = wx_out.get('segments', [])
        r.hypothesis = _normalize(' '.join(s['text'] for s in segments))

        if not r.hypothesis:
            log.debug('[empty_transcription] %s', rel)
            r.flags.append('empty_transcription')
            results.append(r)
            continue

        if r.detected_lang and r.detected_lang != config.WX_LANGUAGE:
            log.debug('[wrong_language] %s  detected=%s', rel, r.detected_lang)
            r.flags.append('wrong_language')

        r.wer = _compute_wer(r.reference, r.hypothesis)
        log.debug('%s  lang=%s  WER=%.3f', rel, r.detected_lang, r.wer)

        if r.wer > config.WX_WER_THRESHOLD:
            r.flags.append('high_wer')

        if align_model is not None and segments:
            try:
                aligned = whisperx.align(segments, align_model, align_meta, audio, device)
                word_scores = [
                    w.get('score', 0.0)
                    for seg in aligned.get('segments', [])
                    for w in seg.get('words', [])
                    if 'score' in w
                ]
                if word_scores:
                    r.avg_word_score = sum(word_scores) / len(word_scores)
            except Exception as e:
                log.debug('Alignment failed for %s: %s', rel, e)

        results.append(r)

    return results


def main():
    data_dir = config.DATA_DIR
    log.info('Starting WhisperX QC | data_dir=%s | limit=%s | log=%s',
             data_dir, config.WX_LIMIT, config.LOG_PATH)

    all_pairs = find_pairs(data_dir)
    valid_pairs = [(srt, mp3) for srt, mp3 in all_pairs if mp3 is not None]
    skipped = len(all_pairs) - len(valid_pairs)
    if skipped:
        log.warning('Skipped %d SRT(s) with no matching .mp3', skipped)

    if not valid_pairs:
        log.error('No valid (srt, mp3) pairs found in %s', data_dir)
        return

    if config.WX_LIMIT and len(valid_pairs) > config.WX_LIMIT:
        random.seed(42)
        valid_pairs = random.sample(valid_pairs, config.WX_LIMIT)
        log.info('Sampled %d / %d pairs (WX_LIMIT=%d)', len(valid_pairs), len(all_pairs), config.WX_LIMIT)

    device, compute_type = _resolve_device()

    base = config.WX_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    try:
        fa = open(f'{base}_all.csv',     'w', newline='', encoding='utf-8')
        fp = open(f'{base}_passed.csv',  'w', newline='', encoding='utf-8')
        ff = open(f'{base}_flagged.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    _hdr = ['mp3', 'srt', 'detected_lang', 'wer', 'avg_word_score']
    wa, wp, wf = csv.writer(fa), csv.writer(fp), csv.writer(ff)
    wa.writerow(_hdr + ['flags', 'passed'])
    wp.writerow(_hdr)
    wf.writerow(_hdr + ['flags', 'reference', 'hypothesis'])

    def _f(v, fmt='.4f'): return format(v, fmt) if v is not None else ''

    n = 0
    n_flagged = 0
    wer_scores: list[float] = []
    align_scores: list[float] = []
    from collections import Counter
    flag_counter: Counter = Counter()
    samples_buf: dict[str, list] = {ft: [] for ft in config.WX_FLAG_ORDER}

    try:
        for r in run(valid_pairs, device, compute_type):
            n += 1
            row_base = [r.mp3_path, r.srt_path, r.detected_lang, _f(r.wer), _f(r.avg_word_score)]
            if r.wer is not None:
                wer_scores.append(r.wer)
            if r.avg_word_score is not None:
                align_scores.append(r.avg_word_score)

            if r.flags:
                n_flagged += 1
                flag_counter.update(r.flags)
                wa.writerow(row_base + ['|'.join(r.flags), 'no'])
                wf.writerow(row_base + ['|'.join(r.flags), r.reference or '', r.hypothesis or ''])
                for ft in r.flags:
                    if len(samples_buf.get(ft, [])) < config.MAX_PRINT:
                        samples_buf.setdefault(ft, []).append(r)
            else:
                wa.writerow(row_base + ['', 'yes'])
                wp.writerow(row_base)
    finally:
        fa.close(); fp.close(); ff.close()

    log.info('')
    log.info('=== WHISPERX QC REPORT ===')
    log.info('Checked files   : %d', n)
    log.info('Flagged files   : %d (%.1f%%)', n_flagged, 100 * n_flagged / n if n else 0)

    if wer_scores:
        wer_scores.sort()
        p = lambda q: wer_scores[int(len(wer_scores) * q / 100)]
        log.info('WER             : mean=%.3f  p50=%.3f  p95=%.3f  max=%.3f',
                 sum(wer_scores) / len(wer_scores), p(50), p(95), wer_scores[-1])

    if align_scores:
        log.info('Avg word score  : %.3f', sum(align_scores) / len(align_scores))

    log.info('')
    log.info('--- Flags breakdown ---')
    for flag, count in flag_counter.most_common():
        log.info('  %-22s %6d  (%.2f%%)', flag, count, 100 * count / n if n else 0)

    for flag_type in config.WX_FLAG_ORDER:
        samples = samples_buf.get(flag_type, [])
        if not samples:
            continue
        total = flag_counter.get(flag_type, 0)
        log.info('')
        log.info('--- %s (%d total, showing up to %d) ---', flag_type, total, config.MAX_PRINT)
        for r in samples:
            rel = os.path.relpath(r.mp3_path, data_dir)
            wer_str = f'WER={r.wer:.2f}' if r.wer is not None else ''
            lang_str = f'lang={r.detected_lang}' if r.detected_lang else ''
            log.warning('[%s] %s  %s %s', flag_type, rel, lang_str, wer_str)
            if flag_type == 'high_wer' and r.reference and r.hypothesis:
                log.warning('  ref: %s', r.reference[:120])
                log.warning('  hyp: %s', r.hypothesis[:120])

    log.info('Saved: %s_{all,passed,flagged}.csv', base)
    log.info('Done.')


if __name__ == '__main__':
    main()
