from __future__ import annotations

import csv
import logging
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # type: ignore[misc]

from . import config
from .txt import MetadataEntry, find_metadata_files, parse_metadata


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


def _resolve_device(rank: int = 0) -> tuple[str, int, str]:
    n = torch.cuda.device_count()
    device = 'cuda' if n > 0 else 'cpu'
    device_index = rank if n > 0 else 0
    compute_type = config.WX_COMPUTE_TYPE if device == 'cuda' else 'int8'
    return device, device_index, compute_type


@dataclass
class WxResult:
    wav_path: str
    source: str
    index: int
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


def run(entries: list[MetadataEntry], device: str, device_index: int, compute_type: str, model: str | None = None):
    # pyannote VAD checkpoints contain omegaconf objects incompatible with weights_only=True
    import functools
    _orig_load = torch.load
    @functools.wraps(_orig_load)
    def _load_compat(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _load_compat

    import whisperx

    model_name = model or config.WX_MODEL
    log.info('Loading WhisperX model: %s  device=%s  device_index=%d  compute_type=%s',
             model_name, device, device_index, compute_type)
    model = whisperx.load_model(
        model_name,
        device,
        device_index=device_index,
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

    for entry in tqdm(entries, desc='whisperx QC', unit='file'):
        rel = os.path.relpath(entry.wav_path, config.DATA_DIR)
        r = WxResult(wav_path=entry.wav_path, source=entry.source, index=entry.index)
        r.reference = _normalize(entry.text)

        if not r.reference:
            r.flags.append('empty_transcription')
            yield r
            continue

        try:
            audio = whisperx.load_audio(entry.wav_path)
            wx_out = model.transcribe(audio, batch_size=config.WX_BATCH_SIZE)
        except Exception as e:
            log.warning('Transcription failed for %s: %s', rel, e)
            r.flags.append('empty_transcription')
            yield r
            continue

        r.detected_lang = wx_out.get('language')
        segments = wx_out.get('segments', [])
        r.hypothesis = _normalize(' '.join(s['text'] for s in segments))

        if not r.hypothesis:
            log.debug('[empty_transcription] %s', rel)
            r.flags.append('empty_transcription')
            yield r
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

        yield r


def _run_shard(rank: int, world_size: int, all_entries: list, base: str, model: str | None = None) -> None:
    device, device_index, compute_type = _resolve_device(rank)
    entries = all_entries[rank::world_size]
    suffix = f'_r{rank}' if world_size > 1 else ''

    log.info('Rank %d/%d | device=%s:%d | entries=%d', rank, world_size, device, device_index, len(entries))

    try:
        fa = open(f'{base}{suffix}_all.csv',     'w', newline='', encoding='utf-8')
        fp = open(f'{base}{suffix}_passed.csv',  'w', newline='', encoding='utf-8')
        ff = open(f'{base}{suffix}_flagged.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    _hdr = ['wav_path', 'source', 'index', 'detected_lang', 'wer', 'avg_word_score']
    wa, wp, wf = csv.writer(fa), csv.writer(fp), csv.writer(ff)
    wa.writerow(_hdr + ['flags', 'passed'])
    wp.writerow(_hdr)
    wf.writerow(_hdr + ['flags', 'reference', 'hypothesis'])

    def _f(v, fmt='.4f'): return format(v, fmt) if v is not None else ''

    n = 0
    n_flagged = 0
    wer_scores: list[float] = []
    align_scores: list[float] = []
    flag_counter: Counter = Counter()
    samples_buf: dict[str, list] = {ft: [] for ft in config.WX_FLAG_ORDER}

    try:
        for r in run(entries, device, device_index, compute_type, model):
            n += 1
            row_base = [r.wav_path, r.source, r.index, r.detected_lang,
                        _f(r.wer), _f(r.avg_word_score)]
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
            fa.flush()
            fp.flush()
            ff.flush()
    finally:
        fa.close(); fp.close(); ff.close()

    if n == 0:
        return

    log.info('')
    log.info('=== WHISPERX QC REPORT (rank %d) ===', rank)
    log.info('Checked entries : %d', n)
    log.info('Flagged entries : %d (%.1f%%)', n_flagged, 100 * n_flagged / n)

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
        log.info('  %-22s %6d  (%.2f%%)', flag, count, 100 * count / n)

    for flag_type in config.WX_FLAG_ORDER:
        samples = samples_buf.get(flag_type, [])
        if not samples:
            continue
        total = flag_counter.get(flag_type, 0)
        log.info('')
        log.info('--- %s (%d total, showing up to %d) ---', flag_type, total, config.MAX_PRINT)
        for r in samples:
            rel = os.path.relpath(r.wav_path, config.DATA_DIR)
            wer_str = f'WER={r.wer:.2f}' if r.wer is not None else ''
            lang_str = f'lang={r.detected_lang}' if r.detected_lang else ''
            log.warning('[%s] %s  %s %s', flag_type, rel, lang_str, wer_str)
            if flag_type == 'high_wer' and r.reference and r.hypothesis:
                log.warning('  ref: %s', r.reference[:120])
                log.warning('  hyp: %s', r.hypothesis[:120])

    log.info('Saved: %s%s_{all,passed,flagged}.csv', base, suffix)
    log.info('Done.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='Max entries to check (0 = all)')
    parser.add_argument('--model', type=str, default=None,
                        help='WhisperX model name or path (default: config.WX_MODEL)')
    args = parser.parse_args()
    limit = args.limit or None

    data_dir = config.DATA_DIR
    log.info('Starting WhisperX QC | data_dir=%s | limit=%s | log=%s',
             data_dir, limit, config.LOG_PATH)

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

    if limit and len(all_entries) > limit:
        random.seed(42)
        all_entries = random.sample(all_entries, limit)
        log.info('Sampled %d entries (limit=%d)', len(all_entries), limit)

    base = config.WX_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    n_gpus = torch.cuda.device_count()
    world_size = max(n_gpus, 1)
    log.info('GPUs detected: %d → %d shard(s)', n_gpus, world_size)

    if world_size > 1:
        mp.spawn(_run_shard, args=(world_size, all_entries, base, args.model), nprocs=world_size, join=True)
    else:
        _run_shard(0, 1, all_entries, base, args.model)


# Example:              python -m ch_ds.wx
# Example (500 clips):  python -m ch_ds.wx --limit 500
# Example (all clips):  python -m ch_ds.wx --limit 0
# Uses all available GPUs automatically; outputs wx_r0_*, wx_r1_*, ... when >1 GPU.
if __name__ == '__main__':
    main()
