"""
Language detection for WAV clips using faster-whisper.

For each clip (or sliding window if clip > LANG_WINDOW_S):
  run faster-whisper language detection, flag if lang != LANG_EXPECTED.
"""
from __future__ import annotations

import csv
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # type: ignore[misc]

from . import config
from .txt import find_metadata_files, parse_metadata


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.lang')
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

SR = 16_000  # faster-whisper expects 16kHz


@dataclass
class WindowResult:
    start_s: float
    end_s: float
    language: str
    probability: float


@dataclass
class FileResult:
    path: str
    duration_s: float | None = None
    windows: list[WindowResult] = field(default_factory=list)
    foreign_windows: list[WindowResult] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _load_audio(wav_path: str) -> np.ndarray | None:
    """Load audio as mono float32 at 16kHz."""
    try:
        import whisperx
        return whisperx.load_audio(wav_path)
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(SR)
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    except Exception as e:
        log.debug('Cannot load audio %s: %s', wav_path, e)
        return None


def _detect_language(model, audio_chunk: np.ndarray, tmp_dir: str) -> tuple[str, float] | None:
    """Write chunk to temp wav, run faster-whisper language detection, return (lang, prob)."""
    tmp_path = os.path.join(tmp_dir, '_chunk.wav')
    try:
        sf.write(tmp_path, audio_chunk, SR)
        segments, info = model.transcribe(
            tmp_path,
            language=None,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        for _ in segments:
            break
        return info.language, info.language_probability
    except Exception as e:
        log.debug('Language detection failed: %s', e)
        return None


def analyze_file(wav_path: str, model, tmp_dir: str) -> FileResult:
    result = FileResult(path=wav_path)
    rel = os.path.relpath(wav_path, config.DATA_DIR)

    audio = _load_audio(wav_path)
    if audio is None:
        result.flags.append('unreadable')
        return result

    result.duration_s = len(audio) / SR
    window_samples = int(config.LANG_WINDOW_S * SR)
    stride_samples = int(config.LANG_STRIDE_S * SR)
    total_samples = len(audio)

    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    if not starts:
        starts = [0]

    log.debug('%s  dur=%.1fs  windows=%d', rel, result.duration_s, len(starts))

    for start in starts:
        end = min(start + window_samples, total_samples)
        chunk = audio[start:end]

        if np.abs(chunk).mean() < 1e-4:
            continue

        det = _detect_language(model, chunk, tmp_dir)
        if det is None:
            continue

        lang, prob = det
        start_s = start / SR
        end_s = end / SR

        if prob < config.LANG_MIN_PROB:
            w = WindowResult(start_s=start_s, end_s=end_s, language=lang, probability=prob)
            result.windows.append(w)
            log.debug('  [low_confidence] %.1fs–%.1fs  lang=%s prob=%.2f', start_s, end_s, lang, prob)
            continue

        w = WindowResult(start_s=start_s, end_s=end_s, language=lang, probability=prob)
        result.windows.append(w)

        if lang != config.LANG_EXPECTED:
            result.foreign_windows.append(w)
            log.debug('  [foreign] %.1fs–%.1fs  lang=%s prob=%.2f', start_s, end_s, lang, prob)

    if result.foreign_windows:
        result.flags.append('foreign_segment')

    return result


def _resolve_device() -> tuple[str, str]:
    device = config.LANG_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu')
    compute_type = config.LANG_COMPUTE_TYPE if device == 'cuda' else 'int8'
    return device, compute_type


def main():
    data_dir = config.DATA_DIR
    log.info('Starting language detection QC | data_dir=%s | log=%s', data_dir, config.LOG_PATH)
    log.info('Window=%.0fs  Stride=%.0fs  MinProb=%.2f  Expected=%s',
             config.LANG_WINDOW_S, config.LANG_STRIDE_S, config.LANG_MIN_PROB, config.LANG_EXPECTED)

    metadata_files = find_metadata_files(data_dir)
    if not metadata_files:
        log.error('No metadata.csv files found in %s', data_dir)
        return

    wav_paths: list[str] = []
    for csv_path in metadata_files:
        for entry in parse_metadata(csv_path):
            wav_paths.append(entry.wav_path)

    if not wav_paths:
        log.error('No WAV entries found in metadata files')
        return

    log.info('Found %d WAV entry/entries', len(wav_paths))

    device, compute_type = _resolve_device()
    log.info('Loading model: %s  device=%s  compute_type=%s', config.LANG_MODEL, device, compute_type)

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(config.LANG_MODEL, device=device, compute_type=compute_type)
    except Exception as e:
        log.error('Failed to load WhisperModel: %s', e)
        return

    base = config.LANG_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    n = 0
    n_flagged = 0
    n_foreign_windows = 0
    lang_counter: Counter = Counter()
    samples_buf: list = []

    try:
        fa = open(f'{base}_all.csv',     'w', newline='', encoding='utf-8')
        fp = open(f'{base}_passed.csv',  'w', newline='', encoding='utf-8')
        ff = open(f'{base}_flagged.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    wa, wp, wf = csv.writer(fa), csv.writer(fp), csv.writer(ff)
    wa.writerow(['path', 'duration_s', 'windows', 'foreign_windows', 'flags', 'passed'])
    wp.writerow(['path', 'duration_s', 'windows'])
    wf.writerow(['file', 'start_s', 'end_s', 'language', 'probability'])

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for wav_path in tqdm(wav_paths, desc='lang QC', unit='file'):
                rel = os.path.relpath(wav_path, data_dir)
                try:
                    r = analyze_file(wav_path, model, tmp_dir)
                except Exception as e:
                    log.error('analyze_file failed for %s: %s', rel, e)
                    continue

                n += 1
                dur = f'{r.duration_s:.2f}' if r.duration_s is not None else ''

                if r.flags:
                    n_flagged += 1
                    wa.writerow([r.path, dur, len(r.windows), len(r.foreign_windows),
                                 '|'.join(r.flags), 'no'])
                    for w in r.foreign_windows:
                        n_foreign_windows += 1
                        lang_counter[w.language] += 1
                        wf.writerow([r.path, f'{w.start_s:.2f}', f'{w.end_s:.2f}',
                                     w.language, f'{w.probability:.4f}'])
                        log.warning('  [%s] %.1fs–%.1fs  lang=%s  prob=%.2f',
                                    rel, w.start_s, w.end_s, w.language, w.probability)
                    if len(samples_buf) < config.MAX_PRINT:
                        samples_buf.append(r)
                else:
                    wa.writerow([r.path, dur, len(r.windows), 0, '', 'yes'])
                    wp.writerow([r.path, dur, len(r.windows)])
    finally:
        fa.close(); fp.close(); ff.close()

    log.info('')
    log.info('=== LANGUAGE DETECTION REPORT ===')
    log.info('Total files         : %d', n)
    log.info('Files with foreign  : %d (%.1f%%)', n_flagged, 100 * n_flagged / n if n else 0)
    log.info('Foreign windows     : %d', n_foreign_windows)

    if lang_counter:
        log.info('')
        log.info('--- Foreign languages detected ---')
        for lang, count in lang_counter.most_common():
            log.info('  %s : %d window(s)', lang, count)

    log.info('')
    log.info('--- Files with foreign segments (showing up to %d) ---', config.MAX_PRINT)
    for r in samples_buf:
        rel = os.path.relpath(r.path, data_dir)
        segments_str = '  '.join(
            f'{w.start_s:.0f}–{w.end_s:.0f}s:{w.language}({w.probability:.2f})'
            for w in r.foreign_windows
        )
        log.warning('%s  →  %s', rel, segments_str)

    log.info('Saved: %s_{all,passed,flagged}.csv', base)
    log.info('Done.')


# Example: python -m ch_ds.lang
if __name__ == '__main__':
    main()
