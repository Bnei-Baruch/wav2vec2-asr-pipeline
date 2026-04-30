"""
Language detection across full MP3 files using a sliding window.

For each window: run faster-whisper language detection (no transcription).
Flags windows where detected language != LANG_EXPECTED.
Catches Russian/other inclusions anywhere in the file.
"""
import csv
import logging
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch

from . import config
from .audio import find_mp3_files


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
    logger.addHandler(ch)
    logger.addHandler(fh)
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


def _load_audio(mp3_path: str) -> np.ndarray | None:
    """Load mp3 as mono float32 at 16kHz."""
    try:
        import whisperx
        return whisperx.load_audio(mp3_path)
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(mp3_path).set_channels(1).set_frame_rate(SR)
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    except Exception as e:
        log.debug('Cannot load audio %s: %s', mp3_path, e)
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
        # consume generator to populate info
        for _ in segments:
            break
        return info.language, info.language_probability
    except Exception as e:
        log.debug('Language detection failed: %s', e)
        return None


def analyze_file(mp3_path: str, model, tmp_dir: str) -> FileResult:
    result = FileResult(path=mp3_path)
    rel = os.path.relpath(mp3_path, config.DATA_DIR)

    audio = _load_audio(mp3_path)
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

        # skip near-silent chunks
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

    mp3_files = find_mp3_files(data_dir)
    if not mp3_files:
        log.error('No .mp3 files found in %s', data_dir)
        return

    log.info('Found %d MP3 file(s)', len(mp3_files))

    device, compute_type = _resolve_device()
    log.info('Loading model: %s  device=%s  compute_type=%s', config.LANG_MODEL, device, compute_type)

    from faster_whisper import WhisperModel
    model = WhisperModel(config.LANG_MODEL, device=device, compute_type=compute_type)

    results: list[FileResult] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, mp3_path in enumerate(mp3_files):
            rel = os.path.relpath(mp3_path, data_dir)
            log.info('[%d/%d] %s', i + 1, len(mp3_files), rel)
            r = analyze_file(mp3_path, model, tmp_dir)
            results.append(r)

            if r.foreign_windows:
                for w in r.foreign_windows:
                    log.warning('  [%s] %.1fs–%.1fs  lang=%s  prob=%.2f',
                                rel, w.start_s, w.end_s, w.language, w.probability)

    n = len(results)
    flagged = [r for r in results if r.flags]
    all_foreign = [w for r in results for w in r.foreign_windows]

    from collections import Counter
    lang_counter = Counter(w.language for w in all_foreign)

    log.info('')
    log.info('=== LANGUAGE DETECTION REPORT ===')
    log.info('Total files         : %d', n)
    log.info('Files with foreign  : %d (%.1f%%)', len(flagged), 100 * len(flagged) / n if n else 0)
    log.info('Foreign windows     : %d', len(all_foreign))

    if lang_counter:
        log.info('')
        log.info('--- Foreign languages detected ---')
        for lang, count in lang_counter.most_common():
            log.info('  %s : %d window(s)', lang, count)

    log.info('')
    log.info('--- Files with foreign segments (showing up to %d) ---', config.MAX_PRINT)
    for r in flagged[:config.MAX_PRINT]:
        rel = os.path.relpath(r.path, data_dir)
        segments_str = '  '.join(
            f'{w.start_s:.0f}–{w.end_s:.0f}s:{w.language}({w.probability:.2f})'
            for w in r.foreign_windows
        )
        log.warning('%s  →  %s', rel, segments_str)

    if config.EXPORT_PATH:
        export_path = config.EXPORT_PATH.replace('.csv', '_lang.csv')
        with open(export_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'start_s', 'end_s', 'language', 'probability'])
            for r in results:
                for w in r.foreign_windows:
                    writer.writerow([r.path, f'{w.start_s:.2f}', f'{w.end_s:.2f}',
                                     w.language, f'{w.probability:.4f}'])
        log.info('Exported %d foreign windows -> %s', len(all_foreign), export_path)

    log.info('Done.')


if __name__ == '__main__':
    main()
