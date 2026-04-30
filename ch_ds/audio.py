from __future__ import annotations

from collections import Counter
import csv
import logging
import os
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # type: ignore[misc]

from . import config


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.audio')
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
class AudioResult:
    path: str
    duration_s: float | None = None
    dbfs: float | None = None
    max_dbfs: float | None = None
    silence_ratio: float | None = None
    channels: int | None = None
    frame_rate: int | None = None
    flags: list[str] = field(default_factory=list)


def find_mp3_files(data_dir: str) -> list[str]:
    result = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith('.mp3'):
                result.append(os.path.join(root, f))
    return result


def _silence_ratio(audio) -> float:
    """Доля чанков тише порога AUDIO_SILENCE_THRESH."""
    chunks = [
        audio[i:i + config.AUDIO_CHUNK_MS]
        for i in range(0, len(audio), config.AUDIO_CHUNK_MS)
    ]
    if not chunks:
        return 1.0
    silent = sum(1 for c in chunks if c.dBFS < config.AUDIO_SILENCE_THRESH)
    return silent / len(chunks)


def check_mp3(path: str) -> AudioResult:
    result = AudioResult(path=path)

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(path)
    except Exception as e:
        log.debug('Unreadable: %s — %s', path, e)
        result.flags.append('unreadable')
        return result

    result.duration_s   = len(audio) / 1000.0
    result.dbfs         = audio.dBFS
    result.max_dbfs     = audio.max_dBFS
    result.channels     = audio.channels
    result.frame_rate   = audio.frame_rate
    result.silence_ratio = _silence_ratio(audio)

    if result.duration_s < config.AUDIO_MIN_DURATION:
        result.flags.append('too_short')

    if result.duration_s > config.AUDIO_MAX_DURATION:
        result.flags.append('too_long')

    if result.dbfs < config.AUDIO_MIN_DBFS:
        result.flags.append('too_quiet')

    if result.max_dbfs >= config.AUDIO_CLIPPING_DBFS:
        result.flags.append('clipping')

    if result.silence_ratio > config.AUDIO_SILENCE_MAX:
        result.flags.append('mostly_silent')

    return result


def _result_detail(r: AudioResult, flag: str, data_dir: str) -> str:
    rel  = os.path.relpath(r.path, data_dir)
    dur  = f'{r.duration_s:.1f}s' if r.duration_s is not None else 'n/a'
    dbfs = f'{r.dbfs:.1f}dBFS' if r.dbfs is not None else 'n/a'
    sil  = f'silence={r.silence_ratio:.0%}' if r.silence_ratio is not None else ''

    extras = {
        'too_short':    f'duration={dur} (min={config.AUDIO_MIN_DURATION}s)',
        'too_long':     f'duration={dur} (max={config.AUDIO_MAX_DURATION}s)',
        'too_quiet':    f'dBFS={dbfs} (min={config.AUDIO_MIN_DBFS})',
        'clipping':     f'max_dBFS={r.max_dbfs:.1f} (threshold={config.AUDIO_CLIPPING_DBFS})',
        'mostly_silent': f'{sil} (max={config.AUDIO_SILENCE_MAX:.0%})',
        'unreadable':   '',
    }
    return f'[{flag}] {rel}  {extras.get(flag, "")}'


def main():
    data_dir = config.DATA_DIR
    log.info('Starting audio QC | data_dir=%s | log=%s', data_dir, config.LOG_PATH)

    mp3_files = find_mp3_files(data_dir)
    if not mp3_files:
        log.error('No .mp3 files found in %s', data_dir)
        return

    log.info('Found %d MP3 file(s)', len(mp3_files))

    base = config.AUDIO_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    n = 0
    n_flagged = 0
    durations: list[float] = []
    dbfs_vals: list[float] = []
    flag_counter: Counter = Counter()
    samples_buf: dict[str, list] = {ft: [] for ft in config.AUDIO_FLAG_ORDER}

    try:
        fa = open(f'{base}_audio_all.csv',     'w', newline='', encoding='utf-8')
        fp = open(f'{base}_audio_passed.csv',  'w', newline='', encoding='utf-8')
        ff = open(f'{base}_audio_flagged.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return

    _hdr = ['path', 'duration_s', 'dbfs', 'silence_ratio']
    wa, wp, wf = csv.writer(fa), csv.writer(fp), csv.writer(ff)
    wa.writerow(_hdr + ['flags', 'passed'])
    wp.writerow(_hdr)
    wf.writerow(['path', 'duration_s', 'dbfs', 'max_dbfs', 'silence_ratio',
                 'channels', 'frame_rate', 'flags'])

    def _f(v, fmt='.2f'): return format(v, fmt) if v is not None else ''

    try:
        for path in tqdm(mp3_files, desc='audio QC', unit='file'):
            rel = os.path.relpath(path, data_dir)
            try:
                r = check_mp3(path)
            except Exception as e:
                log.error('check_mp3 failed for %s: %s', rel, e)
                continue

            n += 1
            if r.duration_s is not None:
                durations.append(r.duration_s)
            if r.dbfs is not None:
                dbfs_vals.append(r.dbfs)

            row_base = [r.path, _f(r.duration_s), _f(r.dbfs, '.1f'), _f(r.silence_ratio, '.3f')]
            if r.flags:
                n_flagged += 1
                flag_counter.update(r.flags)
                wa.writerow(row_base + ['|'.join(r.flags), 'no'])
                wf.writerow([r.path, _f(r.duration_s), _f(r.dbfs, '.1f'), _f(r.max_dbfs, '.1f'),
                             _f(r.silence_ratio, '.3f'), r.channels, r.frame_rate, '|'.join(r.flags)])
                for flag in r.flags:
                    log.debug(_result_detail(r, flag, data_dir))
                    if len(samples_buf.get(flag, [])) < config.MAX_PRINT:
                        samples_buf.setdefault(flag, []).append(r)
                log.info('  FLAGGED %s  flags=%s', rel, r.flags)
            else:
                wa.writerow(row_base + ['', 'yes'])
                wp.writerow(row_base)
                log.debug('  OK  %s  dur=%.1fs  dBFS=%.1f  silence=%.0f%%',
                          rel, r.duration_s or 0, r.dbfs or 0, (r.silence_ratio or 0) * 100)
    finally:
        fa.close(); fp.close(); ff.close()

    log.info('')
    log.info('=== AUDIO QC REPORT ===')
    log.info('Total files     : %d', n)
    log.info('Flagged files   : %d (%.1f%%)', n_flagged, 100 * n_flagged / n if n else 0)

    if durations:
        durations.sort()
        p = lambda q: durations[int(len(durations) * q / 100)]
        log.info('Duration        : min=%.1fs  p50=%.1fs  p95=%.1fs  max=%.1fs  total=%.1fh',
                 durations[0], p(50), p(95), durations[-1], sum(durations) / 3600)

    if dbfs_vals:
        dbfs_vals.sort()
        p2 = lambda q: dbfs_vals[int(len(dbfs_vals) * q / 100)]
        log.info('dBFS            : min=%.1f  p50=%.1f  max=%.1f', dbfs_vals[0], p2(50), dbfs_vals[-1])

    log.info('')
    log.info('--- Flags breakdown ---')
    for flag, count in flag_counter.most_common():
        log.info('  %-16s %6d  (%.2f%%)', flag, count, 100 * count / n if n else 0)

    for flag_type in config.AUDIO_FLAG_ORDER:
        samples = samples_buf.get(flag_type, [])
        if not samples:
            continue
        total = flag_counter.get(flag_type, 0)
        log.info('')
        log.info('--- %s (%d total, showing up to %d) ---', flag_type, total, config.MAX_PRINT)
        for r in samples:
            log.warning(_result_detail(r, flag_type, data_dir))

    log.info('Saved: %s_audio_{all,passed,flagged}.csv', base)
    log.info('Done.')


if __name__ == '__main__':
    main()
