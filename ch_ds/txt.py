from __future__ import annotations

import csv
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass

from . import config
from .punct import check_punct, punct_detail

CYRILLIC     = re.compile(r'[\u0400-\u04ff]')
LATIN        = re.compile(r'[a-zA-Z]')
HTML_TAG     = re.compile(r'<[^>]+>')
TIMESTAMP    = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}')
ENTRY_NUM    = re.compile(r'^\d+$')
REPEATED_WRD = re.compile(r'\b(\w+)\s+\1\b', re.UNICODE)


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('ch_ds.txt')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(config.LOG_PATH, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    base, ext = os.path.splitext(config.LOG_PATH)
    err_path = f'{base}_errors{ext}'
    eh = logging.FileHandler(err_path, encoding='utf-8')
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(eh)
    return logger


log = _setup_logger()


@dataclass
class SrtEntry:
    source: str
    index: int
    start_sec: float
    end_sec: float
    text: str


def _parse_ts(ts: str) -> float:
    h, m, rest = ts.strip().split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: str) -> list[SrtEntry]:
    with open(path, encoding='utf-8-sig') as f:
        content = f.read()

    entries = []
    bad_blocks = 0
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = [l.strip() for l in block.strip().splitlines()]
        if not lines:
            continue

        ts_idx = next((i for i, l in enumerate(lines) if TIMESTAMP.match(l)), None)
        if ts_idx is None:
            continue

        parts = lines[ts_idx].split('-->')
        try:
            start = _parse_ts(parts[0])
            end   = _parse_ts(parts[1])
        except Exception as e:
            bad_blocks += 1
            log.debug('Bad timestamp in %s block %d: %s', path, len(entries), e)
            continue

        text_lines = [l for l in lines[ts_idx + 1:] if l and not ENTRY_NUM.match(l)]
        text = ' '.join(text_lines).strip()

        entries.append(SrtEntry(
            source=path,
            index=len(entries),
            start_sec=start,
            end_sec=end,
            text=text,
        ))

    if bad_blocks:
        log.warning('%s: skipped %d block(s) with unparseable timestamps', path, bad_blocks)

    return entries


def _mp3_duration(mp3_path: str) -> float | None:
    try:
        from pydub import AudioSegment
        return len(AudioSegment.from_mp3(mp3_path)) / 1000.0
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', mp3_path],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        log.warning('Cannot read duration of %s (pydub and ffprobe both failed)', mp3_path)
        return None


def find_pairs(data_dir: str) -> list[tuple[str, str | None]]:
    pairs = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        srt_files = sorted(f for f in files if f.lower().endswith('.srt'))
        mp3_files = sorted(f for f in files if f.lower().endswith('.mp3'))
        mp3_path = os.path.join(root, mp3_files[0]) if mp3_files else None
        for srt_file in srt_files:
            pairs.append((os.path.join(root, srt_file), mp3_path))
    return pairs


def _flag_detail(entry: SrtEntry, flag: str, data_dir: str) -> str:
    rel  = os.path.relpath(entry.source, data_dir)
    dur  = entry.end_sec - entry.start_sec
    cps  = len(entry.text) / dur if dur > 0 else 0
    ts   = f'{entry.start_sec:.2f}s–{entry.end_sec:.2f}s'

    extras = {
        'cps_low':      f'cps={cps:.1f} (min={config.CPS_LOW})',
        'cps_high':     f'cps={cps:.1f} (max={config.CPS_HIGH})',
        'too_short':    f'len={len(entry.text.strip())}',
        'exceeds_audio': f'entry_end={entry.end_sec:.2f}s',
        'bad_timing':   f'duration={dur:.3f}s',
    }
    extra = extras.get(flag) # or punct_detail(entry.text, flag)
    text_preview = entry.text[:80].replace('\n', ' ')
    return f'[{flag}] {rel} entry#{entry.index} {ts}  {extra}  text="{text_preview}"'


def check_entry(entry: SrtEntry, mp3_duration: float | None = None) -> list[str]:
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

    #if REPEATED_WRD.search(text):
    #    flags.append('repeated_word')

    duration = entry.end_sec - entry.start_sec
    if duration <= 0:
        flags.append('bad_timing')
    else:
        cps = len(text) / duration
        if cps < config.CPS_LOW:
            flags.append('cps_low')
        elif cps > config.CPS_HIGH:
            flags.append('cps_high')

    if mp3_duration is not None and entry.end_sec > mp3_duration + config.EXCEEDS_AUDIO_SLACK:
        flags.append('exceeds_audio')

    #flags.extend(check_punct(text))

    return flags


def main():
    data_dir = config.DATA_DIR
    log.info('Starting text QC | data_dir=%s | log=%s', data_dir, config.LOG_PATH)

    pairs = find_pairs(data_dir)
    if not pairs:
        log.error('No .srt files found in %s', data_dir)
        return None

    log.info('Found %d SRT file(s)', len(pairs))

    base = config.TXT_EXPORT_BASE
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    n = 0
    n_flagged = 0
    missing_mp3 = 0
    flag_counter: Counter = Counter()
    text_counter: Counter = Counter()
    samples_buf: dict[str, list] = {ft: [] for ft in config.FLAG_ORDER}

    try:
        fa = open(f'{base}_txt_all.csv',     'w', newline='', encoding='utf-8')
        fp = open(f'{base}_txt_passed.csv',  'w', newline='', encoding='utf-8')
        ff = open(f'{base}_txt_flagged.csv', 'w', newline='', encoding='utf-8')
    except OSError as e:
        log.error('Cannot open output CSV files: %s', e)
        return None

    wa, wp, wf = csv.writer(fa), csv.writer(fp), csv.writer(ff)
    wa.writerow(['source', 'index', 'start_sec', 'end_sec', 'text', 'flags', 'passed'])
    wp.writerow(['source', 'index', 'start_sec', 'end_sec', 'text'])
    wf.writerow(['source', 'index', 'start_sec', 'end_sec', 'text', 'flags'])

    try:
        for srt_path, mp3_path in pairs:
            rel_srt = os.path.relpath(srt_path, data_dir)

            if mp3_path is None:
                log.warning('No .mp3 found alongside: %s', rel_srt)
                missing_mp3 += 1
                mp3_dur = None
            else:
                mp3_dur = _mp3_duration(mp3_path)
                dur_str = f'{mp3_dur:.1f}s' if mp3_dur else 'duration unknown'
                log.info('  SRT: %s  MP3: %s (%s)', rel_srt, os.path.basename(mp3_path), dur_str)

            try:
                entries = parse_srt(srt_path)
            except Exception as e:
                log.error('Failed to parse %s: %s', rel_srt, e)
                continue

            log.info('    parsed %d entries', len(entries))

            for entry in entries:
                n += 1
                text_counter[entry.text] += 1

                try:
                    flags = check_entry(entry, mp3_dur)
                except Exception as e:
                    log.error('check_entry failed for %s entry#%d: %s',
                               entry.source, entry.index, e)
                    continue

                flag_counter.update(flags)
                row = [entry.source, entry.index,
                       f'{entry.start_sec:.3f}', f'{entry.end_sec:.3f}', entry.text]
                if flags:
                    n_flagged += 1
                    wa.writerow(row + ['|'.join(flags), 'no'])
                    wf.writerow(row + ['|'.join(flags)])
                    for f in flags:
                        log.debug(_flag_detail(entry, f, data_dir))
                        if len(samples_buf.get(f, [])) < config.MAX_PRINT:
                            samples_buf.setdefault(f, []).append((entry, flags))
                else:
                    wa.writerow(row + ['', 'yes'])
                    wp.writerow(row)
    finally:
        fa.close(); fp.close(); ff.close()

    if n == 0:
        log.error('No entries parsed — check SRT format')
        return None

    dup_texts = sum(c - 1 for c in text_counter.values() if c > 1)

    log.info('')
    log.info('=== TEXT QC REPORT ===')
    log.info('Total entries   : %d', n)
    log.info('Missing .mp3    : %d', missing_mp3)
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
        for entry, _ in samples:
            log.warning(_flag_detail(entry, flag_type, data_dir))

    log.info('Saved: %s_txt_{all,passed,flagged}.csv', base)
    log.info('Done.')
    return None


if __name__ == '__main__':
    main()
