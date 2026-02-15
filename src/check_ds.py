# scripts/qc_dataset.py
import argparse
import json
from collections import Counter

import numpy as np
from datasets import load_from_disk
from .constants import VOCAB_PATH


def pick_text_col(cols):
    for c in ["transcription", "text", "sentence", "transcript", "labels_text"]:
        if c in cols:
            return c
    raise ValueError(f"Не найден текстовый столбец. Есть: {cols}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_print", type=int, default=20)
    args = p.parse_args()

    ds = load_from_disk("./train")
    cols = ds.column_names
    required = {"input_values", "labels"}
    missing = required - set(cols)
    if missing:
            raise ValueError(f"Нет нужных столбцов: {sorted(missing)}. Есть: {cols}")

    text_col = pick_text_col(cols)

    vocab = json.load(VOCAB_PATH)

    # Разрешенные символы: все токены, кроме спецтокенов вида [XXX]
    allowed = set()
    for tok in vocab.keys():
        if tok.startswith("[") and tok.endswith("]"):
            continue
        allowed.add(tok)

    n = len(ds)
    empty_text = 0
    short_text = 0
    bad_audio = 0

    sample_rates = []
    durations = []
    text_lens = []
    cps = []  # chars per second

    oov_char_counter = Counter()
    oov_sample_ids = []

    text_counter = Counter()

    for i, row in enumerate(ds):
        txt = row[text_col]
        if txt is None:
            txt = ""
        txt = str(txt)

        # Для CTC обычно пробел -> "|"
        txt_for_vocab = txt.replace(" ", "|")

        text_counter[txt] += 1
        if len(txt.strip()) == 0:
            empty_text += 1
        if 0 < len(txt.strip()) <= 2:
            short_text += 1

        # OOV chars
        bad_chars = [ch for ch in txt_for_vocab if ch not in allowed]
        if bad_chars:
            for ch in bad_chars:
                oov_char_counter[ch] += 1
            if len(oov_sample_ids) < args.max_print:
                oov_sample_ids.append(i)

        # Audio stats
        try:
            audio = row["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            dur = len(arr) / sr if sr > 0 else 0.0
        except Exception:
            bad_audio += 1
            continue

        sample_rates.append(sr)
        durations.append(dur)
        text_lens.append(len(txt))
        if dur > 0:
            cps.append(len(txt) / dur)

    durs = np.array(durations) if durations else np.array([0.0])
    cps_arr = np.array(cps) if cps else np.array([0.0])

    # Дубликаты текста
    dup_text_count = sum(1 for _, c in text_counter.items() if c > 1)
    dup_rows = sum(c - 1 for c in text_counter.values() if c > 1)

    # Выбросы по chars/sec
    cps_med = float(np.median(cps_arr))
    cps_p99 = float(np.percentile(cps_arr, 99))
    cps_p1 = float(np.percentile(cps_arr, 1))
    cps_out_low = int((cps_arr < cps_p1).sum())
    cps_out_high = int((cps_arr > cps_p99).sum())

    print("=== DATASET QC ===")
    print(f"rows: {n}")
    print(f"text column: {text_col}")
    print(f"bad audio rows: {bad_audio}")
    print(f"empty text rows: {empty_text} ({empty_text/n:.2%})")
    print(f"very short text rows (1-2 chars): {short_text} ({short_text/n:.2%})")

    print("\n--- Audio ---")
    sr_counts = Counter(sample_rates)
    print(f"sample rates: {dict(sr_counts)}")
    print(
        f"duration sec: min={durs.min():.2f} p50={np.percentile(durs,50):.2f} p95={np.percentile(durs,95):.2f} max={durs.max():.2f}"
    )

    print("\n--- Text ---")
    tl = np.array(text_lens) if text_lens else np.array([0])
    print(
        f"text len chars: min={tl.min()} p50={int(np.percentile(tl,50))} p95={int(np.percentile(tl,95))} max={tl.max()}"
    )
    print(f"duplicate text values: {dup_text_count}, duplicate rows total: {dup_rows}")

    print("\n--- Vocab / OOV ---")
    total_chars = sum(text_lens) if text_lens else 1
    oov_total = sum(oov_char_counter.values())
    print(f"OOV chars total: {oov_total} ({oov_total/total_chars:.4%})")
    if oov_char_counter:
        print("top OOV chars:", oov_char_counter.most_common(20))
        print("sample row ids with OOV:", oov_sample_ids)

    print("\n--- Alignment proxy (chars/sec) ---")
    print(f"chars/sec: p1={cps_p1:.2f} median={cps_med:.2f} p99={cps_p99:.2f}")
    print(f"outliers: low={cps_out_low}, high={cps_out_high}")


if __name__ == "__main__":
    main()
