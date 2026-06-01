#!/usr/bin/env python3
"""Compare two plain-text transcription files line-by-line.

Each file must have one sentence per line. Lines are matched by position.

Usage:
    python compare_texts.py file_a.txt file_b.txt [--label-a A] [--label-b B] [--top N]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from jiwer import process_words

_USE_COLOR = sys.stdout.isatty()
_RED = "\033[31m" if _USE_COLOR else ""
_GRN = "\033[32m" if _USE_COLOR else ""
_YLW = "\033[33m" if _USE_COLOR else ""
_RST = "\033[0m"  if _USE_COLOR else ""


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def word_diff(ref: str, hyp: str) -> tuple[str, str]:
    """Return coloured versions of ref and hyp highlighting word differences."""
    try:
        r = process_words(ref, hyp)
    except Exception:
        return ref, hyp

    ref_words, hyp_words = ref.split(), hyp.split()
    ref_out, hyp_out = [], []
    for alignment in r.alignments:
        for chunk in alignment:
            rw = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            hw = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            if chunk.type == "equal":
                ref_out.append(rw)
                hyp_out.append(hw)
            elif chunk.type == "substitute":
                ref_out.append(f"{_YLW}{rw}{_RST}")
                hyp_out.append(f"{_YLW}{hw}{_RST}")
            elif chunk.type == "delete":
                ref_out.append(f"{_RED}{rw}{_RST}")
            elif chunk.type == "insert":
                hyp_out.append(f"{_GRN}{hw}{_RST}")
    return " ".join(ref_out), " ".join(hyp_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two transcription text files.")
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--top", type=int, default=10, help="Worst N lines to print (default 10)")
    args = parser.parse_args()

    label_a = args.label_a or args.file_a
    label_b = args.label_b or args.file_b

    lines_a = load_lines(args.file_a)
    lines_b = load_lines(args.file_b)

    if len(lines_a) != len(lines_b):
        print(f"Warning: {len(lines_a)} vs {len(lines_b)} lines — truncating to shorter.",
              file=sys.stderr)

    pairs = list(zip(lines_a, lines_b))

    total_sub = total_del = total_ins = total_hits = 0
    subs: Counter = Counter()
    dels: Counter = Counter()
    ins:  Counter = Counter()
    per_sample: list[tuple[float, int, str, str]] = []

    for i, (a, b) in enumerate(pairs):
        try:
            r = process_words(a, b)
        except Exception:
            continue
        total_sub  += r.substitutions
        total_del  += r.deletions
        total_ins  += r.insertions
        total_hits += r.hits

        aw, bw = a.split(), b.split()
        for alignment in r.alignments:
            for chunk in alignment:
                rw = " ".join(aw[chunk.ref_start_idx:chunk.ref_end_idx])
                hw = " ".join(bw[chunk.hyp_start_idx:chunk.hyp_end_idx])
                if chunk.type == "substitute":
                    subs[(rw, hw)] += 1
                elif chunk.type == "delete":
                    for w in aw[chunk.ref_start_idx:chunk.ref_end_idx]:
                        dels[w] += 1
                elif chunk.type == "insert":
                    for w in bw[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                        ins[w] += 1

        per_sample.append((r.wer, i, a, b))

    total_err = total_sub + total_del + total_ins
    total_ref = total_hits + total_sub + total_del
    wer = total_err / total_ref if total_ref else 0.0

    print("=" * 70)
    print(f"  {label_a}  vs  {label_b}")
    print(f"  Lines compared : {len(pairs)}")
    print(f"  Overall WER    : {wer:.4f}  ({wer * 100:.2f}%)")
    print("=" * 70)

    if total_err:
        print(f"\n--- Error breakdown ({total_err} total) ---")
        print(f"  Substitutions : {total_sub:5d}  ({100 * total_sub / total_err:.1f}%)")
        print(f"  Deletions     : {total_del:5d}  ({100 * total_del / total_err:.1f}%)")
        print(f"  Insertions    : {total_ins:5d}  ({100 * total_ins / total_err:.1f}%)")

    if subs:
        print(f"\n--- Top 20 substitutions ({label_a} → {label_b}) ---")
        for (rw, hw), cnt in subs.most_common(20):
            print(f"  {cnt:4d}x  {rw!r:30s} → {hw!r}")

    if dels:
        print(f"\n--- Top 20 words only in {label_a} ---")
        for w, cnt in dels.most_common(20):
            print(f"  {cnt:4d}x  {w!r}")

    if ins:
        print(f"\n--- Top 20 words only in {label_b} ---")
        for w, cnt in ins.most_common(20):
            print(f"  {cnt:4d}x  {w!r}")

    worst = sorted(per_sample, reverse=True)[:args.top]
    if worst:
        print(f"\n--- Top {args.top} most-different lines ---")
        for wer_val, idx, a, b in worst:
            a_col, b_col = word_diff(a, b)
            print(f"\n  Line {idx + 1}  WER={wer_val:.3f}")
            print(f"  {label_a}: {a_col}")
            print(f"  {label_b}: {b_col}")

    print()


if __name__ == "__main__":
    main()
