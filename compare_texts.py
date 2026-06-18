#!/usr/bin/env python3
"""Compare two plain-text transcription files line-by-line.

Each file must have one sentence per line. Lines are matched by position.
No third-party dependencies — uses stdlib difflib only.

Usage:
    python compare_texts.py file_a.txt file_b.txt [--label-a A] [--label-b B] [--top N]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from difflib import SequenceMatcher

_USE_COLOR = sys.stdout.isatty()
_RED = "\033[31m" if _USE_COLOR else ""
_GRN = "\033[32m" if _USE_COLOR else ""
_YLW = "\033[33m" if _USE_COLOR else ""
_RST = "\033[0m"  if _USE_COLOR else ""


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _align(ref_words: list[str], hyp_words: list[str]):
    """Yield (tag, ref_slice, hyp_slice) for each opcode block."""
    sm = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        yield tag, ref_words[i1:i2], hyp_words[j1:j2]


def _wer_counts(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int, int]:
    """Return (hits, substitutions, deletions, insertions)."""
    hits = subs = dels = ins = 0
    for tag, rw, hw in _align(ref_words, hyp_words):
        if tag == "equal":
            hits += len(rw)
        elif tag == "replace":
            n = max(len(rw), len(hw))
            subs += min(len(rw), len(hw))
            if len(rw) > len(hw):
                dels += len(rw) - len(hw)
            else:
                ins += len(hw) - len(rw)
        elif tag == "delete":
            dels += len(rw)
        elif tag == "insert":
            ins += len(hw)
    return hits, subs, dels, ins


def word_diff(ref: str, hyp: str) -> tuple[str, str]:
    """Return colour-highlighted versions of ref and hyp."""
    ref_words, hyp_words = ref.split(), hyp.split()
    ref_out, hyp_out = [], []
    for tag, rw, hw in _align(ref_words, hyp_words):
        if tag == "equal":
            ref_out.extend(rw)
            hyp_out.extend(hw)
        elif tag == "replace":
            ref_out.extend(f"{_YLW}{w}{_RST}" for w in rw)
            hyp_out.extend(f"{_YLW}{w}{_RST}" for w in hw)
        elif tag == "delete":
            ref_out.extend(f"{_RED}{w}{_RST}" for w in rw)
        elif tag == "insert":
            hyp_out.extend(f"{_GRN}{w}{_RST}" for w in hw)
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

    total_hits = total_sub = total_del = total_ins = 0
    subs: Counter = Counter()
    dels: Counter = Counter()
    ins:  Counter = Counter()
    per_sample: list[tuple[float, int, str, str]] = []

    for i, (a, b) in enumerate(pairs):
        aw, bw = a.split(), b.split()
        h, s, d, n = _wer_counts(aw, bw)
        total_hits += h
        total_sub  += s
        total_del  += d
        total_ins  += n

        for tag, rw, hw in _align(aw, bw):
            if tag == "replace":
                subs[(" ".join(rw), " ".join(hw))] += 1
            elif tag == "delete":
                for w in rw:
                    dels[w] += 1
            elif tag == "insert":
                for w in hw:
                    ins[w] += 1

        denom = h + s + d
        wer = (s + d + n) / denom if denom else 0.0
        per_sample.append((wer, i, a, b))

    total_err = total_sub + total_del + total_ins
    total_ref = total_hits + total_sub + total_del
    overall_wer = total_err / total_ref if total_ref else 0.0

    print("=" * 70)
    print(f"  {label_a}  vs  {label_b}")
    print(f"  Lines compared : {len(pairs)}")
    print(f"  Overall WER    : {overall_wer:.4f}  ({overall_wer * 100:.2f}%)")
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
