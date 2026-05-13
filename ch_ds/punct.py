import re

# repeated same mark (,, !! ?? ;;) or wrong dots (.. or ....+) — excludes valid ellipsis ...
PUNCT_REPEATED     = re.compile(r'([,!?;])\1+|(?<!\.)\.{2}(?!\.)|\.{4,}')
# space before punctuation; (?!\Z) skips trailing punct at end of string
PUNCT_SPACE_BEFORE   = re.compile(r'\s[,;.!?](?!\Z)')
# no space after punctuation; (?<=\S) skips punct at start of string (e.g. ?שלום in Hebrew)
# , ! ? : general non-space check; . : only flags when followed by a Hebrew letter
#   so URLs (www.66books.co.il), Latin abbreviations etc. are not flagged
PUNCT_NO_SPACE_AFTER = re.compile(
    r'(?<=\S)[,!?](?=[^\s\d"\')\].׳״…])'   # , ! ? not followed by space
    r'|(?<=\S)\.(?=[א-׿])'          # . followed by Hebrew letter
)
# mojibake: UTF-8 bytes read as Latin-1 — appears as â€ sequences
PUNCT_MOJIBAKE     = re.compile(r'â€|Ã[\x80-\xff]|\x00')
# entry with no Hebrew/Latin/digit characters at all
HEBREW_OR_ALNUM    = re.compile(r'[א-תװ-״\w]', re.UNICODE)
# multiple consecutive spaces
PUNCT_DOUBLE_SPACE = re.compile(r' {2,}')
# invisible characters: zero-width space/joiner/non-joiner, soft hyphen, BOM, word joiner
PUNCT_INVISIBLE    = re.compile(r'[​‌‍­﻿⁠]')
# double dash instead of em-dash
PUNCT_DOUBLE_DASH  = re.compile(r'--')
# colon not followed by space or digit (skips times like 12:30, URLs)
PUNCT_COLON_NO_SPACE = re.compile(r'(?<=\S):(?=[^\s\d/])')
# punctuation at start of string (excluding Hebrew-style ?/! at start)
PUNCT_LEADING      = re.compile(r'^[,;.]')


def _unbalanced_brackets(text: str) -> bool:
    return (
        text.count('(') != text.count(')')
        or text.count('[') != text.count(']')
        or text.count('"') % 2 != 0
    )


def _mixed_quotes(text: str) -> bool:
    return '"' in text and '״' in text


def check_punct(text: str) -> list[str]:
    """Return list of punctuation-related flags for the given text."""
    if not HEBREW_OR_ALNUM.search(text):
        return ['punct_only']

    flags = []
    if PUNCT_REPEATED.search(text):
        flags.append('punct_repeated')
    if PUNCT_SPACE_BEFORE.search(text):
        flags.append('punct_space_before')
    if PUNCT_NO_SPACE_AFTER.search(text):
        flags.append('punct_no_space_after')
    if PUNCT_MOJIBAKE.search(text):
        flags.append('punct_mojibake')
    if PUNCT_DOUBLE_SPACE.search(text):
        flags.append('punct_double_space')
    if PUNCT_INVISIBLE.search(text):
        flags.append('punct_invisible')
    if PUNCT_DOUBLE_DASH.search(text):
        flags.append('punct_double_dash')
    if PUNCT_COLON_NO_SPACE.search(text):
        flags.append('punct_colon_no_space')
    if PUNCT_LEADING.search(text):
        flags.append('punct_leading')
    if _unbalanced_brackets(text):
        flags.append('punct_unbalanced')
    if _mixed_quotes(text):
        flags.append('punct_mixed_quotes')
    return flags


def punct_detail(text: str, flag: str) -> str:
    """Return a short string describing where the punctuation issue was found."""
    if flag == 'punct_repeated':
        m = PUNCT_REPEATED.search(text)
        return f'match="{m.group()}"' if m else ''
    if flag == 'punct_no_space_after':
        m = PUNCT_NO_SPACE_AFTER.search(text)
        return f'match="{m.group()}"' if m else ''
    if flag == 'punct_mojibake':
        m = PUNCT_MOJIBAKE.search(text)
        return f'match="{m.group()}"' if m else ''
    if flag == 'punct_double_space':
        m = PUNCT_DOUBLE_SPACE.search(text)
        return f'at={m.start()}' if m else ''
    if flag == 'punct_invisible':
        m = PUNCT_INVISIBLE.search(text)
        return f'char=U+{ord(m.group()):04X} at={m.start()}' if m else ''
    if flag == 'punct_double_dash':
        m = PUNCT_DOUBLE_DASH.search(text)
        return f'at={m.start()}' if m else ''
    if flag == 'punct_colon_no_space':
        m = PUNCT_COLON_NO_SPACE.search(text)
        return f'match="{m.group()}"' if m else ''
    return ''
