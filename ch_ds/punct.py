import re

# repeated same mark (,, !! ?? ;;) or wrong dots (.. or ....+) — excludes valid ellipsis ...
PUNCT_REPEATED     = re.compile(r'([,!?;])\1+|\.{2}(?!\.)|\.{4,}')
# space before punctuation; (?!\Z) skips trailing punct at end of string
PUNCT_SPACE_BEFORE   = re.compile(r'\s[,;.!?](?!\Z)')
# no space after punctuation; (?<=\S) skips punct at start of string (e.g. ?שלום in Hebrew)
# excludes geresh ׳ / gershayim ״ so Hebrew abbreviations like ד״ר are not flagged
PUNCT_NO_SPACE_AFTER = re.compile(r'(?<=\S)[,.!?](?=[^\s\d"\')\].׳״…])')
# mojibake: UTF-8 bytes read as Latin-1 — appears as â€ sequences
PUNCT_MOJIBAKE     = re.compile(r'â€|Ã[\x80-\xff]|\x00')
# entry with no Hebrew/Latin/digit characters at all
HEBREW_OR_ALNUM    = re.compile(r'[א-תװ-״\w]', re.UNICODE)


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
    return ''
