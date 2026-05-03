DATA_DIR             = "./dataset"
LOG_PATH             = "./ch_ds.log"

# --- output paths (each script writes <BASE>_{all,passed,flagged}.csv) ---
TXT_EXPORT_BASE      = "./results/txt"
AUDIO_EXPORT_BASE    = "./results/audio"
LANG_EXPORT_BASE     = "./results/lang"
WX_EXPORT_BASE       = "./results/wx"
ALL_EXPORT_BASE      = "./results/all"
MAX_PRINT            = 10            # max sample entries shown per flag type

# --- text checks ---
TOO_SHORT_LEN        = 2             # sentence ≤ this char count → too_short

# --- audio checks ---
AUDIO_MIN_DURATION    = 0.5    # seconds: clip shorter than this → too_short
AUDIO_MAX_DURATION    = 30.0   # seconds: clip longer than this → too_long
AUDIO_MIN_DBFS        = -50.0  # dBFS: quieter → too_quiet
AUDIO_CLIPPING_DBFS   = 0.0    # dBFS: louder → clipping
AUDIO_SILENCE_THRESH  = -50.0  # dBFS: threshold for silent chunk detection
AUDIO_SILENCE_MAX     = 0.8    # fraction of silent chunks above which → mostly_silent
AUDIO_CHUNK_MS        = 100    # ms: chunk size for silence analysis

AUDIO_FLAG_ORDER = [
    'unreadable',
    'too_short',
    'too_long',
    'too_quiet',
    'clipping',
    'mostly_silent',
]

# --- language detection ---
LANG_MODEL        = "medium"
LANG_DEVICE       = None        # None = auto
LANG_COMPUTE_TYPE = "float16"
LANG_WINDOW_S     = 30          # analysis window in seconds
LANG_STRIDE_S     = 15          # stride (< window = overlap)
LANG_MIN_PROB     = 0.7         # confidence below this → skip window
LANG_EXPECTED     = "he"

LANG_FLAG_ORDER = [
    'foreign_segment',
    'low_confidence',
    'unreadable',
]

# --- whisperx check ---
WX_MODEL          = "large-v3"
WX_DEVICE         = None        # None = auto
WX_COMPUTE_TYPE   = "float16"
WX_LANGUAGE       = "he"
WX_BATCH_SIZE     = 8
WX_LIMIT          = 200         # max entries to check (None = all)
WX_WER_THRESHOLD  = 0.5
WX_ALIGN          = True

WX_FLAG_ORDER = [
    'wrong_language',
    'high_wer',
    'empty_transcription',
]

# --- all.py ---
ALL_RUN_LANG        = True
# text flags that always reject an entry regardless of ratio
ALL_TXT_FATAL_FLAGS = {'cyrillic', 'html_tag', 'empty'}

FLAG_ORDER = [
    'cyrillic',
    'latin',
    'html_tag',
    'empty',
    'too_short',
    'punct_only',
    'punct_repeated',
    'punct_space_before',
    'punct_no_space_after',
    'punct_mojibake',
]
