DATA_DIR             = "./row_data"
EXPORT_PATH          = None          # путь для CSV с проблемными записями, None — не экспортировать
LOG_PATH             = "./ch_ds.log" # путь к файлу логов
MAX_PRINT            = 10            # сколько примеров показывать на каждый тип флага

CPS_LOW              = 2             # минимум символов/сек (ниже — подозрительно мало текста)
CPS_HIGH             = 35            # максимум символов/сек (выше — текст не вмещается в аудио)
TOO_SHORT_LEN        = 2             # текст ≤ этого числа символов считается слишком коротким
EXCEEDS_AUDIO_SLACK  = 1.0           # допуск в секундах: SRT-конец может выходить за длину mp3

# --- audio checks ---
AUDIO_MIN_DURATION    = 1.0    # секунд: файл короче — слишком короткий
AUDIO_MAX_DURATION    = 600.0  # секунд: файл длиннее — подозрительно
AUDIO_MIN_DBFS        = -50.0  # dBFS: тише — слишком тихий сигнал
AUDIO_CLIPPING_DBFS   = -1.0   # dBFS: громче — клиппинг
AUDIO_SILENCE_THRESH  = -50.0  # dBFS: порог тишины для подсчёта silent-фреймов
AUDIO_SILENCE_MAX     = 0.8    # доля тихих фреймов выше которой — проблема
AUDIO_CHUNK_MS        = 100    # мс: размер чанка для анализа тишины

AUDIO_FLAG_ORDER = [
    'unreadable',
    'too_short',
    'too_long',
    'too_quiet',
    'clipping',
    'mostly_silent',
]

# --- language detection ---
LANG_MODEL        = "medium"    # меньше модель = быстрее, для детекции языка достаточно
LANG_DEVICE       = None        # None = auto
LANG_COMPUTE_TYPE = "float16"   # float16 / int8
LANG_WINDOW_S     = 30          # размер окна анализа в секундах
LANG_STRIDE_S     = 15          # шаг окна (< LANG_WINDOW_S = перекрытие, ловим короткие вкрапления)
LANG_MIN_PROB     = 0.7         # уверенность ниже — детекция ненадёжна, пропускаем окно
LANG_EXPECTED     = "he"        # ожидаемый язык

LANG_FLAG_ORDER = [
    'foreign_segment',
    'low_confidence',
    'unreadable',
]

# --- whisperx check ---
WX_MODEL          = "large-v3"  # faster-whisper model size
WX_DEVICE         = None        # None = auto (cuda если доступна)
WX_COMPUTE_TYPE   = "float16"   # float16 / int8
WX_LANGUAGE       = "he"
WX_BATCH_SIZE     = 8
WX_LIMIT          = 200         # сколько файлов проверять (None = все)
WX_WER_THRESHOLD  = 0.5         # WER выше — файл проблемный
WX_ALIGN          = True        # запускать alignment для word-level scores

WX_FLAG_ORDER = [
    'wrong_language',
    'high_wer',
    'empty_transcription',
]

# --- all.py ---
ALL_OUTPUT_PATH        = "./good_files.csv"  # куда писать прошедшие все проверки
ALL_RUN_LANG           = True                # запускать детекцию языка (медленно, нужен GPU)
ALL_TXT_MAX_FLAG_RATIO = 0.05                # макс. доля проблемных SRT-записей (5%)
# флаги txt которые делают файл плохим независимо от их доли
ALL_TXT_FATAL_FLAGS    = {'cyrillic', 'html_tag', 'bad_timing', 'empty'}

FLAG_ORDER = [
    'cyrillic',
    'latin',
    'html_tag',
    'repeated_word',
    'empty',
    'too_short',
    'bad_timing',
    'exceeds_audio',
    'cps_low',
    'cps_high',
    'punct_only',
    'punct_repeated',
    'punct_space_before',
    'punct_no_space_after',
    'punct_mojibake',
]
