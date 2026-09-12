"""Language names and ISO-639-1 codes.

SOURCE_LANGUAGES: what the UI offers as "language of the video" (Whisper + Argos both cover these).
Target languages are not listed here: they come from the selected TTS backend at runtime.
"""

NAME_TO_CODE: dict[str, str] = {
    "Arabic": "ar",
    "Azerbaijani": "az",
    "Catalan": "ca",
    "Chinese": "zh",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Esperanto": "eo",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Slovak": "sk",
    "Spanish": "es",
    "Swedish": "sv",
    "Turkish": "tr",
    "Ukrainian": "uk",
}

CODE_TO_NAME: dict[str, str] = {code: name for name, code in NAME_TO_CODE.items()}

SOURCE_LANGUAGES: list[dict[str, str]] = [{"code": c, "name": n} for n, c in NAME_TO_CODE.items()]

# ISO-639-2 codes ffmpeg wants for subtitle stream metadata
ISO639_2: dict[str, str] = {
    "ar": "ara",
    "az": "aze",
    "ca": "cat",
    "zh": "zho",
    "cs": "ces",
    "da": "dan",
    "nl": "nld",
    "en": "eng",
    "eo": "epo",
    "fi": "fin",
    "fr": "fra",
    "de": "deu",
    "el": "ell",
    "he": "heb",
    "hi": "hin",
    "hu": "hun",
    "id": "ind",
    "ga": "gle",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "fa": "fas",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "sk": "slk",
    "es": "spa",
    "sv": "swe",
    "tr": "tur",
    "uk": "ukr",
    "ms": "msa",
    "no": "nor",
    "sw": "swa",
}
