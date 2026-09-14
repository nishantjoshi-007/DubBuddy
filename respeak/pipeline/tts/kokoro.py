"""Kokoro-82M, the default TTS backend (flow.md B4.5, decisions D-07 / D-37).

Apache-2.0, 82M parameters, roughly real time on this CPU, eight languages, no voice cloning.
One `KPipeline` per kokoro lang_code is built on first use and kept for the life of the process.
Every voice the repo bundles is offered (plan.md 3.3); `VOICES` still names the one that is used
when a job does not pick one (D-37).
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...config import Settings
from ..types import Voice
from . import TTSError, normalise_lang, write_silence, write_wav

if TYPE_CHECKING:
    from kokoro import KPipeline

log = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"

VOICES: dict[str, tuple[str, str]] = {
    # ISO-639-1 -> (kokoro lang_code, curated default voice)
    "en": ("a", "af_heart"),
    "es": ("e", "ef_dora"),
    "fr": ("f", "ff_siwis"),
    "hi": ("h", "hf_alpha"),
    "it": ("i", "if_sara"),
    "ja": ("j", "jf_alpha"),
    "pt": ("p", "pf_dora"),
    "zh": ("z", "zf_xiaobei"),
}

#: Every voice tensor in `hexgrad/Kokoro-82M/voices/`, grouped by the language it speaks, the
#: curated default (`VOICES`) first. The first letter of an id *is* its kokoro lang_code — English
#: therefore has two, `a` (American) and `b` (British) — and the second is `f` or `m`. The list is
#: static so that `/api/backends` never touches the network; `tests/test_tts.py` checks it against
#: the real repo whenever the snapshot is in the huggingface cache.
VOICE_IDS: dict[str, tuple[str, ...]] = {
    "en": (
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
        "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
        "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ),
    "es": ("ef_dora", "em_alex", "em_santa"),
    "fr": ("ff_siwis",),
    "hi": ("hf_alpha", "hf_beta", "hm_omega", "hm_psi"),
    "it": ("if_sara", "im_nicola"),
    "ja": ("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"),
    "pt": ("pf_dora", "pm_alex", "pm_santa"),
    "zh": (
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ),
}  # fmt: skip

#: The two English lang_codes are the only ones a listener can tell apart by name alone.
_REGIONS: dict[str, str] = {"a": "US", "b": "UK"}
_GENDERS: dict[str, str] = {"f": "female", "m": "male"}


def display_name(voice_id: str) -> str:
    """``'af_heart'`` -> ``'Heart (female, US)'``, ``'zm_yunjian'`` -> ``'Yunjian (male)'``."""
    _, _, given = voice_id.partition("_")
    gender = _GENDERS.get(voice_id[1:2], "")
    region = _REGIONS.get(voice_id[:1], "")
    inside = ", ".join(part for part in (gender, region) if part)
    name = (given or voice_id).replace("_", " ").capitalize()
    return f"{name} ({inside})" if inside else name


VOICES_BY_LANG: dict[str, list[Voice]] = {
    lang: [Voice(id=voice_id, name=display_name(voice_id)) for voice_id in ids]
    for lang, ids in VOICE_IDS.items()
}
"""What `voices()` returns and `/api/backends` publishes: `{lang: [Voice(id, name), ...]}`."""

# Japanese G2P goes through fugashi/MeCab, which needs the unidic dictionary downloaded once.
_DICTIONARY_MARKERS = ("mecab", "unidic", "dicdir", "fugashi", "dictionary")

_UNIDIC_HINT = (
    "kokoro cannot speak Japanese until the unidic dictionary is downloaded: run "
    "`uv run python -m unidic download` once (about 250 MB), then retry"
)


def _as_float32(audio: Any) -> np.ndarray:
    """Torch tensor or array -> a flat float32 numpy array."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class KokoroBackend:
    """`TTSBackend` (respeak/pipeline/tts/base.py) on top of `kokoro.KPipeline`."""

    name: str = "kokoro"
    cloning: bool = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipelines: dict[str, KPipeline] = {}
        self._lock = threading.Lock()

    def languages(self) -> set[str]:
        """ISO-639-1 codes this backend can speak."""
        return set(VOICES)

    def voices(self) -> dict[str, list[Voice]]:
        """Every bundled voice per language, the curated default first (plan.md 3.3).

        A fresh list of the shared `Voice` objects: a caller that sorts or filters its answer must
        not be able to reorder the table every later caller reads.
        """
        return {lang: list(entries) for lang, entries in VOICES_BY_LANG.items()}

    @staticmethod
    def installed() -> bool:
        """Kokoro is a hard dependency, so this is True in any healthy install."""
        return importlib.util.find_spec("kokoro") is not None

    @staticmethod
    def reason() -> str | None:
        """Why the backend is unusable, or None when it is fine."""
        if KokoroBackend.installed():
            return None
        return "not installed: the `kokoro` package is missing; run `uv sync`"

    def voice_for(self, lang: str) -> str:
        """The curated voice used for `lang` when a job picks none (D-37)."""
        return self.resolve_voice(lang)[1]

    def resolve_voice(self, lang: str, voice: str | None = None) -> tuple[str, str]:
        """`('es', None)` -> `('e', 'ef_dora')`; `('en', 'bm_george')` -> `('b', 'bm_george')`.

        Raises `TTSError` for a language Kokoro cannot speak and for a voice that is not one of
        that language's — a voice id carries its own lang_code as its first letter, so accepting
        one from another language would silently synthesise in the wrong one.
        """
        code = normalise_lang(lang)
        if code not in VOICES:
            raise TTSError(f"kokoro cannot speak {lang!r}; it speaks: {', '.join(sorted(VOICES))}")
        wanted = (voice or "").strip()
        if not wanted:
            default = VOICES[code][1]
            return default[0], default
        if wanted not in VOICE_IDS[code]:
            known = ", ".join(VOICE_IDS[code])
            raise TTSError(f"kokoro has no voice {wanted!r} for {code!r}; its voices are: {known}")
        return wanted[0], wanted

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
        """Write a 24 kHz mono WAV of `text` spoken in `lang` to `out` and return it.

        `voice` is one of this language's ids from `voices()`, or None for the curated default.
        `reference_wav` is ignored: Kokoro does not clone voices. Empty or unspeakable text
        becomes a short silence so that one blank segment cannot fail a whole job.
        """
        lang_code, chosen = self.resolve_voice(lang, voice)
        if reference_wav is not None:
            log.debug("kokoro does not clone; ignoring reference clip %s", reference_wav)
        if not text.strip():
            log.warning("empty text for %s: writing silence", Path(out).name)
            return write_silence(out)

        pipeline = self._pipeline(lang_code)
        try:
            results = pipeline(text, voice=chosen)
            chunks = [_as_float32(r.audio) for r in results if r.audio is not None]
        except Exception as exc:
            raise TTSError(
                f"kokoro failed to speak {lang!r} text {text[:60]!r} as {chosen!r}: {exc}"
            ) from exc

        chunks = [chunk for chunk in chunks if chunk.size]
        if not chunks:
            log.warning("kokoro produced no audio for %r (%s): writing silence", text[:60], lang)
            return write_silence(out)
        return write_wav(np.concatenate(chunks), out)

    def _pipeline(self, lang_code: str) -> KPipeline:
        """The cached `KPipeline` for one kokoro lang_code, built on first use."""
        with self._lock:
            pipeline = self._pipelines.get(lang_code)
            if pipeline is not None:
                return pipeline
            try:
                from kokoro import KPipeline as _KPipeline
            except ImportError as exc:  # pragma: no cover - kokoro is a hard dependency
                raise TTSError(f"the `kokoro` package is not importable: {exc}") from exc

            device = self._settings.resolved_device()
            log.info("loading kokoro pipeline lang_code=%s device=%s", lang_code, device)
            try:
                pipeline = _KPipeline(lang_code=lang_code, repo_id=REPO_ID, device=device)
            except Exception as exc:
                raise self._load_error(lang_code, exc) from exc
            self._pipelines[lang_code] = pipeline
            return pipeline

    @staticmethod
    def _load_error(lang_code: str, exc: Exception) -> TTSError:
        """Turn a KPipeline construction failure into an error the user can act on."""
        message = str(exc).lower()
        if lang_code == "j" and any(marker in message for marker in _DICTIONARY_MARKERS):
            return TTSError(f"{_UNIDIC_HINT} (original error: {exc})")
        return TTSError(f"kokoro failed to load its {lang_code!r} pipeline: {exc}")


__all__ = ["REPO_ID", "VOICES", "VOICES_BY_LANG", "VOICE_IDS", "KokoroBackend", "display_name"]
