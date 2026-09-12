"""Kokoro-82M, the default TTS backend (flow.md B4.5, decisions D-07 / D-37).

Apache-2.0, 82M parameters, roughly real time on this CPU, eight languages, no voice cloning.
One `KPipeline` per language is built on first use and kept for the life of the process; each
language gets one curated voice (D-37 — a voice picker is Phase 3).
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...config import Settings
from . import TTSError, normalise_lang, write_silence, write_wav

if TYPE_CHECKING:
    from kokoro import KPipeline

log = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"

VOICES: dict[str, tuple[str, str]] = {
    # ISO-639-1 -> (kokoro lang_code, curated voice)
    "en": ("a", "af_heart"),
    "es": ("e", "ef_dora"),
    "fr": ("f", "ff_siwis"),
    "hi": ("h", "hf_alpha"),
    "it": ("i", "if_sara"),
    "ja": ("j", "jf_alpha"),
    "pt": ("p", "pf_dora"),
    "zh": ("z", "zf_xiaobei"),
}

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
        """The curated voice used for `lang` (D-37)."""
        return self._resolve(lang)[1]

    def synthesize(self, text: str, lang: str, reference_wav: Path | None, out: Path) -> Path:
        """Write a 24 kHz mono WAV of `text` spoken in `lang` to `out` and return it.

        `reference_wav` is ignored: Kokoro does not clone voices. Empty or unspeakable text
        becomes a short silence so that one blank segment cannot fail a whole job.
        """
        lang_code, voice = self._resolve(lang)
        if reference_wav is not None:
            log.debug("kokoro does not clone; ignoring reference clip %s", reference_wav)
        if not text.strip():
            log.warning("empty text for %s: writing silence", Path(out).name)
            return write_silence(out)

        pipeline = self._pipeline(lang_code)
        try:
            results = pipeline(text, voice=voice)
            chunks = [_as_float32(r.audio) for r in results if r.audio is not None]
        except Exception as exc:
            raise TTSError(f"kokoro failed to speak {lang!r} text {text[:60]!r}: {exc}") from exc

        chunks = [chunk for chunk in chunks if chunk.size]
        if not chunks:
            log.warning("kokoro produced no audio for %r (%s): writing silence", text[:60], lang)
            return write_silence(out)
        return write_wav(np.concatenate(chunks), out)

    def _resolve(self, lang: str) -> tuple[str, str]:
        """`'es'` -> `('e', 'ef_dora')`, raising `TTSError` for a language Kokoro cannot speak."""
        entry = VOICES.get(normalise_lang(lang))
        if entry is None:
            raise TTSError(f"kokoro cannot speak {lang!r}; it speaks: {', '.join(sorted(VOICES))}")
        return entry

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


__all__ = ["REPO_ID", "VOICES", "KokoroBackend"]
