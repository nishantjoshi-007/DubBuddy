"""Chatterbox Multilingual, the optional cloning backend (flow.md B4.5, decisions D-07 / D-27).

MIT, ~500M parameters, 23 languages, zero-shot cloning from `reference.wav` — and a GPU in
practice. It ships as the `clone` extra, so it is usually absent: nothing here is imported at
module scope, the class always constructs, and `installed()` / `reason()` let
`available_backends()` explain what to run.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ...config import Settings
from . import SAMPLE_RATE, TTSError, normalise_lang, write_silence, write_wav

log = logging.getLogger(__name__)

NOT_INSTALLED_REASON = "not installed: run `uv sync --extra clone`"

LANGUAGES: frozenset[str] = frozenset(
    {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    }
)  # fmt: skip


def _model_class() -> Any:
    """Import `ChatterboxMultilingualTTS` lazily. Raises ImportError when the extra is absent."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS


def _import_problem() -> str | None:
    """None when chatterbox can be used, otherwise a one-line reason for the UI."""
    if importlib.util.find_spec("chatterbox") is None:
        return NOT_INSTALLED_REASON
    try:
        _model_class()
    except Exception as exc:  # pragma: no cover - needs a broken `clone` extra
        return f"{NOT_INSTALLED_REASON} (the package is present but `chatterbox.mtl_tts` failed: {exc})"
    return None


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Numpy fallback when torchaudio is unavailable: linear interpolation onto the new grid."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    count = max(1, int(round(audio.size * dst_rate / float(src_rate))))
    src_times = np.arange(audio.size, dtype=np.float64) / float(src_rate)
    dst_times = np.arange(count, dtype=np.float64) / float(dst_rate)
    return np.interp(dst_times, src_times, audio).astype(np.float32)


def _to_mono_24k(wav: Any, src_rate: int) -> np.ndarray:
    """Model output (torch tensor or array, any rate) -> flat float32 mono at 24 kHz."""
    rate = src_rate
    if hasattr(wav, "detach"):
        tensor = wav.detach().cpu()
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=0)
        if rate != SAMPLE_RATE:
            try:
                import torchaudio

                tensor = torchaudio.functional.resample(tensor, rate, SAMPLE_RATE)
                rate = SAMPLE_RATE
            except Exception as exc:  # pragma: no cover - torchaudio ships with chatterbox
                log.debug("torchaudio resample unavailable (%s); using the numpy fallback", exc)
        audio = np.asarray(tensor.numpy(), dtype=np.float32)
    else:
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
    return _resample_linear(audio.reshape(-1), rate, SAMPLE_RATE)


class ChatterboxBackend:
    """`TTSBackend` (respeak/pipeline/tts/base.py) on top of `ChatterboxMultilingualTTS`."""

    name: str = "chatterbox"
    cloning: bool = True

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()

    def languages(self) -> set[str]:
        """ISO-639-1 codes this backend can speak."""
        return set(LANGUAGES)

    @staticmethod
    def installed() -> bool:
        """True only when the `clone` extra is present and importable."""
        return _import_problem() is None

    @staticmethod
    def reason() -> str | None:
        """Why the backend is unusable, or None when it is fine."""
        return _import_problem()

    def synthesize(self, text: str, lang: str, reference_wav: Path | None, out: Path) -> Path:
        """Write a 24 kHz mono WAV of `text` spoken in `lang` to `out` and return it.

        When `reference_wav` is given the model clones that speaker; otherwise it uses its own
        default voice. Empty text becomes a short silence, as with every backend.
        """
        code = normalise_lang(lang)
        if code not in LANGUAGES:
            raise TTSError(f"chatterbox cannot speak {lang!r}; it speaks: {', '.join(sorted(LANGUAGES))}")
        if not text.strip():
            log.warning("empty text for %s: writing silence", Path(out).name)
            return write_silence(out)
        prompt: str | None = None
        if reference_wav is not None:
            if not Path(reference_wav).exists():
                raise TTSError(f"reference clip {reference_wav} does not exist")
            prompt = str(reference_wav)

        model = self._load()
        try:
            wav = model.generate(text, language_id=code, audio_prompt_path=prompt)
        except Exception as exc:
            raise TTSError(f"chatterbox failed to speak {lang!r} text {text[:60]!r}: {exc}") from exc

        audio = _to_mono_24k(wav, int(getattr(model, "sr", SAMPLE_RATE)))
        if audio.size == 0:
            log.warning("chatterbox produced no audio for %r (%s): writing silence", text[:60], lang)
            return write_silence(out)
        return write_wav(audio, out)

    def _load(self) -> Any:
        """The multilingual model, loaded once per process on `Settings.resolved_device()`."""
        with self._lock:
            if self._model is not None:
                return self._model
            problem = _import_problem()
            if problem is not None:
                raise TTSError(f"tts backend 'chatterbox' is unavailable: {problem}")
            device = self._settings.resolved_device()
            log.info("loading chatterbox multilingual model on %s", device)
            try:
                self._model = _model_class().from_pretrained(device=device)
            except Exception as exc:  # pragma: no cover - needs the `clone` extra
                raise TTSError(f"chatterbox failed to load its model on {device}: {exc}") from exc
            return self._model


__all__ = ["LANGUAGES", "NOT_INSTALLED_REASON", "ChatterboxBackend"]
