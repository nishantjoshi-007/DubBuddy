"""TTS backend registry (docs/flow.md B4.5).

`available_backends()` answers `GET /api/backends`; `get_backend()` hands the speak stage a
ready backend chosen by `TTS_BACKEND` or the per-job override. The backend modules are
imported lazily, so importing this package stays cheap for the web process — nothing here pulls
torch or a model into memory.

This module also owns the small pieces every backend shares: `TTSError`, the 24 kHz output
contract and the WAV writer.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from ...config import Settings
from ..types import BackendInfo, Voice
from .base import TTSBackend

if TYPE_CHECKING:  # concrete types for the registry only; never imported at runtime from here
    from .chatterbox import ChatterboxBackend
    from .kokoro import KokoroBackend

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
"""Every backend writes mono WAV at this rate; B4.6 (fit) and B4.8 (mux) assume it."""

SILENCE_SECONDS = 0.2
"""What an empty segment becomes, so one blank line never fails a job."""

BACKEND_NAMES: tuple[str, ...] = ("kokoro", "chatterbox")


class TTSError(RuntimeError):
    """A TTS backend could not do what was asked.

    Unknown backend name, missing optional extra, unsupported language, model or synthesis
    failure. Backends raise this instead of returning None (decisions: "silent None").
    """


def normalise_lang(lang: str) -> str:
    """`'ZH-CN'` -> `'zh'`. Backends speak ISO-639-1; regional suffixes are dropped."""
    return lang.strip().lower().replace("_", "-").split("-")[0]


def write_wav(samples: Any, out: Path) -> Path:
    """Write float samples as a 24 kHz mono 16-bit WAV at `out` and return `out`."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise TTSError(f"refusing to write an empty wav to {out}")
    if not np.all(np.isfinite(audio)):
        raise TTSError(f"synthesized audio for {out} contains NaN or infinity")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), pcm, SAMPLE_RATE, subtype="PCM_16")
    return destination


def write_silence(out: Path, seconds: float = SILENCE_SECONDS) -> Path:
    """Write `seconds` of digital silence at `out`. Used for empty or unspeakable text."""
    return write_wav(np.zeros(max(1, int(SAMPLE_RATE * seconds)), dtype=np.float32), out)


_lock = threading.Lock()
_cache: dict[str, tuple[Settings, TTSBackend]] = {}


def _construct(name: str, settings: Settings) -> KokoroBackend | ChatterboxBackend:
    """Build a backend object. Cheap: no model is loaded and no optional import is forced."""
    if name == "kokoro":
        from .kokoro import KokoroBackend as _Kokoro

        return _Kokoro(settings)
    if name == "chatterbox":
        from .chatterbox import ChatterboxBackend as _Chatterbox

        return _Chatterbox(settings)
    raise TTSError(f"unknown backend {name!r}; known backends: {', '.join(BACKEND_NAMES)}")


def _voices_of(backend: TTSBackend) -> dict[str, list[Voice]]:
    """`backend.voices()`, or `{}` when that backend cannot answer.

    `available_backends()` is behind a route the page loads on every visit, and a backend that
    offers no voices is an ordinary state (Chatterbox clones instead). One backend whose voice
    table is broken therefore degrades to "no preset voices" and a loud log line, rather than
    taking `GET /api/backends` — and with it the whole form — down with it.
    """
    try:
        return backend.voices()
    except Exception:
        log.exception("backend %r could not list its voices; offering none", getattr(backend, "name", "?"))
        return {}


def available_backends(settings: Settings) -> dict[str, BackendInfo]:
    """Describe every backend Respeak knows about, installed or not (docs/flow.md B4.5).

    `installed` is what the UI gates the backend selector on; `reason` says what to run when a
    backend is missing. `languages` is always the full set the backend speaks, so the UI can tell
    a user which languages an extra would unlock. `voices` feeds the voice picker: `{lang: [Voice]}`,
    empty for a backend that clones the original speaker.
    """
    infos: dict[str, BackendInfo] = {}
    for name in BACKEND_NAMES:
        backend = _construct(name, settings)
        installed = backend.installed()
        infos[name] = BackendInfo(
            name=backend.name,
            installed=installed,
            languages=backend.languages(),
            cloning=backend.cloning,
            reason=None if installed else backend.reason(),
            voices=_voices_of(backend),
        )
    return infos


def get_backend(name: str, settings: Settings) -> TTSBackend:
    """Return the installed backend called `name`, cached for the life of the process.

    Raises `TTSError` for an unknown name or for a backend whose package is missing.
    """
    key = (name or "").strip().lower()
    if key not in BACKEND_NAMES:
        raise TTSError(f"unknown backend {name!r}; known backends: {', '.join(BACKEND_NAMES)}")
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] is settings:
            return cached[1]
        backend = _construct(key, settings)
        if not backend.installed():
            raise TTSError(f"tts backend {key!r} is unavailable: {backend.reason()}")
        log.debug("tts backend %r ready (cloning=%s)", key, backend.cloning)
        _cache[key] = (settings, backend)
        return backend


def clear_backend_cache() -> None:
    """Drop the cached backends (and the models they hold). For tests and shutdown."""
    with _lock:
        _cache.clear()


__all__ = [
    "BACKEND_NAMES",
    "SAMPLE_RATE",
    "SILENCE_SECONDS",
    "TTSBackend",
    "TTSError",
    "available_backends",
    "clear_backend_cache",
    "get_backend",
    "normalise_lang",
    "write_silence",
    "write_wav",
]
