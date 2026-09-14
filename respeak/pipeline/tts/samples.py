"""Voice previews for the picker (plan.md 3.3, flow.md B6).

"Dora (female)" and "Alex (male)" tell a person almost nothing about what they are about to hear, so
the form offers a play button next to the voice select and this module answers it: one short neutral
sentence per language, synthesised once per (backend, language, voice) and then cached on disk under
``DATA_DIR/voice_samples/``.

Deliberately *not* under ``DATA_DIR/jobs/``: the TTL sweeper (``respeak.jobs.sweep``) walks only
``JobStore.list_ids()``, i.e. the directories inside ``jobs_dir``, so a cache that lives beside it is
never swept — which is what we want. A sample is the same forever; regenerating it on every visit
would cost a model load per click.

Synchronous, like everything else in ``respeak/pipeline/``: the route that calls this is a plain
``def``, so it runs in FastAPI's threadpool and a first-time synthesis never touches the event loop.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from pathlib import Path

from ...config import Settings
from . import available_backends, get_backend, normalise_lang

log = logging.getLogger(__name__)

SAMPLE_TEXT: dict[str, str] = {
    # One short, neutral, self-describing sentence per language Kokoro speaks. Short because it is
    # synthesised on demand behind a click, and neutral because the point is the voice, not the words.
    "en": "Hi, this is how my voice sounds.",
    "es": "Hola, así suena mi voz.",
    "fr": "Bonjour, voici ma voix.",
    "hi": "नमस्ते, मेरी आवाज़ ऐसी है।",
    "it": "Ciao, questa è la mia voce.",
    "ja": "こんにちは、これが私の声です。",
    "pt": "Olá, esta é a minha voz.",
    "zh": "你好，这是我的声音。",
}
"""ISO-639-1 -> the sentence every voice of that language says in its preview."""

SAMPLES_DIRNAME = "voice_samples"
"""``DATA_DIR/<this>/<backend>/<lang>/<voice>.wav``. Outside ``jobs/``, so the sweeper ignores it."""

#: A path segment may only be a plain name. Every value that reaches :func:`sample_path` has already
#: been matched against the backend's own tables, so this can never fire in practice — it is here so
#: that the function is safe on its own terms and a future caller cannot turn a voice id into a path.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_locks_guard = threading.Lock()
#: One lock per sample file: two browsers clicking the same voice at once must synthesise once.
_locks: dict[str, threading.Lock] = {}


def sample_path(settings: Settings, backend: str, lang: str, voice: str) -> Path:
    """Where the preview of one voice lives: ``DATA_DIR/voice_samples/<backend>/<lang>/<voice>.wav``.

    Raises ``ValueError`` for anything that is not a plain name, so no caller can build a path that
    leaves the cache directory.
    """
    for label, segment in (("backend", backend), ("language", lang), ("voice", voice)):
        if not _SAFE_SEGMENT.match(segment or ""):
            raise ValueError(f"{segment!r} is not a usable {label} name for a voice sample")
    return Path(settings.data_dir) / SAMPLES_DIRNAME / backend / lang / f"{voice}.wav"


def check_sample(settings: Settings, backend_name: str, lang: str, voice: str) -> tuple[str, str, str]:
    """Validate a ``(backend, lang, voice)`` triple against the live tables; return it normalised.

    Raises ``ValueError`` naming what is actually on offer. This is the only thing standing between
    three URL path segments and the filesystem, so it matches the voice id against the backend's own
    table exactly rather than sanitising it.
    """
    name = (backend_name or "").strip().lower()
    infos = available_backends(settings)
    info = infos.get(name)
    if info is None:
        raise ValueError(f"unknown backend {name!r}; known backends: {', '.join(sorted(infos))}")
    if not info.voices:
        raise ValueError(f"the {info.name} backend has no preset voices to preview")
    if not info.installed:
        raise ValueError(info.reason or f"the {name} backend is not installed")

    code = normalise_lang(lang or "")
    offered = info.voices.get(code) or []
    if not offered:
        speaks = ", ".join(sorted(info.voices))
        raise ValueError(f"the {name} backend has no voices for {code!r}; it has voices for: {speaks}")

    wanted = (voice or "").strip()
    if wanted not in {entry.id for entry in offered}:
        ids = ", ".join(entry.id for entry in offered)
        raise ValueError(f"the {name} backend has no voice {wanted!r} for {code!r}; its voices are: {ids}")

    if code not in SAMPLE_TEXT:
        langs = ", ".join(sorted(SAMPLE_TEXT))
        raise ValueError(f"there is no preview sentence for {code!r}; there is one for: {langs}")
    return name, code, wanted


def _lock_for(path: Path) -> threading.Lock:
    """The lock guarding one sample file. Bounded by the size of the voice tables."""
    key = str(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def _is_ready(path: Path) -> bool:
    """True when a complete sample is already on disk (a zero-byte file is not one)."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def ensure_sample(settings: Settings, backend_name: str, lang: str, voice: str) -> Path:
    """The cached preview WAV for one voice, synthesising it once if it is not there yet.

    Raises ``ValueError`` for a triple the backend does not offer (nothing is written), and whatever
    the backend raises — ``TTSError`` — when synthesis itself fails.

    Two browsers clicking the same voice at the same moment must cost one synthesis, not two, and
    neither may be handed a half-written file: the work happens under a per-path lock, into a temp
    file in the same directory, and only ``os.replace`` makes it visible under its real name.
    """
    name, code, voice_id = check_sample(settings, backend_name, lang, voice)
    path = sample_path(settings, name, code, voice_id)
    if _is_ready(path):
        return path

    with _lock_for(path):
        if _is_ready(path):  # another thread synthesised it while we waited
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        # The suffix stays .wav: soundfile picks its format from the extension.
        tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex}.wav"
        log.info("synthesising the %s/%s/%s voice sample", name, code, voice_id)
        try:
            get_backend(name, settings).synthesize(SAMPLE_TEXT[code], code, None, tmp, voice=voice_id)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return path


__all__ = [
    "SAMPLES_DIRNAME",
    "SAMPLE_TEXT",
    "check_sample",
    "ensure_sample",
    "sample_path",
]
