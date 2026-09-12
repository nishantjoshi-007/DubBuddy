"""Speech recognition with faster-whisper (flow.md B4.3, decisions D-06).

One `WhisperModel` per (model, device, compute_type) per process: loading `small` costs seconds and the
worker pool reuses the same interpreter for every job.  Everything here is synchronous — the pipeline runs
in a worker thread, never on the event loop.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from respeak.config import Settings
from respeak.pipeline.types import Segment, Transcript, Word

if TYPE_CHECKING:  # pragma: no cover - import cost is paid lazily at runtime
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

#: Segments shorter than this are glued onto the next one: a 0.3 s "Yeah." makes a TTS clip that cannot be
#: fitted to its slot (B4.6) and a subtitle cue nobody can read (B4.7).
MIN_SEGMENT_SECONDS = 0.6

#: compute_type used on CUDA when the operator left WHISPER_COMPUTE at its default (D-06).
CUDA_COMPUTE_TYPE = "float16"

ProgressCallback = Callable[[float], None]

_models: dict[tuple[str, str, str], WhisperModel] = {}
_models_lock = threading.Lock()


class ASRError(RuntimeError):
    """Transcription could not be performed. Never swallowed, never signalled by returning None."""


def compute_type_for(settings: Settings) -> str:
    """int8 on CPU, float16 on CUDA — unless WHISPER_COMPUTE was set to something other than its default."""
    default = Settings.model_fields["whisper_compute"].default
    if settings.resolved_device() == "cuda" and settings.whisper_compute == default:
        return CUDA_COMPUTE_TYPE
    return settings.whisper_compute


def load_model(settings: Settings) -> WhisperModel:
    """Return the process-wide `WhisperModel` for these settings, loading it on first use.

    Cached per (model, device, compute_type) so `WHISPER_MODEL` can change between jobs without leaking
    a second copy of the same weights.
    """
    device = settings.resolved_device()
    compute_type = compute_type_for(settings)
    key = (settings.whisper_model, device, compute_type)
    with _models_lock:
        model = _models.get(key)
        if model is not None:
            return model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - faster-whisper is a hard dependency
            raise ASRError(f"faster-whisper is not importable: {exc}") from exc

        kwargs: dict[str, object] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": os.cpu_count() or 0,
        }
        if settings.model_cache_dir is not None:
            kwargs["download_root"] = str(settings.model_cache_dir)
        log.info("loading whisper model %s on %s (%s)", settings.whisper_model, device, compute_type)
        try:
            model = WhisperModel(settings.whisper_model, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            raise ASRError(
                f"could not load whisper model {settings.whisper_model!r} "
                f"on {device} with compute_type {compute_type!r}: {exc}"
            ) from exc
        _models[key] = model
        return model


def transcribe(
    wav: Path,
    language: str | None,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Transcribe a 16 kHz mono wav into a `Transcript` (flow.md B4.3).

    `language` is the Advanced-fold override (ISO-639-1); None auto-detects.  `progress`, when given, is
    called with a 0–1 fraction as segments arrive, and once with 1.0 at the end.
    """
    wav = Path(wav)
    if not wav.is_file():
        raise ASRError(f"audio file not found: {wav}")

    model = load_model(settings)
    try:
        raw_segments, info = model.transcribe(
            str(wav),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
    except Exception as exc:
        raise ASRError(f"whisper failed on {wav.name}: {exc}") from exc

    total = float(getattr(info, "duration", 0.0) or 0.0)
    kept: list[Segment] = []
    try:
        for raw in raw_segments:
            text = (raw.text or "").strip()
            if text:
                kept.append(
                    Segment(
                        start=float(raw.start),
                        end=float(raw.end),
                        text=text,
                        words=_words_of(raw),
                    )
                )
            if progress is not None and total > 0:
                progress(min(1.0, max(0.0, float(raw.end) / total)))
    except Exception as exc:
        raise ASRError(f"whisper failed while decoding {wav.name}: {exc}") from exc

    segments = _merge_short(kept)
    if progress is not None:
        progress(1.0)

    detected = language or getattr(info, "language", None)
    if not detected:
        raise ASRError(f"whisper reported no language for {wav.name}")
    log.info(
        "transcribed %s: %d segments, language=%s (%.0f%% confidence)",
        wav.name,
        len(segments),
        detected,
        100.0 * float(getattr(info, "language_probability", 0.0) or 0.0),
    )
    return Transcript(language=detected, segments=segments)


def _words_of(raw: object) -> list[Word]:
    """faster-whisper gives `words=None` when a segment carried no timed tokens."""
    words = getattr(raw, "words", None) or []
    out: list[Word] = []
    for word in words:
        text = (word.word or "").strip()
        if text:
            out.append(Word(start=float(word.start), end=float(word.end), text=text))
    return out


def _merge_short(segments: list[Segment]) -> list[Segment]:
    """Glue every segment shorter than MIN_SEGMENT_SECONDS onto the next one (the last onto the previous)."""
    merged: list[Segment] = []
    pending: Segment | None = None
    for segment in segments:
        if pending is not None:
            segment = _join(pending, segment)
            pending = None
        if segment.end - segment.start < MIN_SEGMENT_SECONDS:
            pending = segment
            continue
        merged.append(segment)
    if pending is not None:
        if merged:
            merged[-1] = _join(merged[-1], pending)
        else:
            merged.append(pending)
    return merged


def _join(first: Segment, second: Segment) -> Segment:
    return Segment(
        start=first.start,
        end=max(first.end, second.end),
        text=f"{first.text} {second.text}".strip(),
        words=[*first.words, *second.words],
    )
