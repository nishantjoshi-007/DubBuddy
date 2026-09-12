"""The orchestrator: one job, eight stages, one `out.mp4` (flow.md B4.1–B4.8, B7; plan.md 1.1, 1.10).

`run_job()` is the synchronous function :class:`respeak.jobs.JobRunner` calls in a worker thread — no
asyncio anywhere below this line. Every stage marks itself in `status.json` *before* it does any work, so
when it raises, the runner's `f"{step}: {exc}"` names the stage that actually failed. Nothing is caught
here that a user should see: exceptions propagate, and the job ends `failed` with a readable message.

    probe .02 → fetch .10 → transcribe .30 → translate .40 → speak .60 → fit .75
              → subtitles .80 → mux .95 → finish 1.0

Long stages (transcribe, speak) push a fraction into the range between their own progress value and the
next stage's, so the bar keeps moving inside them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from respeak.config import Settings
from respeak.jobs import JobStore
from respeak.pipeline import asr, audio, ffmpeg, inputs, mux, subtitles
from respeak.pipeline.translate import ArgosTranslator, Translator, normalize_code
from respeak.pipeline.tts import get_backend
from respeak.pipeline.types import Cue, Placed, Probe, Segment

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------- files

UPLOAD_FILENAME = "upload.bin"
SOURCE_MP4 = "source.mp4"
SOURCE_WAV = "source.wav"
REFERENCE_WAV = "reference.wav"
DUBBED_WAV = "dubbed.wav"
SUBS_SRT = "subs.srt"
OUTPUT_MP4 = "out.mp4"
SEGMENT_GLOB = "seg_*.wav"

#: Everything the job directory may keep once the job is done (flow.md B7).
KEPT_FILES: frozenset[str] = frozenset({OUTPUT_MP4, SUBS_SRT, "status.json"})

# --------------------------------------------------------------------------------------------- stages

STEPS: tuple[str, ...] = (
    "probe",
    "fetch",
    "transcribe",
    "translate",
    "speak",
    "fit",
    "subtitles",
    "mux",
    "finish",
)

STEP_PROGRESS: dict[str, float] = {
    "probe": 0.02,
    "fetch": 0.10,
    "transcribe": 0.30,
    "translate": 0.40,
    "speak": 0.60,
    "fit": 0.75,
    "subtitles": 0.80,
    "mux": 0.95,
    "finish": 1.0,
}

#: Smallest progress change worth an fsync'd write of status.json.
PROGRESS_STEP = 0.01

#: A slot this short cannot be fitted (`audio.fit` needs a positive target); ASR already merges the worst.
MIN_SLOT_SECONDS = 0.1

#: Trimming less than this off the tail of the dub is rounding, not something to warn a user about.
TRIM_WARNING_SECONDS = 0.25


class PipelineError(RuntimeError):
    """The job cannot go on: no speech, a bad language pair, a stage that produced nothing usable."""


# --------------------------------------------------------------------------------------------------
# One translator per process: Argos installs packages into a shared directory and caches routes.
# --------------------------------------------------------------------------------------------------

_translator_lock = threading.Lock()
_translator: ArgosTranslator | None = None


def get_translator(settings: Settings) -> Translator:
    """The process-wide :class:`ArgosTranslator`, built on first use (rebuilt if `settings` changed)."""
    global _translator
    with _translator_lock:
        if _translator is None or _translator.settings is not settings:
            _translator = ArgosTranslator(settings)
        return _translator


# --------------------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------------------


def run_job(job_id: str, settings: Settings, store: JobStore) -> None:
    """Dub one job from `status.json` to `out.mp4`. Raises on any failure; the runner records it."""
    job_dir = store.path(job_id)
    status = store.require(job_id)
    source: dict[str, Any] = dict(status.get("source") or {})
    options: dict[str, Any] = dict(status.get("options") or {})
    timings: dict[str, float] = {}
    started = time.monotonic()
    log.info("job %s starting: %s → %s", job_id, source.get("type"), options.get("to_lang"))

    # -- probe ------------------------------------------------------------------------------------
    with _stage(store, job_id, "probe", timings):
        to_lang = _target_language(options)
        from_lang = _source_language(options, to_lang)
        burn = bool(options.get("burn_subtitles", True))
        backend_name = str(options.get("backend") or settings.tts_backend)
        probe = _probe(source, job_dir, settings)
        title = probe.title or _fallback_title(source)
        if title:
            store.update(job_id, title=title)

    # -- fetch ------------------------------------------------------------------------------------
    with _stage(store, job_id, "fetch", timings):
        source_mp4 = _fetch(source, job_dir, settings)
        source_wav, reference_wav = inputs.extract_audio(source_mp4, job_dir)
        video_seconds = _video_seconds(source_mp4, probe)

    # -- transcribe -------------------------------------------------------------------------------
    with _stage(store, job_id, "transcribe", timings):
        transcript = asr.transcribe(
            source_wav, from_lang, settings, progress=_progress_reporter(store, job_id, "transcribe")
        )
        detected = normalize_code(transcript.language)
        store.update(job_id, detected_language=detected)
        _check_pair(detected, to_lang)
        segments = transcript.segments
        if not segments:
            raise PipelineError(
                "no speech was found in this video, so there is nothing to dub (is it music or silence?)"
            )
        log.info("job %s: %d segments, %s detected", job_id, len(segments), detected)

    # -- translate --------------------------------------------------------------------------------
    with _stage(store, job_id, "translate", timings):
        translator = get_translator(settings)
        translator.ensure_pair(detected, to_lang)
        texts = translator.translate([segment.text for segment in segments], detected, to_lang)
        spoken = _translated_segments(segments, texts)

    # -- speak ------------------------------------------------------------------------------------
    with _stage(store, job_id, "speak", timings):
        backend = get_backend(backend_name, settings)
        reference = reference_wav if backend.cloning else None
        report = _progress_reporter(store, job_id, "speak")
        clips: list[Path] = []
        for index, segment in enumerate(spoken, start=1):
            clip = job_dir / f"seg_{index:04d}.wav"
            backend.synthesize(segment.text, to_lang, reference, clip)
            clips.append(clip)
            report(index / len(spoken))

    # -- fit --------------------------------------------------------------------------------------
    with _stage(store, job_id, "fit", timings):
        fitted: list[tuple[Path, float]] = []
        for index, (segment, clip) in enumerate(zip(spoken, clips, strict=True), start=1):
            slot = max(segment.end - segment.start, MIN_SLOT_SECONDS)
            fitted.append(audio.fit(clip, slot, job_dir / f"seg_{index:04d}_fit.wav"))
        placed = audio.place(spoken, fitted)
        assembled = audio.assemble(placed, video_seconds, job_dir / DUBBED_WAV)
        dubbed_wav = assembled.path
        warnings = assemble_warnings(assembled)
        if warnings:
            log.warning("job %s: %s", job_id, " ".join(warnings))
            store.update(job_id, warnings=warnings)

    # -- subtitles --------------------------------------------------------------------------------
    with _stage(store, job_id, "subtitles", timings):
        subs_srt = subtitles.build_srt(_cues(placed), job_dir / SUBS_SRT)

    # -- mux --------------------------------------------------------------------------------------
    with _stage(store, job_id, "mux", timings):
        out_mp4 = mux.mux(source_mp4, dubbed_wav, subs_srt, burn, to_lang, job_dir / OUTPUT_MP4)
        if not out_mp4.is_file() or out_mp4.stat().st_size == 0:
            raise PipelineError(f"ffmpeg wrote no output for job {job_id}")

    # -- finish -----------------------------------------------------------------------------------
    with _stage(store, job_id, "finish", timings):
        cleanup(job_dir)
        store.update(job_id, state="done", output=OUTPUT_MP4, progress=1.0, error=None)

    log.info(
        "job %s done in %.1f s (%s)",
        job_id,
        time.monotonic() - started,
        ", ".join(f"{step} {timings[step]:.1f}s" for step in STEPS if step in timings),
    )


# --------------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------------


def _probe(source: dict[str, Any], job_dir: Path, settings: Settings) -> Probe:
    """B4.1: metadata before any download, and the duration cap enforced for both source types."""
    kind = source.get("type")
    if kind == "youtube":
        probe = inputs.probe_youtube(_require_url(source), settings)
        inputs.enforce_duration(probe, settings)  # probe_youtube does not enforce it itself
        return probe
    if kind == "upload":
        return inputs.validate_upload(job_dir / UPLOAD_FILENAME, settings)  # enforces the cap itself
    raise PipelineError(f"unknown source type {kind!r}; expected 'youtube' or 'upload'")


def _fetch(source: dict[str, Any], job_dir: Path, settings: Settings) -> Path:
    """B4.2: either source type ends as `<jobdir>/source.mp4`."""
    kind = source.get("type")
    if kind == "youtube":
        return inputs.fetch_youtube(_require_url(source), job_dir, settings)
    upload = job_dir / UPLOAD_FILENAME
    if not upload.is_file():
        raise PipelineError(f"the uploaded file is missing from {job_dir}")
    return inputs.remux(upload, job_dir / SOURCE_MP4)


def cleanup(job_dir: Path) -> list[str]:
    """B7: delete every intermediate, keep `out.mp4`, `subs.srt` and `status.json`. Never fatal."""
    targets = [job_dir / name for name in (UPLOAD_FILENAME, SOURCE_WAV, REFERENCE_WAV, DUBBED_WAV)]
    targets += sorted(job_dir.glob("source.*"))  # source.mp4, plus any container yt-dlp left behind
    targets += sorted(job_dir.glob(SEGMENT_GLOB))
    removed: list[str] = []
    for path in targets:
        if path.name in KEPT_FILES or not path.is_file():
            continue
        try:
            path.unlink()
        except OSError as exc:  # pragma: no cover - a locked file must not fail a finished job
            log.warning("could not delete %s: %s", path, exc)
            continue
        removed.append(path.name)
    log.debug("cleaned %d intermediate file(s) from %s", len(removed), job_dir)
    return removed


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


@contextmanager
def _stage(store: JobStore, job_id: str, step: str, timings: dict[str, float]) -> Iterator[None]:
    """Mark the step *before* the work (so a failure is named right) and log what it cost."""
    store.mark(job_id, "running", step=step, progress=STEP_PROGRESS[step])
    start = time.monotonic()
    try:
        yield
    finally:
        timings[step] = time.monotonic() - start
        log.info("job %s: %s took %.2f s", job_id, step, timings[step])


def _progress_reporter(store: JobStore, job_id: str, step: str) -> Callable[[float], None]:
    """A 0–1 callback that moves the bar between this step's value and the next one's."""
    start = STEP_PROGRESS[step]
    end = STEP_PROGRESS[STEPS[STEPS.index(step) + 1]]
    state = {"last": start}

    def report(fraction: float) -> None:
        value = start + (end - start) * min(1.0, max(0.0, float(fraction)))
        if value - state["last"] < PROGRESS_STEP:  # one fsync per percent, not per segment
            return
        state["last"] = value
        store.update(job_id, progress=round(value, 3))

    return report


def _target_language(options: dict[str, Any]) -> str:
    raw = options.get("to_lang")
    if not raw:
        raise PipelineError("this job has no target language (to_lang)")
    return normalize_code(str(raw))


def _source_language(options: dict[str, Any], to_lang: str) -> str | None:
    """The Advanced-fold override, or None for auto-detection (flow.md B4.3, D-32)."""
    raw = options.get("from_lang")
    if raw is None or not str(raw).strip():
        return None
    from_lang = normalize_code(str(raw))
    _check_pair(from_lang, to_lang)
    return from_lang


def _check_pair(src: str, dst: str) -> None:
    """Argos raises for src == dst; say it in words a user can act on, before the work starts."""
    if src == dst:
        raise PipelineError(f"the video is already in {src!r}; pick a different target language")


def _require_url(source: dict[str, Any]) -> str:
    url = str(source.get("url") or "").strip()
    if not url:
        raise PipelineError("this job is a YouTube job but carries no URL")
    return url


def _fallback_title(source: dict[str, Any]) -> str | None:
    """An upload with no container title is named after the file the user picked."""
    filename = source.get("filename")
    return Path(str(filename)).stem if filename else None


def _video_seconds(source_mp4: Path, probe: Probe) -> float:
    """The real length of the downloaded file; YouTube's metadata duration is only a hint."""
    try:
        seconds = ffmpeg.duration(source_mp4)
    except Exception as exc:  # pragma: no cover - ffprobe already read this file once
        log.warning("could not measure %s (%s); using the probed duration", source_mp4.name, exc)
        seconds = probe.duration
    if seconds <= 0:
        raise PipelineError(f"{source_mp4.name} has no duration; the download looks broken")
    return seconds


def _translated_segments(segments: list[Segment], texts: list[str]) -> list[Segment]:
    """New segments carrying the translated text — `audio.place()` copies `Segment.text` into the cues."""
    if len(texts) != len(segments):
        raise PipelineError(f"the translator returned {len(texts)} texts for {len(segments)} segments")
    return [
        Segment(start=segment.start, end=segment.end, text=(text or "").strip() or segment.text)
        for segment, text in zip(segments, texts, strict=True)
    ]


def _cues(placed: list[Placed]) -> list[Cue]:
    """B4.7: the subtitle times are the times the dubbed clips actually got."""
    return [Cue(start=item.start, end=item.end, text=item.text) for item in placed]


def assemble_warnings(result: audio.AssembleResult) -> list[str]:
    """Plain sentences for whatever the video's length cost the dub; empty when nothing was lost.

    The job still succeeds — the browser shows these as a notice next to the finished video, because
    a dub whose last sentences fell off the end must not look identical to a perfect one.
    """
    messages: list[str] = []
    if result.dropped_clips:
        count = result.dropped_clips
        messages.append(
            f"{count} segment{'' if count == 1 else 's'} ({result.dropped_seconds:.1f} s of speech) "
            "did not fit before the video ended and were cut."
        )
    if result.trimmed_seconds >= TRIM_WARNING_SECONDS:
        messages.append(
            f"The dub was {result.trimmed_seconds:.1f} s longer than the video and was trimmed to fit."
        )
    return messages


__all__ = [
    "KEPT_FILES",
    "OUTPUT_MP4",
    "STEPS",
    "STEP_PROGRESS",
    "SUBS_SRT",
    "PipelineError",
    "assemble_warnings",
    "cleanup",
    "get_translator",
    "run_job",
]
