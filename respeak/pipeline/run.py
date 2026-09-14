"""The orchestrator: one job, eight stages, one `out.mp4` (flow.md B4.1–B4.8, B7; plan.md 1.1, 1.10).

`run_job()` is the synchronous function :class:`respeak.jobs.JobRunner` calls in a worker thread — no
asyncio anywhere below this line. Every stage marks itself in `status.json` *before* it does any work, so
when it raises, the runner's `f"{step}: {exc}"` names the stage that actually failed. Nothing is caught
here that a user should see: exceptions propagate, and the job ends `failed` with a readable message.

    probe .02 → fetch .10 → transcribe .30 → translate .40 → speak .60 → fit .75
              → subtitles .80 → mux .95 → finish 1.0

The fit stage never cuts speech to make it fit (decisions.md D-49): when the translation runs long it
slows the *picture* down instead, and the one factor `stretch` travels from `audio.stretch_factor()`
through `place()` and `assemble()` into `mux()`, so audio, subtitles and video share one timeline.

Long stages move the bar inside their own range instead of sitting still (plan.md 3.1): fetch .02–.10,
transcribe .30–.40, speak .60–.75, mux .80–.95 — see `STEP_RANGE`. Every stage also writes
`status.detail`, one short sentence saying what is happening right now ("downloading 3.1 MB of 7.4 MB",
"speaking segment 3 of 12"); it is cleared when the job finishes.
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
from respeak.pipeline.tts.base import TTSBackend
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

#: The stages that report progress while they work: (bar on entry, bar when they hand over).
#: `STEP_PROGRESS` stays the handover value; fetch and mux start their range at the value the stage
#: before them left behind, so the bar creeps through a download or an encode instead of jumping.
STEP_RANGE: dict[str, tuple[float, float]] = {
    "fetch": (0.02, 0.10),
    "transcribe": (0.30, 0.40),
    "speak": (0.60, 0.75),
    "mux": (0.80, 0.95),
}

#: Smallest progress change worth an fsync'd write of status.json.
PROGRESS_STEP = 0.01

#: A slot this short cannot be fitted (`audio.fit` needs a positive target); ASR already merges the worst.
MIN_SLOT_SECONDS = 0.1

#: Trimming less than this off the tail of the dub is rounding, not something to warn a user about.
TRIM_WARNING_SECONDS = 0.25

#: D-49 step 4: once the video is as slow as it may go, the sentences that still overrun are sped up
#: this hard — fast, but still intelligible — before anything is allowed to be cut.
LAST_RESORT_SPEEDUP = 1.5

#: A slowdown smaller than this (in percent) is not worth a sentence in `warnings`.
STRETCH_WARNING_PERCENT = 0.5

#: A cue may end this far past the end of the picture before it counts as speech that did not fit.
OVERFLOW_TOLERANCE_SECONDS = 0.05

#: How much of a sentence a warning quotes before it trails off.
WARNING_QUOTE_CHARS = 40


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
    is_youtube = source.get("type") == "youtube"
    timings: dict[str, float] = {}
    started = time.monotonic()
    log.info("job %s starting: %s → %s", job_id, source.get("type"), options.get("to_lang"))

    # -- probe ------------------------------------------------------------------------------------
    with _stage(store, job_id, "probe", timings, "probing the URL" if is_youtube else "checking the file"):
        to_lang = _target_language(options)
        from_lang = _source_language(options, to_lang)
        burn = bool(options.get("burn_subtitles", True))
        backend_name = str(options.get("backend") or settings.tts_backend)
        voice = _voice(options)
        probe = _probe(source, job_dir, settings)
        title = probe.title or _fallback_title(source)
        if title:
            store.update(job_id, title=title)

    # -- fetch ------------------------------------------------------------------------------------
    with _stage(store, job_id, "fetch", timings, "downloading" if is_youtube else "reading the upload"):
        source_mp4 = _fetch(source, job_dir, settings, _fetch_reporter(store, job_id))
        _report(store, job_id, "fetch", 1.0, "extracting the audio")
        source_wav, reference_wav = inputs.extract_audio(source_mp4, job_dir)
        video_seconds = _video_seconds(source_mp4, probe)

    # -- transcribe -------------------------------------------------------------------------------
    with _stage(store, job_id, "transcribe", timings, "transcribing"):
        transcript = asr.transcribe(
            source_wav,
            from_lang,
            settings,
            progress=_progress_reporter(
                store, job_id, "transcribe", _clock_detail("transcribed", video_seconds)
            ),
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
    with _stage(store, job_id, "translate", timings, _plural("translating", len(segments), "segment")):
        translator = get_translator(settings)
        translator.ensure_pair(detected, to_lang)
        texts = translator.translate([segment.text for segment in segments], detected, to_lang)
        spoken = _translated_segments(segments, texts)

    # -- speak ------------------------------------------------------------------------------------
    with _stage(store, job_id, "speak", timings, "preparing the voice"):
        backend = get_backend(backend_name, settings)
        reference = _reference(store, job_id, backend, source_wav, segments, reference_wav, job_dir)
        total = len(spoken)
        clips: list[Path] = []
        for index, segment in enumerate(spoken, start=1):
            # Before and after each sentence: the bar would otherwise stop one segment short of the
            # top of the speak range and jump, and on a one-segment job it would never move at all.
            _report(store, job_id, "speak", (index - 1) / total, f"speaking segment {index} of {total}")
            clip = job_dir / f"seg_{index:04d}.wav"
            backend.synthesize(segment.text, to_lang, reference, clip, voice=voice)
            clips.append(clip)
            _report(store, job_id, "speak", index / total)

    # -- fit --------------------------------------------------------------------------------------
    with _stage(store, job_id, "fit", timings, "fitting the audio to the video"):
        slots = [max(segment.end - segment.start, MIN_SLOT_SECONDS) for segment in spoken]
        fitted = _fit_clips(clips, slots, job_dir, hi=settings.max_speech_speedup)
        # D-49: how much slower the picture would have to run for every sentence to fit whole.
        stretch = audio.stretch_factor(spoken, [seconds for _, seconds in fitted], video_seconds)
        speech_cap = settings.max_speech_speedup
        if stretch > settings.max_video_stretch:
            stretch = settings.max_video_stretch
            speech_cap = LAST_RESORT_SPEEDUP
            store.update(job_id, detail="speeding up the sentences that still do not fit")
            fitted = _fit_clips(
                clips,
                [slot * stretch for slot in slots],
                job_dir,
                hi=speech_cap,
                keep=fitted,  # only the clips that still overrun their stretched slot are redone
            )
        total_seconds = stretch * video_seconds
        placed = audio.place(spoken, fitted, stretch=stretch)
        assembled = audio.assemble(placed, total_seconds, job_dir / DUBBED_WAV)
        dubbed_wav = assembled.path
        log.info(
            "job %s: fit %d clips onto %.2f s of video (stretch %.4f, speech cap %.2f)",
            job_id,
            len(fitted),
            video_seconds,
            stretch,
            speech_cap,
        )
        warnings = fit_warnings(placed, total_seconds, stretch, speech_cap, assembled)
        if warnings:
            log.warning("job %s: %s", job_id, " ".join(warnings))
            store.update(job_id, warnings=warnings)

    # -- subtitles --------------------------------------------------------------------------------
    with _stage(store, job_id, "subtitles", timings, "writing subtitles"):
        subs_srt = subtitles.build_srt(_cues(placed, total_seconds), job_dir / SUBS_SRT)

    # -- mux --------------------------------------------------------------------------------------
    with _stage(store, job_id, "mux", timings, "encoding the video"):
        out_mp4 = mux.mux(
            source_mp4,
            dubbed_wav,
            subs_srt,
            burn,
            to_lang,
            job_dir / OUTPUT_MP4,
            stretch=stretch,
            progress=_progress_reporter(store, job_id, "mux", _clock_detail("encoding", total_seconds)),
        )
        if not out_mp4.is_file() or out_mp4.stat().st_size == 0:
            raise PipelineError(f"ffmpeg wrote no output for job {job_id}")

    # -- finish -----------------------------------------------------------------------------------
    with _stage(store, job_id, "finish", timings, "tidying up"):
        cleanup(job_dir)
        store.update(job_id, state="done", output=OUTPUT_MP4, progress=1.0, error=None, detail=None)

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


def _fetch(
    source: dict[str, Any],
    job_dir: Path,
    settings: Settings,
    progress: inputs.DownloadProgress | None = None,
) -> Path:
    """B4.2: either source type ends as `<jobdir>/source.mp4`."""
    kind = source.get("type")
    if kind == "youtube":
        return inputs.fetch_youtube(_require_url(source), job_dir, settings, progress)
    upload = job_dir / UPLOAD_FILENAME
    if not upload.is_file():
        raise PipelineError(f"the uploaded file is missing from {job_dir}")
    return inputs.remux(upload, job_dir / SOURCE_MP4)


def _reference(
    store: JobStore,
    job_id: str,
    backend: TTSBackend,
    source_wav: Path,
    segments: list[Segment],
    fallback: Path,
    job_dir: Path,
) -> Path | None:
    """The speaker clip for a cloning backend: the busiest 30 s of speech (plan.md 3.7), else None.

    Backends that do not clone get None and the cut is skipped, which saves the second it costs.
    Without segments — which `run_job` never allows this far — the loudest-window clip from
    `extract_audio` stands in.
    """
    if not backend.cloning:
        return None
    if not segments:
        return fallback
    store.update(job_id, detail="picking a reference clip")
    return inputs.reference_from_segments(source_wav, segments, job_dir / REFERENCE_WAV)


def _fit_clips(
    clips: list[Path],
    targets: list[float],
    job_dir: Path,
    *,
    hi: float,
    keep: list[tuple[Path, float]] | None = None,
) -> list[tuple[Path, float]]:
    """B4.6: speed each spoken clip towards its slot, at most by `hi`; return (path, real length).

    Every pass starts from the *original* clip, so raising the cap on the second pass (D-49 step 4)
    speeds that sentence up 1.5× in total, not 1.3 × 1.5. With `keep`, the clips that already fit
    their (new, stretched) target are left exactly as they were and only the overrunning ones are
    re-encoded — usually one or two of them.
    """
    fitted: list[tuple[Path, float]] = []
    for index, (clip, target) in enumerate(zip(clips, targets, strict=True), start=1):
        previous = keep[index - 1] if keep is not None else None
        if previous is not None and previous[1] <= target:
            fitted.append(previous)
            continue
        fitted.append(audio.fit(clip, target, job_dir / f"seg_{index:04d}_fit.wav", hi=hi))
    return fitted


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
def _stage(
    store: JobStore,
    job_id: str,
    step: str,
    timings: dict[str, float],
    detail: str | None = None,
) -> Iterator[None]:
    """Mark the step *before* the work (so a failure is named right) and log what it cost.

    One write carries the state, the step, the bar's value on entry and the sentence for the step.
    """
    store.update(job_id, state="running", step=step, progress=_step_range(step)[0], detail=detail)
    start = time.monotonic()
    try:
        yield
    finally:
        timings[step] = time.monotonic() - start
        log.info("job %s: %s took %.2f s", job_id, step, timings[step])


def _step_range(step: str) -> tuple[float, float]:
    """(bar on entry, bar on handover) for a step — the same number twice for the quick ones."""
    if step in STEP_RANGE:
        return STEP_RANGE[step]
    return STEP_PROGRESS[step], STEP_PROGRESS[step]


def _interpolate(step: str, fraction: float) -> float:
    """Where the bar sits `fraction` of the way through `step`."""
    start, end = _step_range(step)
    return start + (end - start) * min(1.0, max(0.0, float(fraction)))


def _report(store: JobStore, job_id: str, step: str, fraction: float, detail: str | None = None) -> None:
    """Write the bar (and, when given, the sentence) for a step that counts its own work."""
    fields: dict[str, Any] = {"progress": round(_interpolate(step, fraction), 3)}
    if detail is not None:
        fields["detail"] = detail
    store.update(job_id, **fields)


def _throttled_writer(store: JobStore, job_id: str, step: str) -> Callable[[float, str | None], None]:
    """Writes the bar (and a sentence) for a step, at most once per percent of the whole job.

    Three rules beyond the percent, all of them there because one stage (fetch) reports several
    files through one writer — yt-dlp counts the video 0→1, then the audio 0→1, then the merge:

    * a *changed sentence* is always written, even when the bar cannot move. Otherwise the page
      freezes on "downloading 7.4 MB of 7.4 MB" for the whole second file and the whole merge.
    * the bar never goes **backwards**: the second file restarting at 0.05 must not undo the first.
    * a *final* write (fraction 1.0, the end of a range) is never throttled, so `mux` really does
      land on 0.95 before `finish` rather than a percent short of it.
    """
    state: dict[str, Any] = {"last": _step_range(step)[0], "detail": None}

    def write(fraction: float, detail: str | None) -> None:
        last = float(state["last"])
        value = max(last, _interpolate(step, fraction))
        moved = value - last >= PROGRESS_STEP
        spoke = detail is not None and detail != state["detail"]
        topped = fraction >= 1.0 and value > last  # the end of the range, not written yet
        if not (moved or spoke or topped):  # one fsync per percent, not per callback
            return
        state["last"] = value
        fields: dict[str, Any] = {"progress": round(value, 3)}
        if detail is not None:
            fields["detail"] = detail
            state["detail"] = detail
        store.update(job_id, **fields)

    return write


def _progress_reporter(
    store: JobStore,
    job_id: str,
    step: str,
    detail: Callable[[float], str] | None = None,
) -> Callable[[float], None]:
    """A 0–1 callback that moves the bar inside this step's range, with an optional detail line."""
    write = _throttled_writer(store, job_id, step)

    def report(fraction: float) -> None:
        clamped = min(1.0, max(0.0, float(fraction)))
        write(clamped, detail(clamped) if detail is not None else None)

    return report


def _fetch_reporter(store: JobStore, job_id: str) -> inputs.DownloadProgress:
    """yt-dlp's (fraction, sentence) hook → the bar inside the fetch range, throttled the same way."""
    write = _throttled_writer(store, job_id, "fetch")

    def report(fraction: float, detail: str) -> None:
        write(fraction, detail)

    return report


def _clock_detail(verb: str, total_seconds: float) -> Callable[[float], str]:
    """'transcribed 0:42 of 2:10' from a 0–1 fraction of `total_seconds`."""

    def detail(fraction: float) -> str:
        return f"{verb} {_clock(fraction * total_seconds)} of {_clock(total_seconds)}"

    return detail


def _clock(seconds: float) -> str:
    """63.4 → '1:03'; 3723 → '1:02:03'."""
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _plural(verb: str, count: int, noun: str) -> str:
    """'translating 12 segments' / 'translating 1 segment'."""
    return f"{verb} {count} {noun}{'' if count == 1 else 's'}"


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


def _voice(options: dict[str, Any]) -> str | None:
    """The voice id picked in the form, or None for the backend's default (flow.md B5, plan.md 3.3)."""
    raw = options.get("voice")
    if raw is None:
        return None
    voice = str(raw).strip()
    return voice or None


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


def _cues(placed: list[Placed], total_seconds: float) -> list[Cue]:
    """B4.7: the subtitle times are the times the dubbed clips actually got, clipped to the picture.

    A cue can only run past the end when speech had to be cut after all (D-49's last resort), and a
    subtitle that outlives the video makes the finished file *claim* a length the picture does not
    have — mp4 duration is the longest stream, subtitles included.
    """
    cues: list[Cue] = []
    for item in placed:
        if item.start >= total_seconds:  # this sentence is not in the dub either; assemble dropped it
            continue
        cues.append(Cue(start=item.start, end=min(item.end, total_seconds), text=item.text))
    return cues


def fit_warnings(
    placed: list[Placed],
    total_seconds: float,
    stretch: float,
    speech_cap: float,
    assembled: audio.AssembleResult,
) -> list[str]:
    """Everything the browser should be told about the timeline, in the order it should read it (D-49).

    One informational line when the picture was slowed, then — only if speech *still* did not fit,
    which the cap makes rare — the sentences that ran off the end and how much was lost.
    """
    messages: list[str] = []
    slowed = stretch_warning(stretch)
    if slowed is not None:
        messages.append(slowed)
    overflow = overflow_warning(placed, total_seconds, stretch, speech_cap)
    if overflow is None:
        return messages + assemble_warnings(assembled)
    messages.append(overflow)
    if assembled.lost_seconds >= TRIM_WARNING_SECONDS:
        messages.append(cut_sentence(assembled.lost_seconds))
    return messages


def stretch_warning(stretch: float) -> str | None:
    """The line a slowed video earns: "The video was slowed by 12 % so all the speech fits."

    None when the change is under half a percent — nobody needs to be told about 0.3 %.
    """
    percent = (float(stretch) - 1.0) * 100.0
    if percent < STRETCH_WARNING_PERCENT:
        return None
    return f"The video was slowed by {percent:.0f} % so all the speech fits."


def overflow_warning(
    placed: list[Placed],
    total_seconds: float,
    stretch: float,
    speech_cap: float,
) -> str | None:
    """Name the sentences that ran past the end of the (already slowed) picture; None when none did."""
    over = [item for item in placed if item.end > total_seconds + OVERFLOW_TOLERANCE_SECONDS]
    if not over:
        return None
    slowed = round((float(stretch) - 1.0) * 100)
    reason = (
        f"even after slowing the video {slowed} % and speeding speech to {speech_cap:g}×"
        if slowed >= 1
        else "before the video ended"
    )
    quoted = ", ".join(_quote(item.text) for item in over)
    return f"{len(over)} sentence{'' if len(over) == 1 else 's'} could not fit {reason}: {quoted}"


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
        messages.append(cut_sentence(result.trimmed_seconds))
    return messages


def cut_sentence(seconds: float) -> str:
    """Say what was lost, not how long the dub was: "The last 2.2 s of speech were cut."."""
    return f"The last {seconds:.1f} s of speech were cut to fit the video."


def _quote(text: str, limit: int = WARNING_QUOTE_CHARS) -> str:
    """A sentence as it appears inside a warning: one line, trailing off after `limit` characters."""
    clean = " ".join(str(text).split())
    if len(clean) > limit:
        clean = clean[:limit].rstrip() + "…"
    return f"'{clean}'"


__all__ = [
    "KEPT_FILES",
    "LAST_RESORT_SPEEDUP",
    "OUTPUT_MP4",
    "STEPS",
    "STEP_PROGRESS",
    "STEP_RANGE",
    "SUBS_SRT",
    "PipelineError",
    "assemble_warnings",
    "cleanup",
    "cut_sentence",
    "fit_warnings",
    "get_translator",
    "overflow_warning",
    "run_job",
    "stretch_warning",
]
