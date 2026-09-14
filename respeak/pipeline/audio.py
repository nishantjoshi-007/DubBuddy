"""Fit each spoken clip to its slot, place it on the timeline, assemble `dubbed.wav`.

docs/flow.md B4.6: per-segment `atempo`, never a global speed change, and never cut speech —
slow the *picture* down instead (docs/decisions.md, "When the dub is longer than the video").

    slot_i    = seg.end - seg.start
    factor_i  = min(clip_seconds / slot_i, 1.3) when the clip is too long, else 1.0
                                                            pitch-preserving, chained beyond 0.5–2.0
    s         = max(1, max_i Σ_{j≥i} d_j / (V − start_i))   the smallest video slowdown that fits
    start_i   = max(s · seg.start, prev_end)                a long clip pushes the next one later
    end_i     = start_i + fitted_seconds

A clip that is *shorter* than its slot is left alone: stretching speech to fill a gap sounds drunk,
and the silence after it is exactly the pause the original speaker took.

`stretch_factor()` answers "how much would the video have to be slowed for all of this speech to fit?"
in closed form; the caller caps it (`MAX_VIDEO_STRETCH`) and passes what it settled on to `place()`,
so picture and speech stay aligned on the stretched timeline.

`assemble()` mixes the fitted clips onto silence of exactly `total_seconds` (the caller passes
`s · video_seconds`), so `dubbed.wav` is always the length of the picture it will be muxed onto — the
mux never has to guess. It reports what did not fit, because a dub that quietly loses its last three
sentences is the kind of failure nobody notices until it is published.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from respeak.pipeline import ffmpeg
from respeak.pipeline.types import Placed, Segment

log = logging.getLogger(__name__)

ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0
"""What a single `atempo` filter accepts; anything outside is expressed as a chain."""

SAMPLE_RATE = 24_000
"""Everything downstream of the TTS backends is 24 kHz mono."""


def fit(
    clip: Path | str,
    target_seconds: float,
    out: Path | str,
    lo: float | None = None,
    hi: float = 1.3,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[Path, float]:
    """Speed `clip` up towards `target_seconds`, at most by `hi`; return (written path, real length).

    A clip longer than its slot is sped up, never past `hi`: what does not fit stays too long and
    pushes the following segments later (`place()`), which sounds far better than a chipmunk. A clip
    *shorter* than its slot is left at its own speed — `lo` is an optional lower bound for a caller
    that really does want the clip stretched to fill the gap.
    """
    src = Path(clip)
    dest = Path(out)
    if target_seconds <= 0:
        raise ValueError(f"fit() needs a positive target, got {target_seconds!r} for {src.name}")
    if hi <= 0 or (lo is not None and (lo <= 0 or hi < lo)):
        raise ValueError(f"fit() needs 0 < lo <= hi, got lo={lo!r} hi={hi!r}")
    clip_seconds = ffmpeg.duration(src)
    if clip_seconds <= 0:
        raise ValueError(f"{src} is empty; nothing to fit")
    ratio = clip_seconds / target_seconds
    if ratio > 1.0:
        factor = min(ratio, hi)
    elif lo is not None:
        factor = max(ratio, lo)
    else:
        factor = 1.0
    chain = atempo_chain(factor)
    filters = ",".join(f"atempo={part:.6f}" for part in chain)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-i",
            str(src),
            "-filter:a",
            filters,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    actual = ffmpeg.duration(dest)
    log.debug(
        "fit %s: %.3f s → %.3f s (target %.3f s, factor %.3f via %s)",
        src.name,
        clip_seconds,
        actual,
        target_seconds,
        factor,
        filters,
    )
    return dest, actual


def atempo_chain(factor: float) -> list[float]:
    """Split a tempo factor into steps `atempo` accepts (each within 0.5–2.0)."""
    if factor <= 0:
        raise ValueError(f"tempo factor must be positive, got {factor!r}")
    parts: list[float] = []
    remaining = float(factor)
    while remaining > ATEMPO_MAX:
        parts.append(ATEMPO_MAX)
        remaining /= ATEMPO_MAX
    while remaining < ATEMPO_MIN:
        parts.append(ATEMPO_MIN)
        remaining /= ATEMPO_MIN
    parts.append(remaining)
    return parts


def stretch_factor(
    segments: Sequence[Segment],
    fitted_seconds: Sequence[float],
    video_seconds: float,
) -> float:
    """The smallest factor the video must be slowed by for every clip to fit (docs/flow.md B4.6).

        s = max(1, max_i  Σ_{j≥i} d_j / (V − start_i))

    Why this is the whole answer: on a timeline slowed by `s` the cascade ends at
    `max_i (s·start_i + Σ_{j≥i} d_j)`, so "the last word lands before the picture stops" is
    `s·start_i + Σ_{j≥i} d_j ≤ s·V` for every sentence — one division each, no search.

    1.0 means everything already fits. A sentence that starts after the video ends cannot be rescued
    by any slowdown (`s·start_i > s·V` for every `s`), so it is skipped as a divisor — its clip still
    counts towards the tail every earlier sentence has to carry.
    """
    if len(segments) != len(fitted_seconds):
        raise ValueError(
            f"stretch_factor() got {len(segments)} segments but {len(fitted_seconds)} clip lengths"
        )
    if video_seconds <= 0:
        raise ValueError(f"stretch_factor() needs a positive video length, got {video_seconds!r}")
    lengths = [float(seconds) for seconds in fitted_seconds]
    if any(seconds < 0 for seconds in lengths):
        raise ValueError(f"stretch_factor() got a negative clip length in {lengths!r}")
    factor = 1.0
    tail = 0.0  # Σ_{j≥i} d_j, accumulated from the back so the whole sweep is one pass.
    for segment, seconds in zip(reversed(segments), reversed(lengths), strict=True):
        tail += seconds
        room = video_seconds - float(segment.start)
        if room <= 0:  # starts at or after the end of the picture: never fits, never a divisor
            continue
        factor = max(factor, tail / room)
    return factor


def place(
    segments: list[Segment],
    fitted: list[tuple[Path, float]],
    *,
    stretch: float = 1.0,
) -> list[Placed]:
    """Lay the fitted clips on the timeline, never overlapping (docs/flow.md B4.6).

    `segments[i].text` is what the subtitle will say, so callers pass the *translated* segments.
    `stretch` is the factor the picture is being slowed by: every original start moves to
    `stretch · seg.start`, which keeps each sentence over the shot it belongs to; the cascade that
    pushes a long clip into the next slot is unchanged.
    """
    if len(segments) != len(fitted):
        raise ValueError(f"place() got {len(segments)} segments but {len(fitted)} fitted clips")
    if stretch <= 0:
        raise ValueError(f"place() needs a positive stretch factor, got {stretch!r}")
    placed: list[Placed] = []
    prev_end = 0.0
    for segment, (path, seconds) in zip(segments, fitted, strict=True):
        if seconds < 0:
            raise ValueError(f"fitted clip {path} reports a negative length ({seconds})")
        start = max(float(stretch) * float(segment.start), prev_end)
        end = start + float(seconds)
        placed.append(Placed(path=Path(path), start=start, end=end, text=segment.text))
        prev_end = end
    return placed


class AssembleResult(NamedTuple):
    """`dubbed.wav` and what the video's length cost it (docs/flow.md B4.6)."""

    path: Path
    dropped_seconds: float = 0.0
    """Speech that started after the video ended and is not in the dub at all."""
    trimmed_seconds: float = 0.0
    """Speech cut off the end of clips that only partly fitted."""
    dropped_clips: int = 0
    """How many segments were dropped whole."""

    @property
    def lost_seconds(self) -> float:
        return self.dropped_seconds + self.trimmed_seconds


def assemble(
    placed: list[Placed],
    total_seconds: float,
    out: Path | str,
    sample_rate: int = SAMPLE_RATE,
) -> AssembleResult:
    """Mix the placed clips onto silence, writing exactly `total_seconds` of 24 kHz mono WAV.

    Returns the path together with what did not fit, so the job can say so instead of shipping a
    silently truncated dub.
    """
    dest = Path(out)
    if total_seconds <= 0:
        raise ValueError(f"assemble() needs a positive duration, got {total_seconds!r}")
    frames = int(round(total_seconds * sample_rate))
    canvas = np.zeros(frames, dtype=np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dropped_frames = 0
    trimmed_frames = 0
    dropped_clips = 0
    with tempfile.TemporaryDirectory(prefix="respeak-assemble-") as scratch:
        for index, item in enumerate(placed):
            samples = _mono_samples(item.path, sample_rate, Path(scratch) / f"r{index:05d}.wav")
            offset = int(round(item.start * sample_rate))
            if offset >= frames:
                log.warning(
                    "clip %s starts at %.3f s, past the %.3f s end; dropped",
                    item.path.name,
                    item.start,
                    total_seconds,
                )
                dropped_frames += int(samples.size)
                dropped_clips += 1
                continue
            room = frames - offset
            if samples.size > room:
                log.info(
                    "clip %s trimmed by %.3f s to fit the video",
                    item.path.name,
                    (samples.size - room) / sample_rate,
                )
                trimmed_frames += int(samples.size - room)
                samples = samples[:room]
            canvas[offset : offset + samples.size] += samples
    np.clip(canvas, -1.0, 1.0, out=canvas)
    sf.write(str(dest), canvas, sample_rate, subtype="PCM_16")
    return AssembleResult(
        path=dest,
        dropped_seconds=dropped_frames / sample_rate,
        trimmed_seconds=trimmed_frames / sample_rate,
        dropped_clips=dropped_clips,
    )


def _mono_samples(path: Path, sample_rate: int, scratch: Path) -> np.ndarray:
    """Read a clip as mono float32 at `sample_rate`, resampling through ffmpeg when it differs."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"fitted clip {source} is missing; cannot assemble the dub")
    with sf.SoundFile(str(source)) as handle:
        rate = handle.samplerate
    if rate != sample_rate:
        ffmpeg.run(
            ["-i", str(source), "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(scratch)]
        )
        source = scratch
    data, _ = sf.read(str(source), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        return data.mean(axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(data[:, 0], dtype=np.float32)
