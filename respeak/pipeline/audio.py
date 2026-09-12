"""Fit each spoken clip to its slot, place it on the timeline, assemble `dubbed.wav`.

flow.md B4.6, plan.md 1.7, decisions.md D-09 (per-segment `atempo`, never a global speed change).

    slot_i    = seg.end - seg.start
    factor_i  = clamp(clip_seconds / slot_i, 0.8, 1.3)      pitch-preserving, chained beyond 0.5–2.0
    start_i   = max(seg.start, prev_end)                    a long clip pushes the next one later
    end_i     = start_i + fitted_seconds

`assemble()` mixes the fitted clips onto silence of exactly `total_seconds`, so `dubbed.wav` is always
the length of the video — the mux never has to guess.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

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
    lo: float = 0.8,
    hi: float = 1.3,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[Path, float]:
    """Speed `clip` towards `target_seconds` within [lo, hi] and return (written path, real length).

    The factor is deliberately clamped: a clip that cannot fit its slot stays too long and pushes the
    following segments later (`place()`), which sounds far better than a chipmunk.
    """
    src = Path(clip)
    dest = Path(out)
    if target_seconds <= 0:
        raise ValueError(f"fit() needs a positive target, got {target_seconds!r} for {src.name}")
    if lo <= 0 or hi < lo:
        raise ValueError(f"fit() needs 0 < lo <= hi, got lo={lo!r} hi={hi!r}")
    clip_seconds = ffmpeg.duration(src)
    if clip_seconds <= 0:
        raise ValueError(f"{src} is empty; nothing to fit")
    factor = min(max(clip_seconds / target_seconds, lo), hi)
    chain = atempo_chain(factor)
    filters = ",".join(f"atempo={part:.6f}" for part in chain)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-i", str(src),
            "-filter:a", filters,
            "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(dest),
        ]
    )
    actual = ffmpeg.duration(dest)
    log.debug(
        "fit %s: %.3f s → %.3f s (target %.3f s, factor %.3f via %s)",
        src.name, clip_seconds, actual, target_seconds, factor, filters,
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


def place(segments: list[Segment], fitted: list[tuple[Path, float]]) -> list[Placed]:
    """Lay the fitted clips on the timeline, never overlapping (flow.md B4.6).

    `segments[i].text` is what the subtitle will say, so callers pass the *translated* segments.
    """
    if len(segments) != len(fitted):
        raise ValueError(f"place() got {len(segments)} segments but {len(fitted)} fitted clips")
    placed: list[Placed] = []
    prev_end = 0.0
    for segment, (path, seconds) in zip(segments, fitted, strict=True):
        if seconds < 0:
            raise ValueError(f"fitted clip {path} reports a negative length ({seconds})")
        start = max(float(segment.start), prev_end)
        end = start + float(seconds)
        placed.append(Placed(path=Path(path), start=start, end=end, text=segment.text))
        prev_end = end
    return placed


def assemble(
    placed: list[Placed],
    total_seconds: float,
    out: Path | str,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Mix the placed clips onto silence and write exactly `total_seconds` of 24 kHz mono WAV."""
    dest = Path(out)
    if total_seconds <= 0:
        raise ValueError(f"assemble() needs a positive duration, got {total_seconds!r}")
    frames = int(round(total_seconds * sample_rate))
    canvas = np.zeros(frames, dtype=np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="respeak-assemble-") as scratch:
        for index, item in enumerate(placed):
            samples = _mono_samples(item.path, sample_rate, Path(scratch) / f"r{index:05d}.wav")
            offset = int(round(item.start * sample_rate))
            if offset >= frames:
                log.warning("clip %s starts at %.3f s, past the %.3f s end; dropped",
                            item.path.name, item.start, total_seconds)
                continue
            room = frames - offset
            if samples.size > room:
                log.info("clip %s trimmed by %.3f s to fit the video",
                         item.path.name, (samples.size - room) / sample_rate)
                samples = samples[:room]
            canvas[offset:offset + samples.size] += samples
    np.clip(canvas, -1.0, 1.0, out=canvas)
    sf.write(str(dest), canvas, sample_rate, subtype="PCM_16")
    return dest


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
