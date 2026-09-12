"""Build `subs.srt` from the placed, translated cues (flow.md B4.7, plan.md 1.8).

The times come straight from `audio.place()`, so a cue always covers the audio it describes — there is
no second transcription pass, and no truncation to whole seconds (Part A's bug).
"""

from __future__ import annotations

import logging
import textwrap
from datetime import timedelta
from pathlib import Path

import srt

from respeak.pipeline.types import Cue

log = logging.getLogger(__name__)

MIN_CUE_SECONDS = 0.3
"""Shorter than this, a cue is collapsed into its neighbour: it would only flash."""

WRAP_WIDTH = 42
"""Characters per subtitle line before wrapping on a word boundary."""

MAX_LINES = 2
"""Never show more than two lines at once."""


def build_srt(
    cues: list[Cue],
    out: Path | str,
    *,
    min_seconds: float = MIN_CUE_SECONDS,
    width: int = WRAP_WIDTH,
    max_lines: int = MAX_LINES,
) -> Path:
    """Write `cues` as an SRT file with millisecond precision and return its path."""
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged = collapse_short(cues, min_seconds=min_seconds)
    subtitles: list[srt.Subtitle] = []
    for index, cue in enumerate(merged, start=1):
        start_ms = max(0, round(cue.start * 1000))
        end_ms = max(start_ms + 1, round(cue.end * 1000))
        subtitles.append(
            srt.Subtitle(
                index=index,
                start=timedelta(milliseconds=start_ms),
                end=timedelta(milliseconds=end_ms),
                content=wrap_text(cue.text, width=width, max_lines=max_lines),
            )
        )
    dest.write_text(srt.compose(subtitles), encoding="utf-8")
    log.debug("wrote %d cues to %s", len(subtitles), dest)
    return dest


def collapse_short(cues: list[Cue], *, min_seconds: float = MIN_CUE_SECONDS) -> list[Cue]:
    """Merge cues shorter than `min_seconds` into a neighbour; drop cues with no text."""
    usable = [c for c in cues if c.text and c.text.strip()]
    for cue in usable:
        if cue.end < cue.start:
            raise ValueError(f"cue ends before it starts: {cue.start:.3f} → {cue.end:.3f} ({cue.text!r})")
    kept: list[Cue] = []
    pending: Cue | None = None  # a short cue waiting for the next one to carry it
    for cue in usable:
        text = cue.text.strip()
        start, end = cue.start, cue.end
        if pending is not None:
            start = min(start, pending.start)
            text = f"{pending.text} {text}".strip()
            pending = None
        if end - start >= min_seconds:
            kept.append(Cue(start=start, end=end, text=text))
            continue
        if kept:  # glue it onto the cue before it
            previous = kept[-1]
            kept[-1] = Cue(
                start=previous.start,
                end=max(previous.end, end),
                text=f"{previous.text} {text}".strip(),
            )
        else:  # nothing before it yet: let the next cue carry it
            pending = Cue(start=start, end=end, text=text)
    if pending is not None:  # the only cue in the file, and it is short
        kept.append(Cue(start=pending.start, end=pending.start + min_seconds, text=pending.text))
    return kept


def wrap_text(text: str, *, width: int = WRAP_WIDTH, max_lines: int = MAX_LINES) -> str:
    """Hard-wrap on word boundaries to at most `max_lines` lines of about `width` characters."""
    collapsed = " ".join(str(text).split())
    if not collapsed:
        return ""
    limit = max(1, width)
    lines = textwrap.wrap(collapsed, width=limit, break_long_words=False, break_on_hyphens=False)
    while len(lines) > max_lines and limit < len(collapsed):
        limit += max(1, width // 4)
        lines = textwrap.wrap(collapsed, width=limit, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:  # only reachable for text without spaces; never drop words
        head = lines[: max_lines - 1]
        head.append(" ".join(lines[max_lines - 1 :]))
        lines = head
    return "\n".join(lines)
