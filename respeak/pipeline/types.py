"""Shared types for the pipeline (flow.md B4).

Owned by the orchestrator; stages import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(slots=True)
class Transcript:
    language: str
    segments: list[Segment]


@dataclass(slots=True)
class Probe:
    duration: float
    width: int
    height: int
    title: str | None
    has_video: bool


@dataclass(slots=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Placed:
    path: Path
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Voice:
    id: str
    name: str


@dataclass(slots=True)
class BackendInfo:
    name: str
    installed: bool
    languages: set[str]
    cloning: bool
    reason: str | None = None
    voices: dict[str, list[Voice]] = field(default_factory=dict)
