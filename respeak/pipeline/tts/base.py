"""The TTS backend contract (flow.md B4).

Owned by the orchestrator; kokoro.py and chatterbox.py implement it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..types import Voice


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    cloning: bool

    def languages(self) -> set[str]:
        """ISO-639-1 codes this backend can speak."""
        ...

    def voices(self) -> dict[str, list[Voice]]:
        """Preset voices per language code. Empty for backends that clone the original speaker."""
        ...

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
        """Write a 24 kHz mono WAV for `text` in `lang` to `out` and return it.

        `reference_wav` is a short clip of the original speaker; backends without cloning ignore it.
        `voice` is one of this backend's voice ids for `lang`, or None for the backend's default.
        """
        ...
