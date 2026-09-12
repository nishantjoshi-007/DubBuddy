"""TTS backend registry (flow.md B4.5).

WP-D replaces the placeholder table with real backends.
"""

from __future__ import annotations

from ...config import Settings
from ..types import BackendInfo
from .base import TTSBackend

# Placeholder until WP-D lands: what Kokoro speaks, without importing it.
_KOKORO_LANGUAGES = {"en", "es", "fr", "hi", "it", "ja", "pt", "zh"}


def available_backends(settings: Settings) -> dict[str, BackendInfo]:
    return {
        "kokoro": BackendInfo(name="kokoro", installed=True, languages=set(_KOKORO_LANGUAGES), cloning=False),
        "chatterbox": BackendInfo(
            name="chatterbox",
            installed=False,
            languages=set(),
            cloning=True,
            reason="not installed: run `uv sync --extra clone`",
        ),
    }


def get_backend(name: str, settings: Settings) -> TTSBackend:  # pragma: no cover - WP-D
    raise NotImplementedError("TTS backends arrive with WP-D")


__all__ = ["TTSBackend", "available_backends", "get_backend"]
