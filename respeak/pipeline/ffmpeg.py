"""ffmpeg / ffprobe / deno helpers (flow.md B4).

WP-B extends this module; ensure_binaries() is used at startup.
"""

from __future__ import annotations

import logging
import shutil

log = logging.getLogger(__name__)


def ensure_binaries() -> dict[str, bool]:
    """Put the pip-provided ffmpeg/ffprobe/deno on PATH when the system has none; report what is reachable."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        try:
            import static_ffmpeg

            static_ffmpeg.add_paths()
        except Exception as exc:  # pragma: no cover - only when the package is missing
            log.warning("static_ffmpeg unavailable: %s", exc)
    if shutil.which("deno") is None:
        try:
            import deno  # noqa: F401 - the package installs a `deno` console script next to python

        except Exception as exc:  # pragma: no cover
            log.warning("deno package unavailable: %s", exc)
    found = {name: shutil.which(name) is not None for name in ("ffmpeg", "ffprobe", "deno")}
    missing = [k for k, v in found.items() if not v]
    if missing:
        log.warning("binaries not found on PATH: %s", ", ".join(missing))
    return found
