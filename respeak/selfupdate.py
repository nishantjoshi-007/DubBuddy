"""yt-dlp self-update and the version report.

Video sites change more often than this repo is released, and a broken download is almost always
fixed by a newer yt-dlp. `YTDLP_AUTO_UPDATE=true` (the default inside the Docker image) therefore asks the
installer for the newest yt-dlp at startup.

Nothing here raises: an update is a convenience, never a reason for the server not to start. Everything is
plain and synchronous; `main.py` runs :func:`update_ytdlp` in a daemon thread so startup never waits on it.

Versions are read with :mod:`importlib.metadata`, which only parses the `*.dist-info` on disk — asking
`torch.__version__` would import half a gigabyte of model code onto the web process (docs/flow.md B2:
the server answers in under a second).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import time

from . import __version__

log = logging.getLogger(__name__)

PACKAGE = "yt-dlp"

#: Reported by `/api/health` and logged at startup; "respeak" is added from `__version__`.
PACKAGES: tuple[str, ...] = ("yt-dlp", "faster-whisper", "kokoro", "torch")

UNKNOWN = "unknown"

#: `uv pip install` prints ` + yt-dlp==2026.8.19`, pip prints `Successfully installed yt-dlp-2026.8.19`.
_VERSION_IN_OUTPUT = re.compile(r"yt[-_]dlp[-=]=?([0-9][^\s,;)]*)", re.IGNORECASE)


def package_version(name: str) -> str:
    """The installed version of `name`, or ``"unknown"`` when its metadata is not on disk."""
    try:
        return importlib.metadata.version(name)
    except Exception as exc:  # PackageNotFoundError, or a half-written dist-info during an update
        log.debug("no installed version for %s: %s", name, exc)
        return UNKNOWN


def ytdlp_version() -> str:
    """The installed yt-dlp version, or ``"unknown"``. Never raises."""
    return package_version(PACKAGE)


def component_versions() -> dict[str, str]:
    """`{"respeak": ..., "yt-dlp": ..., "faster-whisper": ..., "kokoro": ..., "torch": ...}`.

    Cheap enough for `/api/health` on every request: five dist-info reads, no heavy import.
    """
    versions = {"respeak": __version__}
    versions.update({name: package_version(name) for name in PACKAGES})
    return versions


def version_from_output(text: str) -> str | None:
    """The yt-dlp version an installer reported in its output, or None if it named none."""
    match = _VERSION_IN_OUTPUT.search(text or "")
    return match.group(1) if match else None


def _pip_available() -> bool:
    """True when `python -m pip` can run in this interpreter (uv venvs often have no pip)."""
    try:
        return importlib.util.find_spec("pip") is not None
    except Exception:  # a broken sys.path entry must not stop the server
        return False


def _installer_command() -> list[str] | None:
    """The command that upgrades yt-dlp in *this* interpreter, or None when there is no installer.

    uv is preferred and is what the Docker image has; `--python sys.executable` keeps it pointed at the
    running venv even when uv would otherwise pick the project's own.
    """
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, "-U", PACKAGE]
    if _pip_available():
        return [sys.executable, "-m", "pip", "install", "-U", PACKAGE]
    return None


def update_ytdlp(timeout: int = 120) -> str | None:
    """Upgrade yt-dlp in place; return the version now installed, or None if nothing was installed.

    Never raises and never exits: a failed upgrade (offline, read-only site-packages, no installer)
    is logged and the server carries on with the yt-dlp it already has.

    The new code is picked up without a restart because `respeak/pipeline/inputs.py` imports `yt_dlp`
    lazily, inside `probe_url()` / `fetch_url()`: as long as the upgrade finishes before the
    first link job is picked up by a worker thread, that job imports the new version. A job that is
    already running keeps the module it imported — that one does need a restart.
    """
    before = ytdlp_version()
    command = _installer_command()
    if command is None:
        log.warning(
            "YTDLP_AUTO_UPDATE is on but neither uv nor pip is available; staying on yt-dlp %s", before
        )
        return None

    log.info("updating yt-dlp (installed: %s) with: %s", before, " ".join(command))
    started = time.monotonic()
    try:
        completed = subprocess.run(  # a fixed command, no shell, nothing from a request
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("the yt-dlp update timed out after %s s; staying on %s", timeout, before)
        return None
    except Exception as exc:  # OSError when the installer vanished between which() and run()
        log.warning("the yt-dlp update could not be started (%s); staying on %s", exc, before)
        return None

    elapsed = time.monotonic() - started
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    if completed.returncode != 0:
        log.warning(
            "the yt-dlp update failed (exit %s after %.1f s); staying on %s: %s",
            completed.returncode,
            elapsed,
            before,
            _tail(output),
        )
        return None

    # site-packages changed underneath us; importlib caches directory listings by mtime.
    importlib.invalidate_caches()
    after = ytdlp_version()
    if after == UNKNOWN:
        after = version_from_output(output) or before
    if after == before:
        log.info("yt-dlp is already up to date at %s (%.1f s)", after, elapsed)
    else:
        log.info("yt-dlp updated %s -> %s (%.1f s)", before, after, elapsed)
    return after


def _tail(output: str, lines: int = 5) -> str:
    """The last few lines of an installer's output, for a log message that still fits on a screen."""
    kept = [line for line in output.splitlines() if line.strip()][-lines:]
    return " | ".join(kept) if kept else "(no output)"


__all__ = [
    "PACKAGE",
    "PACKAGES",
    "UNKNOWN",
    "component_versions",
    "package_version",
    "update_ytdlp",
    "version_from_output",
    "ytdlp_version",
]
