"""ffmpeg / ffprobe / deno helpers (flow.md B4).

Every piece of media work in the pipeline goes through one of these subprocess wrappers: there is no
moviepy, no OpenCV and no pydub anywhere (decisions.md D-05). Failures raise `FFmpegError` carrying the
tail of stderr — nothing here ever returns None to signal a problem.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from respeak.pipeline.types import Probe

log = logging.getLogger(__name__)

STDERR_TAIL_LINES = 30
"""How many trailing lines of stderr a FFmpegError carries."""

_TOOLS = ("ffmpeg", "ffprobe")


class FFmpegError(RuntimeError):
    """An ffmpeg or ffprobe invocation failed, timed out, or was not found on PATH."""


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


def run(args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg (or ffprobe) and return the completed process; raise `FFmpegError` on any failure.

    `args` may start with "ffmpeg"/"ffprobe" to choose the tool; anything else is treated as ffmpeg
    arguments. `-hide_banner` is always added, plus `-nostdin -y` for ffmpeg (ffprobe rejects both).
    """
    if not args:
        raise ValueError("ffmpeg.run() needs at least one argument")
    binary, rest = _split_binary(args)
    flags = ["-hide_banner"] if binary == "ffprobe" else ["-hide_banner", "-nostdin", "-y"]
    cmd = [binary, *flags, *(str(a) for a in rest)]
    if shutil.which(binary) is None:
        ensure_binaries()
    log.debug("running: %s", shlex.join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(
            f"{binary} is not on PATH. Install it, or make sure ensure_binaries() ran first."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{binary} timed out after {timeout} s: {shlex.join(cmd)}") from exc
    if proc.returncode != 0:
        raise FFmpegError(_failure_message(binary, cmd, proc.returncode, proc.stderr))
    return proc


def probe(path: Path | str) -> dict[str, Any]:
    """`ffprobe -print_format json -show_format -show_streams` as a dict."""
    src = Path(path)
    if not src.exists():
        raise FFmpegError(f"cannot probe {src}: file does not exist")
    proc = run(
        ["ffprobe", "-loglevel", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)]
    )
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned unreadable JSON for {src}: {exc}") from exc
    if not isinstance(info, dict):
        raise FFmpegError(f"ffprobe returned unexpected JSON for {src}: {type(info).__name__}")
    return info


def duration(path: Path | str) -> float:
    """Length of the file in seconds (container duration, falling back to the longest stream)."""
    return _duration_of(probe(path), Path(path))


def probe_summary(path: Path | str) -> Probe:
    """Duration, frame size, container title and whether the file carries a real video stream."""
    src = Path(path)
    info = probe(src)
    streams = video_streams(info)
    first = streams[0] if streams else {}
    return Probe(
        duration=_duration_of(info, src),
        width=int(first.get("width") or 0),
        height=int(first.get("height") or 0),
        title=_container_title(info),
        has_video=bool(streams),
    )


def video_streams(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Real video streams from a `probe()` dict — cover art (`attached_pic`) does not count."""
    out: list[dict[str, Any]] = []
    for stream in info.get("streams") or []:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        disposition = stream.get("disposition") or {}
        if disposition.get("attached_pic"):
            continue
        out.append(stream)
    return out


def audio_streams(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Audio streams from a `probe()` dict."""
    return [
        s for s in (info.get("streams") or []) if isinstance(s, dict) and s.get("codec_type") == "audio"
    ]


def _split_binary(args: list[str]) -> tuple[str, list[str]]:
    head = str(args[0])
    if Path(head).name in _TOOLS or Path(head).stem in _TOOLS:
        return head, [str(a) for a in args[1:]]
    return "ffmpeg", [str(a) for a in args]


def _failure_message(binary: str, cmd: list[str], code: int, stderr: str) -> str:
    tail = "\n".join((stderr or "").strip().splitlines()[-STDERR_TAIL_LINES:])
    body = tail or "(no stderr)"
    return f"{binary} failed with exit code {code}\ncommand: {shlex.join(cmd)}\n{body}"


def _duration_of(info: dict[str, Any], src: Path) -> float:
    candidates: list[float] = []
    raw = (info.get("format") or {}).get("duration")
    value = _as_float(raw)
    if value is not None:
        candidates.append(value)
    for stream in info.get("streams") or []:
        if not isinstance(stream, dict):
            continue
        value = _as_float(stream.get("duration"))
        if value is not None:
            candidates.append(value)
    if not candidates:
        raise FFmpegError(f"ffprobe reported no duration for {src}")
    return max(candidates)


def _as_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:  # NaN / inf / negative
        return None
    return value


def _container_title(info: dict[str, Any]) -> str | None:
    tags = (info.get("format") or {}).get("tags") or {}
    if not isinstance(tags, dict):
        return None
    for key, value in tags.items():
        if str(key).lower() == "title" and isinstance(value, str) and value.strip():
            return value.strip()
    return None
