"""Input stage: YouTube and upload both end as `source.mp4` + `source.wav` + `reference.wav`.

flow.md B4.1 / B4.2, plan.md 1.3, decisions.md D-11 (limits), D-23/D-26 (uploads).

One `extract_info` call answers the probe question for YouTube (never the three round-trips of the old
system), and every failure raises `InputError` with a message a user can act on.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from respeak.config import Settings
from respeak.pipeline import ffmpeg
from respeak.pipeline.ffmpeg import FFmpegError
from respeak.pipeline.types import Probe

log = logging.getLogger(__name__)

REFERENCE_SECONDS = 30.0
"""Longest speaker reference clip handed to a cloning TTS backend (flow.md B4.2)."""

SOURCE_SAMPLE_RATE = 16_000
"""What faster-whisper wants (flow.md B4.2)."""

REFERENCE_SAMPLE_RATE = 24_000
"""What the TTS backends speak at (flow.md B4 `TTSBackend`)."""

MP4_VIDEO_CODECS: frozenset[str] = frozenset({"h264", "hevc"})
MP4_AUDIO_CODECS: frozenset[str] = frozenset({"aac", "mp3"})
"""What an `.mp4` may hold and still play in a browser `<video>` (flow.md B4.8)."""


class InputError(Exception):
    """The source cannot be used: too long, not a video, private, blocked, or undownloadable."""


# --------------------------------------------------------------------------------------------------
# URL safety
# --------------------------------------------------------------------------------------------------


def check_public_url(url: str) -> None:
    """Raise `InputError` unless `url` is an http(s) URL pointing at a public address.

    yt-dlp falls back to a generic extractor for anything it does not recognise, so without this a
    posted `http://169.254.169.254/latest/meta-data/` would be fetched by the server — cloud
    metadata, the LAN, or a service on localhost. The hostname is resolved here and every address it
    answers with has to be routable. A host that does not resolve at all is left to yt-dlp: it can
    reach nothing either.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise InputError("The URL must start with http:// or https://.")
    try:
        host = parsed.hostname
    except ValueError as exc:  # a malformed IPv6 literal
        raise InputError(f"That URL has an unreadable host name ({exc}).") from exc
    if not host:
        raise InputError("That URL has no host name.")

    literal = _as_ip(host)
    if literal is not None:
        addresses = [literal]
    else:
        addresses = _resolve(host)
        if not addresses:
            log.info("could not resolve %s; leaving the decision to yt-dlp", host)
            return
    for address in addresses:
        if not _is_public(address):
            raise InputError(
                f"{host} points at {address}, which is on this machine or a private network. "
                "Respeak only downloads from public addresses."
            )


def _as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The host as an IP address when it is written as a literal, else None."""
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every address `host` resolves to; an empty list when the name does not resolve at all."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        log.info("getaddrinfo(%s) failed: %s", host, exc)
        return []
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        candidate = _as_ip(str(info[4][0]))
        if candidate is not None:
            addresses.append(candidate)
    return addresses


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """False for loopback, private, link-local, multicast, reserved and unspecified addresses."""
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return not (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def enforce_duration(probe: Probe, settings: Settings) -> None:
    """Raise `InputError` when the source is longer than `MAX_VIDEO_SECONDS` (D-11)."""
    limit = float(settings.max_video_seconds)
    if probe.duration > limit:
        raise InputError(
            f"Video is {probe.duration:.0f} s long; the limit is {settings.max_video_seconds} s. "
            "Trim it or raise MAX_VIDEO_SECONDS."
        )


# --------------------------------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------------------------------


def probe_youtube(url: str, settings: Settings) -> Probe:
    """One `extract_info(download=False)` → title, duration, frame size. Playlists are rejected."""
    check_public_url(url)  # api._validate checks this too; the pipeline never trusts its caller
    ffmpeg.ensure_binaries()  # yt-dlp needs deno as its JS runtime
    import yt_dlp

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    _add_cookiefile(opts, settings)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises DownloadError and friends
        raise InputError(_readable_ytdlp_error(url, exc)) from exc
    if not isinstance(info, dict):
        raise InputError(f"YouTube returned nothing usable for {url}.")
    if info.get("_type") == "playlist" or info.get("entries") is not None:
        raise InputError("That link is a playlist or channel. Paste the URL of a single video.")
    duration = info.get("duration")
    if duration is None:
        raise InputError(
            "That video has no duration (a live stream or premiere?). Respeak needs a finished video."
        )
    try:
        seconds = float(duration)
    except (TypeError, ValueError) as exc:
        raise InputError(f"YouTube reported an unreadable duration ({duration!r}).") from exc
    title = info.get("title")
    return Probe(
        duration=seconds,
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        title=title if isinstance(title, str) and title.strip() else None,
        has_video=True,
    )


def fetch_youtube(url: str, dest_dir: Path, settings: Settings) -> Path:
    """Download at most `MAX_HEIGHT` and return `dest_dir/source.mp4` (D-11, flow.md B4.2)."""
    ffmpeg.ensure_binaries()
    import yt_dlp

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "source.mp4"
    height = int(settings.max_height)
    fmt = (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]"
    )
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "retries": 3,
        "fragment_retries": 3,
        "noprogress": True,
    }
    _add_cookiefile(opts, settings)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise InputError(_readable_ytdlp_error(url, exc)) from exc
    if target.exists() and target.stat().st_size > 0:
        # The `bestvideo+bestaudio` fallback merges whatever YouTube served — often VP9 + Opus —
        # into the .mp4 container, and that plays in almost no <video> element.
        return ensure_mp4_codecs(target)
    produced = sorted(p for p in dest_dir.glob("source.*") if p.is_file() and p.stat().st_size > 0)
    if not produced:
        raise InputError(f"yt-dlp finished but wrote no file for {url}.")
    other = produced[0]
    log.info("yt-dlp produced %s; re-muxing to source.mp4", other.name)
    remux(other, target)
    other.unlink(missing_ok=True)
    return target


def _add_cookiefile(opts: dict[str, Any], settings: Settings) -> None:
    cookies = settings.ytdlp_cookies_file
    if cookies is None:
        return
    path = Path(cookies)
    if not path.exists():
        log.warning("YTDLP_COOKIES_FILE points at %s, which does not exist; ignoring it", path)
        return
    opts["cookiefile"] = str(path)


def _readable_ytdlp_error(url: str, exc: Exception) -> str:
    text = str(exc).replace("ERROR: ", "").strip()
    low = text.lower()
    if "not a bot" in low or "sign in to confirm" in low:
        return (
            "YouTube asked this machine to prove it is not a bot. Set YTDLP_COOKIES_FILE to a cookies "
            f"export from a signed-in browser, or upload the file instead. ({text})"
        )
    if "private video" in low:
        return f"That video is private. ({text})"
    if "members-only" in low or "join this channel" in low:
        return f"That video is members-only. ({text})"
    if "age" in low and "confirm" in low:
        return f"That video is age-restricted; cookies are needed. ({text})"
    if "unavailable" in low or "removed" in low or "does not exist" in low:
        return f"That video is unavailable. ({text})"
    if "is not a valid url" in low or "unsupported url" in low:
        return f"{url} is not a URL yt-dlp can handle. ({text})"
    return f"YouTube download failed for {url}: {text}"


# --------------------------------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------------------------------


def validate_upload(path: Path | str, settings: Settings) -> Probe:
    """ffprobe must succeed on the upload, show exactly one video stream, and fit the limits (D-26)."""
    src = Path(path)
    if not src.exists() or src.stat().st_size == 0:
        raise InputError(f"The uploaded file is missing or empty ({src.name}).")
    size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise InputError(f"The upload is {size_mb:.0f} MB; the limit is {settings.max_upload_mb} MB.")
    try:
        info = ffmpeg.probe(src)
    except FFmpegError as exc:
        raise InputError(
            f"That file is not a video ffmpeg can read ({src.name}). Try MP4, MOV, MKV or WebM."
        ) from exc
    streams = ffmpeg.video_streams(info)
    if not streams:
        raise InputError(f"{src.name} has no video stream (audio-only files are not supported yet).")
    if len(streams) > 1:
        raise InputError(f"{src.name} has {len(streams)} video streams; Respeak handles one.")
    if not ffmpeg.audio_streams(info):
        raise InputError(f"{src.name} has no audio stream, so there is nothing to dub.")
    try:
        probe = ffmpeg.probe_summary(src)
    except FFmpegError as exc:
        raise InputError(f"{src.name} has no readable duration; the file looks damaged.") from exc
    enforce_duration(probe, settings)
    return probe


def remux(src: Path | str, dest: Path | str) -> Path:
    """Stream-copy into a faststart MP4; re-encode only when the copy is refused (D-26)."""
    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg.run(["-i", str(source), "-c", "copy", "-movflags", "+faststart", str(target)])
    except FFmpegError as exc:
        log.info("stream copy failed for %s, re-encoding: %s", source.name, exc)
    else:
        # A stream copy is happy to put VP9 + Opus (a WebM upload) inside an .mp4; the browser is not.
        return ensure_mp4_codecs(target)
    target.unlink(missing_ok=True)
    ffmpeg.run(
        [
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    return target


def ensure_mp4_codecs(path: Path | str) -> Path:
    """Return a file with MP4-native codecs, re-encoding `path` to h264 + aac when it is not one.

    The dub is muxed back into the source container later (flow.md B4.8), and with `burn=False` the
    video stream is copied, so a VP9/Opus source would come out the other end as an `out.mp4` that
    Safari and most phones refuse to play. The re-encode is written to a sibling file; when the
    input is already an `.mp4` it takes its place, so `source.mp4` stays `source.mp4`.
    """
    src = Path(path)
    info = ffmpeg.probe(src)
    video = ffmpeg.video_streams(info)
    audio = ffmpeg.audio_streams(info)
    video_codec = str((video[0] if video else {}).get("codec_name") or "").lower()
    audio_codec = str((audio[0] if audio else {}).get("codec_name") or "").lower()
    if video_codec in MP4_VIDEO_CODECS and audio_codec in MP4_AUDIO_CODECS:
        return src
    if not video:
        # Nothing to re-encode into a video file; validate_upload() is the one that says so.
        log.warning("%s has no video stream; leaving it alone", src.name)
        return src

    recoded = src.with_name(f"{src.stem}.h264.mp4")
    log.info(
        "%s holds %s + %s, which mp4 cannot carry safely; re-encoding to h264 + aac",
        src.name,
        video_codec or "no video",
        audio_codec or "no audio",
    )
    ffmpeg.run(
        [
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(recoded),
        ]
    )
    if src.suffix.lower() == ".mp4":
        os.replace(recoded, src)
        return src
    return recoded


# --------------------------------------------------------------------------------------------------
# Audio extraction
# --------------------------------------------------------------------------------------------------


def extract_audio(source_mp4: Path | str, dest_dir: Path | str) -> tuple[Path, Path]:
    """→ (`source.wav` 16 kHz mono for ASR, `reference.wav` 24 kHz mono, the loudest ≤ 30 s window)."""
    source = Path(source_mp4)
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_wav = out_dir / "source.wav"
    reference_wav = out_dir / "reference.wav"

    ffmpeg.run(
        [
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SOURCE_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(source_wav),
        ]
    )
    total = ffmpeg.duration(source_wav)
    if total <= 0:
        raise InputError(f"{source.name} carries no audio to dub.")
    window = min(REFERENCE_SECONDS, total)
    start = _loudest_window_start(source_wav, window, total)
    ffmpeg.run(
        [
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{window:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(REFERENCE_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(reference_wav),
        ]
    )
    if not reference_wav.exists() or reference_wav.stat().st_size == 0:
        # Some containers dislike an input seek; fall back to the head of the extracted WAV.
        ffmpeg.run(
            [
                "-i",
                str(source_wav),
                "-t",
                f"{window:.3f}",
                "-ac",
                "1",
                "-ar",
                str(REFERENCE_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(reference_wav),
            ]
        )
    return source_wav, reference_wav


def _loudest_window_start(wav: Path, window: float, total: float) -> float:
    """Start of the loudest `window` seconds, measured one second at a time; 0.0 when in doubt."""
    if window >= total:
        return 0.0
    try:
        import numpy as np
        import soundfile as sf

        energies: list[float] = []
        with sf.SoundFile(str(wav)) as handle:
            rate = handle.samplerate
            for block in handle.blocks(blocksize=rate, dtype="float32", always_2d=True):
                energies.append(float(np.mean(np.square(block))) if block.size else 0.0)
        span = max(1, int(math.floor(window)))
        if len(energies) <= span:
            return 0.0
        best_index, best_score = 0, -1.0
        running = sum(energies[:span])
        best_score = running
        for i in range(1, len(energies) - span + 1):
            running += energies[i + span - 1] - energies[i - 1]
            if running > best_score:
                best_score, best_index = running, i
        start = float(best_index)
    except Exception as exc:  # pragma: no cover - analysis is an optimisation, never a hard failure
        log.info("loudest-window analysis failed for %s (%s); using the first %.0f s", wav, exc, window)
        return 0.0
    return max(0.0, min(start, total - window))
