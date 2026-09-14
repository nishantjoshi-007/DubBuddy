"""JSON API (flow.md B3, B5).

Every route here is cheap: the job routes read or write one small file and return. The work happens in
:class:`respeak.jobs.JobRunner`'s thread pool, never on the event loop.

The routes are plain ``def``, not ``async def``, on purpose: FastAPI then runs them in its threadpool,
so reading a status file, asking torch about CUDA or writing a 500 MB upload never blocks the loop.

The router carries full paths (``/api/…`` plus the one root-level ``POST /jobs``) so that ``main.py``
can include it unchanged.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
import shutil
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, NamedTuple
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings, get_settings
from .jobs import JobStore, get_runner, get_store
from .lang_codes import NAME_TO_CODE, SOURCE_LANGUAGES
from .pipeline.inputs import InputError, check_public_url
from .pipeline.tts import TTSError, available_backends
from .pipeline.tts.samples import ensure_sample
from .pipeline.types import BackendInfo
from .selfupdate import component_versions

log = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_FILENAME = "upload.bin"
UPLOAD_CHUNK_BYTES = 1024 * 1024
#: What the multipart envelope around the file itself may reasonably add (headers, other fields).
FORM_OVERHEAD_BYTES = 1024 * 1024
#: The one route whose body may be huge; the size guard only looks at this path.
CREATE_JOB_PATH = "/jobs"
_TRUE = {"true", "1", "on", "yes"}
_FALSE = {"false", "0", "off", "no"}
_AUTO = {"", "auto", "none", "null"}
_UNSAFE_NAME = re.compile(r"[^\w.\- ]+", re.UNICODE)
_SOURCE_CODES = frozenset(NAME_TO_CODE.values())


class Rejected(Exception):
    """A request that must not create (or must undo) a job."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def _norm_lang(raw: str | None) -> str:
    """``'pt-BR'`` → ``'pt'``. Backends and Argos both speak plain ISO-639-1."""
    return (raw or "").strip().lower().replace("_", "-").split("-")[0]


def safe_name(raw: str | None, fallback: str = "") -> str:
    """A filename-safe version of a client filename or a video title (decisions.md D-26).

    Separators and punctuation become ``_``; leading dots and underscores go, so the result can never
    be ``.``, ``..`` or a path. Titles keep their letters, including non-Latin ones.
    """
    if not raw:
        return fallback
    name = _UNSAFE_NAME.sub("_", str(raw)).strip(" ._")
    return name[:120] or fallback


# --------------------------------------------------------------------------- rate limit

#: What ``RATE_LIMIT_JOBS`` may be counted per, in seconds (``"10/hour"``, ``"3/minute"``, ``"100/day"``).
RATE_UNITS: dict[str, float] = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}
_RATE_RE = re.compile(r"^(\d+)\s*/\s*(\d*)\s*([a-z]+)$", re.IGNORECASE)
#: Forget a client whose bucket has been full and untouched for this many refill windows.
BUCKET_IDLE_WINDOWS = 2.0
#: Only bother pruning once the table is this big; a self-hosted server never gets there. It is also
#: the hard cap: a flood from thousands of addresses is evicted back down to it, oldest first.
BUCKET_PRUNE_AT = 1024


class RateLimit(NamedTuple):
    """``count`` job submissions allowed per ``seconds``."""

    count: int
    seconds: float


def parse_rate_limit(raw: str | None) -> RateLimit | None:
    """``"10/hour"`` → ``RateLimit(10, 3600.0)``; empty → ``None`` (the limit is off).

    Also accepts a multiple (``"5/2 hours"``) and a plural unit. Raises ``ValueError`` on anything
    else: a typo must not quietly leave a public deployment unlimited (plan.md 3.5).
    """
    text = (raw or "").strip()
    if not text:
        return None
    match = _RATE_RE.match(text)
    if match is None:
        raise ValueError(f"{raw!r} is not a rate like '10/hour', '3/minute' or '100/day'")
    count, multiple, unit = int(match.group(1)), int(match.group(2) or 1), match.group(3).lower()
    seconds = RATE_UNITS.get(unit.rstrip("s"))  # "hours" and "hour" both count per 3600 s
    if seconds is None:
        raise ValueError(f"{raw!r} counts per {unit!r}; use one of {', '.join(sorted(RATE_UNITS))}")
    if count < 1 or multiple < 1:
        raise ValueError(f"{raw!r} allows no jobs at all; use a count of 1 or more, or leave it empty")
    return RateLimit(count=count, seconds=seconds * multiple)


class RateLimiter:
    """One in-memory token bucket per client address (flow.md B6, plan.md 3.5).

    A bucket starts full, so the first ``count`` submissions from a fresh address always go through;
    it refills continuously at ``count / seconds`` per second, which is what makes the limit reset by
    itself without a timer. Thread-safe because FastAPI runs ``POST /jobs`` in its threadpool.

    Keys are parsed IP strings (:func:`client_address`), never raw header text, and the table is
    capped at ``BUCKET_PRUNE_AT`` entries: an IPv6 flood could otherwise name 2**64 distinct clients
    and the dict would be the leak. Forgetting a bucket only ever gives that address a *full* bucket
    back, so eviction is safe — it costs one extra allowed submission, not an unbounded process.
    """

    def __init__(self, limit: RateLimit) -> None:
        self.limit = limit
        self.rate = limit.count / limit.seconds
        self._lock = threading.Lock()
        #: client address -> (tokens left, when it was last seen)
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, client: str, now: float | None = None) -> float:
        """Take one token: ``0.0`` when the request may proceed, else the seconds left to wait."""
        moment = time.monotonic() if now is None else float(now)
        capacity = float(self.limit.count)
        with self._lock:
            tokens, seen = self._buckets.get(client, (capacity, moment))
            tokens = min(capacity, tokens + max(0.0, moment - seen) * self.rate)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, moment)
                self._prune(moment)
                return 0.0
            self._buckets[client] = (tokens, moment)
            return (1.0 - tokens) / self.rate

    def _prune(self, moment: float) -> None:
        """Drop addresses whose bucket has long since refilled, then cap the table.

        Called with the lock held. Pruning by age is the normal case; the cap is the backstop for a
        flood that arrives faster than ``BUCKET_IDLE_WINDOWS`` can age it out, and evicts the
        least-recently-seen entries — the ones whose buckets are closest to full anyway.
        """
        if len(self._buckets) <= BUCKET_PRUNE_AT:
            return
        idle = self.limit.seconds * BUCKET_IDLE_WINDOWS
        stale = [client for client, (_, seen) in self._buckets.items() if moment - seen > idle]
        for client in stale:
            del self._buckets[client]
        log.debug("rate limiter forgot %d idle client(s)", len(stale))

        excess = len(self._buckets) - BUCKET_PRUNE_AT
        if excess <= 0:
            return
        oldest = sorted(self._buckets, key=lambda client: self._buckets[client][1])[:excess]
        for client in oldest:
            del self._buckets[client]
        log.warning(
            "rate limiter table hit %d entries; evicted the %d oldest bucket(s)",
            BUCKET_PRUNE_AT + excess,
            excess,
        )


_rate_lock = threading.Lock()
#: ``(the RATE_LIMIT_JOBS string it was built from, the limiter)`` — rebuilt when the setting changes.
_rate_state: tuple[str, RateLimiter | None] | None = None


def get_limiter(settings: Settings) -> RateLimiter | None:
    """The process-wide limiter, or ``None`` when ``RATE_LIMIT_JOBS`` is empty. Raises on a typo."""
    global _rate_state
    raw = (settings.rate_limit_jobs or "").strip()
    with _rate_lock:
        if _rate_state is None or _rate_state[0] != raw:
            limit = parse_rate_limit(raw)
            if limit is not None:
                log.info("rate limiting POST /jobs: %d per %.0f s per client", limit.count, limit.seconds)
            _rate_state = (raw, RateLimiter(limit) if limit is not None else None)
        return _rate_state[1]


def reset_rate_limiter() -> None:
    """Drop the limiter and every bucket in it (tests, and a lifespan that restarts in-process)."""
    global _rate_state
    with _rate_lock:
        _rate_state = None


def parse_ip(raw: str | None) -> str | None:
    """``raw`` as a normalised IP string, or ``None`` when it is not an address at all.

    A bucket key has to be an address and nothing else: unparsed header text would let one flooder
    own as many buckets as it can type, and it is what ends up in the log line for a 429.
    """
    text = (raw or "").strip().strip("[]")  # "[2001:db8::1]" is a legal X-Forwarded-For entry
    if not text:
        return None
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def client_address(request: Request, trust_proxy: bool) -> str:
    """The address the limit is counted against (flow.md B6).

    ``X-Forwarded-For`` is a client-supplied header: honouring it without a proxy in front would let
    anyone reset their own bucket by inventing an address, so it is read only with ``TRUST_PROXY``.

    Read the **last** entry, not the first. A proxy *appends* the peer it saw, so the header a real
    deployment receives is ``<whatever the client sent>, <the address our proxy actually observed>``;
    trusting the leftmost entry would hand every flooder an unlimited supply of fresh buckets for the
    price of one header. Anything that does not parse as an IP falls back to the real peer.
    """
    if trust_proxy:
        forwarded = parse_ip((request.headers.get("x-forwarded-for") or "").rsplit(",", 1)[-1])
        if forwarded is not None:
            return forwarded
    client = request.client
    return parse_ip(client.host if client is not None else None) or "unknown"


def _rate_limit_response(request: Request, settings: Settings) -> JSONResponse | None:
    """``None`` when this address may submit, otherwise the 429 to return instead."""
    try:
        limiter = get_limiter(settings)
    except ValueError as exc:
        # Fail closed and loudly: a limit that cannot be parsed must not read as "no limit".
        log.error("RATE_LIMIT_JOBS is not usable (%s); refusing to create jobs", exc)
        return _error(500, f"this server's RATE_LIMIT_JOBS setting is not valid: {exc}")
    if limiter is None:
        return None
    client = client_address(request, settings.trust_proxy)
    wait = limiter.check(client)
    if wait <= 0.0:
        return None
    minutes = max(1, math.ceil(wait / 60.0))
    log.info("rate limit reached by %s; %.0f s to wait", client, wait)
    plural = "" if minutes == 1 else "s"
    return JSONResponse(
        status_code=429,
        content={"error": f"too many jobs from this address; try again in {minutes} minute{plural}"},
        headers={"Retry-After": str(max(1, math.ceil(wait)))},
    )


# --------------------------------------------------------------------------- info


@router.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "name": "respeak",
        "version": __version__,
        "device": settings.resolved_device(),
        "tts_backend": settings.tts_backend,
        "whisper_model": settings.whisper_model,
        "limits": settings.limits(),
        "jobs_dir": str(settings.jobs_dir),
        "binaries": {name: shutil.which(name) is not None for name in ("ffmpeg", "ffprobe", "deno")},
        # dist-info reads only (respeak.selfupdate) — importing torch here would cost a second and
        # half a gigabyte on a route that is polled. With YTDLP_AUTO_UPDATE on, this is where an
        # operator checks that the container really did pick up a newer yt-dlp (plan.md 2.0).
        "versions": component_versions(),
    }


@router.get("/api/backends")
def backends() -> dict[str, Any]:
    settings = get_settings()
    infos = available_backends(settings)
    return {
        "default": settings.tts_backend,
        "backends": [
            {
                "name": info.name,
                "installed": info.installed,
                "languages": sorted(info.languages),
                "cloning": info.cloning,
                "reason": info.reason,
                # {lang: [{id, name}]} — the Phase 3 voice picker. Empty for a cloning backend, so
                # the form can simply hide the select when the chosen language has no entry.
                "voices": {
                    lang: [{"id": voice.id, "name": voice.name} for voice in entries]
                    for lang, entries in sorted(info.voices.items())
                    if entries
                },
            }
            for info in infos.values()
        ],
        "source_languages": SOURCE_LANGUAGES,
        "allow_uploads": settings.allow_uploads,
    }


#: How long a browser may keep a voice sample. A voice never changes, so this is generous on purpose.
SAMPLE_CACHE_CONTROL = "public, max-age=86400"


@router.get("/api/voices/{backend}/{lang}/{voice}")
def voice_sample(backend: str, lang: str, voice: str) -> Any:
    """A short WAV of one voice saying one sentence, so the picker can be listened to (plan.md 3.3).

    A plain ``def``: the first request for a voice may load a model and synthesise for a second or
    two, and that belongs in the threadpool, never on the event loop. Afterwards it is a cached file.

    The three path segments are never allowed near the filesystem unchecked — ``ensure_sample``
    matches the voice id against the backend's own table — so a bad triple is a 400. A backend with
    no preset voices at all (Chatterbox clones the original speaker instead) is a 404: there is
    nothing here to preview and there never will be.
    """
    settings = get_settings()
    info = available_backends(settings).get((backend or "").strip().lower())
    if info is not None and not info.voices:
        return _error(404, f"the {info.name} backend has no preset voices to preview")
    try:
        path = ensure_sample(settings, backend, lang, voice)
    except ValueError as exc:
        return _error(400, str(exc))
    except TTSError as exc:
        log.exception("could not synthesise the %s/%s/%s voice sample", backend, lang, voice)
        return _error(500, f"could not synthesise a sample of this voice: {exc}")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": SAMPLE_CACHE_CONTROL})


# --------------------------------------------------------------------------- create


def check_voice(info: BackendInfo, target: str, voice: str | None) -> str | None:
    """The voice id to store, or ``None`` for the backend's own default (plan.md 3.3).

    Raises :class:`Rejected` naming the ids that *are* available, because an unknown id in a form
    is almost always a stale page or a typed CLI flag.
    """
    picked = (voice or "").strip()
    if not picked:
        return None
    offered = info.voices.get(target, [])
    if not offered:
        raise Rejected(400, f"the {info.name} backend has no preset voices for '{target}'")
    if picked not in {entry.id for entry in offered}:
        ids = ", ".join(entry.id for entry in offered)
        raise Rejected(
            400, f"the {info.name} backend has no voice '{picked}' for '{target}'; its voices are: {ids}"
        )
    return picked


def check_source_url(url: str | None) -> str:
    """The URL a YouTube job may be created from, or :class:`Rejected` (flow.md B4.1, F9).

    The host must resolve to a public address: yt-dlp's generic extractor would otherwise happily
    fetch ``http://169.254.169.254/…`` — anything this machine can reach — on a caller's behalf.
    The pipeline checks again in ``inputs.probe_youtube``; this one keeps the job from existing.
    """
    link = (url or "").strip()
    if not link:
        raise Rejected(400, "a YouTube URL is required")
    parsed = urlparse(link)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise Rejected(400, "the URL must start with http:// or https://")
    try:
        check_public_url(link)
    except InputError as exc:
        raise Rejected(400, str(exc)) from exc
    return link


def check_options(
    settings: Settings,
    to_lang: str | None,
    from_lang: str | None = None,
    backend: str | None = None,
    voice: str | None = None,
) -> dict[str, Any]:
    """Backend, target language, source-language override and voice — the part of the job options
    the form and the CLI check identically (flow.md B3, plan.md 3.2/3.3).

    Returns the ``options`` fragment for :meth:`respeak.jobs.JobStore.create`; raises
    :class:`Rejected` with a message meant to be shown to a person, whichever front end asked.
    """
    name = (backend or "").strip().lower() or settings.tts_backend
    infos = available_backends(settings)
    info = infos.get(name)
    if info is None:
        raise Rejected(400, f"unknown backend '{name}'; installed backends: {', '.join(sorted(infos))}")
    if not info.installed:
        raise Rejected(400, info.reason or f"the {name} backend is not installed")

    target = _norm_lang(to_lang)
    if not target:
        raise Rejected(400, "to_lang is required")
    if target not in info.languages:
        speaks = ", ".join(sorted(info.languages))
        raise Rejected(400, f"the {name} backend cannot speak '{target}'; it speaks: {speaks}")

    source = (from_lang or "").strip().lower()
    origin: str | None = None if source in _AUTO else _norm_lang(source)
    if origin is not None and origin not in _SOURCE_CODES:
        raise Rejected(400, f"unknown source language '{origin}'")
    if origin is not None and origin == target:
        raise Rejected(400, "the source and target languages are the same")

    return {
        "to_lang": target,
        "from_lang": origin,
        "backend": name,
        "voice": check_voice(info, target, voice),
    }


def _validate(
    settings: Settings,
    source_type: str | None,
    url: str | None,
    to_lang: str | None,
    from_lang: str | None,
    backend: str | None,
    burn_subtitles: str | None,
    file: UploadFile | None,
    voice: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the form before anything is created or downloaded. Returns ``(source, options)``."""
    kind = (source_type or "").strip().lower()
    if kind not in {"youtube", "upload"}:
        raise Rejected(400, "source_type must be 'youtube' or 'upload'")

    options = check_options(settings, to_lang, from_lang, backend, voice)

    burn_raw = (burn_subtitles if burn_subtitles is not None else "true").strip().lower()
    if burn_raw in _TRUE or burn_raw == "":
        burn = True
    elif burn_raw in _FALSE:
        burn = False
    else:
        raise Rejected(400, "burn_subtitles must be 'true' or 'false'")

    link: str | None = None
    filename: str | None = None
    if kind == "youtube":
        link = check_source_url(url)
    else:
        if not settings.allow_uploads:
            raise Rejected(400, "file uploads are disabled on this server")
        if file is None or not (file.filename or "").strip():
            raise Rejected(400, "a video file is required")
        filename = safe_name(file.filename, fallback="upload")

    source_info = {"type": kind, "url": link, "filename": filename}
    return source_info, {**options, "burn_subtitles": burn}


def _save_upload(file: UploadFile, dest: Path, max_upload_mb: int) -> int:
    """Copy the upload to ``dest`` in chunks, enforcing MAX_UPLOAD_MB (decisions.md D-26).

    Synchronous on purpose: the route is a plain ``def``, so this runs in FastAPI's threadpool and
    half a gigabyte of disk writes never sits on the event loop.
    """
    limit = max(1, int(max_upload_mb)) * 1024 * 1024
    written = 0
    spooled = file.file  # the SpooledTemporaryFile starlette already parsed the multipart body into
    spooled.seek(0)
    with open(dest, "wb") as handle:
        while True:
            chunk = spooled.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise Rejected(413, f"the upload is larger than the {max_upload_mb} MB limit")
            handle.write(chunk)
    if written == 0:
        raise Rejected(400, "the uploaded file is empty")
    return written


async def limit_upload_size(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Refuse a rate-limited or over-large ``POST /jobs`` before its body is spooled.

    Without this, starlette parses (and writes to a temp file) the whole multipart body before the
    route ever sees it, so a 4 GB upload costs 4 GB of disk before the 413, and a flood of uploads
    costs disk before the 429. Bodies without a ``Content-Length`` (chunked) are still caught by the
    streamed check in :func:`_save_upload`. The limiter is one dict lookup under a lock, cheap enough
    for the event loop.

    Size first, then the limit: a request that can never become a job must not spend a token, or a
    browser that picks one too-large file would eat the whole hour's allowance on a 413.
    """
    if request.method == "POST" and request.url.path == CREATE_JOB_PATH:
        settings = get_settings()
        limit = max(1, int(settings.max_upload_mb)) * 1024 * 1024 + FORM_OVERHEAD_BYTES
        declared = _content_length(request)
        if declared is not None and declared > limit:
            log.info("refusing a %d byte POST /jobs: over the %d MB limit", declared, settings.max_upload_mb)
            return _error(413, f"the upload is larger than the {settings.max_upload_mb} MB limit")
        limited = _rate_limit_response(request, settings)
        if limited is not None:
            return limited
    return await call_next(request)


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@router.post("/jobs", status_code=201)
def create_job(
    request: Request,
    source_type: Annotated[str | None, Form()] = None,
    url: Annotated[str | None, Form()] = None,
    to_lang: Annotated[str | None, Form()] = None,
    from_lang: Annotated[str | None, Form()] = None,
    backend: Annotated[str | None, Form()] = None,
    voice: Annotated[str | None, Form()] = None,
    burn_subtitles: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> JSONResponse:
    """Validate, create the job directory, hand it to the pool, answer immediately (flow.md B3)."""
    settings = get_settings()
    # The rate limit was already applied in limit_upload_size(), before the body was parsed.
    try:
        source, options = _validate(
            settings, source_type, url, to_lang, from_lang, backend, burn_subtitles, file, voice
        )
    except Rejected as exc:
        return _error(exc.status_code, exc.message)

    store = get_store(settings)
    job_id = store.create(source=source, options=options)
    if source["type"] == "upload":
        assert file is not None  # _validate guarantees it
        try:
            _save_upload(file, store.path(job_id) / UPLOAD_FILENAME, settings.max_upload_mb)
        except Rejected as exc:
            store.delete(job_id)
            return _error(exc.status_code, exc.message)
        except OSError as exc:
            store.delete(job_id)
            return _error(500, f"could not store the upload: {exc}")

    get_runner(settings).submit(job_id)
    return JSONResponse(status_code=201, content={"id": job_id, "url": f"/jobs/{job_id}"})


# --------------------------------------------------------------------------- read


def _load(store: JobStore, job_id: str) -> dict[str, Any]:
    """The job's status dict, or a :class:`Rejected` the caller turns into a JSON error."""
    try:
        status = store.get(job_id)
    except ValueError as exc:  # unreadable status.json
        raise Rejected(500, f"job {job_id} has an unreadable status file: {exc}") from exc
    if status is None:
        raise Rejected(404, f"no job {job_id}")
    return status


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """The status dict exactly as it is on disk (flow.md B5)."""
    store = get_store(get_settings())
    try:
        return JSONResponse(content=_load(store, job_id))
    except Rejected as exc:
        return _error(exc.status_code, exc.message)


@router.get("/api/jobs/{job_id}/download")
def job_download(job_id: str) -> Any:
    """The finished video. 404 while it is not done, 409 when the file has been swept away."""
    settings = get_settings()
    store = get_store(settings)
    try:
        status = _load(store, job_id)
    except Rejected as exc:
        return _error(exc.status_code, exc.message)

    output = status.get("output")
    if status.get("state") != "done" or not output:
        return _error(404, f"job {job_id} has no output yet (state: {status.get('state')})")

    path = store.path(job_id) / Path(str(output)).name
    if not path.is_file():
        return _error(409, "the output file is gone; the job directory was cleaned up")

    title = safe_name(status.get("title")) or safe_name(status.get("source", {}).get("filename"))
    target = str(status.get("options", {}).get("to_lang") or "dub")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{title or job_id}-{target}.mp4",
    )


__all__ = [
    "SAMPLE_CACHE_CONTROL",
    "UPLOAD_FILENAME",
    "RateLimit",
    "RateLimiter",
    "Rejected",
    "check_options",
    "check_source_url",
    "check_voice",
    "client_address",
    "get_limiter",
    "limit_upload_size",
    "parse_ip",
    "parse_rate_limit",
    "reset_rate_limiter",
    "router",
    "safe_name",
    "voice_sample",
]
