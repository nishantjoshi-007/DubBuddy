"""JSON API (flow.md B3, B5).

Every route here is cheap: the job routes read or write one small file and return. The work happens in
:class:`respeak.jobs.JobRunner`'s thread pool, never on the event loop.

The routes are plain ``def``, not ``async def``, on purpose: FastAPI then runs them in its threadpool,
so reading a status file, asking torch about CUDA or writing a 500 MB upload never blocks the loop.

The router carries full paths (``/api/…`` plus the one root-level ``POST /jobs``) so that ``main.py``
can include it unchanged.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings, get_settings
from .jobs import JobStore, get_runner, get_store
from .lang_codes import NAME_TO_CODE, SOURCE_LANGUAGES
from .pipeline.inputs import InputError, check_public_url
from .pipeline.tts import available_backends
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


class _Rejected(Exception):
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
            }
            for info in infos.values()
        ],
        "source_languages": SOURCE_LANGUAGES,
        "allow_uploads": settings.allow_uploads,
    }


# --------------------------------------------------------------------------- create


def _validate(
    settings: Settings,
    source_type: str | None,
    url: str | None,
    to_lang: str | None,
    from_lang: str | None,
    backend: str | None,
    burn_subtitles: str | None,
    file: UploadFile | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the form before anything is created or downloaded. Returns ``(source, options)``."""
    kind = (source_type or "").strip().lower()
    if kind not in {"youtube", "upload"}:
        raise _Rejected(400, "source_type must be 'youtube' or 'upload'")

    name = (backend or "").strip().lower() or settings.tts_backend
    infos = available_backends(settings)
    info = infos.get(name)
    if info is None:
        raise _Rejected(400, f"unknown backend '{name}'; installed backends: {', '.join(sorted(infos))}")
    if not info.installed:
        raise _Rejected(400, info.reason or f"the {name} backend is not installed")

    target = _norm_lang(to_lang)
    if not target:
        raise _Rejected(400, "to_lang is required")
    if target not in info.languages:
        speaks = ", ".join(sorted(info.languages))
        raise _Rejected(400, f"the {name} backend cannot speak '{target}'; it speaks: {speaks}")

    source = (from_lang or "").strip().lower()
    origin: str | None = None if source in _AUTO else _norm_lang(source)
    if origin is not None and origin not in _SOURCE_CODES:
        raise _Rejected(400, f"unknown source language '{origin}'")
    if origin is not None and origin == target:
        raise _Rejected(400, "the source and target languages are the same")

    burn_raw = (burn_subtitles if burn_subtitles is not None else "true").strip().lower()
    if burn_raw in _TRUE or burn_raw == "":
        burn = True
    elif burn_raw in _FALSE:
        burn = False
    else:
        raise _Rejected(400, "burn_subtitles must be 'true' or 'false'")

    link: str | None = None
    filename: str | None = None
    if kind == "youtube":
        link = (url or "").strip()
        if not link:
            raise _Rejected(400, "a YouTube URL is required")
        parsed = urlparse(link)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise _Rejected(400, "the URL must start with http:// or https://")
        try:
            # yt-dlp's generic extractor would happily fetch http://169.254.169.254/… for us.
            check_public_url(link)
        except InputError as exc:
            raise _Rejected(400, str(exc)) from exc
    else:
        if not settings.allow_uploads:
            raise _Rejected(400, "file uploads are disabled on this server")
        if file is None or not (file.filename or "").strip():
            raise _Rejected(400, "a video file is required")
        filename = safe_name(file.filename, fallback="upload")

    source_info = {"type": kind, "url": link, "filename": filename}
    options = {"to_lang": target, "from_lang": origin, "backend": name, "burn_subtitles": burn}
    return source_info, options


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
                raise _Rejected(413, f"the upload is larger than the {max_upload_mb} MB limit")
            handle.write(chunk)
    if written == 0:
        raise _Rejected(400, "the uploaded file is empty")
    return written


async def limit_upload_size(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Refuse an over-large ``POST /jobs`` from its ``Content-Length``, before the body is spooled.

    Without this, starlette parses (and writes to a temp file) the whole multipart body before the
    route ever sees it, so a 4 GB upload costs 4 GB of disk before the 413. Bodies without a
    ``Content-Length`` (chunked) are still caught by the streamed check in :func:`_save_upload`.
    """
    if request.method == "POST" and request.url.path == CREATE_JOB_PATH:
        settings = get_settings()
        limit = max(1, int(settings.max_upload_mb)) * 1024 * 1024 + FORM_OVERHEAD_BYTES
        declared = _content_length(request)
        if declared is not None and declared > limit:
            log.info("refusing a %d byte POST /jobs: over the %d MB limit", declared, settings.max_upload_mb)
            return _error(413, f"the upload is larger than the {settings.max_upload_mb} MB limit")
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
    source_type: Annotated[str | None, Form()] = None,
    url: Annotated[str | None, Form()] = None,
    to_lang: Annotated[str | None, Form()] = None,
    from_lang: Annotated[str | None, Form()] = None,
    backend: Annotated[str | None, Form()] = None,
    burn_subtitles: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> JSONResponse:
    """Validate, create the job directory, hand it to the pool, answer immediately (flow.md B3)."""
    settings = get_settings()
    try:
        source, options = _validate(
            settings, source_type, url, to_lang, from_lang, backend, burn_subtitles, file
        )
    except _Rejected as exc:
        return _error(exc.status_code, exc.message)

    store = get_store(settings)
    job_id = store.create(source=source, options=options)
    if source["type"] == "upload":
        assert file is not None  # _validate guarantees it
        try:
            _save_upload(file, store.path(job_id) / UPLOAD_FILENAME, settings.max_upload_mb)
        except _Rejected as exc:
            store.delete(job_id)
            return _error(exc.status_code, exc.message)
        except OSError as exc:
            store.delete(job_id)
            return _error(500, f"could not store the upload: {exc}")

    get_runner(settings).submit(job_id)
    return JSONResponse(status_code=201, content={"id": job_id, "url": f"/jobs/{job_id}"})


# --------------------------------------------------------------------------- read


def _load(store: JobStore, job_id: str) -> dict[str, Any]:
    """The job's status dict, or a :class:`_Rejected` the caller turns into a JSON error."""
    try:
        status = store.get(job_id)
    except ValueError as exc:  # unreadable status.json
        raise _Rejected(500, f"job {job_id} has an unreadable status file: {exc}") from exc
    if status is None:
        raise _Rejected(404, f"no job {job_id}")
    return status


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """The status dict exactly as it is on disk (flow.md B5)."""
    store = get_store(get_settings())
    try:
        return JSONResponse(content=_load(store, job_id))
    except _Rejected as exc:
        return _error(exc.status_code, exc.message)


@router.get("/api/jobs/{job_id}/download")
def job_download(job_id: str) -> Any:
    """The finished video. 404 while it is not done, 409 when the file has been swept away."""
    settings = get_settings()
    store = get_store(settings)
    try:
        status = _load(store, job_id)
    except _Rejected as exc:
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


__all__ = ["UPLOAD_FILENAME", "limit_upload_size", "router", "safe_name"]
