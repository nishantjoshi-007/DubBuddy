"""JSON API (flow.md B3, B5).

Every route here is cheap: the job routes read or write one small file and return. The work happens in
:class:`respeak.jobs.JobRunner`'s thread pool, never on the event loop.

The router carries full paths (``/api/…`` plus the one root-level ``POST /jobs``) so that ``main.py``
can include it unchanged.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings, get_settings
from .jobs import JobStore, get_runner, get_store
from .lang_codes import NAME_TO_CODE, SOURCE_LANGUAGES
from .pipeline.tts import available_backends

log = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_FILENAME = "upload.bin"
UPLOAD_CHUNK_BYTES = 1024 * 1024
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
async def health() -> dict[str, Any]:
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
    }


@router.get("/api/backends")
async def backends() -> dict[str, Any]:
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
    else:
        if not settings.allow_uploads:
            raise _Rejected(400, "file uploads are disabled on this server")
        if file is None or not (file.filename or "").strip():
            raise _Rejected(400, "a video file is required")
        filename = safe_name(file.filename, fallback="upload")

    source_info = {"type": kind, "url": link, "filename": filename}
    options = {"to_lang": target, "from_lang": origin, "backend": name, "burn_subtitles": burn}
    return source_info, options


async def _save_upload(file: UploadFile, dest: Path, max_upload_mb: int) -> int:
    """Stream the upload to ``dest`` in chunks, enforcing MAX_UPLOAD_MB (decisions.md D-26)."""
    limit = max(1, int(max_upload_mb)) * 1024 * 1024
    written = 0
    with open(dest, "wb") as handle:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise _Rejected(413, f"the upload is larger than the {max_upload_mb} MB limit")
            handle.write(chunk)
    if written == 0:
        raise _Rejected(400, "the uploaded file is empty")
    return written


@router.post("/jobs", status_code=201)
async def create_job(
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
            await _save_upload(file, store.path(job_id) / UPLOAD_FILENAME, settings.max_upload_mb)
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


__all__ = ["UPLOAD_FILENAME", "router", "safe_name"]
