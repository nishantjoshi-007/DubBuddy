"""JSON API (flow.md B3). Phase 0 ships health and backends; jobs arrive with WP-A."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, HTTPException

from . import __version__
from .config import get_settings
from .lang_codes import SOURCE_LANGUAGES
from .pipeline.tts import available_backends

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "name": "respeak",
        "version": __version__,
        "device": settings.resolved_device(),
        "tts_backend": settings.tts_backend,
        "whisper_model": settings.whisper_model,
        "limits": settings.limits(),
        "binaries": {name: shutil.which(name) is not None for name in ("ffmpeg", "ffprobe", "deno")},
    }


@router.get("/backends")
async def backends():
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


@router.post("/jobs", status_code=501)
async def create_job_placeholder():
    raise HTTPException(status_code=501, detail="Job submission arrives in Phase 1 (WP-A).")
