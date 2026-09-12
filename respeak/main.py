"""Application factory. Run with: uvicorn respeak.main:app"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__, api, pages
from .config import get_settings
from .pipeline.ffmpeg import ensure_binaries

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("respeak")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    ensure_binaries()
    log.info(
        "respeak %s ready (device=%s, tts=%s, whisper=%s)",
        __version__,
        settings.resolved_device(),
        settings.tts_backend,
        settings.whisper_model,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Respeak", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.include_router(pages.router)
    app.include_router(api.router)
    return app


app = create_app()
