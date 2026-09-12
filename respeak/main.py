"""Application factory. Run with: uvicorn respeak.main:app"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__, api, pages
from .config import get_settings
from .jobs import get_runner, get_store, start_sweeper
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
    store = get_store(settings)
    runner = get_runner(settings, run_fn=_pipeline_run_fn())
    sweeper = start_sweeper(store, settings, runner=runner)
    log.info(
        "respeak %s ready (device=%s, tts=%s, whisper=%s, jobs=%s)",
        __version__,
        settings.resolved_device(),
        settings.tts_backend,
        settings.whisper_model,
        settings.jobs_dir,
    )
    try:
        yield
    finally:
        sweeper.stop.set()
        runner.shutdown(wait=False)


def _pipeline_run_fn():
    """The real pipeline, or None while ``respeak/pipeline/run.py`` does not exist yet.

    Only that one missing module is tolerated: a broken install (no torch, no soundfile, a typo in an
    import) must fail here, with its own traceback, instead of silently leaving every job to the
    runner's placeholder.
    """
    try:
        from .pipeline.run import run_job
    except ModuleNotFoundError as exc:
        if exc.name != "respeak.pipeline.run":
            raise
        log.warning("respeak.pipeline.run is missing; jobs will fail until it exists")
        return None
    return run_job


def create_app() -> FastAPI:
    app = FastAPI(title="Respeak", version=__version__, lifespan=lifespan)
    app.middleware("http")(api.limit_upload_size)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.include_router(pages.router)
    app.include_router(api.router)
    return app


app = create_app()
