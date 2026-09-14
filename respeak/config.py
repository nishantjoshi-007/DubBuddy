"""Runtime configuration. Every value can be set in the environment or in a .env file."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tts_backend: Literal["kokoro", "chatterbox"] = "kokoro"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_model: str = "small"
    whisper_compute: str = "int8"

    max_video_seconds: int = 900
    max_upload_mb: int = 500
    max_height: int = 720
    max_concurrent_jobs: int = 1
    job_ttl_minutes: int = 60

    data_dir: Path = Path("./data")
    model_cache_dir: Path | None = None

    allow_uploads: bool = True
    ytdlp_cookies_file: Path | None = None
    #: Update yt-dlp at startup. Off from source, on in the Docker image: video sites change more
    #: often than this repo is released, and the fix is almost always a newer yt-dlp.
    ytdlp_auto_update: bool = False
    #: Comma-separated ISO-639-1 codes `python -m respeak.prewarm` downloads models for (Argos pairs
    #: through English, Kokoro voices, and the Japanese dictionary when "ja" is in the list).
    prewarm_languages: str = "en,es"

    max_speech_speedup: float = 1.3  # a sentence may be spoken up to this much faster to fit its slot
    max_video_stretch: float = 1.15  # the video may be slowed by up to this factor so all speech fits
    rate_limit_jobs: str = ""  # e.g. "10/hour"; empty = off
    trust_proxy: bool = False  # honour X-Forwarded-For for rate limiting

    log_level: str = "INFO"

    #: Memoised answer of :meth:`resolved_device` (see there).
    _device: str | None = PrivateAttr(default=None)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def resolved_device(self) -> str:
        """'cpu' or 'cuda'. 'auto' becomes 'cuda' only when torch can actually see a GPU.

        Importing torch and asking CUDA costs about a second, and `/api/health` calls this on every
        request, so the answer is computed once per :class:`Settings` instance and cached.
        """
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    def _detect_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            log.info("torch could not be asked about CUDA (%s); using the CPU", exc)
            return "cpu"

    def limits(self) -> dict[str, int]:
        return {
            "max_video_seconds": self.max_video_seconds,
            "max_upload_mb": self.max_upload_mb,
            "max_height": self.max_height,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "job_ttl_minutes": self.job_ttl_minutes,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
