"""Runtime configuration. Every value can be set in the environment or in a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    log_level: str = "INFO"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def resolved_device(self) -> str:
        """'cpu' or 'cuda'. 'auto' becomes 'cuda' only when torch can actually see a GPU."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
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
