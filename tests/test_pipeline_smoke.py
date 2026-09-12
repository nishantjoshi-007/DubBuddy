"""WP-F: the whole pipeline on a checked-in clip (flow.md B4.1–B4.8, B5, B7; plan.md 1.1, 1.10, 1.12).

Offline once the caches the README describes exist (whisper `base`, Kokoro-82M, the Argos en->es
package); without them and without the network every test here skips instead of failing.

`tests/fixtures/sample.mp4` (10.2 s, 640x360 15 fps, h264 + aac, 466 KB) was generated once with:

    uv run python -c "
    from pathlib import Path
    from respeak.config import Settings
    from respeak.pipeline.tts.kokoro import KokoroBackend
    KokoroBackend(Settings()).synthesize(
        'Respeak turns a video into another language, keeping the timing of the original speaker. '
        'This short clip exists so the tests have real speech to transcribe.',
        'en', None, Path('speech_en.wav'))"

    ffmpeg -f lavfi -i testsrc2=size=640x360:rate=15:duration=10.2 -i speech_en.wav \
           -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p -g 30 \
           -c:a aac -b:a 64k -shortest -movflags +faststart tests/fixtures/sample.mp4

Only the MP4 is checked in; the intermediate WAV is not (tests/fixtures/speech_en.wav belongs to WP-C
and is a different, longer clip).
"""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path
from typing import Any

import pytest
import srt as srt_lib

from respeak.config import Settings
from respeak.jobs import JobRunner, JobStore
from respeak.pipeline import ffmpeg
from respeak.pipeline.audio import AssembleResult
from respeak.pipeline.run import KEPT_FILES, OUTPUT_MP4, SUBS_SRT, assemble_warnings, run_job
from respeak.pipeline.tts import TTSError

ffmpeg.ensure_binaries()

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURES / "sample.mp4"
WHISPER_MODEL = "base"  # cached on the dev machine; `small` is the product default (D-06)
KOKORO_FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/ef_dora.pt")

#: Words that appear in almost any Spanish sentence; the SRT must not still be English.
SPANISH_MARKERS = (" de ", " la ", " el ", " un ", " una ", " en ", " que ", " los ", " las ")


# --------------------------------------------------------------------------------------------------
# availability — everything expensive skips rather than fails
# --------------------------------------------------------------------------------------------------


def _has_network(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _whisper_cached(name: str) -> bool:
    try:
        from huggingface_hub import constants
    except ImportError:  # pragma: no cover - huggingface_hub ships with faster-whisper
        return False
    hub = Path(os.environ.get("HF_HUB_CACHE") or constants.HF_HUB_CACHE)
    return (hub / f"models--Systran--faster-whisper-{name}").is_dir()


def _kokoro_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # pragma: no cover - huggingface_hub ships with kokoro
        return False
    return all(isinstance(try_to_load_from_cache("hexgrad/Kokoro-82M", f), str) for f in KOKORO_FILES)


def _argos_installed(from_code: str, to_code: str) -> bool:
    from argostranslate import package as argos_package

    return any(
        p.from_code == from_code and p.to_code == to_code for p in argos_package.get_installed_packages()
    )


needs_models = pytest.mark.skipif(
    not (_whisper_cached(WHISPER_MODEL) and _kokoro_cached()) and not _has_network(),
    reason=f"whisper {WHISPER_MODEL} / Kokoro-82M are not cached and there is no network",
)
needs_en_es = pytest.mark.skipif(
    not _argos_installed("en", "es") and not _has_network("argos-net.com"),
    reason="the Argos en->es package is not installed and there is no network to download it",
)


# --------------------------------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A throwaway DATA_DIR, the small whisper model and Kokoro — explicit, so no .env can change it."""
    return Settings(
        data_dir=tmp_path,
        whisper_model=WHISPER_MODEL,
        tts_backend="kokoro",
        max_video_seconds=60,
        max_upload_mb=10,
    )


@pytest.fixture
def store(settings: Settings) -> JobStore:
    return JobStore(settings.jobs_dir)


def make_upload_job(store: JobStore, **options: Any) -> str:
    """A job of type upload with `sample.mp4` already streamed to `<jobdir>/upload.bin` (api.py does this)."""
    job_id = store.create(
        source={"type": "upload", "url": None, "filename": "sample.mp4"},
        options={"to_lang": "es", "from_lang": None, "backend": "kokoro", "burn_subtitles": True, **options},
    )
    shutil.copyfile(SAMPLE_MP4, store.path(job_id) / "upload.bin")
    return job_id


def stream_of(path: Path, codec_type: str) -> dict[str, Any]:
    """The first stream of a kind, or a failed assertion naming what the file actually holds."""
    info = ffmpeg.probe(path)
    streams = [s for s in info["streams"] if s.get("codec_type") == codec_type]
    kinds = [s.get("codec_type") for s in info["streams"]]
    assert streams, f"{path.name} has no {codec_type} stream (only {kinds})"
    return streams[0]


def stream_seconds(stream: dict[str, Any]) -> float:
    return float(stream.get("duration") or 0.0)


def video_md5(path: Path) -> str:
    """Hash the video packets without decoding: equal hashes prove `-c:v copy` really copied."""
    return ffmpeg.run(["-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"]).stdout.strip()


def cues_of(srt_path: Path) -> list[srt_lib.Subtitle]:
    return list(srt_lib.parse(srt_path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------------------------------


@needs_models
@needs_en_es
def test_upload_job_runs_end_to_end(settings: Settings, store: JobStore) -> None:
    """en -> es with burnt-in subtitles: done, one playable file, the intermediates gone (B4, B7)."""
    job_id = make_upload_job(store)
    job_dir = store.path(job_id)

    run_job(job_id, settings, store)

    status = store.require(job_id)
    assert status["state"] == "done"
    assert status["error"] is None
    assert status["step"] == "finish"
    assert status["progress"] == 1.0
    assert status["output"] == OUTPUT_MP4
    assert status["download_url"] == f"/api/jobs/{job_id}/download"
    assert status["detected_language"] == "en"
    assert status["title"] == "sample"  # the container carries no title, so the filename is used
    assert status["warnings"] == []  # nothing was cut off this dub

    # B7: out.mp4, subs.srt and status.json — nothing else survives.
    assert {p.name for p in job_dir.iterdir()} == set(KEPT_FILES)

    out = job_dir / OUTPUT_MP4
    assert out.stat().st_size > 0

    cues = cues_of(job_dir / SUBS_SRT)
    assert len(cues) >= 1
    spoken = " ".join(cue.content for cue in cues)
    assert spoken.strip()
    assert any(marker in f" {spoken.lower()} ".replace("\n", " ") for marker in SPANISH_MARKERS), spoken

    # B4.8: h264 video, aac audio, a soft mov_text subtitle track.
    video = stream_of(out, "video")
    audio = stream_of(out, "audio")
    subtitle = stream_of(out, "subtitle")
    assert video["codec_name"] == "h264"
    assert audio["codec_name"] == "aac"
    assert subtitle["codec_name"] == "mov_text"
    assert subtitle.get("tags", {}).get("language") == "spa"

    # B4.6: the dub is exactly as long as the picture.
    assert stream_seconds(audio) == pytest.approx(stream_seconds(video), abs=0.2)


@needs_models
@needs_en_es
def test_burn_off_copies_the_video_stream(settings: Settings, store: JobStore) -> None:
    """burn_subtitles=False must not re-encode the picture (B4.8: `-map 0:v -c:v copy`)."""
    job_id = make_upload_job(store, burn_subtitles=False)
    job_dir = store.path(job_id)

    run_job(job_id, settings, store)

    assert store.require(job_id)["state"] == "done"
    out = job_dir / OUTPUT_MP4
    assert stream_of(out, "video")["codec_name"] == stream_of(SAMPLE_MP4, "video")["codec_name"] == "h264"
    assert video_md5(out) == video_md5(SAMPLE_MP4), "the video was re-encoded instead of copied"
    # The subtitles are still there as a soft track, they are just not painted on.
    assert stream_of(out, "subtitle")["codec_name"] == "mov_text"
    assert len(cues_of(job_dir / SUBS_SRT)) >= 1


# --------------------------------------------------------------------------------------------------
# warnings (F5) — a dub that did not fit must say so; no model needed
# --------------------------------------------------------------------------------------------------


def test_assemble_warnings_are_empty_when_everything_fitted() -> None:
    assert assemble_warnings(AssembleResult(path=Path("dubbed.wav"))) == []
    rounding = AssembleResult(path=Path("dubbed.wav"), trimmed_seconds=0.01)
    assert assemble_warnings(rounding) == []


def test_assemble_warnings_name_the_segments_and_the_seconds() -> None:
    result = AssembleResult(
        path=Path("dubbed.wav"), dropped_seconds=12.4, trimmed_seconds=1.5, dropped_clips=3
    )
    messages = assemble_warnings(result)
    assert len(messages) == 2
    assert messages[0] == ("3 segments (12.4 s of speech) did not fit before the video ended and were cut.")
    assert "1.5 s" in messages[1]
    assert all(message.endswith(".") for message in messages)


def test_assemble_warnings_count_one_segment_in_the_singular() -> None:
    result = AssembleResult(path=Path("dubbed.wav"), dropped_seconds=2.0, dropped_clips=1)
    assert assemble_warnings(result)[0].startswith("1 segment (2.0 s of speech)")


@needs_models
@needs_en_es
def test_a_dub_that_does_not_fit_is_reported_in_the_status(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job still finishes, but status.warnings tells the browser what was cut (F5)."""
    from respeak.pipeline import audio as audio_module

    real_assemble = audio_module.assemble

    def half_length(placed, total_seconds, out, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Pretend the video ended after a third of its real length: the tail cannot fit.
        return real_assemble(placed, max(0.5, total_seconds / 3.0), out, *args, **kwargs)

    monkeypatch.setattr("respeak.pipeline.run.audio.assemble", half_length)
    job_id = make_upload_job(store)

    run_job(job_id, settings, store)

    status = store.require(job_id)
    assert status["state"] == "done"
    assert status["error"] is None
    assert status["warnings"], "a truncated dub must not look like a perfect one"
    assert any("did not fit" in w or "trimmed" in w for w in status["warnings"]), status["warnings"]


# --------------------------------------------------------------------------------------------------
# failure
# --------------------------------------------------------------------------------------------------


class BoomBackend:
    """A TTS backend that fails the way a missing model would (flow.md B8: never a silent None)."""

    name = "boom"
    cloning = False

    def languages(self) -> set[str]:
        return {"es"}

    def synthesize(self, text: str, lang: str, reference_wav: Path | None, out: Path) -> Path:
        raise TTSError("kokoro ran out of voices")


@needs_models
@needs_en_es
def test_speak_failure_is_recorded_against_the_speak_step(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner turns the exception into `"<step>: <message>"`, and the step must be the failing one."""
    monkeypatch.setattr("respeak.pipeline.run.get_backend", lambda name, settings: BoomBackend())
    job_id = make_upload_job(store)

    runner = JobRunner(store, settings, run_fn=run_job)
    try:
        runner.submit(job_id).result(timeout=300)
    finally:
        runner.shutdown()

    status = store.require(job_id)
    assert status["state"] == "failed"
    assert status["step"] == "speak"
    assert status["error"].startswith("speak:"), status["error"]
    assert "ran out of voices" in status["error"]
    assert not (store.path(job_id) / OUTPUT_MP4).exists()
    # A failed job keeps its intermediates: cleanup only ever runs on success (B7).
    assert (store.path(job_id) / "source.mp4").exists()
