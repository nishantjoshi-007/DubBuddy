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
import soundfile as sf
import srt as srt_lib

from respeak.config import Settings
from respeak.jobs import JobRunner, JobStore
from respeak.pipeline import ffmpeg, inputs
from respeak.pipeline import run as run_module
from respeak.pipeline.audio import AssembleResult
from respeak.pipeline.run import KEPT_FILES, OUTPUT_MP4, SUBS_SRT, assemble_warnings, run_job
from respeak.pipeline.tts import TTSError
from respeak.pipeline.types import Segment, Transcript, Voice

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
    assert status["detail"] is None  # 3.1: the per-stage sentence is cleared when the job ends

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
# per-stage progress and detail (plan.md 3.1) — the real media path, fake models, so this always runs
# --------------------------------------------------------------------------------------------------


#: What the fake transcript claims is in `sample.mp4` (10.2 s of English).
FAKE_SEGMENTS: tuple[tuple[float, float, str], ...] = (
    (0.5, 3.0, "Respeak turns a video into another language."),
    (3.5, 6.0, "It keeps the timing of the original speaker."),
    (6.5, 9.5, "This clip exists so the tests have real speech."),
)


class FakeBackend:
    """A TTS backend that writes a real 24 kHz clip per segment without loading a model."""

    name = "fake"

    def __init__(self, cloning: bool = False) -> None:
        self.cloning = cloning
        self.calls: list[tuple[str, str, Path | None, str | None]] = []
        self.reference_audio: tuple[int, int, float] | None = None

    def languages(self) -> set[str]:
        return {"en", "es"}

    def voices(self) -> dict[str, list[Voice]]:
        return {}

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
        self.calls.append((text, lang, reference_wav, voice))
        if reference_wav is not None and self.reference_audio is None:
            with sf.SoundFile(str(reference_wav)) as handle:
                self.reference_audio = (handle.samplerate, handle.channels, len(handle) / handle.samplerate)
        ffmpeg.run(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=330:sample_rate=24000:duration=1.2",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(out),
            ]
        )
        return out


class FakeTranslator:
    """`ArgosTranslator` without Argos: the pipeline only needs 1:1 texts back."""

    def ensure_pair(self, src: str, dst: str) -> None:
        return None

    def translate(self, texts: list[str], src: str, dst: str) -> list[str]:
        return [f"en {dst}: {text}" for text in texts]


def fake_models(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    """Replace whisper, Argos and the TTS backend; every ffmpeg step stays real."""

    def transcribe(wav: Path, language: str | None, settings: Settings, progress: Any = None) -> Transcript:
        if progress is not None:
            progress(0.5)
            progress(1.0)
        return Transcript(
            language=language or "en",
            segments=[Segment(start=s, end=e, text=t) for s, e, t in FAKE_SEGMENTS],
        )

    monkeypatch.setattr("respeak.pipeline.run.asr.transcribe", transcribe)
    monkeypatch.setattr("respeak.pipeline.run.get_translator", lambda settings: FakeTranslator())
    monkeypatch.setattr("respeak.pipeline.run.get_backend", lambda name, settings: backend)


def record_status_writes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every `JobStore.update(**fields)` the pipeline makes, in order, still written to disk."""
    written: list[dict[str, Any]] = []
    real = JobStore.update

    def spy(self: JobStore, job_id: str, **fields: Any) -> dict[str, Any]:
        written.append(dict(fields))
        return real(self, job_id, **fields)

    monkeypatch.setattr(JobStore, "update", spy)
    return written


def test_every_stage_writes_a_detail_and_the_bar_only_moves_forward(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3.1: `detail` names what is happening at least eight different times, `progress` never dips."""
    backend = FakeBackend()
    fake_models(monkeypatch, backend)
    job_id = make_upload_job(store, voice="ef_dora")
    written = record_status_writes(monkeypatch)

    run_job(job_id, settings, store)

    details = [fields["detail"] for fields in written if "detail" in fields]
    spoken = {detail for detail in details if detail}
    assert len(spoken) >= 8, sorted(spoken)
    assert "checking the file" in spoken
    assert "transcribing" in spoken
    assert any(detail.startswith("transcribed ") and " of 0:10" in detail for detail in spoken), spoken
    assert "translating 3 segments" in spoken
    assert {"speaking segment 1 of 3", "speaking segment 2 of 3", "speaking segment 3 of 3"} <= spoken
    assert "writing subtitles" in spoken
    assert "encoding the video" in spoken
    assert details[-1] is None, "the detail line must be cleared when the job finishes"

    values = [fields["progress"] for fields in written if "progress" in fields]
    assert values == sorted(values), values
    assert values[0] <= 0.02 and values[-1] == 1.0
    assert len(values) >= 10, values  # "the bar moves at least ten times between submit and done"

    status = store.require(job_id)
    assert (status["state"], status["detail"], status["progress"]) == ("done", None, 1.0)
    # 3.3 wiring: whatever the form picked reaches every synthesize() call.
    assert {voice for _, _, _, voice in backend.calls} == {"ef_dora"}
    assert [reference for _, _, reference, _ in backend.calls] == [None] * 3  # this backend cannot clone


def test_the_fetch_bar_keeps_talking_through_every_file_yt_dlp_downloads(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3: yt-dlp downloads the video, then the audio, then merges — all through one writer.

    The first file reaching 1.0 pins the bar at the top of the fetch range, so throttling on
    "has the bar moved?" alone silently dropped every sentence after it and the page froze on
    "downloading 7.4 MB of 7.4 MB" for the rest of the download.
    """
    job_id = make_upload_job(store)
    written = record_status_writes(monkeypatch)
    write = run_module._throttled_writer(store, job_id, "fetch")

    hook: list[tuple[float, str]] = [
        (0.5, "downloading 3.7 MB of 7.4 MB"),
        (1.0, "downloading 7.4 MB of 7.4 MB"),
        (1.0, "merging the download"),
        (0.05, "downloading 0.1 MB of 1.1 MB"),  # the audio file starts over at nearly nothing
        (0.5, "downloading 0.6 MB of 1.1 MB"),
        (1.0, "downloading 1.1 MB of 1.1 MB"),
        (1.0, "merging the download"),
    ]
    for fraction, detail in hook:
        write(fraction, detail)

    details = [fields["detail"] for fields in written if "detail" in fields]
    assert set(details) == {detail for _, detail in hook}, details

    values = [fields["progress"] for fields in written if "progress" in fields]
    assert values == sorted(values), values  # the audio restarting at 0.05 must not rewind the bar
    assert values[-1] == pytest.approx(run_module.STEP_RANGE["fetch"][1])
    assert store.require(job_id)["detail"] == "merging the download"


def test_the_bar_reaches_the_top_of_the_speak_and_mux_ranges(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F6: speak reported only `(i-1)/n`, so it handed over a whole segment short of its range,
    and mux's closing `progress(1.0)` was thrown away by the throttle a fraction below 0.95."""
    backend = FakeBackend()
    fake_models(monkeypatch, backend)
    job_id = make_upload_job(store)
    written = record_status_writes(monkeypatch)

    run_job(job_id, settings, store)

    step: str | None = None
    bar: list[tuple[str | None, float]] = []
    for fields in written:
        if "step" in fields:
            step = fields["step"]
        if "progress" in fields:
            bar.append((step, fields["progress"]))

    speak = [value for name, value in bar if name == "speak"]
    assert speak, bar
    assert speak[0] == pytest.approx(run_module.STEP_RANGE["speak"][0], abs=0.001)
    assert speak[-1] == pytest.approx(0.75, abs=0.001), speak

    mux = [value for name, value in bar if name == "mux"]
    assert any(value == pytest.approx(0.95, abs=0.001) for value in mux), mux
    assert [value for name, value in bar if name == "finish"][-1] == 1.0


def test_a_cloning_backend_gets_a_reference_cut_from_the_transcript(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3.7: `reference_from_segments` replaces the loudest-RMS clip — and only for cloning backends."""
    backend = FakeBackend(cloning=True)
    fake_models(monkeypatch, backend)
    cuts: list[tuple[str, list[tuple[float, float]], str]] = []
    real_cut = inputs.reference_from_segments

    def spy(source_wav: Path, segments: Any, out: Path, *args: Any, **kwargs: Any) -> Path:
        cuts.append(
            (Path(source_wav).name, [(s.start, s.end) for s in segments], Path(out).name),
        )
        return real_cut(source_wav, segments, out, *args, **kwargs)

    monkeypatch.setattr("respeak.pipeline.run.inputs.reference_from_segments", spy)
    job_id = make_upload_job(store)

    run_job(job_id, settings, store)

    assert store.require(job_id)["state"] == "done"
    assert len(cuts) == 1, "the reference is cut once per job, after transcription"
    assert cuts[0][0] == "source.wav"  # the ASR wav, not the video
    assert cuts[0][1] == [(start, end) for start, end, _ in FAKE_SEGMENTS]
    assert cuts[0][2] == "reference.wav"
    assert backend.reference_audio is not None
    rate, channels, seconds = backend.reference_audio
    assert (rate, channels) == (24_000, 1)
    assert 0 < seconds <= inputs.REFERENCE_SECONDS + 0.05
    assert {reference for _, _, reference, _ in backend.calls} == {store.path(job_id) / "reference.wav"}


def test_a_backend_that_cannot_clone_never_pays_for_a_reference_cut(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(cloning=False)
    fake_models(monkeypatch, backend)
    called: list[Path] = []
    monkeypatch.setattr(
        "respeak.pipeline.run.inputs.reference_from_segments",
        lambda source_wav, segments, out, *a, **kw: called.append(Path(out)) or Path(out),
    )
    job_id = make_upload_job(store)

    run_job(job_id, settings, store)

    assert called == []
    assert store.require(job_id)["state"] == "done"


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
    # D-49: the sentence says what was *cut*, not how much longer than the video the dub was.
    assert messages[1] == "The last 1.5 s of speech were cut to fit the video."
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
# D-49 — speech that does not fit slows the video down instead of being cut (no model needed)
# --------------------------------------------------------------------------------------------------

STARTS: tuple[float, ...] = tuple(start for start, _, _ in FAKE_SEGMENTS)
SLOTS: tuple[float, ...] = tuple(end - start for start, end, _ in FAKE_SEGMENTS)


class RatioBackend(FakeBackend):
    """A backend whose clip for each sentence is `ratio ×` the slot the original speaker used.

    `ratio` is what a real expansion looks like measured against the source timing: English → Hindi
    lands around 1.2–1.3 before any speed-up, and a wordy sentence far past that.
    """

    def __init__(self, ratio: float) -> None:
        super().__init__()
        self.ratio = ratio
        self.spoken_seconds: list[float] = []

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
        seconds = self.ratio * SLOTS[len(self.spoken_seconds) % len(SLOTS)]
        self.spoken_seconds.append(seconds)
        self.calls.append((text, lang, reference_wav, voice))
        ffmpeg.run(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=330:sample_rate=24000:duration={seconds:.3f}",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(out),
            ]
        )
        return out


def expected_stretch(ratio: float, cap: float, video_seconds: float) -> float:
    """D-49's closed form, worked out here from the fake clip lengths instead of from `audio`.

    A clip `ratio ×` its slot is sped up by at most `cap`, so it still covers `ratio / cap` slots;
    `s` is then the worst `Σ_{j≥i} d_j / (V − start_i)` over the sentences.
    """
    fitted = [slot * max(1.0, ratio / cap) for slot in SLOTS]
    factors = [
        sum(fitted[index:]) / (video_seconds - start)
        for index, start in enumerate(STARTS)
        if start < video_seconds
    ]
    return max([1.0, *factors])


def test_speech_that_overruns_slows_the_video_instead_of_being_cut(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-49 steps 1–3: clips 1.7× their slots survive whole because the picture runs ~8 % slower.

    (1.25× — the average English → Hindi expansion — needs no slowdown at all here: the 1.3× speech
    cap and the pauses between the sentences already swallow it. It takes ~1.6× to move the picture.)
    """
    backend = RatioBackend(1.7)
    fake_models(monkeypatch, backend)
    video_seconds = ffmpeg.duration(SAMPLE_MP4)
    stretch = expected_stretch(1.7, settings.max_speech_speedup, video_seconds)
    assert 1.0 < stretch <= settings.max_video_stretch, stretch
    job_id = make_upload_job(store)
    job_dir = store.path(job_id)

    run_job(job_id, settings, store)

    status = store.require(job_id)
    assert status["state"] == "done"
    assert status["error"] is None
    percent = round((stretch - 1.0) * 100)
    assert status["warnings"] == [f"The video was slowed by {percent} % so all the speech fits."]
    assert not any("cut" in warning for warning in status["warnings"]), status["warnings"]

    out = job_dir / OUTPUT_MP4
    measured = ffmpeg.duration(out)
    print(f"\n[D-49 fits] s={stretch:.4f} source={video_seconds:.2f}s out={measured:.2f}s")
    assert measured == pytest.approx(stretch * video_seconds, abs=0.2)
    assert measured > video_seconds + 0.3  # the picture really is longer than it was

    # The subtitles live on the same stretched timeline, so the last cue is still on screen.
    last = cues_of(job_dir / SUBS_SRT)[-1]
    assert last.end.total_seconds() <= measured + 0.1
    assert last.start.total_seconds() > STARTS[-1]  # moved later by the slowdown, not left behind


def test_speech_that_overruns_even_the_slowest_video_is_named_in_the_warnings(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-49 step 4: the cap holds at 15 %, speech goes to 1.5×, and what is still lost is named."""
    backend = RatioBackend(3.5)
    fake_models(monkeypatch, backend)
    video_seconds = ffmpeg.duration(SAMPLE_MP4)
    assert expected_stretch(3.5, settings.max_speech_speedup, video_seconds) > settings.max_video_stretch
    stretch = settings.max_video_stretch
    job_id = make_upload_job(store, burn_subtitles=False)
    job_dir = store.path(job_id)

    run_job(job_id, settings, store)

    status = store.require(job_id)
    assert status["state"] == "done"
    warnings = status["warnings"]
    assert warnings[0] == "The video was slowed by 15 % so all the speech fits."
    named = next(w for w in warnings if "could not fit" in w)
    assert "slowing the video 15 % and speeding speech to 1.5×" in named
    assert "'en es: " in named, named  # the sentences themselves are quoted, truncated
    assert any(w.startswith("The last ") and "were cut to fit the video." in w for w in warnings), warnings

    out = job_dir / OUTPUT_MP4
    measured = ffmpeg.duration(out)
    print(f"\n[D-49 capped] s={stretch:.4f} source={video_seconds:.2f}s out={measured:.2f}s {warnings}")
    assert measured == pytest.approx(stretch * video_seconds, abs=0.2)
    # Even with burn off, a stretched picture must be re-encoded — a stream copy cannot apply setpts.
    assert stream_of(out, "video")["codec_name"] == "h264"
    assert video_md5(out) != video_md5(SAMPLE_MP4), "the stretched video was copied, not re-encoded"


def test_speech_that_fits_leaves_the_video_exactly_as_it_was(
    settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case pays nothing for D-49: no stretch, no warning, and still a stream copy."""
    backend = RatioBackend(0.9)
    fake_models(monkeypatch, backend)
    video_seconds = ffmpeg.duration(SAMPLE_MP4)
    assert expected_stretch(0.9, settings.max_speech_speedup, video_seconds) == 1.0
    job_id = make_upload_job(store, burn_subtitles=False)
    job_dir = store.path(job_id)

    run_job(job_id, settings, store)

    status = store.require(job_id)
    assert status["state"] == "done"
    assert status["warnings"] == []
    measured = ffmpeg.duration(job_dir / OUTPUT_MP4)
    print(f"\n[D-49 fits already] s=1.0 source={video_seconds:.2f}s out={measured:.2f}s")
    assert measured == pytest.approx(video_seconds, abs=0.2)
    assert video_md5(job_dir / OUTPUT_MP4) == video_md5(SAMPLE_MP4), "the picture was touched anyway"


# --------------------------------------------------------------------------------------------------
# failure
# --------------------------------------------------------------------------------------------------


class BoomBackend:
    """A TTS backend that fails the way a missing model would (flow.md B8: never a silent None)."""

    name = "boom"
    cloning = False

    def languages(self) -> set[str]:
        return {"es"}

    def voices(self) -> dict[str, list[Voice]]:
        return {}

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
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
