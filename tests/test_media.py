"""WP-B: ffmpeg helpers, inputs, fit/assemble, subtitles, mux (flow.md B4.1/B4.2/B4.6/B4.7/B4.8).

Every fixture is generated with ffmpeg at test time, so the suite is hermetic and offline. The one
YouTube test only probes metadata and skips itself when the network is unreachable.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import srt as srt_lib

from respeak.config import Settings
from respeak.pipeline import audio, ffmpeg, inputs, mux, subtitles
from respeak.pipeline.ffmpeg import FFmpegError
from respeak.pipeline.inputs import InputError
from respeak.pipeline.types import Cue, Probe, Segment

ffmpeg.ensure_binaries()

SAMPLE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", 19 s


# --------------------------------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------------------------------


def make_video(
    path: Path,
    seconds: float = 10.0,
    size: str = "640x360",
    vcodec: str = "mpeg4",
    audio_filter: str | None = None,
    title: str | None = None,
) -> Path:
    """A testsrc2 video with a sine tone, written with ffmpeg (`mpeg4` so `-c:v copy` is provable)."""
    args = [
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate=25:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={seconds}",
    ]
    if audio_filter:
        args += ["-af", audio_filter]
    args += ["-c:v", vcodec, "-q:v", "6", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k"]
    if title:
        args += ["-metadata", f"title={title}"]
    args += ["-shortest", str(path)]
    ffmpeg.run(args)
    return path


def make_tone(path: Path, seconds: float, freq: int = 440, rate: int = 24_000) -> Path:
    ffmpeg.run(
        [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate={rate}:duration={seconds}",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def wav_info(path: Path) -> tuple[int, int, float]:
    """(sample rate, channels, seconds) straight from the WAV header."""
    with sf.SoundFile(str(path)) as handle:
        return handle.samplerate, handle.channels, len(handle) / handle.samplerate


def rms(path: Path, start: float = 0.0, end: float | None = None) -> float:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    lo = int(start * rate)
    hi = len(mono) if end is None else min(len(mono), int(end * rate))
    chunk = mono[lo:hi]
    return float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0


def codecs(path: Path) -> dict[str, list[str]]:
    """codec_type → codec names, from ffprobe."""
    found: dict[str, list[str]] = {}
    for stream in ffmpeg.probe(path)["streams"]:
        found.setdefault(str(stream["codec_type"]), []).append(str(stream["codec_name"]))
    return found


def gray_frame(path: Path, at: float, raw: Path) -> np.ndarray:
    """One frame as an 8-bit grayscale array (via rawvideo; no image library needed)."""
    probe = ffmpeg.probe_summary(path)
    ffmpeg.run(
        [
            "-ss",
            f"{at:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            str(raw),
        ]
    )
    return np.fromfile(raw, dtype=np.uint8).reshape(probe.height, probe.width)


def has_network(host: str = "www.youtube.com", port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=4):
            return True
    except OSError:
        return False


NETWORK = has_network()


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_video(tmp_path_factory.mktemp("media") / "sample.mp4", title="Sample Clip")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------------------------------
# ffmpeg.py
# --------------------------------------------------------------------------------------------------


def test_run_reports_failures_with_the_stderr_tail(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError) as excinfo:
        ffmpeg.run(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4")])
    message = str(excinfo.value)
    assert "ffmpeg failed with exit code" in message
    assert "No such file" in message


def test_run_needs_arguments() -> None:
    with pytest.raises(ValueError):
        ffmpeg.run([])


def test_probe_and_duration(sample_video: Path) -> None:
    info = ffmpeg.probe(sample_video)
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    assert ffmpeg.duration(sample_video) == pytest.approx(10.0, abs=0.25)


def test_probe_summary_describes_the_file(sample_video: Path) -> None:
    probe = ffmpeg.probe_summary(sample_video)
    assert (probe.width, probe.height) == (640, 360)
    assert probe.has_video is True
    assert probe.title == "Sample Clip"
    assert probe.duration == pytest.approx(10.0, abs=0.25)


def test_probe_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError, match="does not exist"):
        ffmpeg.probe(tmp_path / "ghost.mp4")


def test_probe_summary_marks_audio_only_files(tmp_path: Path) -> None:
    tone = make_tone(tmp_path / "tone.wav", 2.0)
    probe = ffmpeg.probe_summary(tone)
    assert probe.has_video is False
    assert probe.width == 0


def test_run_progress_follows_out_time_to_the_end_of_the_encode(tmp_path: Path) -> None:
    """3.1: `-progress pipe:1` lines become seconds of output, in order, never past the real length."""
    source = make_video(tmp_path / "src.mp4", seconds=4.0, size="160x120")
    seen: list[float] = []
    proc = ffmpeg.run_progress(
        ["-i", str(source), "-c:v", "libx264", "-preset", "ultrafast", str(tmp_path / "out.mp4")],
        seen.append,
    )
    assert proc.returncode == 0
    assert seen, "ffmpeg printed no progress block at all"
    assert seen == sorted(seen), seen  # microseconds, parsed once per block, never backwards
    assert seen[-1] == pytest.approx(4.0, abs=0.3)
    assert ffmpeg.duration(tmp_path / "out.mp4") == pytest.approx(4.0, abs=0.3)


def test_run_progress_reports_a_failure_like_run_does(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError) as excinfo:
        ffmpeg.run_progress(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4")], lambda _: None)
    assert "ffmpeg failed with exit code" in str(excinfo.value)
    assert "No such file" in str(excinfo.value)


def test_run_progress_refuses_ffprobe() -> None:
    with pytest.raises(ValueError, match="no -progress"):
        ffmpeg.run_progress(["ffprobe", "-i", "x.mp4"], lambda _: None)


# --------------------------------------------------------------------------------------------------
# inputs.py — uploads
# --------------------------------------------------------------------------------------------------


def test_validate_upload_accepts_a_normal_clip(sample_video: Path, settings: Settings) -> None:
    probe = inputs.validate_upload(sample_video, settings)
    assert probe.has_video and probe.duration == pytest.approx(10.0, abs=0.25)


def test_validate_upload_rejects_a_long_video(sample_video: Path) -> None:
    with pytest.raises(InputError) as excinfo:
        inputs.validate_upload(sample_video, Settings(max_video_seconds=5))
    message = str(excinfo.value)
    assert "10 s" in message and "5 s" in message  # the message names both numbers


def test_validate_upload_rejects_audio_only(tmp_path: Path, settings: Settings) -> None:
    tone = make_tone(tmp_path / "audio_only.wav", 2.0)
    with pytest.raises(InputError, match="no video stream"):
        inputs.validate_upload(tone, settings)


def test_validate_upload_rejects_junk(tmp_path: Path, settings: Settings) -> None:
    # ffmpeg's bintext demuxer "succeeds" on random bytes, so the real gate is: no audio, nothing to dub.
    junk = tmp_path / "notes.bin"
    junk.write_bytes(os.urandom(64_000))
    with pytest.raises(InputError, match="nothing to dub"):
        inputs.validate_upload(junk, settings)


def test_validate_upload_rejects_an_empty_file(tmp_path: Path, settings: Settings) -> None:
    empty = tmp_path / "upload.bin"
    empty.write_bytes(b"")
    with pytest.raises(InputError, match="missing or empty"):
        inputs.validate_upload(empty, settings)


def test_validate_upload_rejects_an_oversized_file(sample_video: Path) -> None:
    with pytest.raises(InputError, match="the limit is 0 MB"):
        inputs.validate_upload(sample_video, Settings(max_upload_mb=0))


def test_enforce_duration_names_both_numbers(settings: Settings) -> None:
    probe = Probe(duration=1200.0, width=1, height=1, title=None, has_video=True)
    with pytest.raises(InputError, match="1200 s"):
        inputs.enforce_duration(probe, settings)
    inputs.enforce_duration(Probe(10.0, 1, 1, None, True), settings)  # under the limit: silent


def test_remux_turns_an_odd_container_into_source_mp4(tmp_path: Path, settings: Settings) -> None:
    made = make_video(tmp_path / "clip.mkv", seconds=3.0, size="320x240")
    source = made.rename(tmp_path / "upload.bin")  # exactly how an upload lands on disk (D-26)
    dest = inputs.remux(source, tmp_path / "source.mp4")
    assert dest.exists()
    probe = inputs.validate_upload(dest, settings)
    assert probe.has_video and probe.duration == pytest.approx(3.0, abs=0.25)


def make_webm(path: Path, seconds: float = 2.0) -> Path:
    """A tiny VP9 + Opus WebM — exactly what `bestvideo+bestaudio` hands back for many videos."""
    ffmpeg.run(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x120:rate=10:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={seconds}",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "60k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-shortest",
            str(path),
        ]
    )
    return path


def test_ensure_mp4_codecs_reencodes_vp9_and_opus(tmp_path: Path) -> None:
    webm = make_webm(tmp_path / "vp9.webm")
    assert codecs(webm)["video"] == ["vp9"] and codecs(webm)["audio"] == ["opus"]
    fixed = inputs.ensure_mp4_codecs(webm)
    assert fixed != webm and fixed.suffix == ".mp4"
    assert codecs(fixed)["video"] == ["h264"]
    assert codecs(fixed)["audio"] == ["aac"]


def test_ensure_mp4_codecs_leaves_an_h264_file_alone(sample_video: Path, tmp_path: Path) -> None:
    h264 = tmp_path / "already.mp4"
    ffmpeg.run(["-i", str(sample_video), "-t", "1", "-c:v", "libx264", "-c:a", "aac", str(h264)])
    before = h264.stat().st_mtime_ns
    assert inputs.ensure_mp4_codecs(h264) == h264
    assert h264.stat().st_mtime_ns == before  # not rewritten


def test_remux_of_a_webm_upload_yields_a_playable_mp4(tmp_path: Path) -> None:
    """F7: a stream copy would have put VP9 + Opus inside source.mp4, and out.mp4 after it."""
    upload = make_webm(tmp_path / "clip.webm").rename(tmp_path / "upload.bin")
    dest = inputs.remux(upload, tmp_path / "source.mp4")
    assert dest == tmp_path / "source.mp4"  # the caller's filename survives the re-encode
    assert codecs(dest)["video"] == ["h264"]
    assert codecs(dest)["audio"] == ["aac"]
    assert sorted(p.name for p in tmp_path.glob("source*")) == ["source.mp4"]


# --------------------------------------------------------------------------------------------------
# inputs.py — URL safety (F9)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://169.254.169.254/latest",
        "http://10.0.0.5:8080/",
        "https://192.168.1.1/admin",
        "http://[::1]:9000/",
        "http://0.0.0.0/",
    ],
)
def test_check_public_url_rejects_addresses_inside_the_house(url: str) -> None:
    with pytest.raises(InputError, match="private network|http"):
        inputs.check_public_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "", "http://"])
def test_check_public_url_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(InputError):
        inputs.check_public_url(url)


def test_check_public_url_accepts_a_public_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hostname is resolved, not guessed — so the test resolves it too (no network needed)."""
    calls: list[str] = []

    def fake_getaddrinfo(host: str, *args: object, **kwargs: object) -> list[tuple]:
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.72.206", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    inputs.check_public_url(SAMPLE_URL)
    assert calls == ["www.youtube.com"]


def test_check_public_url_rejects_a_name_that_resolves_inwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DNS-rebinding shape: a public-looking name that answers with a private address."""

    def fake_getaddrinfo(host: str, *args: object, **kwargs: object) -> list[tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(InputError, match="private network"):
        inputs.check_public_url("https://metadata.example.com/latest/meta-data/")


def test_check_public_url_lets_an_unresolvable_name_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing resolves, nothing can be reached: yt-dlp reports it in its own words."""

    def fake_getaddrinfo(host: str, *args: object, **kwargs: object) -> list[tuple]:
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    inputs.check_public_url("https://not-a-real-host.example/watch?v=1")


# --------------------------------------------------------------------------------------------------
# inputs.py — audio extraction
# --------------------------------------------------------------------------------------------------


def test_extract_audio_writes_asr_and_reference_wavs(sample_video: Path, tmp_path: Path) -> None:
    source_wav, reference_wav = inputs.extract_audio(sample_video, tmp_path)
    assert source_wav.name == "source.wav" and reference_wav.name == "reference.wav"
    rate, channels, seconds = wav_info(source_wav)
    assert (rate, channels) == (16_000, 1)
    assert seconds == pytest.approx(10.0, abs=0.25)
    rate, channels, seconds = wav_info(reference_wav)
    assert (rate, channels) == (24_000, 1)
    assert seconds <= inputs.REFERENCE_SECONDS + 0.05


def test_extract_audio_reference_prefers_the_loudest_window(tmp_path: Path) -> None:
    # 40 s: silent for the first 20 s, a tone afterwards. The best 30 s window starts at 10 s.
    quiet_first = make_video(
        tmp_path / "quiet_first.mp4",
        seconds=40.0,
        size="160x120",
        audio_filter="volume=volume=0:enable='lt(t,20)'",
    )
    source_wav, reference_wav = inputs.extract_audio(quiet_first, tmp_path)
    assert wav_info(reference_wav)[2] == pytest.approx(30.0, abs=0.1)
    assert rms(reference_wav) > rms(source_wav, 0.0, 30.0) * 1.3


# --------------------------------------------------------------------------------------------------
# inputs.py — the reference clip chosen from the transcript (plan.md 3.7)
# --------------------------------------------------------------------------------------------------


def test_speech_window_start_picks_the_busiest_thirty_seconds() -> None:
    segments = [Segment(20.0, 25.0, "a"), Segment(26.0, 30.0, "b"), Segment(31.0, 40.0, "c")]
    assert inputs.speech_window_start(segments, 30.0, 60.0) == pytest.approx(20.0, abs=1.0)


def test_speech_window_start_falls_back_to_the_front() -> None:
    assert inputs.speech_window_start([], 30.0, 60.0) == 0.0  # no speech at all
    assert inputs.speech_window_start([Segment(5.0, 9.0, "a")], 30.0, 20.0) == 0.0  # shorter than a window


def test_speech_window_start_never_runs_past_the_end() -> None:
    """A sentence 5 s before the end cannot start a 30 s window there; the window is pulled back."""
    start = inputs.speech_window_start([Segment(55.0, 58.0, "late")], 30.0, 60.0)
    assert start == pytest.approx(30.0)


def test_reference_from_segments_cuts_the_window_holding_the_speech(tmp_path: Path) -> None:
    """60 s, tone only between 20 s and 40 s: the clip must start at the speech, not at the file."""
    source_wav = tmp_path / "source.wav"
    ffmpeg.run(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=60",
            "-af",
            "volume=volume=0:enable='not(between(t,20,40))'",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(source_wav),
        ]
    )
    segments = [Segment(20.0, 25.0, "a"), Segment(26.0, 30.0, "b"), Segment(31.0, 40.0, "c")]

    reference = inputs.reference_from_segments(source_wav, segments, tmp_path / "reference.wav")

    rate, channels, seconds = wav_info(reference)
    assert (rate, channels) == (24_000, 1)
    assert seconds <= inputs.REFERENCE_SECONDS + 0.05
    assert seconds == pytest.approx(30.0, abs=0.1)
    # The window is [20, 50]: its first 20 s carry the tone, its last 10 s the silence after it.
    quiet = rms(source_wav, 0.0, 19.0)
    assert rms(reference, 0.0, 19.0) > max(0.02, quiet * 10), "the reference starts before the speech does"
    assert rms(reference, 21.0, 30.0) < 0.01


def test_reference_from_segments_beats_the_loudest_window_on_a_noisy_intro(tmp_path: Path) -> None:
    """plan.md 3.7 'done when': 20 s of loud music first, quieter speech after — speech must win."""
    source_wav = tmp_path / "source.wav"
    ffmpeg.run(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=200:sample_rate=16000:duration=60",
            "-af",
            "volume=volume=0.05:enable='gt(t,20)',volume=volume=0:enable='gt(t,45)'",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(source_wav),
        ]
    )
    segments = [Segment(21.0, 30.0, "hello"), Segment(31.0, 44.0, "world")]
    loud = rms(source_wav, 0.0, 20.0)

    reference = inputs.reference_from_segments(source_wav, segments, tmp_path / "reference.wav")

    assert wav_info(reference)[0] == 24_000
    # The loudest 30 s window starts at 0; this one starts at the first sentence instead.
    assert rms(reference, 0.0, 20.0) < loud / 5, "the music won the window again"
    assert rms(reference, 0.0, 20.0) > 0.001, "the reference holds no sound at all"


def test_reference_from_segments_reports_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="is missing"):
        inputs.reference_from_segments(tmp_path / "ghost.wav", [], tmp_path / "reference.wav")


# --------------------------------------------------------------------------------------------------
# audio.py — fit, place, assemble
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factor",
    [0.1, 0.25, 0.5, 0.8, 1.0, 1.3, 2.0, 4.0, 5.0],
)
def test_atempo_chain_stays_in_range_and_multiplies_back(factor: float) -> None:
    chain = audio.atempo_chain(factor)
    assert chain, "the chain is never empty"
    assert all(audio.ATEMPO_MIN <= part <= audio.ATEMPO_MAX for part in chain)
    assert float(np.prod(chain)) == pytest.approx(factor, rel=1e-9)


def test_atempo_chain_uses_one_filter_inside_the_range() -> None:
    assert audio.atempo_chain(1.3) == [1.3]
    assert audio.atempo_chain(4.0) == [2.0, 2.0]
    assert audio.atempo_chain(0.25) == [0.5, 0.5]


def test_atempo_chain_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        audio.atempo_chain(0.0)


def test_fit_clamps_a_clip_that_is_far_too_long(tmp_path: Path) -> None:
    clip = make_tone(tmp_path / "clip.wav", 4.0)
    fitted, seconds = audio.fit(clip, 1.0, tmp_path / "fit_hi.wav")
    assert seconds == pytest.approx(4.0 / 1.3, abs=0.1)  # clamped at hi=1.3, not squeezed to 1 s
    assert wav_info(fitted)[:2] == (24_000, 1)


def test_fit_never_stretches_a_clip_that_is_shorter_than_its_slot(tmp_path: Path) -> None:
    """A short clip keeps its own speed; the rest of the slot is the pause the speaker took."""
    clip = make_tone(tmp_path / "short.wav", 1.0)
    fitted, seconds = audio.fit(clip, 4.0, tmp_path / "fit_lo.wav")
    assert seconds == pytest.approx(1.0, abs=0.05)
    assert wav_info(fitted)[:2] == (24_000, 1)


def test_fit_stretches_only_when_a_caller_asks_for_a_lower_bound(tmp_path: Path) -> None:
    clip = make_tone(tmp_path / "short_lo.wav", 1.0)
    _, seconds = audio.fit(clip, 4.0, tmp_path / "fit_lo_explicit.wav", lo=0.8)
    assert seconds == pytest.approx(1.0 / 0.8, abs=0.1)


def test_fit_chains_atempo_beyond_the_single_filter_range(tmp_path: Path) -> None:
    clip = make_tone(tmp_path / "long.wav", 4.0)
    _, seconds = audio.fit(clip, 1.0, tmp_path / "fit_chain.wav", lo=0.5, hi=4.0)
    assert seconds == pytest.approx(1.0, abs=0.1)  # factor 4.0 → atempo=2,atempo=2


def test_fit_rejects_an_empty_slot(tmp_path: Path) -> None:
    clip = make_tone(tmp_path / "c.wav", 1.0)
    with pytest.raises(ValueError, match="positive target"):
        audio.fit(clip, 0.0, tmp_path / "never.wav")


def test_place_pushes_a_late_clip_into_the_next_slot() -> None:
    segments = [
        Segment(start=0.0, end=2.0, text="one"),
        Segment(start=2.0, end=4.0, text="two"),
        Segment(start=10.0, end=12.0, text="three"),
    ]
    fitted = [(Path("a.wav"), 3.0), (Path("b.wav"), 1.0), (Path("c.wav"), 2.0)]
    placed = audio.place(segments, fitted)
    assert [(p.start, p.end) for p in placed] == [(0.0, 3.0), (3.0, 4.0), (10.0, 12.0)]
    assert [p.text for p in placed] == ["one", "two", "three"]


def test_place_checks_the_lists_line_up() -> None:
    with pytest.raises(ValueError, match="fitted clips"):
        audio.place([Segment(0.0, 1.0, "a")], [])


def test_assemble_is_exactly_the_video_length(tmp_path: Path) -> None:
    first = make_tone(tmp_path / "s1.wav", 1.0)
    second = make_tone(tmp_path / "s2.wav", 1.0, freq=660)
    placed = audio.place(
        [Segment(1.0, 2.0, "one"), Segment(4.0, 5.0, "two")],
        [(first, 1.0), (second, 1.0)],
    )
    result = audio.assemble(placed, 7.5, tmp_path / "dubbed.wav")
    assert (result.dropped_clips, result.dropped_seconds, result.trimmed_seconds) == (0, 0.0, 0.0)
    dubbed = result.path
    rate, channels, seconds = wav_info(dubbed)
    assert (rate, channels) == (24_000, 1)
    assert seconds == pytest.approx(7.5, abs=0.005)
    assert ffmpeg.duration(dubbed) == pytest.approx(7.5, abs=0.01)
    assert rms(dubbed, 0.0, 0.9) < 1e-4  # silence before the first clip
    assert rms(dubbed, 1.1, 1.9) > 0.05  # the clip itself (lavfi sine peaks at 0.125)
    assert rms(dubbed, 6.0, 7.5) < 1e-4  # silence after the last clip


def test_assemble_trims_a_clip_that_runs_past_the_end(tmp_path: Path) -> None:
    clip = make_tone(tmp_path / "tail.wav", 3.0)
    placed = audio.place([Segment(1.0, 2.0, "x")], [(clip, 3.0)])
    result = audio.assemble(placed, 2.0, tmp_path / "short.wav")
    assert wav_info(result.path)[2] == pytest.approx(2.0, abs=0.005)
    assert rms(result.path, 1.1, 2.0) > 0.05
    # F5: the caller has to be able to say that 2 s of speech was cut off.
    assert result.trimmed_seconds == pytest.approx(2.0, abs=0.01)
    assert (result.dropped_clips, result.dropped_seconds) == (0, 0.0)


def test_assemble_reports_clips_that_start_after_the_video_ends(tmp_path: Path) -> None:
    inside = make_tone(tmp_path / "inside.wav", 1.0)
    late = make_tone(tmp_path / "late.wav", 2.0, freq=660)
    later = make_tone(tmp_path / "later.wav", 1.5, freq=880)
    placed = audio.place(
        [Segment(0.0, 1.0, "a"), Segment(6.0, 8.0, "b"), Segment(9.0, 10.5, "c")],
        [(inside, 1.0), (late, 2.0), (later, 1.5)],
    )
    result = audio.assemble(placed, 5.0, tmp_path / "cut.wav")
    assert result.dropped_clips == 2
    assert result.dropped_seconds == pytest.approx(3.5, abs=0.02)
    assert result.trimmed_seconds == 0.0
    assert result.lost_seconds == pytest.approx(3.5, abs=0.02)
    assert wav_info(result.path)[2] == pytest.approx(5.0, abs=0.005)


def test_assemble_on_an_empty_timeline_is_silence(tmp_path: Path) -> None:
    result = audio.assemble([], 1.5, tmp_path / "silent.wav")
    assert wav_info(result.path)[2] == pytest.approx(1.5, abs=0.005)
    assert rms(result.path) == 0.0
    assert result.lost_seconds == 0.0


# --------------------------------------------------------------------------------------------------
# subtitles.py
# --------------------------------------------------------------------------------------------------


def test_build_srt_parses_and_keeps_milliseconds(tmp_path: Path) -> None:
    cues = [Cue(0.5, 2.25, "Hola mundo"), Cue(2.5, 4.125, "Segunda línea")]
    path = subtitles.build_srt(cues, tmp_path / "subs.srt")
    parsed = list(srt_lib.parse(path.read_text(encoding="utf-8")))
    assert [s.content for s in parsed] == ["Hola mundo", "Segunda línea"]
    assert parsed[0].start.total_seconds() == pytest.approx(0.5)
    assert parsed[0].end.total_seconds() == pytest.approx(2.25)
    assert parsed[1].end.total_seconds() == pytest.approx(4.125)
    assert "00:00:02,250" in path.read_text(encoding="utf-8")


def test_build_srt_collapses_a_flash_cue(tmp_path: Path) -> None:
    cues = [Cue(0.0, 1.0, "first"), Cue(1.0, 1.1, "oh"), Cue(1.2, 2.5, "second")]
    parsed = list(srt_lib.parse(subtitles.build_srt(cues, tmp_path / "s.srt").read_text("utf-8")))
    assert len(parsed) == 2
    assert parsed[0].content == "first oh"
    assert parsed[0].end.total_seconds() == pytest.approx(1.1)


def test_build_srt_collapses_a_leading_flash_cue(tmp_path: Path) -> None:
    cues = [Cue(0.0, 0.1, "hm"), Cue(0.2, 2.0, "then this")]
    parsed = list(srt_lib.parse(subtitles.build_srt(cues, tmp_path / "s.srt").read_text("utf-8")))
    assert len(parsed) == 1
    assert parsed[0].content == "hm then this"
    assert parsed[0].start.total_seconds() == pytest.approx(0.0)


def test_build_srt_drops_empty_cues(tmp_path: Path) -> None:
    cues = [Cue(0.0, 1.0, "  "), Cue(1.0, 2.0, "kept")]
    parsed = list(srt_lib.parse(subtitles.build_srt(cues, tmp_path / "s.srt").read_text("utf-8")))
    assert [s.content for s in parsed] == ["kept"]


def test_build_srt_wraps_long_lines_on_word_boundaries(tmp_path: Path) -> None:
    long_text = "Esta es una frase muy larga que no cabe en una sola linea de subtitulo y hay que partirla"
    path = subtitles.build_srt([Cue(0.0, 4.0, long_text)], tmp_path / "wrap.srt")
    content = list(srt_lib.parse(path.read_text(encoding="utf-8")))[0].content
    lines = content.splitlines()
    assert 1 < len(lines) <= subtitles.MAX_LINES
    assert " ".join(lines) == long_text  # no word is lost
    assert max(len(line) for line in lines) <= 52  # about 42, widened only to stay within two lines


def test_wrap_text_never_exceeds_two_lines() -> None:
    text = " ".join(["palabra"] * 40)
    assert len(subtitles.wrap_text(text).splitlines()) <= subtitles.MAX_LINES


def test_build_srt_rejects_a_backwards_cue(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ends before it starts"):
        subtitles.build_srt([Cue(2.0, 1.0, "backwards")], tmp_path / "bad.srt")


# --------------------------------------------------------------------------------------------------
# mux.py
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def dubbed(tmp_path: Path) -> Path:
    return audio.assemble(
        audio.place([Segment(1.0, 3.0, "hola")], [(make_tone(tmp_path / "seg.wav", 2.0), 2.0)]),
        10.0,
        tmp_path / "dubbed.wav",
    ).path


def test_mux_with_burn_reencodes_to_h264_and_attaches_mov_text(
    sample_video: Path, dubbed: Path, tmp_path: Path
) -> None:
    subs = subtitles.build_srt([Cue(1.0, 3.0, "Hola mundo")], tmp_path / "subs.srt")
    out = mux.mux(sample_video, dubbed, subs, burn=True, lang="es", out=tmp_path / "out.mp4")
    streams = codecs(out)
    assert streams["video"] == ["h264"]
    assert streams["audio"] == ["aac"]
    assert streams["subtitle"] == ["mov_text"]
    tags = next(s for s in ffmpeg.probe(out)["streams"] if s["codec_type"] == "subtitle")["tags"]
    assert tags["language"] == "spa"
    assert ffmpeg.duration(out) == pytest.approx(10.0, abs=0.5)


def test_mux_without_burn_copies_the_source_video(sample_video: Path, dubbed: Path, tmp_path: Path) -> None:
    subs = subtitles.build_srt([Cue(1.0, 3.0, "Hola mundo")], tmp_path / "subs.srt")
    out = mux.mux(sample_video, dubbed, subs, burn=False, lang="hi", out=tmp_path / "soft.mp4")
    streams = codecs(out)
    assert streams["video"] == codecs(sample_video)["video"] == ["mpeg4"]  # -c:v copy, no re-encode
    assert streams["subtitle"] == ["mov_text"]
    tags = next(s for s in ffmpeg.probe(out)["streams"] if s["codec_type"] == "subtitle")["tags"]
    assert tags["language"] == "hin"


def test_mux_without_subtitles_still_replaces_the_audio(
    sample_video: Path, dubbed: Path, tmp_path: Path
) -> None:
    out = mux.mux(sample_video, dubbed, None, burn=False, lang="es", out=tmp_path / "plain.mp4")
    streams = codecs(out)
    assert "subtitle" not in streams
    assert streams["audio"] == ["aac"]


def test_mux_burns_unicode_subtitles_from_an_awkward_path(
    sample_video: Path, dubbed: Path, tmp_path: Path
) -> None:
    tricky = tmp_path / "it's a:dir, with [brackets]"
    tricky.mkdir()
    cues = [
        Cue(0.5, 2.0, "Hola mundo, ¿qué tal?"),
        Cue(2.0, 3.5, "नमस्ते दुनिया"),
        Cue(3.5, 5.0, "مرحبا بالعالم"),
        Cue(5.0, 6.5, "你好，世界"),
    ]
    subs = subtitles.build_srt(cues, tricky / "subs.srt")
    out = mux.mux(sample_video, dubbed, subs, burn=True, lang="zh-cn", out=tricky / "out.mp4")
    assert out.exists() and out.stat().st_size > 0
    tags = next(s for s in ffmpeg.probe(out)["streams"] if s["codec_type"] == "subtitle")["tags"]
    assert tags["language"] == "zho"


def test_burned_subtitles_are_drawn_near_the_bottom(sample_video: Path, dubbed: Path, tmp_path: Path) -> None:
    """Same encode twice, once with a cue on screen: the difference must sit in the lower band."""
    visible = subtitles.build_srt([Cue(0.5, 4.0, "Hola mundo, esto es una prueba")], tmp_path / "on.srt")
    later = subtitles.build_srt([Cue(50.0, 52.0, "Hola mundo, esto es una prueba")], tmp_path / "off.srt")
    with_text = mux.mux(sample_video, dubbed, visible, burn=True, lang="es", out=tmp_path / "on.mp4")
    without = mux.mux(sample_video, dubbed, later, burn=True, lang="es", out=tmp_path / "off.mp4")
    a = gray_frame(with_text, 2.0, tmp_path / "a.raw")
    b = gray_frame(without, 2.0, tmp_path / "b.raw")
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
    height = difference.shape[0]
    bottom = float(difference[int(height * 0.75) :].mean())
    top = float(difference[: height // 2].mean())
    assert bottom > 1.0, "no burned text found in the bottom quarter of the frame"
    assert bottom > top * 5, f"text is not bottom-positioned (bottom {bottom:.2f} vs top {top:.2f})"


def test_mux_refuses_to_burn_without_subtitles(sample_video: Path, dubbed: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a subtitle file"):
        mux.mux(sample_video, dubbed, None, burn=True, lang="es", out=tmp_path / "never.mp4")


def test_mux_reports_progress_that_climbs_to_one(sample_video: Path, dubbed: Path, tmp_path: Path) -> None:
    """3.1: with a callback, mux parses `-progress` and always finishes on 1.0."""
    subs = subtitles.build_srt([Cue(1.0, 3.0, "Hola mundo")], tmp_path / "subs.srt")
    seen: list[float] = []
    out = mux.mux(
        sample_video, dubbed, subs, burn=True, lang="es", out=tmp_path / "out.mp4", progress=seen.append
    )
    assert out.is_file() and out.stat().st_size > 0
    assert seen, "no progress was reported at all"
    assert seen == sorted(seen), seen
    assert all(0.0 <= value <= 1.0 for value in seen), seen
    assert seen[-1] == 1.0
    assert ffmpeg.duration(out) == pytest.approx(10.0, abs=0.5)


def test_mux_without_burn_still_reports_progress(sample_video: Path, dubbed: Path, tmp_path: Path) -> None:
    """A stream copy can finish before ffmpeg prints a block; the bar must still reach the end."""
    seen: list[float] = []
    mux.mux(
        sample_video, dubbed, None, burn=False, lang="es", out=tmp_path / "copy.mp4", progress=seen.append
    )
    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_mux_reports_a_missing_input(sample_video: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dubbed audio"):
        mux.mux(sample_video, tmp_path / "ghost.wav", None, burn=False, lang="es", out=tmp_path / "x.mp4")


def test_iso639_2_maps_ui_codes_and_falls_back() -> None:
    assert mux.iso639_2("es") == "spa"
    assert mux.iso639_2("ZH-CN") == "zho"
    assert mux.iso639_2("hin") == "hin"
    assert mux.iso639_2("klingon") == "und"


def test_escape_filter_path_quotes_the_specials() -> None:
    escaped = mux.escape_filter_path("/tmp/it's a:dir/subs.srt")
    assert escaped.startswith("'") and escaped.endswith("'")
    assert r"\:" in escaped


# --------------------------------------------------------------------------------------------------
# inputs.py — the download progress hook (offline: yt-dlp itself is replaced)
# --------------------------------------------------------------------------------------------------


class FakeYoutubeDL:
    """Enough of `yt_dlp.YoutubeDL` to prove the hook wiring: it reports bytes, then writes the file."""

    #: 3.1 MB of 7.4 MB, then the whole thing — the numbers the assertions below spell out.
    EVENTS = (
        {"status": "downloading", "downloaded_bytes": 3_250_585, "total_bytes": 7_759_462},
        {"status": "downloading", "downloaded_bytes": 7_759_462, "total_bytes": 7_759_462},
        {"status": "finished"},
    )
    source: Path

    def __init__(self, opts: dict[str, object]) -> None:
        self.opts = opts

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def download(self, urls: list[str]) -> None:
        hooks: list[Any] = list(self.opts.get("progress_hooks") or [])  # type: ignore[arg-type]
        for hook in hooks:
            for event in self.EVENTS:
                hook(dict(event))
        target = Path(str(self.opts["outtmpl"]).replace("%(ext)s", "mp4"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(type(self).source, target)


def test_fetch_youtube_reports_bytes_through_the_progress_hook(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3.1: yt-dlp's progress_hooks become (fraction, "downloading 3.1 MB of 7.4 MB")."""
    FakeYoutubeDL.source = make_video(tmp_path / "served.mp4", seconds=2.0, size="160x120", vcodec="libx264")
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    seen: list[tuple[float, str]] = []

    out = inputs.fetch_youtube(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        tmp_path / "job",
        settings,
        lambda fraction, detail: seen.append((fraction, detail)),
    )

    assert out.name == "source.mp4" and out.stat().st_size > 0
    assert [fraction for fraction, _ in seen] == sorted(fraction for fraction, _ in seen)
    assert seen[0] == (pytest.approx(0.419, abs=0.01), "downloading 3.1 MB of 7.4 MB")
    assert seen[1] == (1.0, "downloading 7.4 MB of 7.4 MB")
    assert seen[-1] == (1.0, "merging the download")


def test_fetch_youtube_survives_a_hook_that_cannot_write(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed status write is cosmetic: it must never turn into "YouTube download failed"."""
    FakeYoutubeDL.source = make_video(tmp_path / "served.mp4", seconds=2.0, size="160x120", vcodec="libx264")
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)

    def explode(fraction: float, detail: str) -> None:
        raise OSError("status.json is on a full disk")

    out = inputs.fetch_youtube("https://www.youtube.com/watch?v=x", tmp_path / "job", settings, explode)
    assert out.is_file()


# --------------------------------------------------------------------------------------------------
# YouTube (network) — metadata only, never a download
# --------------------------------------------------------------------------------------------------


def test_probe_youtube_refuses_a_local_url_before_touching_the_network(settings: Settings) -> None:
    """F9 defence in depth: the pipeline checks the URL again, not only api._validate."""
    with pytest.raises(InputError, match="private network"):
        inputs.probe_youtube("http://127.0.0.1:8000/internal.mp4", settings)


@pytest.mark.slow
@pytest.mark.skipif(not NETWORK, reason="no network: youtube.com:443 unreachable")
def test_probe_youtube_reads_title_and_duration(settings: Settings) -> None:
    probe = inputs.probe_youtube(SAMPLE_URL, settings)
    assert probe.has_video
    assert probe.duration == pytest.approx(19.0, abs=2.0)
    assert probe.title and "zoo" in probe.title.lower()


@pytest.mark.slow
@pytest.mark.skipif(not NETWORK, reason="no network: youtube.com:443 unreachable")
def test_probe_youtube_reports_a_bad_url_clearly(settings: Settings) -> None:
    with pytest.raises(InputError):
        inputs.probe_youtube("https://www.youtube.com/watch?v=respeak_no_such_video", settings)
