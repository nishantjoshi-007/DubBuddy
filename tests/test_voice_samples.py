"""WP-3E: the voice-preview cache behind ``GET /api/voices/{backend}/{lang}/{voice}``.

Everything here is offline except the one ``slow`` test at the bottom, which really asks Kokoro to
speak and is skipped unless the model is already in the huggingface cache. The rest drives
:func:`respeak.pipeline.tts.samples.ensure_sample` with a fake backend, because what is being tested
is the caching and the locking, not the synthesiser.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from respeak.config import Settings
from respeak.jobs import JobStore, sweep
from respeak.pipeline.tts import TTSError, write_wav
from respeak.pipeline.tts import samples as samples_module
from respeak.pipeline.tts.kokoro import VOICE_IDS, VOICES
from respeak.pipeline.tts.samples import SAMPLE_TEXT, check_sample, ensure_sample, sample_path

# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Real Settings whose DATA_DIR is a throwaway directory."""
    return Settings(data_dir=tmp_path)


class FakeBackend:
    """A backend that writes a fixed tone and counts how often it was asked to."""

    name = "kokoro"
    cloning = False

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.delay = delay
        self.fail: Exception | None = None
        self._lock = threading.Lock()

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
        with self._lock:
            self.calls.append((text, lang, voice))
        if self.delay:
            time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        return write_wav(np.full(2400, 0.25, dtype=np.float32), out)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Replace the real synthesiser, keeping the real ``available_backends`` validation."""
    fake = FakeBackend()
    monkeypatch.setattr(samples_module, "get_backend", lambda _name, _settings: fake)
    return fake


# --------------------------------------------------------------------------- the sentences


def test_sample_text_covers_every_kokoro_language() -> None:
    """A language the picker offers must have something for its voices to say."""
    assert set(SAMPLE_TEXT) == set(VOICES)
    assert all(text.strip() for text in SAMPLE_TEXT.values())
    assert all(len(text) <= 60 for text in SAMPLE_TEXT.values()), "a preview is one short sentence"


# --------------------------------------------------------------------------- paths


def test_the_cache_lives_beside_the_jobs_directory_not_inside_it(settings: Settings) -> None:
    """The TTL sweeper walks ``jobs_dir`` only, so a sample cached outside it is never deleted."""
    path = sample_path(settings, "kokoro", "es", "ef_dora")
    assert path == settings.data_dir / "voice_samples" / "kokoro" / "es" / "ef_dora.wav"
    assert settings.jobs_dir not in path.parents

    # And prove it rather than trusting the layout: a sample older than any TTL survives a sweep.
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFF....WAVE")
    old = time.time() - 30 * 24 * 3600
    os.utime(path, (old, old))
    store = JobStore(settings.jobs_dir)
    assert sweep(store, ttl_minutes=0) == []
    assert path.is_file()


@pytest.mark.parametrize("segment", ["..", "../escape", "a/b", "", ".hidden", "x" * 80])
def test_a_path_segment_that_is_not_a_plain_name_is_refused(settings: Settings, segment: str) -> None:
    with pytest.raises(ValueError):
        sample_path(settings, "kokoro", "es", segment)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("lang", "voice", "fragment"),
    [
        ("es", "bogus", "no voice 'bogus'"),
        ("es", "am_adam", "no voice 'am_adam'"),  # a real id, but English
        ("es", "", "no voice ''"),
        ("zz", "ef_dora", "no voices for 'zz'"),
    ],
)
def test_an_impossible_voice_is_refused_and_nothing_is_written(
    settings: Settings, backend: FakeBackend, lang: str, voice: str, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        ensure_sample(settings, "kokoro", lang, voice)
    assert backend.calls == []
    assert not (settings.data_dir / "voice_samples").exists()


def test_an_unknown_backend_is_refused(settings: Settings, backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="unknown backend 'banana'"):
        ensure_sample(settings, "banana", "es", "ef_dora")
    assert backend.calls == []


def test_a_backend_with_no_preset_voices_is_refused(settings: Settings, backend: FakeBackend) -> None:
    """Chatterbox clones the original speaker; there is nothing to preview."""
    with pytest.raises(ValueError, match="no preset voices"):
        ensure_sample(settings, "chatterbox", "es", "ef_dora")
    assert backend.calls == []


def test_check_sample_normalises_the_triple_it_returns(settings: Settings) -> None:
    assert check_sample(settings, " KOKORO ", "es-MX", "em_alex") == ("kokoro", "es", "em_alex")


# --------------------------------------------------------------------------- caching


def test_a_sample_is_synthesised_once_and_then_reused(settings: Settings, backend: FakeBackend) -> None:
    first = ensure_sample(settings, "kokoro", "es", "ef_dora")
    assert first.is_file() and first.stat().st_size > 0
    assert backend.calls == [(SAMPLE_TEXT["es"], "es", "ef_dora")]

    again = ensure_sample(settings, "kokoro", "es", "ef_dora")
    assert again == first
    assert len(backend.calls) == 1, "the cached file must be served, not made again"

    other = ensure_sample(settings, "kokoro", "es", "em_alex")
    assert other != first
    assert len(backend.calls) == 2, "a different voice is a different sample"


def test_nothing_but_the_finished_wav_is_left_in_the_cache(settings: Settings, backend: FakeBackend) -> None:
    """The temp file the synthesiser writes into must never survive under its own name."""
    path = ensure_sample(settings, "kokoro", "es", "ef_dora")
    assert sorted(p.name for p in path.parent.iterdir()) == ["ef_dora.wav"]


def test_a_failed_synthesis_leaves_no_file_to_be_served_later(
    settings: Settings, backend: FakeBackend
) -> None:
    """Fail loudly: a half-written sample cached as a success would be silent audio forever."""
    backend.fail = TTSError("kokoro exploded")
    with pytest.raises(TTSError, match="exploded"):
        ensure_sample(settings, "kokoro", "es", "ef_dora")
    assert list((settings.data_dir / "voice_samples" / "kokoro" / "es").iterdir()) == []

    backend.fail = None
    assert ensure_sample(settings, "kokoro", "es", "ef_dora").is_file()
    assert len(backend.calls) == 2, "the failure must not have been cached"


def test_an_empty_file_is_treated_as_missing(settings: Settings, backend: FakeBackend) -> None:
    """A zero-byte leftover (a crash between create and write) must be regenerated, not served."""
    path = sample_path(settings, "kokoro", "es", "ef_dora")
    path.parent.mkdir(parents=True)
    path.touch()
    assert ensure_sample(settings, "kokoro", "es", "ef_dora") == path
    assert len(backend.calls) == 1
    assert path.stat().st_size > 0


def test_concurrent_callers_synthesise_exactly_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two browsers clicking the same voice at once cost one synthesis and get one whole file."""
    slow = FakeBackend(delay=0.2)
    monkeypatch.setattr(samples_module, "get_backend", lambda _name, _settings: slow)

    workers = 8
    start = threading.Barrier(workers)
    results: list[Path] = []
    errors: list[Exception] = []
    guard = threading.Lock()

    def ask() -> None:
        try:
            start.wait(10)
            path = ensure_sample(settings, "kokoro", "es", "ef_dora")
            size = path.stat().st_size
        except Exception as exc:  # reported through `errors`, so the assert names it
            with guard:
                errors.append(exc)
            return
        with guard:
            results.append(path)
            assert size > 0, "a half-written sample was served"

    threads = [threading.Thread(target=ask, name=f"preview-{n}") for n in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(results) == workers
    assert len(set(results)) == 1
    assert len(slow.calls) == 1, f"{len(slow.calls)} threads synthesised the same sample"


# --------------------------------------------------------------------------- the real thing


def _kokoro_snapshot() -> bool:
    """True when hexgrad/Kokoro-82M is already in the huggingface cache (no download needed)."""
    roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ]
    return any(root and Path(root, "models--hexgrad--Kokoro-82M").is_dir() for root in roots)


@pytest.mark.slow
@pytest.mark.skipif(not _kokoro_snapshot(), reason="Kokoro-82M is not in the huggingface cache")
def test_kokoro_really_speaks_the_spanish_sample(settings: Settings) -> None:
    """End to end, no fakes: the default Spanish voice says the Spanish sentence."""
    import soundfile as sf

    assert "ef_dora" in VOICE_IDS["es"]
    path = ensure_sample(settings, "kokoro", "es", "ef_dora")
    assert path == settings.data_dir / "voice_samples" / "kokoro" / "es" / "ef_dora.wav"

    audio, rate = sf.read(str(path))
    assert rate == 24_000
    assert audio.ndim == 1, "the fit and mux stages assume mono"
    assert 0.5 < len(audio) / rate < 15.0, "a preview is one short sentence"
    assert float(np.max(np.abs(audio))) > 0.01, "the sample is silence"
