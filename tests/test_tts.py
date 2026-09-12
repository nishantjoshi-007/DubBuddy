"""WP-D: the TTS backends and their registry (flow.md B4.5).

The Kokoro tests need hexgrad/Kokoro-82M plus the two voices they use; those are in the
huggingface cache on a normal dev box. When neither the cache nor the network has them the
synthesis tests skip instead of failing.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import soundfile as sf

from respeak.config import Settings, get_settings
from respeak.pipeline.tts import (
    SAMPLE_RATE,
    TTSBackend,
    TTSError,
    available_backends,
    get_backend,
)
from respeak.pipeline.tts.chatterbox import LANGUAGES as CHATTERBOX_LANGUAGES
from respeak.pipeline.tts.chatterbox import NOT_INSTALLED_REASON, ChatterboxBackend
from respeak.pipeline.tts.kokoro import VOICES, KokoroBackend

KOKORO_FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/ef_dora.pt")


def _kokoro_cached() -> bool:
    """True when every file the synthesis tests need is already in the huggingface cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # pragma: no cover - huggingface_hub ships with kokoro
        return False
    return all(isinstance(try_to_load_from_cache("hexgrad/Kokoro-82M", name), str) for name in KOKORO_FILES)


def _has_network() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2.0).close()
        return True
    except OSError:
        return False


KOKORO_AVAILABLE = _kokoro_cached() or _has_network()
needs_kokoro_model = pytest.mark.skipif(
    not KOKORO_AVAILABLE, reason="Kokoro-82M is neither cached nor downloadable"
)
needs_chatterbox_missing = pytest.mark.skipif(
    ChatterboxBackend.installed(), reason="the `clone` extra is installed here"
)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="module")
def kokoro(settings: Settings) -> KokoroBackend:
    backend = get_backend("kokoro", settings)
    assert isinstance(backend, KokoroBackend)
    return backend


def _wav_facts(path: Path) -> tuple[int, int, float]:
    info = sf.info(str(path))
    return info.samplerate, info.channels, info.frames / info.samplerate


# --- registry ---------------------------------------------------------------------------------


def test_available_backends_shape(settings: Settings) -> None:
    infos = available_backends(settings)
    assert set(infos) == {"kokoro", "chatterbox"}
    for name, info in infos.items():
        assert info.name == name
        assert isinstance(info.installed, bool)
        assert isinstance(info.languages, set) and info.languages
        assert info.reason is None or isinstance(info.reason, str)
    assert infos["kokoro"].installed is True
    assert infos["kokoro"].cloning is False
    assert infos["kokoro"].reason is None
    assert infos["chatterbox"].cloning is True
    assert infos["chatterbox"].languages == set(CHATTERBOX_LANGUAGES)


@needs_chatterbox_missing
def test_chatterbox_reports_not_installed_with_the_uv_hint(settings: Settings) -> None:
    info = available_backends(settings)["chatterbox"]
    assert info.installed is False
    assert info.reason is not None
    assert "uv sync --extra clone" in info.reason

    with pytest.raises(TTSError) as raised:
        get_backend("chatterbox", settings)
    assert "uv sync --extra clone" in str(raised.value)


def test_get_backend_rejects_an_unknown_name(settings: Settings) -> None:
    with pytest.raises(TTSError, match="unknown backend"):
        get_backend("fish", settings)


def test_get_backend_is_cached_per_process(settings: Settings) -> None:
    assert get_backend("kokoro", settings) is get_backend("KOKORO ", settings)


def test_kokoro_satisfies_the_protocol(kokoro: KokoroBackend) -> None:
    assert isinstance(kokoro, TTSBackend)
    assert kokoro.name == "kokoro"
    assert kokoro.cloning is False


# --- kokoro -----------------------------------------------------------------------------------


def test_kokoro_languages_and_voices(kokoro: KokoroBackend) -> None:
    assert kokoro.languages() == {"en", "es", "fr", "hi", "it", "ja", "pt", "zh"}
    assert set(VOICES) == kokoro.languages()
    assert kokoro.voice_for("es") == "ef_dora"
    assert kokoro.voice_for("en") == "af_heart"
    assert kokoro.voice_for("ZH-CN") == "zf_xiaobei"  # regional suffixes are normalised away


def test_kokoro_refuses_a_language_it_cannot_speak(kokoro: KokoroBackend) -> None:
    with pytest.raises(TTSError, match="cannot speak"):
        kokoro.synthesize("Guten Tag", "de", None, Path("/tmp/never-written.wav"))


def test_kokoro_japanese_dictionary_error_names_the_unidic_command() -> None:
    """A missing unidic dictionary must tell the user the one command that fixes it."""
    error = KokoroBackend._load_error("j", RuntimeError("Failed initializing MeCab ... -d .../unidic/dicdir"))
    assert isinstance(error, TTSError)
    assert "uv run python -m unidic download" in str(error)


def test_kokoro_other_load_errors_are_wrapped_plainly() -> None:
    error = KokoroBackend._load_error("e", RuntimeError("boom"))
    assert "unidic" not in str(error)
    assert "boom" in str(error)


@needs_kokoro_model
@pytest.mark.parametrize(
    ("lang", "text"),
    [("es", "Hola, esto es una prueba de doblaje."), ("en", "Hello, this is a dubbing test.")],
)
def test_kokoro_synthesizes_24k_mono(kokoro: KokoroBackend, tmp_path: Path, lang: str, text: str) -> None:
    out = kokoro.synthesize(text, lang, None, tmp_path / f"seg_{lang}.wav")
    assert out.exists() and out.stat().st_size > 0
    rate, channels, seconds = _wav_facts(out)
    assert rate == SAMPLE_RATE
    assert channels == 1
    assert 0.5 < seconds < 10.0


@needs_kokoro_model
def test_kokoro_ignores_a_reference_clip(kokoro: KokoroBackend, tmp_path: Path) -> None:
    """Kokoro does not clone: a reference wav must neither be read nor cause a failure."""
    out = kokoro.synthesize("Buenos dias.", "es", tmp_path / "missing-reference.wav", tmp_path / "ref.wav")
    assert _wav_facts(out)[0] == SAMPLE_RATE


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_text_becomes_a_short_silence(kokoro: KokoroBackend, tmp_path: Path, text: str) -> None:
    out = kokoro.synthesize(text, "es", None, tmp_path / "nested" / "empty.wav")
    rate, channels, seconds = _wav_facts(out)
    assert (rate, channels) == (SAMPLE_RATE, 1)
    assert 0.1 < seconds < 0.5
    samples, _ = sf.read(str(out))
    assert abs(samples).max() == 0.0


# --- chatterbox (never loads the model: the extra is not installed here) -----------------------


def test_chatterbox_language_table(settings: Settings) -> None:
    backend = ChatterboxBackend(settings)
    assert backend.name == "chatterbox"
    assert backend.cloning is True
    assert isinstance(backend, TTSBackend)
    assert len(backend.languages()) == 23
    assert {"ar", "en", "sw", "zh"} <= backend.languages()


def test_chatterbox_refuses_a_language_before_loading_anything(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(TTSError, match="cannot speak"):
        ChatterboxBackend(settings).synthesize("hello", "cs", None, tmp_path / "x.wav")


@needs_chatterbox_missing
def test_chatterbox_synthesize_explains_the_missing_extra(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(TTSError) as raised:
        ChatterboxBackend(settings).synthesize("hello", "en", None, tmp_path / "x.wav")
    assert NOT_INSTALLED_REASON in str(raised.value)
