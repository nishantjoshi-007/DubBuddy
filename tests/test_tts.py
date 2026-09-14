"""WP-D and WP-3B: the TTS backends, their registry and the voice table (flow.md B4.5, plan.md 3.3).

The Kokoro tests need hexgrad/Kokoro-82M plus the voices they actually speak with; those are in the
huggingface cache on a normal dev box. When neither the cache nor the network has them the synthesis
tests skip instead of failing, and the table-integrity test skips unless it can see the real repo.
"""

from __future__ import annotations

import hashlib
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
from respeak.pipeline.tts.kokoro import (
    REPO_ID,
    VOICE_IDS,
    VOICES,
    VOICES_BY_LANG,
    KokoroBackend,
    display_name,
)
from respeak.pipeline.types import Voice

KOKORO_FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/ef_dora.pt")
ALL_VOICE_IDS = [voice_id for ids in VOICE_IDS.values() for voice_id in ids]


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


def _repo_voice_ids() -> set[str] | None:
    """Every ``voices/*.pt`` in the real Kokoro repo, or None when nothing can answer offline.

    A complete snapshot in the huggingface cache answers without a request; a partial one (the usual
    case — only the voices this machine has spoken were downloaded) cannot prove an id is *missing*,
    so the hub is asked instead, and the test skips when there is no network either.
    """
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache
    except Exception:  # pragma: no cover - huggingface_hub ships with kokoro
        return None
    cached = try_to_load_from_cache(REPO_ID, "config.json")
    if isinstance(cached, str):
        voices_dir = Path(cached).parent / "voices"
        local = {path.stem for path in voices_dir.glob("*.pt")} if voices_dir.is_dir() else set()
        if len(local) >= len(ALL_VOICE_IDS):  # a full snapshot; no request needed
            return local
    if not _has_network():
        return None
    try:
        files = HfApi().list_repo_files(REPO_ID)
    except Exception:  # pragma: no cover - the hub answered the connection check but not this
        return None
    return {name.removeprefix("voices/").removesuffix(".pt") for name in files if name.endswith(".pt")}


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


def test_available_backends_carries_the_voice_tables(settings: Settings) -> None:
    """The registry fills `BackendInfo.voices`; a cloning backend offers none (plan.md 3.3)."""
    infos = available_backends(settings)
    kokoro = infos["kokoro"]
    assert set(kokoro.voices) == kokoro.languages
    assert kokoro.voices["es"][0] == Voice(id="ef_dora", name="Dora (female)")
    assert all(isinstance(voice, Voice) for voices in kokoro.voices.values() for voice in voices)
    assert infos["chatterbox"].voices == {}, "chatterbox clones the speaker; it has no preset voices"


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
    assert isinstance(kokoro, TTSBackend)  # name, cloning, languages(), voices(), synthesize()
    assert kokoro.name == "kokoro"
    assert kokoro.cloning is False
    assert isinstance(kokoro.voices(), dict)


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


# --- the voice table (plan.md 3.3) -------------------------------------------------------------


def test_the_voice_table_covers_every_language_default_first(kokoro: KokoroBackend) -> None:
    voices = kokoro.voices()
    assert set(voices) == kokoro.languages()
    for lang, entries in voices.items():
        lang_code, default = VOICES[lang]
        assert entries, f"{lang} lists no voice at all"
        assert entries[0].id == default, f"{lang} must keep its curated default first"
        assert default.startswith(lang_code), f"{default} is not a {lang_code!r} voice"
        assert [voice.id for voice in entries] == list(VOICE_IDS[lang])
        assert len({voice.id for voice in entries}) == len(entries), f"{lang} repeats a voice"
        # every ISO code happens to begin with its kokoro lang_code; English has two (US and UK)
        prefixes = ("a", "b") if lang == "en" else (lang[0],)
        for voice in entries:
            assert voice.id.startswith(prefixes), f"{voice.id} is not a {lang} voice"
            assert voice.id[1] in "fm", f"{voice.id} says neither female nor male"
            assert voice.name and voice.name[0].isupper()


def test_voice_ids_are_unique_across_languages() -> None:
    assert len(set(ALL_VOICE_IDS)) == len(ALL_VOICE_IDS)


def test_display_name_reads_like_a_person() -> None:
    assert display_name("af_heart") == "Heart (female, US)"
    assert display_name("am_adam") == "Adam (male, US)"
    assert display_name("bf_emma") == "Emma (female, UK)"
    assert display_name("zm_yunjian") == "Yunjian (male)"
    assert display_name("ef_dora") == "Dora (female)"


def test_voices_is_a_copy_not_the_table(kokoro: KokoroBackend) -> None:
    """A caller that sorts its answer must not reorder the table everyone else reads."""
    taken = kokoro.voices()
    assert taken == VOICES_BY_LANG
    assert taken["es"] is not VOICES_BY_LANG["es"]
    taken["es"].clear()
    taken.pop("en")
    assert kokoro.voices()["es"][0].id == "ef_dora"
    assert "en" in kokoro.voices()


def test_the_voice_table_only_lists_voices_the_repo_really_has() -> None:
    """Every id must be a file in hexgrad/Kokoro-82M, or a job picking it fails at synthesis."""
    repo = _repo_voice_ids()
    if repo is None:
        pytest.skip("the Kokoro repo is neither fully cached nor reachable")
    missing = sorted(set(ALL_VOICE_IDS) - repo)
    assert missing == [], f"VOICE_IDS lists voices that do not exist: {missing}"


def test_resolve_voice_picks_the_default_and_honours_a_choice(kokoro: KokoroBackend) -> None:
    assert kokoro.resolve_voice("es") == ("e", "ef_dora")
    assert kokoro.resolve_voice("ES-mx", "  em_alex ") == ("e", "em_alex")
    # the first letter of the id is the kokoro lang_code, so a British voice switches the pipeline
    assert kokoro.resolve_voice("en") == ("a", "af_heart")
    assert kokoro.resolve_voice("en", "bm_george") == ("b", "bm_george")


def test_an_unknown_voice_is_refused_with_the_list(kokoro: KokoroBackend, tmp_path: Path) -> None:
    with pytest.raises(TTSError, match="no voice 'bogus'") as raised:
        kokoro.synthesize("Hola", "es", None, tmp_path / "never.wav", voice="bogus")
    assert "ef_dora" in str(raised.value)
    assert not (tmp_path / "never.wav").exists()


def test_a_voice_from_another_language_is_refused(kokoro: KokoroBackend, tmp_path: Path) -> None:
    """`am_adam` speaks English; used for Spanish it would silently be an American accent."""
    with pytest.raises(TTSError, match="no voice 'am_adam'"):
        kokoro.synthesize("Hola", "es", None, tmp_path / "never.wav", voice="am_adam")


@needs_kokoro_model
def test_two_voices_produce_two_different_dubs(kokoro: KokoroBackend, tmp_path: Path) -> None:
    """plan.md 3.3 "Done when": two jobs with different voices sound different (D-37)."""
    text = "Hello, this is a dubbing test."
    default = kokoro.synthesize(text, "en", None, tmp_path / "default.wav")
    picked = kokoro.synthesize(text, "en", None, tmp_path / "adam.wav", voice="am_adam")
    assert _wav_facts(picked)[:2] == (SAMPLE_RATE, 1)
    assert hashlib.sha256(default.read_bytes()).digest() != hashlib.sha256(picked.read_bytes()).digest()


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
