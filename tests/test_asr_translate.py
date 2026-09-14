"""faster-whisper transcription (B4.3) and Argos translation through English (B4.4).

Offline once the caches described in the README exist: whisper `base` under the HF hub cache and the Argos
en->es package under ~/.local/share/argos-translate.  Anything needing a download skips instead of failing.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from respeak.config import Settings
from respeak.pipeline import asr
from respeak.pipeline.asr import MIN_SEGMENT_SECONDS, compute_type_for, load_model, transcribe
from respeak.pipeline.ffmpeg import ensure_binaries
from respeak.pipeline.translate import (
    ArgosTranslator,
    TranslationError,
    Translator,
    normalize_code,
)
from respeak.pipeline.types import Segment

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_WAV = FIXTURES / "speech_en.wav"
WHISPER_MODEL = "base"  # cached on the dev machine; `small` is the product default (D-06)


# --------------------------------------------------------------------------------------------------
# availability checks — every expensive test skips rather than fails
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


def _argos_installed(from_code: str, to_code: str) -> bool:
    from argostranslate import package as argos_package

    return any(
        p.from_code == from_code and p.to_code == to_code for p in argos_package.get_installed_packages()
    )


needs_whisper = pytest.mark.skipif(
    not _whisper_cached(WHISPER_MODEL) and not _has_network(),
    reason=f"whisper {WHISPER_MODEL} is not cached and there is no network to download it",
)
needs_en_es = pytest.mark.skipif(
    not _argos_installed("en", "es") and not _has_network("argos-net.com"),
    reason="the Argos en->es package is not installed and there is no network to download it",
)
needs_pivot_download = pytest.mark.skipif(
    not (_argos_installed("es", "en") and _argos_installed("en", "fr")) and not _has_network("argos-net.com"),
    reason="the es->en / en->fr Argos packages are not installed and there is no network",
)


@pytest.fixture(scope="session")
def speech_wav() -> Path:
    """A 19 s English sample, 16 kHz mono — exactly what B4.2 hands to B4.3."""
    if not SPEECH_WAV.exists():  # pragma: no cover - the fixture is committed
        _build_speech_fixture()
    if not SPEECH_WAV.exists():  # pragma: no cover
        pytest.skip(f"missing speech fixture {SPEECH_WAV}")
    return SPEECH_WAV


def _build_speech_fixture() -> None:  # pragma: no cover - only when the committed wav is absent
    """Re-create the fixture from RESPEAK_TEST_SPEECH with ffmpeg (how the committed file was made)."""
    source = os.environ.get("RESPEAK_TEST_SPEECH")
    if not source or not Path(source).is_file():
        return
    ensure_binaries()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    FIXTURES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(SPEECH_WAV),
        ],
        check=True,
    )


@pytest.fixture(scope="session")
def asr_settings() -> Settings:
    return Settings(whisper_model=WHISPER_MODEL)


# --------------------------------------------------------------------------------------------------
# asr.py
# --------------------------------------------------------------------------------------------------


@needs_whisper
def test_transcribe_autodetects_english(speech_wav: Path, asr_settings: Settings) -> None:
    fractions: list[float] = []
    transcript = transcribe(speech_wav, None, asr_settings, progress=fractions.append)

    assert transcript.language == "en"
    assert len(transcript.segments) >= 3
    assert all(seg.text.strip() for seg in transcript.segments)
    assert any(seg.words for seg in transcript.segments)

    previous_end = -1.0
    for seg in transcript.segments:
        assert seg.end > seg.start
        assert seg.start >= previous_end - 1e-6, "segments must not overlap or run backwards"
        assert seg.end - seg.start >= MIN_SEGMENT_SECONDS
        previous_end = seg.end
        for word in seg.words:
            assert seg.start - 1e-3 <= word.start <= word.end <= seg.end + 1e-3
            assert word.text == word.text.strip() and word.text

    text = " ".join(seg.text for seg in transcript.segments).lower()
    assert "elephants" in text, text

    assert fractions and fractions == sorted(fractions)
    assert 0.0 <= fractions[0] <= 1.0
    assert fractions[-1] == 1.0


@needs_whisper
def test_transcribe_with_forced_language(speech_wav: Path, asr_settings: Settings) -> None:
    transcript = transcribe(speech_wav, "en", asr_settings)
    assert transcript.language == "en"
    assert len(transcript.segments) >= 3
    assert "elephants" in " ".join(s.text for s in transcript.segments).lower()


@needs_whisper
def test_load_model_is_cached_per_process(asr_settings: Settings) -> None:
    assert load_model(asr_settings) is load_model(asr_settings)


def test_transcribe_raises_on_a_missing_file(tmp_path: Path, asr_settings: Settings) -> None:
    with pytest.raises(asr.ASRError, match="not found"):
        transcribe(tmp_path / "nope.wav", None, asr_settings)


def test_compute_type_follows_device() -> None:
    assert compute_type_for(Settings(device="cpu")) == "int8"
    assert compute_type_for(Settings(device="cpu", whisper_compute="float32")) == "float32"

    class _Cuda(Settings):
        def resolved_device(self) -> str:
            return "cuda"

    assert compute_type_for(_Cuda()) == "float16"
    assert compute_type_for(_Cuda(whisper_compute="int8_float16")) == "int8_float16"


def test_short_segments_merge_forward() -> None:
    merged = asr._merge_short(
        [
            Segment(0.0, 0.3, "Hi."),
            Segment(0.3, 2.0, "There you are."),
            Segment(2.0, 4.0, "Long enough."),
        ]
    )
    assert [(s.start, s.end, s.text) for s in merged] == [
        (0.0, 2.0, "Hi. There you are."),
        (2.0, 4.0, "Long enough."),
    ]


def test_a_trailing_short_segment_merges_backwards() -> None:
    merged = asr._merge_short([Segment(0.0, 3.0, "Long."), Segment(3.0, 3.2, "Bye.")])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end, merged[0].text) == (0.0, 3.2, "Long. Bye.")


# --------------------------------------------------------------------------------------------------
# translate.py
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def translator() -> ArgosTranslator:
    return ArgosTranslator(Settings())


def test_argos_satisfies_the_translator_protocol(translator: ArgosTranslator) -> None:
    assert isinstance(translator, Translator)


def test_normalize_code_strips_regions() -> None:
    assert normalize_code("zh-CN") == "zh"
    assert normalize_code(" PT_br ") == "pt"
    assert normalize_code("EN") == "en"
    for bad in ("", "eng", "e", "1a"):
        with pytest.raises(TranslationError):
            normalize_code(bad)


@needs_en_es
def test_translate_en_to_es_keeps_one_to_one(translator: ArgosTranslator) -> None:
    texts = ["The elephants have really long trunks.", "That is pretty much all there is to say."]
    out = translator.translate(texts, "en", "es")

    assert len(out) == len(texts)
    assert all(line.strip() for line in out)
    assert all(line == line.strip() for line in out)
    assert out != texts
    assert "elefantes" in out[0].lower(), out[0]


@needs_en_es
def test_blank_input_stays_blank_and_empty_list_is_empty(translator: ArgosTranslator) -> None:
    assert translator.translate([], "en", "es") == []
    out = translator.translate(["", "   ", "Good morning."], "en", "es")
    assert len(out) == 3
    assert out[0] == "" and out[1] == ""
    assert out[2].strip()


@needs_en_es
def test_region_tagged_codes_work(translator: ArgosTranslator) -> None:
    assert translator.translate(["Good morning."], "EN", "es-ES")[0].strip()


@needs_pivot_download
def test_pivot_pair_composes_through_english(translator: ArgosTranslator) -> None:
    """es -> fr exists only as es->en->fr; Argos composes it once both hops are installed."""
    translator.ensure_pair("es", "fr")
    out = translator.translate(["Los elefantes son muy grandes."], "es", "fr")
    assert len(out) == 1
    assert "éléphant" in out[0].lower(), out[0]
    assert out[0] != "Los elefantes son muy grandes."


@needs_pivot_download
def test_es_to_en_is_real_english_not_int8_garbage(translator: ArgosTranslator) -> None:
    """The es->en 1.9 model decodes to garbage under int8; I run models at their shipped precision."""
    import argostranslate.settings as argos_settings

    translator.ensure_pair("es", "en")
    assert argos_settings.compute_type == "default"
    out = translator.translate(["Los elefantes son muy grandes."], "es", "en")[0]
    assert "elephant" in out.lower(), out
    assert "mainstre" not in out


def test_ensure_pair_rejects_an_impossible_pair(translator: ArgosTranslator) -> None:
    with pytest.raises(TranslationError, match="zz"):
        translator.ensure_pair("zz", "en")


def test_ensure_pair_rejects_same_source_and_target(translator: ArgosTranslator) -> None:
    with pytest.raises(TranslationError, match="both 'es'"):
        translator.ensure_pair("es", "es")


def test_supported_targets_covers_the_ui_languages(translator: ArgosTranslator) -> None:
    from respeak.lang_codes import NAME_TO_CODE

    targets = translator.supported_targets("en")
    if not targets:  # pragma: no cover - only without an index and without network
        pytest.skip("no Argos package index available")
    assert "es" in targets and "en" not in targets
    missing = {code for code in NAME_TO_CODE.values() if code != "en"} - targets
    assert not missing, f"Argos index does not cover UI languages: {sorted(missing)}"
