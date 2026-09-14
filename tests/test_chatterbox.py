"""WP-3D: the Chatterbox cloning backend (flow.md B4.5, decisions D-07 / D-27).

Chatterbox ships as the optional `clone` extra, so almost everything here runs whether or not the
extra is installed: the language table, the protocol shape, the pre-flight errors and the promise
that `installed()` / `reason()` never drag torch into the web process are all answered without
importing the package.

The one test that actually speaks is marked `slow` and skips unless the extra imports *and*
ResembleAI/chatterbox is already in the huggingface cache (about 3 GB); tests never download it.
Run the fast ones alone with `uv run pytest tests/test_chatterbox.py -m "not slow"`.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from respeak.config import Settings, get_settings
from respeak.pipeline.tts import SAMPLE_RATE, TTSBackend, TTSError, available_backends
from respeak.pipeline.tts.chatterbox import (
    LANGUAGES,
    ChatterboxBackend,
    _pkg_resources_stub,
    _resample_linear,
    _to_mono_24k,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "speech_en.wav"

#: What `ChatterboxMultilingualTTS.from_pretrained()` pulls from the hub (mtl_tts.py, allow_patterns).
CHECKPOINT_FILES = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)


def _model_cached() -> bool:
    """True when every checkpoint file is already on disk. Tests must never start a 3 GB download."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # pragma: no cover - huggingface_hub ships with kokoro
        return False
    return all(
        isinstance(try_to_load_from_cache("ResembleAI/chatterbox", name), str) for name in CHECKPOINT_FILES
    )


needs_extra = pytest.mark.skipif(not ChatterboxBackend.installed(), reason="the `clone` extra is absent")
needs_model = pytest.mark.skipif(
    not (ChatterboxBackend.installed() and _model_cached()),
    reason="ResembleAI/chatterbox is not in the huggingface cache (tests never download it)",
)


def _import_mtl_tts_or_skip() -> Any:
    """`chatterbox.mtl_tts`, or a skip explaining why the installed extra cannot be imported.

    `installed()` only proves the files are there. A mismatched torch/torchaudio pair (the PyPI
    torchaudio wheel is a CUDA build and dies with `libcudart.so.*` next to a `+cpu` torch) fails
    with OSError rather than ImportError, so this catches everything.
    """
    try:
        return importlib.import_module("chatterbox.mtl_tts")
    except Exception as exc:  # pragma: no cover - depends on the installed wheels
        pytest.skip(f"the `clone` extra is installed but not importable: {exc}")


@pytest.fixture(scope="module")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="module")
def backend(settings: Settings) -> ChatterboxBackend:
    """Constructing the backend is always cheap: no import, no model, no network."""
    return ChatterboxBackend(settings)


@pytest.fixture(scope="module")
def reference_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Eight seconds of the English fixture as 24 kHz mono — what B4.7 hands the cloning backend."""
    from respeak.pipeline.ffmpeg import ensure_binaries

    ensure_binaries()
    out = tmp_path_factory.mktemp("reference") / "reference.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(FIXTURE), "-t", "8", "-ac", "1", "-ar", "24000",
         str(out)],
        check=True,
        timeout=60,
    )  # fmt: skip
    return out


def _wav_facts(path: Path) -> tuple[int, int, float]:
    info = sf.info(str(path))
    return info.samplerate, info.channels, info.frames / info.samplerate


# --- protocol ---------------------------------------------------------------------------------


def test_backend_satisfies_the_tts_protocol(backend: ChatterboxBackend) -> None:
    assert isinstance(backend, TTSBackend)
    assert backend.name == "chatterbox"
    assert backend.cloning is True


def test_voices_is_empty_because_the_backend_clones(backend: ChatterboxBackend) -> None:
    """Phase 3 (flow.md B4): an empty mapping is how the UI knows to hide the voice picker."""
    assert backend.voices() == {}


def test_synthesize_takes_the_phase_3_voice_keyword() -> None:
    signature = inspect.signature(ChatterboxBackend.synthesize)
    assert list(signature.parameters) == ["self", "text", "lang", "reference_wav", "out", "voice"]
    assert signature.parameters["voice"].default is None


def test_language_table_matches_the_23_the_model_speaks(backend: ChatterboxBackend) -> None:
    assert backend.languages() == set(LANGUAGES)
    assert len(LANGUAGES) == 23
    assert {"ar", "en", "es", "sw", "zh"} <= set(LANGUAGES)


@needs_extra
@pytest.mark.slow
def test_language_table_matches_the_installed_package() -> None:
    """Our copy of the table must not drift from the one the model validates against."""
    mtl_tts = _import_mtl_tts_or_skip()
    assert set(mtl_tts.SUPPORTED_LANGUAGES) == set(LANGUAGES)


@needs_extra
@pytest.mark.slow
def test_the_perth_watermarker_is_repaired_even_when_perth_was_imported_first() -> None:
    """chatterbox watermarks every clip, and on setuptools >= 81 that watermarker goes missing.

    `perth/perth_net/__init__.py` needs `pkg_resources`, which setuptools 84 no longer ships;
    `perth/__init__.py` swallows the ImportError and leaves `PerthImplicitWatermarker = None`,
    and the model constructor then dies with `TypeError: 'NoneType' object is not callable`.
    A fresh interpreter that imports perth *before* us is the case that regressed once.
    """
    code = (
        "import perth\n"
        "from respeak.pipeline.tts.chatterbox import _ensure_perth_watermarker\n"
        "try:\n"
        "    _ensure_perth_watermarker()\n"
        "except Exception as exc:\n"
        "    print('UNIMPORTABLE', type(exc).__name__, exc)\n"
        "else:\n"
        "    print('REPAIRED', callable(perth.PerthImplicitWatermarker))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    answer = proc.stdout.strip().splitlines()[-1]
    if answer.startswith("UNIMPORTABLE"):  # pragma: no cover - depends on the installed wheels
        pytest.skip(f"resemble-perth is installed but not importable: {answer}")
    assert answer == "REPAIRED True", proc.stdout


@needs_extra
@pytest.mark.slow
def test_the_pkg_resources_stub_does_not_outlive_the_perth_import(tmp_path: Path) -> None:
    """F2: the stub answers `resource_filename` and nothing else, and `sys.modules` is global.

    jieba — which misaki pulls in for Kokoro's Chinese voices — does `import pkg_resources` and
    then calls `pkg_resources.resource_stream()` to open its dictionary, so a stub left behind by
    a chatterbox import broke every `zh` job in a completely unrelated backend. TMPDIR points at an
    empty directory so that a jieba.cache built by an earlier run cannot hide the failure.
    """
    code = (
        "import sys\n"
        "from respeak.pipeline.tts.chatterbox import _ensure_perth_watermarker\n"
        "try:\n"
        "    _ensure_perth_watermarker()\n"
        "except Exception as exc:\n"
        "    print('UNIMPORTABLE', type(exc).__name__, exc)\n"
        "    raise SystemExit(0)\n"
        "print('LEAKED', 'pkg_resources' in sys.modules)\n"
        "import jieba\n"
        "print('JIEBA', ''.join(jieba.lcut('\\u4f60\\u597d\\u4e16\\u754c')))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    if any(line.startswith("UNIMPORTABLE") for line in lines):  # pragma: no cover - broken wheels
        pytest.skip(f"resemble-perth is installed but not importable: {lines[-1]}")
    assert "LEAKED False" in lines, proc.stdout
    assert "JIEBA 你好世界" in lines, proc.stdout


def test_the_stub_never_shadows_a_real_pkg_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """F2: on a venv that still ships setuptools' pkg_resources, the stub must not be installed."""
    real = types.ModuleType("pkg_resources")
    monkeypatch.setitem(sys.modules, "pkg_resources", real)
    with _pkg_resources_stub():
        assert sys.modules["pkg_resources"] is real
    assert sys.modules["pkg_resources"] is real


def test_the_stub_is_installed_and_removed_again_when_nothing_provides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2: inside the block perth can find `resource_filename`; outside, the name is gone again."""
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **kw: None)
    with _pkg_resources_stub():
        shim = sys.modules["pkg_resources"]
        assert callable(shim.resource_filename)
        assert not hasattr(shim, "resource_stream"), "a fuller stub would be a lie, not a fix"
        assert Path(shim.resource_filename("respeak", "config.py")).is_file()
    assert "pkg_resources" not in sys.modules


# --- pre-flight errors (no model, no import) ----------------------------------------------------


def test_refuses_a_language_it_cannot_speak(backend: ChatterboxBackend, tmp_path: Path) -> None:
    with pytest.raises(TTSError, match="cannot speak"):
        backend.synthesize("Ahoj", "cs", None, tmp_path / "never.wav")


def test_refuses_a_reference_clip_that_is_not_there(backend: ChatterboxBackend, tmp_path: Path) -> None:
    with pytest.raises(TTSError, match="does not exist"):
        backend.synthesize("Hola", "es", tmp_path / "gone.wav", tmp_path / "never.wav")


@pytest.mark.parametrize("text", ["", "  \n\t "])
def test_empty_text_becomes_a_short_silence(backend: ChatterboxBackend, tmp_path: Path, text: str) -> None:
    """One blank translated line must not fail a job, and must not load a 3 GB model either."""
    out = backend.synthesize(text, "es", None, tmp_path / "nested" / "empty.wav", voice="ignored")
    rate, channels, seconds = _wav_facts(out)
    assert (rate, channels) == (SAMPLE_RATE, 1)
    assert 0.1 < seconds < 0.5
    assert abs(sf.read(str(out))[0]).max() == 0.0


# --- registry ----------------------------------------------------------------------------------


@needs_extra
def test_available_backends_reports_the_installed_extra(settings: Settings) -> None:
    info = available_backends(settings)["chatterbox"]
    assert info.installed is True
    assert info.reason is None
    assert info.cloning is True
    assert info.languages == set(LANGUAGES)


def test_installed_and_reason_never_import_torch_or_chatterbox() -> None:
    """`GET /api/backends` calls both on every request; neither may cost an 8 s torch import.

    A fresh interpreter is the only honest witness: by this point the test session has torch loaded.
    """
    code = (
        "import sys\n"
        "from respeak.pipeline.tts.chatterbox import ChatterboxBackend\n"
        "print(ChatterboxBackend.installed(), ChatterboxBackend.reason())\n"
        "heavy = [m for m in sys.modules if m == 'torch' or m.startswith(('torch.', 'chatterbox'))]\n"
        "print(','.join(sorted(heavy)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    answer, heavy = proc.stdout.splitlines()[:2]
    assert answer.startswith("True" if ChatterboxBackend.installed() else "False")
    assert heavy == "", f"installed()/reason() pulled in {heavy}"


# --- audio conversion helpers -------------------------------------------------------------------


def test_resample_linear_lands_on_the_new_grid() -> None:
    audio = np.sin(np.linspace(0.0, 4.0 * np.pi, 48_000, dtype=np.float32))
    out = _resample_linear(audio, 48_000, SAMPLE_RATE)
    assert out.dtype == np.float32
    assert out.size == 24_000
    assert _resample_linear(audio, SAMPLE_RATE, SAMPLE_RATE) is audio


def test_to_mono_24k_flattens_and_resamples_arrays() -> None:
    """The model returns (1, n) at 24 kHz; anything wider or at another rate must still work."""
    stereo = np.array([[0.0, 1.0, 0.0, -1.0], [0.0, -1.0, 0.0, 1.0]], dtype=np.float32)
    assert _to_mono_24k(stereo, SAMPLE_RATE).tolist() == [0.0, 0.0, 0.0, 0.0]
    at_48k = np.zeros((1, 48_000), dtype=np.float32)
    assert _to_mono_24k(at_48k, 48_000).size == 24_000


# --- the real thing (needs the extra and the cached checkpoint) ----------------------------------


@needs_model
@pytest.mark.slow
def test_clones_the_reference_speaker_into_24k_mono(
    backend: ChatterboxBackend, reference_wav: Path, tmp_path: Path
) -> None:
    """CPU synthesis of one sentence takes minutes here; that is what `slow` means."""
    _import_mtl_tts_or_skip()
    text = "Hola, estamos frente a los elefantes."
    out = backend.synthesize(text, "es", reference_wav, tmp_path / "clone.wav")

    assert out.exists() and out.stat().st_size > 0
    rate, channels, seconds = _wav_facts(out)
    assert rate == SAMPLE_RATE
    assert channels == 1
    assert 1.0 < seconds < 20.0
    samples, _ = sf.read(str(out))
    assert 0.02 < abs(samples).max() <= 1.0, "the clip is silent or clipped"
    assert backend._load().sr == SAMPLE_RATE, "chatterbox already speaks at our output rate"
