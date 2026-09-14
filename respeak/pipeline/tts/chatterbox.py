"""Chatterbox Multilingual, the optional cloning backend (flow.md B4.5, decisions D-07 / D-27).

MIT, ~500M parameters, 23 languages, zero-shot cloning from `reference.wav` — and a GPU in
practice. It ships as the `clone` extra, so it is usually absent: nothing here is imported at
module scope, the class always constructs, and `installed()` / `reason()` let
`available_backends()` explain what to run.

Measured on my 4-core laptop CPU: 3.0 GB of checkpoints from ResembleAI/chatterbox on
first use, 25-50 s to load them, then 70-90 s per sentence — about 25x slower than real time,
against roughly real time for Kokoro. The UI says "GPU recommended" for a reason.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ...config import Settings
from ..types import Voice
from . import SAMPLE_RATE, TTSError, normalise_lang, write_silence, write_wav

log = logging.getLogger(__name__)

NOT_INSTALLED_REASON = "not installed: run `uv sync --extra clone`"

LANGUAGES: frozenset[str] = frozenset(
    {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    }
)  # fmt: skip


@contextmanager
def _pkg_resources_stub() -> Iterator[None]:
    """Give resemble-perth the one `pkg_resources` function it asks for, *only* while it imports.

    chatterbox watermarks every clip it generates with resemble-perth, and
    `perth/perth_net/__init__.py` opens with `from pkg_resources import resource_filename`.
    setuptools 81 deprecated that module and setuptools 84 — what a Python 3.12+ venv gets — no
    longer ships it, so the import fails, `perth/__init__.py` swallows the ImportError and leaves
    `perth.PerthImplicitWatermarker = None`, and `ChatterboxMultilingualTTS.__init__` then dies
    with `TypeError: 'NoneType' object is not callable`.

    perth uses it once, to find a directory inside its own package, so we supply exactly that
    rather than drop the watermark or pin setuptools back.

    Why a context manager and not a one-way `sys.modules` entry (review finding F2): the stub
    answers to `resource_filename` and nothing else, and it is *global*. jieba — which misaki
    pulls in for Kokoro's Chinese voices — starts with `import pkg_resources` and then calls
    `pkg_resources.resource_stream` to open its dictionary, so a leaked stub turned every `zh`
    job into `AttributeError: module 'pkg_resources' has no attribute 'resource_stream'`, in a
    backend that has nothing to do with chatterbox. Importers keep the *function* they bound, not
    the module, so taking the stub back out afterwards costs perth nothing.
    """
    if "pkg_resources" in sys.modules or importlib.util.find_spec("pkg_resources") is not None:
        yield  # a real pkg_resources is present: never shadow it
        return

    def resource_filename(package: str, resource: str) -> str:
        module = sys.modules.get(package)
        origin = getattr(module, "__file__", None)
        if origin is None:
            spec = importlib.util.find_spec(package)
            origin = None if spec is None else spec.origin
        if origin is None:
            raise ImportError(f"cannot locate {package!r} to resolve {resource!r}")
        return str(Path(origin).parent / resource)

    log.debug("setuptools no longer ships pkg_resources; installing the stub resemble-perth needs")
    shim = types.ModuleType("pkg_resources")
    shim.resource_filename = resource_filename  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim
    try:
        yield
    finally:
        if sys.modules.get("pkg_resources") is shim:
            del sys.modules["pkg_resources"]
            log.debug("removed the pkg_resources stub again; it must not outlive the perth import")


def _ensure_perth_watermarker() -> None:
    """Make sure `perth.PerthImplicitWatermarker` is a class, not the None its import guard leaves.

    The stub only helps if it is in place before anything imports `perth`; a test, the CLI or a
    prewarm run may have imported chatterbox first and got the broken module, which then stays in
    `sys.modules`. So repair it afterwards too: the failed submodule is not cached, so importing it
    again with the stub in place succeeds.
    """
    with _pkg_resources_stub():
        import perth

        if getattr(perth, "PerthImplicitWatermarker", None) is not None:
            return
        from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

        log.debug("restoring perth.PerthImplicitWatermarker, which perth imported before the stub existed")
        perth.PerthImplicitWatermarker = PerthImplicitWatermarker


def _model_class() -> Any:
    """Import `ChatterboxMultilingualTTS` lazily. Raises ImportError when the extra is absent."""
    _ensure_perth_watermarker()
    with _pkg_resources_stub():  # chatterbox pulls in more of the resemble stack on its way down
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS


def _import_problem() -> str | None:
    """None when the `clone` extra is present, otherwise a one-line reason for the UI.

    This answers `GET /api/backends`, so it must stay cheap and it must not pull torch into the
    web process. `find_spec("chatterbox.mtl_tts")` would do both: locating a *sub*module executes
    the parent package, and `chatterbox/__init__.py` imports `chatterbox.tts`, which imports torch
    (about eight seconds, and it trips tests/test_api.py's "the api never loads torch" check). So
    we locate the top-level package — which `find_spec` never executes — and look for the module
    file on disk. The real import happens in `_load()`, in the worker thread.
    """
    try:
        spec = importlib.util.find_spec("chatterbox")
    except Exception as exc:  # pragma: no cover - needs a broken `clone` extra
        return f"{NOT_INSTALLED_REASON} (the package is present but not importable: {exc})"
    if spec is None:
        return NOT_INSTALLED_REASON
    roots = spec.submodule_search_locations or ()
    if not any((Path(root) / "mtl_tts.py").is_file() for root in roots):
        return NOT_INSTALLED_REASON
    return None


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Numpy fallback when torchaudio is unavailable: linear interpolation onto the new grid."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    count = max(1, int(round(audio.size * dst_rate / float(src_rate))))
    src_times = np.arange(audio.size, dtype=np.float64) / float(src_rate)
    dst_times = np.arange(count, dtype=np.float64) / float(dst_rate)
    return np.interp(dst_times, src_times, audio).astype(np.float32)


def _to_mono_24k(wav: Any, src_rate: int) -> np.ndarray:
    """Model output (torch tensor or array, any rate) -> flat float32 mono at 24 kHz."""
    rate = src_rate
    if hasattr(wav, "detach"):
        tensor = wav.detach().cpu()
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=0)
        if rate != SAMPLE_RATE:
            try:
                import torchaudio

                tensor = torchaudio.functional.resample(tensor, rate, SAMPLE_RATE)
                rate = SAMPLE_RATE
            except Exception as exc:  # pragma: no cover - torchaudio ships with chatterbox
                log.debug("torchaudio resample unavailable (%s); using the numpy fallback", exc)
        audio = np.asarray(tensor.numpy(), dtype=np.float32)
    else:
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
    return _resample_linear(audio.reshape(-1), rate, SAMPLE_RATE)


class ChatterboxBackend:
    """`TTSBackend` (respeak/pipeline/tts/base.py) on top of `ChatterboxMultilingualTTS`."""

    name: str = "chatterbox"
    cloning: bool = True

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._builtin_conds: Any | None = None
        self._lock = threading.Lock()

    def languages(self) -> set[str]:
        """ISO-639-1 codes this backend can speak."""
        return set(LANGUAGES)

    def voices(self) -> dict[str, list[Voice]]:
        """Always empty: Chatterbox has no preset voices, it clones the speaker in `reference.wav`.

        `available_backends()` reports this as `voices: {}`, which is how the UI knows to hide the
        voice picker for this backend (flow.md B4).
        """
        return {}

    @staticmethod
    def installed() -> bool:
        """True only when the `clone` extra is present and importable."""
        return _import_problem() is None

    @staticmethod
    def reason() -> str | None:
        """Why the backend is unusable, or None when it is fine."""
        return _import_problem()

    def synthesize(
        self, text: str, lang: str, reference_wav: Path | None, out: Path, voice: str | None = None
    ) -> Path:
        """Write a 24 kHz mono WAV of `text` spoken in `lang` to `out` and return it.

        When `reference_wav` is given the model clones that speaker; otherwise it uses its own
        default voice. `voice` is ignored — this backend has no preset voices (`voices()` is
        empty), and a stale id from another backend must not fail a job. Empty text becomes a
        short silence, as with every backend.
        """
        if voice:
            log.debug("ignoring voice %r: chatterbox clones the reference clip instead", voice)
        code = normalise_lang(lang)
        if code not in LANGUAGES:
            raise TTSError(f"chatterbox cannot speak {lang!r}; it speaks: {', '.join(sorted(LANGUAGES))}")
        if not text.strip():
            log.warning("empty text for %s: writing silence", Path(out).name)
            return write_silence(out)
        prompt: str | None = None
        if reference_wav is not None:
            if not Path(reference_wav).exists():
                raise TTSError(f"reference clip {reference_wav} does not exist")
            prompt = str(reference_wav)

        model = self._load()
        if prompt is None and self._builtin_conds is None:
            raise TTSError("this chatterbox checkpoint has no built-in voice: pass a reference clip")
        try:
            # `generate()` is not re-entrant: `prepare_conditionals()` stores the cloned speaker on
            # the model itself (`model.conds`) and `generate()` then rewrites its emotion vector, so
            # two concurrent jobs would swap voices mid-sentence. One model per process, one
            # sentence at a time; MAX_CONCURRENT_JOBS is 1 by default anyway.
            with self._lock:
                if prompt is None:
                    # Without a prompt `generate()` reuses whatever speaker the *previous* call
                    # cloned — the backend is cached for the life of the process, so job 2 would be
                    # voiced by job 1's speaker. Put the checkpoint's own voice back first.
                    model.conds = self._builtin_conds
                wav = model.generate(text, language_id=code, audio_prompt_path=prompt)
        except Exception as exc:
            raise TTSError(f"chatterbox failed to speak {lang!r} text {text[:60]!r}: {exc}") from exc

        audio = _to_mono_24k(wav, int(getattr(model, "sr", SAMPLE_RATE)))
        if audio.size == 0:
            log.warning("chatterbox produced no audio for %r (%s): writing silence", text[:60], lang)
            return write_silence(out)
        return write_wav(audio, out)

    def _load(self) -> Any:
        """The multilingual model, loaded once per process on `Settings.resolved_device()`."""
        with self._lock:
            if self._model is not None:
                return self._model
            problem = _import_problem()
            if problem is not None:
                raise TTSError(f"tts backend 'chatterbox' is unavailable: {problem}")
            try:
                model_class = _model_class()
            except Exception as exc:  # pragma: no cover - needs a broken `clone` extra
                raise TTSError(
                    f"the `clone` extra is installed but chatterbox cannot be imported: {exc}"
                ) from exc
            device = self._settings.resolved_device()
            log.info("loading chatterbox multilingual model on %s", device)
            try:
                self._model = model_class.from_pretrained(device=device)
            except Exception as exc:  # pragma: no cover - needs the `clone` extra
                raise TTSError(f"chatterbox failed to load its model on {device}: {exc}") from exc
            # `conds.pt` from the checkpoint: the voice used when a job has no reference clip.
            # Keep it, because the first clone overwrites `model.conds` for good.
            self._builtin_conds = getattr(self._model, "conds", None)
            return self._model


__all__ = ["LANGUAGES", "NOT_INSTALLED_REASON", "ChatterboxBackend"]
