"""Download everything a job needs before the first job needs it (plan.md 2.3).

    uv run python -m respeak.prewarm --languages en,es,fr

Run from source to make the first dub fast, or at image build time with `--build-arg PREWARM=1` so a
fresh container downloads nothing (plan.md 2.3 "Done when"). Four caches are involved:

    faster-whisper weights   $HF_HOME (huggingface hub cache)   `WHISPER_MODEL`, or --whisper
    Argos packages           $XDG_DATA_HOME/argos-translate     one xx->en and one en->xx per language
    Kokoro voices            $HF_HOME                           one curated voice per language (D-37)
    unidic (Japanese G2P)    inside the installed `unidic`      only when "ja" is in the list

Every step is optional and every step is tolerant: a language Argos has no package for, or a voice the
network refuses today, is logged and the run continues. The exit code is 1 if anything failed, so a
build can still be told to care. Steps run in dependency order — the Japanese dictionary is downloaded
before Kokoro is asked to speak Japanese, because without it that step fails (see `tts/kokoro.py`).

Synchronous and single-threaded on purpose: this is a command, not part of the server.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - the real import happens inside the step that needs it
    from .pipeline.translate import Translator

log = logging.getLogger("respeak.prewarm")

#: Argos only ships `xx<->en` packages, so every pair is installed against English (flow.md B4.4).
PIVOT = "en"

OK = "ok"
SKIPPED = "skipped"
FAILED = "failed"

#: One short sentence per language Kokoro speaks; synthesising it downloads that voice.
SAMPLES: dict[str, str] = {
    "en": "This is a short warm-up sentence.",
    "es": "Esta es una frase corta de preparación.",
    "fr": "Voici une courte phrase de préchauffage.",
    "hi": "यह एक छोटा सा वाक्य है।",
    "it": "Questa è una breve frase di riscaldamento.",
    "ja": "これは短い準備の文です。",
    "pt": "Esta é uma frase curta de aquecimento.",
    "zh": "这是一个简短的热身句子。",
}


@dataclass
class Step:
    """One unit of warming: what it was, how it went, how long it took."""

    name: str
    status: str
    seconds: float = 0.0
    detail: str = ""


def parse_languages(raw: str) -> list[str]:
    """``"en, ES ,,pt-BR"`` -> ``["en", "es", "pt"]`` — lower case, de-duplicated, order kept."""
    out: list[str] = []
    for piece in str(raw or "").split(","):
        code = piece.strip().lower().replace("_", "-").split("-")[0]
        if code and code not in out:
            out.append(code)
    return out


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """The CLI. Defaults come from :class:`respeak.config.Settings`, so `.env` configures prewarm too."""
    settings = settings if settings is not None else get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m respeak.prewarm",
        description="Download the Whisper model, Argos packages, Kokoro voices and the Japanese "
        "dictionary ahead of the first job.",
    )
    parser.add_argument(
        "--whisper",
        default=settings.whisper_model,
        metavar="MODEL",
        help="faster-whisper model to download (default: %(default)s, from WHISPER_MODEL)",
    )
    parser.add_argument(
        "--languages",
        default=settings.prewarm_languages,
        metavar="CODES",
        help="comma-separated ISO-639-1 codes to warm (default: %(default)s, from PREWARM_LANGUAGES)",
    )
    parser.add_argument("--skip-whisper", action="store_true", help="do not download the Whisper model")
    parser.add_argument("--skip-argos", action="store_true", help="do not install Argos packages")
    parser.add_argument("--skip-kokoro", action="store_true", help="do not download Kokoro voices")
    parser.add_argument("--skip-unidic", action="store_true", help="do not download the unidic dictionary")
    return parser


# --------------------------------------------------------------------------------------------- steps


def _warm_whisper(settings: Settings, model: str) -> str:
    """Construct the `WhisperModel`, which downloads the weights on first use (asr.py `load_model`)."""
    from .pipeline.asr import compute_type_for, load_model

    warmed = settings.model_copy(update={"whisper_model": model})
    load_model(warmed)
    return f"{model} on {warmed.resolved_device()} ({compute_type_for(warmed)})"


def _warm_argos_pair(translator: Translator, src: str, dst: str) -> str:
    """Install the one Argos package behind `src`->`dst` (the translator does the download)."""
    translator.ensure_pair(src, dst)
    return f"{src}->{dst} ready"


def download_voice_tensors(lang: str) -> int:
    """Fetch every voice file Kokoro has for `lang` into the HF cache (each is ~0.5 MB).

    Kokoro loads a voice tensor from the Hub the first time it is asked for it, so without this a
    voice picked from the menu on an offline machine fails at `speak`. Returns how many were fetched.
    """
    from huggingface_hub import hf_hub_download

    from .pipeline.tts.kokoro import REPO_ID, VOICE_IDS

    count = 0
    for voice in VOICE_IDS.get(lang, ()):
        hf_hub_download(repo_id=REPO_ID, filename=f"voices/{voice}.pt")
        count += 1
    return count


def _warm_kokoro_voice(settings: Settings, lang: str, out_dir: Path) -> str:
    """Download all of `lang`'s voice tensors, then speak one sentence to prove the pipeline loads."""
    from .pipeline.tts import get_backend

    fetched = download_voice_tensors(lang)
    backend = get_backend("kokoro", settings)
    text = SAMPLES.get(lang)
    if text is None:  # a voice was added to kokoro.VOICES without a sample sentence here
        log.warning("no sample sentence for %r; warming it with the English one", lang)
        text = SAMPLES["en"]
    out = out_dir / f"prewarm_{lang}.wav"
    backend.synthesize(text, lang, None, out)
    size = out.stat().st_size if out.is_file() else 0
    voice = getattr(backend, "voice_for", lambda _lang: "?")(lang)
    return f"{fetched} voices cached; spoke with {voice}, {size} bytes"


def unidic_ready() -> bool:
    """True when the MeCab dictionary Japanese G2P needs is already unpacked."""
    import unidic

    directory = Path(unidic.DICDIR)
    return directory.is_dir() and any(directory.iterdir())


def _warm_unidic() -> str:
    """`python -m unidic download`, ~250 MB, only when the dictionary directory is missing or empty."""
    if unidic_ready():
        return "already downloaded"
    completed = subprocess.run(
        [sys.executable, "-m", "unidic", "download"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"`python -m unidic download` exited {completed.returncode}: {' | '.join(tail)}")
    if not unidic_ready():
        raise RuntimeError("`python -m unidic download` succeeded but the dictionary directory is empty")
    return "downloaded"


# ----------------------------------------------------------------------------------------- the run


def _run(steps: list[Step], name: str, work: Callable[[], str]) -> bool:
    """Run one step, timing it and turning any failure into a `FAILED` row instead of a traceback."""
    log.info("%s ...", name)
    started = time.monotonic()
    try:
        detail = work() or ""
    except Exception as exc:
        elapsed = time.monotonic() - started
        log.error("%s failed after %.1f s: %s", name, elapsed, exc)
        steps.append(Step(name, FAILED, elapsed, str(exc).replace("\n", " ")[:200]))
        return False
    elapsed = time.monotonic() - started
    log.info("%s ok in %.1f s%s", name, elapsed, f" ({detail})" if detail else "")
    steps.append(Step(name, OK, elapsed, detail))
    return True


def _print_summary(steps: list[Step]) -> None:
    """The final table; printed, not logged, so it survives LOG_LEVEL=WARNING."""
    width = max([len(step.name) for step in steps] + [4])
    print()
    print(f"{'step'.ljust(width)}  {'status':<8} {'time':>7}  detail")
    print("-" * (width + 26))
    for step in steps:
        print(f"{step.name.ljust(width)}  {step.status:<8} {step.seconds:>6.1f}s  {step.detail}")
    failed = [step.name for step in steps if step.status == FAILED]
    print()
    if failed:
        print(f"{len(failed)} of {len(steps)} steps failed: {', '.join(failed)}")
    else:
        print(f"all {len(steps)} steps finished without an error")


def main(argv: Sequence[str] | None = None) -> int:
    """Warm every cache the options ask for. Returns 0, or 1 when any step failed."""
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    languages = parse_languages(args.languages)
    log.info("prewarming: whisper=%s languages=%s", args.whisper, ",".join(languages) or "(none)")
    steps: list[Step] = []

    # (a) speech recognition -----------------------------------------------------------------------
    if args.skip_whisper:
        steps.append(Step(f"whisper {args.whisper}", SKIPPED, detail="--skip-whisper"))
    else:
        _run(steps, f"whisper {args.whisper}", lambda: _warm_whisper(settings, args.whisper))

    # (b) translation ------------------------------------------------------------------------------
    pairs = [(code, PIVOT) for code in languages if code != PIVOT]
    pairs += [(PIVOT, code) for code in languages if code != PIVOT]
    pairs.sort()
    if args.skip_argos:
        steps.append(Step("argos", SKIPPED, detail="--skip-argos"))
    elif not pairs:
        steps.append(Step("argos", SKIPPED, detail=f"nothing to install for {languages or ['(none)']}"))
    else:
        from .pipeline.translate import ArgosTranslator

        translator = ArgosTranslator(settings)
        for src, dst in pairs:
            _run(steps, f"argos {src}->{dst}", lambda s=src, d=dst: _warm_argos_pair(translator, s, d))

    # (c) the Japanese dictionary, before Kokoro is asked to speak Japanese ------------------------
    if "ja" not in languages:
        pass  # nothing needs it; `unidic download` is 250 MB
    elif args.skip_unidic:
        steps.append(Step("unidic", SKIPPED, detail="--skip-unidic"))
    else:
        _run(steps, "unidic", _warm_unidic)

    # (d) text to speech ---------------------------------------------------------------------------
    if args.skip_kokoro:
        steps.append(Step("kokoro", SKIPPED, detail="--skip-kokoro"))
    else:
        from .pipeline.tts.kokoro import VOICES

        speakable = [code for code in languages if code in VOICES]
        if not speakable:
            spoken = ", ".join(sorted(VOICES))
            steps.append(Step("kokoro", SKIPPED, detail=f"none of {languages or ['(none)']} in {spoken}"))
        else:
            with tempfile.TemporaryDirectory(prefix="respeak-prewarm-") as tmp:
                out_dir = Path(tmp)
                for code in speakable:
                    _run(steps, f"kokoro {code}", lambda c=code: _warm_kokoro_voice(settings, c, out_dir))

    _print_summary(steps)
    return 1 if any(step.status == FAILED for step in steps) else 0


if __name__ == "__main__":  # pragma: no cover - exercised by `python -m respeak.prewarm`
    raise SystemExit(main())
