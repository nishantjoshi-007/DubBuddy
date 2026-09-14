"""WP-3B: the `respeak` command (flow.md B2, B6; plan.md 3.2).

Everything here is offline and instant except `test_dub_writes_a_playable_file`, which runs the whole
pipeline on `tests/fixtures/sample.mp4` with the `base` Whisper model; it skips when the caches it
needs are missing and there is no network to fill them.
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from respeak import cli
from respeak.config import get_settings
from respeak.jobs import JobStore

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURES / "sample.mp4"
WHISPER_MODEL = "base"  # cached on the dev machine; `small` is the product default (D-06)
KOKORO_FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/ef_dora.pt")


# --------------------------------------------------------------------------- availability


def _has_network(host: str = "huggingface.co", port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3.0):
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


needs_everything = pytest.mark.skipif(
    not (_whisper_cached(WHISPER_MODEL) and _kokoro_cached() and _argos_installed("en", "es"))
    and not _has_network(),
    reason=f"whisper {WHISPER_MODEL} / Kokoro / the Argos en->es package are not cached, and no network",
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway DATA_DIR and the small Whisper model, with the settings cache reset around it."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WHISPER_MODEL", WHISPER_MODEL)
    monkeypatch.setenv("TTS_BACKEND", "kokoro")
    monkeypatch.setenv("MAX_VIDEO_SECONDS", "60")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def run(*argv: str) -> int:
    return cli.main(list(argv))


# --------------------------------------------------------------------------- the parser


@pytest.mark.parametrize("argv", [["--help"], ["dub", "--help"], ["serve", "--help"]])
def test_help_exits_zero(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("usage: respeak")


def test_the_top_level_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "dub" in out and "serve" in out and "prewarm" in out


def _subparser_help(parser: argparse.ArgumentParser, name: str) -> str:
    (subcommands,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return subcommands.choices[name].format_help()


def test_dub_help_documents_every_flag() -> None:
    """plan.md 3.2 "Done when": `--help` documents every flag."""
    dub_help = _subparser_help(cli.build_parser(), "dub")
    for flag in ("--to", "--from", "--backend", "--voice", "--no-burn", "--out", "--data-dir"):
        assert flag in dub_help, f"{flag} is undocumented"
    assert "URL-OR-FILE" in dub_help


def test_dub_flags_land_where_they_are_read() -> None:
    args = cli.build_parser().parse_args(
        ["dub", "clip.mp4", "--to", "ES", "--from", "en", "--backend", "kokoro",
         "--voice", "ef_dora", "--no-burn", "--out", "x.mp4", "--data-dir", "d"]
    )  # fmt: skip
    assert (args.input, args.to_lang, args.from_lang) == ("clip.mp4", "ES", "en")
    assert (args.backend, args.voice, args.no_burn) == ("kokoro", "ef_dora", True)
    assert (args.out, args.data_dir) == ("x.mp4", "d")


def test_serve_defaults_to_loopback() -> None:
    args = cli.build_parser().parse_args(["serve"])
    assert (args.host, args.port, args.reload) == (cli.DEFAULT_HOST, cli.DEFAULT_PORT, False)
    assert cli.DEFAULT_HOST == "127.0.0.1", "a laptop must not be published to the network by default"


def test_serve_runs_uvicorn_with_the_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(app=app, **kwargs))
    assert run("serve", "--host", "0.0.0.0", "--port", "9001", "--reload") == 0
    assert seen == {
        "app": "respeak.main:app",
        "host": "0.0.0.0",
        "port": 9001,
        "reload": True,
        "forwarded_allow_ips": "127.0.0.1",  # TRUST_PROXY is off by default
    }


def test_prewarm_passes_its_flags_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from respeak import prewarm as prewarm_module

    seen: list[list[str]] = []
    monkeypatch.setattr(prewarm_module, "main", lambda argv: seen.append(list(argv)) or 0)
    assert run("prewarm", "--languages", "en,fr", "--skip-whisper") == 0
    assert seen == [["--languages", "en,fr", "--skip-whisper"]]


def test_an_unknown_flag_is_an_error_not_a_silent_ignore() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["dub", "clip.mp4", "--to", "es", "--bogus"])
    assert raised.value.code == 2


def test_a_missing_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([])
    assert raised.value.code == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://www.youtube.com/watch?v=x", True),
        ("http://example.com/v.mp4", True),
        ("tests/fixtures/sample.mp4", False),
        ("/home/me/clip.mp4", False),
        ("~/clip.mp4", False),
    ],
)
def test_looks_like_url(text: str, expected: bool) -> None:
    assert cli.looks_like_url(text) is expected


# --------------------------------------------------------------------------- dub: refusals


def test_dub_refuses_a_file_that_is_not_there(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("dub", str(env / "nope.mp4"), "--to", "es") == 1
    assert "neither an http(s) URL nor a file" in capsys.readouterr().err


def test_dub_refuses_a_language_the_backend_cannot_speak(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("dub", str(SAMPLE_MP4), "--to", "zz") == 1
    assert "cannot speak 'zz'" in capsys.readouterr().err


def test_dub_refuses_an_unknown_voice_and_names_the_real_ones(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("dub", str(SAMPLE_MP4), "--to", "es", "--voice", "am_adam") == 1
    error = capsys.readouterr().err
    assert "am_adam" in error and "ef_dora" in error


def test_dub_refuses_a_url_that_is_not_public(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """F9 again: yt-dlp would fetch whatever this machine can reach, CLI or not."""
    assert run("dub", "http://127.0.0.1:8000/secret", "--to", "es") == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_dub_creates_nothing_when_it_refuses(env: Path) -> None:
    run("dub", str(SAMPLE_MP4), "--to", "zz")
    assert not (env / "data" / "jobs").exists() or JobStore(env / "data" / "jobs").list_ids() == []


# --------------------------------------------------------------------------- progress and naming


def test_progress_store_prints_one_line_per_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = cli.ProgressStore(tmp_path / "jobs")
    job_id = store.create(source={"type": "upload"}, options={"to_lang": "es"})
    store.mark(job_id, "running", step="speak", progress=0.6)
    store.update(job_id, detail="segment 1 of 2")
    store.update(job_id, detail="segment 1 of 2")  # nothing changed: nothing printed
    store.update(job_id, warnings=["2 segments did not fit."])
    store.update(job_id, warnings=["2 segments did not fit."])

    lines = [line for line in capsys.readouterr().err.splitlines() if line]
    assert lines == [
        "[ 60%] speak",
        "[ 60%] speak: segment 1 of 2",
        "warning: 2 segments did not fit.",
    ]


def test_the_default_output_name_is_title_and_language() -> None:
    status = {"title": "Hello / World", "source": {"filename": "clip.mp4"}}
    assert cli._output_name(status, "es") == "Hello _ World-es.mp4"
    assert cli._output_name({"source": {"filename": "clip.mp4"}}, "fr") == "clip-fr.mp4"
    assert cli._output_name({}, "pt") == f"{cli.FALLBACK_STEM}-pt.mp4"


def test_out_may_be_a_directory(tmp_path: Path) -> None:
    status = {"title": "Clip"}
    assert cli._destination(str(tmp_path), status, "es") == tmp_path / "Clip-es.mp4"
    assert cli._destination(str(tmp_path / "x.mp4"), status, "es") == tmp_path / "x.mp4"


def test_a_local_file_is_linked_not_copied_when_it_can_be(tmp_path: Path) -> None:
    """A 500 MB upload must not be duplicated on disk just to enter the job directory."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    dest = tmp_path / "upload.bin"
    cli._place_upload(source, dest)
    assert dest.read_bytes() == b"video"
    assert dest.stat().st_ino == source.stat().st_ino
    dest.unlink()
    assert source.exists(), "the job's cleanup must never delete the user's own file"


# --------------------------------------------------------------------------- the real thing


@needs_everything
@pytest.mark.slow
def test_dub_writes_a_playable_file(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """plan.md 3.2 "Done when": `respeak dub <file> --to es --out …` produces a file, offline."""
    from respeak.pipeline import ffmpeg

    out = env / "dubbed.mp4"
    code = run("dub", str(SAMPLE_MP4), "--to", "es", "--no-burn", "--out", str(out))
    captured = capsys.readouterr()
    assert code == 0, captured.err

    assert out.is_file() and out.stat().st_size > 0
    assert captured.out.strip().endswith("dubbed.mp4"), "stdout carries the path and nothing else"
    assert "[100%] finish" in captured.err
    assert ffmpeg.duration(out) > 1.0

    store = JobStore(env / "data" / "jobs")
    (job_id,) = store.list_ids()
    status = store.require(job_id)
    assert status["state"] == "done"
    assert status["options"] == {
        "to_lang": "es",
        "from_lang": None,
        "backend": "kokoro",
        "voice": None,
        "burn_subtitles": False,
    }
    assert (store.path(job_id) / "subs.srt").is_file(), "the subtitles stay next to the job"
