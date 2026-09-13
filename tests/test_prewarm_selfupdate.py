"""P2-A: yt-dlp self-update, the version report and `python -m respeak.prewarm` (plan.md 2.0 / 2.3).

Offline and fast by construction: every installer call is a monkeypatched ``subprocess.run`` and every
prewarm run here skips the steps that would download something. Nothing in this file touches the
network, loads a model, or imports torch.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from respeak import main as main_module
from respeak import prewarm, selfupdate
from respeak.config import Settings
from respeak.main import app
from respeak.selfupdate import (
    PACKAGE,
    UNKNOWN,
    component_versions,
    update_ytdlp,
    version_from_output,
    ytdlp_version,
)

# --------------------------------------------------------------------------- versions


def test_ytdlp_version_is_a_string() -> None:
    version = ytdlp_version()
    assert isinstance(version, str)
    assert version and version != UNKNOWN, "yt-dlp is a hard dependency; it must report a version"


def test_package_version_of_something_uninstalled_is_unknown() -> None:
    """Never raises, never returns None: a missing package is reported, not signalled (D-xx)."""
    assert selfupdate.package_version("definitely-not-installed-xyz") == UNKNOWN


def test_component_versions_has_the_five_reported_keys() -> None:
    versions = component_versions()
    assert set(versions) == {"respeak", "yt-dlp", "faster-whisper", "kokoro", "torch"}
    assert all(isinstance(value, str) and value for value in versions.values())
    assert versions["torch"] != UNKNOWN, "torch must be installed without being imported here"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (" + yt-dlp==2026.8.19", "2026.8.19"),  # uv pip install
        ("Successfully installed yt-dlp-2026.8.19", "2026.8.19"),  # pip install
        ("Requirement already satisfied: yt-dlp==2026.8.19 in /app/.venv", "2026.8.19"),
        ("nothing to see here", None),
        ("", None),
    ],
)
def test_version_from_output(output: str, expected: str | None) -> None:
    assert version_from_output(output) == expected


# --------------------------------------------------------------------------- update_ytdlp


class FakeRun:
    """Stands in for ``subprocess.run``: records the command, answers with a canned result."""

    def __init__(self, returncode: int = 0, stdout: str = "", raises: BaseException | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.raises = raises
        self.commands: list[list[str]] = []
        self.kwargs: dict[str, object] = {}

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, stdout=self.stdout, stderr="")


def test_update_ytdlp_runs_an_installer_and_returns_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun(stdout=" + yt-dlp==2099.1.1")
    monkeypatch.setattr(selfupdate.subprocess, "run", fake)

    result = update_ytdlp(timeout=7)

    assert result == ytdlp_version(), "with the metadata unchanged, the installed version is reported"
    assert len(fake.commands) == 1
    command = fake.commands[0]
    assert command[-1] == PACKAGE and "-U" in command
    assert sys.executable in command, "the upgrade must target the running interpreter"
    assert fake.kwargs["timeout"] == 7
    assert fake.kwargs["check"] is False, "a non-zero exit must be handled, not raised"


def test_update_ytdlp_falls_back_to_the_installer_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the dist-info cannot be read back, the version the installer printed is used."""
    monkeypatch.setattr(selfupdate.subprocess, "run", FakeRun(stdout="Successfully installed yt-dlp-9.9.9"))
    monkeypatch.setattr(selfupdate, "ytdlp_version", lambda: UNKNOWN)

    assert update_ytdlp() == "9.9.9"


def test_update_ytdlp_returns_none_when_the_installer_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = FakeRun(returncode=1, stdout="no matching distribution")
    monkeypatch.setattr(selfupdate.subprocess, "run", failed)
    assert update_ytdlp() is None


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(cmd="uv", timeout=1),
        OSError("uv vanished"),
        MemoryError("not even this"),
    ],
)
def test_update_ytdlp_never_raises(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """A failed self-update must never be able to stop the server from starting."""
    monkeypatch.setattr(selfupdate.subprocess, "run", FakeRun(raises=error))
    assert update_ytdlp() is None


def test_update_ytdlp_without_an_installer_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(selfupdate.shutil, "which", lambda _name: None)
    monkeypatch.setattr(selfupdate, "_pip_available", lambda: False)
    monkeypatch.setattr(selfupdate.subprocess, "run", lambda *a, **k: calls.append(a))

    assert update_ytdlp() is None
    assert calls == [], "no installer means no subprocess at all"


def test_update_ytdlp_prefers_uv_over_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    uv_path = "/usr/local/bin/uv"
    monkeypatch.setattr(selfupdate.shutil, "which", lambda name: uv_path if name == "uv" else None)
    command = selfupdate._installer_command()
    assert command is not None
    assert command[:4] == ["/usr/local/bin/uv", "pip", "install", "--python"]


def test_pip_is_the_fallback_when_uv_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selfupdate.shutil, "which", lambda _name: None)
    monkeypatch.setattr(selfupdate, "_pip_available", lambda: True)
    assert selfupdate._installer_command() == [sys.executable, "-m", "pip", "install", "-U", PACKAGE]


# --------------------------------------------------------------------------- startup wiring


def test_auto_update_is_off_unless_asked() -> None:
    assert main_module._start_ytdlp_update(Settings(ytdlp_auto_update=False)) is None


def test_auto_update_runs_in_a_daemon_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never on the event loop and never blocking startup (plan.md 2.0)."""
    done = threading.Event()
    seen: dict[str, object] = {}

    def fake_update() -> None:
        current = threading.current_thread()
        seen["thread"] = current
        seen["daemon"] = current.daemon
        done.set()

    monkeypatch.setattr(main_module, "update_ytdlp", fake_update)
    thread = main_module._start_ytdlp_update(Settings(ytdlp_auto_update=True))

    assert thread is not None
    assert done.wait(10), "the update thread never ran"
    thread.join(timeout=10)
    assert seen["thread"] is not threading.current_thread()
    assert seen["daemon"] is True


def test_health_reports_every_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guard against a developer .env that turns the self-update on: no test may reach the network.
    monkeypatch.setattr(main_module, "update_ytdlp", lambda *a, **k: None)
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    versions = response.json()["versions"]
    assert set(versions) == {"respeak", "yt-dlp", "faster-whisper", "kokoro", "torch"}
    assert versions["yt-dlp"] == ytdlp_version()


# --------------------------------------------------------------------------- prewarm


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en,es", ["en", "es"]),
        (" EN , es ,, pt-BR ", ["en", "es", "pt"]),
        ("en,en,EN", ["en"]),
        ("", []),
    ],
)
def test_parse_languages(raw: str, expected: list[str]) -> None:
    assert prewarm.parse_languages(raw) == expected


def test_parser_defaults_come_from_settings() -> None:
    settings = Settings(whisper_model="base", prewarm_languages="en,ja")
    args = prewarm.build_parser(settings).parse_args([])
    assert args.whisper == "base"
    assert args.languages == "en,ja"
    assert not any((args.skip_whisper, args.skip_argos, args.skip_kokoro, args.skip_unidic))


ALL_SKIPS = ["--skip-whisper", "--skip-argos", "--skip-kokoro", "--skip-unidic"]


def test_main_with_every_skip_flag_exits_zero_and_prints_a_summary(capsys: pytest.CaptureFixture) -> None:
    assert prewarm.main([*ALL_SKIPS, "--languages", "en,ja"]) == 0
    out = capsys.readouterr().out
    assert "step" in out and "status" in out
    assert out.count(prewarm.SKIPPED) == 4, out
    assert "all 4 steps finished without an error" in out


def test_unidic_is_only_warmed_for_japanese(capsys: pytest.CaptureFixture) -> None:
    assert prewarm.main(["--skip-whisper", "--skip-argos", "--skip-kokoro", "--languages", "en,es"]) == 0
    assert "unidic" not in capsys.readouterr().out


def test_a_failed_step_is_reported_and_sets_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """One unreachable download must not stop the others (plan.md 2.3: log and continue)."""

    def boom(_settings: Settings, _model: str) -> str:
        raise RuntimeError("the model registry is offline")

    monkeypatch.setattr(prewarm, "_warm_whisper", boom)

    code = prewarm.main(["--skip-argos", "--skip-kokoro", "--skip-unidic", "--languages", "en"])

    out = capsys.readouterr().out
    assert code == 1
    assert prewarm.FAILED in out
    assert "the model registry is offline" in out
    assert "1 of 3 steps failed" in out
