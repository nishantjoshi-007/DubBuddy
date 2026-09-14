"""Job store, job runner, routes and sweeper (flow.md B3, B5, B7).

Everything here is offline: no model, no network, no ffmpeg. The pipeline itself is stubbed, which is
the point of injecting ``run_fn`` into :class:`respeak.jobs.JobRunner`.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from respeak import api as api_module
from respeak import jobs as jobs_module
from respeak import main as main_module
from respeak.config import Settings, get_settings
from respeak.jobs import (
    JobNotFoundError,
    JobRunner,
    JobStore,
    parse_iso,
    placeholder_run,
    start_sweeper,
    sweep,
)
from respeak.lang_codes import NAME_TO_CODE
from respeak.main import app
from respeak.pipeline.tts import available_backends

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Real Settings pointed at tmp_path, with the singletons reset around each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOW_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "2")
    monkeypatch.setenv("JOB_TTL_MINUTES", "60")
    monkeypatch.setenv("RATE_LIMIT_JOBS", "")
    monkeypatch.setenv("TRUST_PROXY", "false")
    get_settings.cache_clear()
    jobs_module.reset_state()
    api_module.reset_rate_limiter()  # the token buckets are process-wide; no test inherits another's
    yield get_settings()
    jobs_module.reset_state()
    api_module.reset_rate_limiter()
    get_settings.cache_clear()


class StubRunner:
    """Records submissions instead of running anything (keeps the API tests deterministic)."""

    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, job_id: str) -> None:
        self.submitted.append(job_id)


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> StubRunner:
    stub = StubRunner()
    monkeypatch.setattr("respeak.api.get_runner", lambda _settings: stub)
    return stub


@pytest.fixture
def client(settings: Settings, runner: StubRunner) -> TestClient:
    # No `with`: the app lifespan (binary checks) is irrelevant to these routes.
    return TestClient(app)


def store_for(settings: Settings) -> JobStore:
    return JobStore(settings.jobs_dir)


def read_status(settings: Settings, job_id: str) -> dict[str, Any]:
    return json.loads((settings.jobs_dir / job_id / "status.json").read_text(encoding="utf-8"))


YOUTUBE_FORM = {
    "source_type": "youtube",
    "url": "https://www.youtube.com/watch?v=abc12345678",
    "to_lang": "es",
    "from_lang": "en",
    "burn_subtitles": "true",
}


# --------------------------------------------------------------------------- POST /jobs


def test_youtube_job_is_created_queued_and_submitted(
    client: TestClient, settings: Settings, runner: StubRunner
) -> None:
    response = client.post("/jobs", data=YOUTUBE_FORM)
    assert response.status_code == 201, response.text
    body = response.json()
    job_id = body["id"]
    assert body["url"] == f"/jobs/{job_id}"
    assert runner.submitted == [job_id]

    status = read_status(settings, job_id)
    assert status["id"] == job_id
    assert status["warnings"] == []  # F5: the key exists from the first write
    assert status["state"] == "queued"
    assert status["step"] is None
    assert status["progress"] == 0.0
    assert status["error"] is None
    assert status["title"] is None
    assert status["detected_language"] is None
    assert status["output"] is None
    assert status["download_url"] is None
    assert status["source"] == {"type": "youtube", "url": YOUTUBE_FORM["url"], "filename": None}
    assert status["options"] == {
        "to_lang": "es",
        "from_lang": "en",
        "backend": "kokoro",
        "voice": None,  # no voice picked: the backend uses its curated default (plan.md 3.3)
        "burn_subtitles": True,
    }
    assert parse_iso(status["created_at"]) <= parse_iso(status["updated_at"])


def test_youtube_job_accepts_a_multipart_body_with_an_empty_file_part(client: TestClient) -> None:
    """A browser FormData is always multipart; an empty file input must not break the URL path."""
    response = client.post(
        "/jobs",
        data=YOUTUBE_FORM,
        files={"file": ("", b"", "application/octet-stream")},
    )
    assert response.status_code == 201, response.text


def test_upload_job_streams_the_file_and_sanitises_its_name(client: TestClient, settings: Settings) -> None:
    payload = b"fake mp4 bytes" * 100
    response = client.post(
        "/jobs",
        data={"source_type": "upload", "to_lang": "fr", "burn_subtitles": "false"},
        files={"file": ("../../my clip!.mp4", payload, "video/mp4")},
    )
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]

    assert (settings.jobs_dir / job_id / "upload.bin").read_bytes() == payload
    status = read_status(settings, job_id)
    assert status["source"]["type"] == "upload"
    assert status["source"]["url"] is None
    assert status["source"]["filename"] == "my clip_.mp4"
    assert status["options"]["burn_subtitles"] is False
    assert status["options"]["from_lang"] is None


@pytest.mark.parametrize(
    ("form", "fragment"),
    [
        ({"source_type": "youtube", "to_lang": "es"}, "URL is required"),
        ({"source_type": "youtube", "url": "ftp://x/y", "to_lang": "es"}, "http"),
        ({"source_type": "banana", "url": "https://x/y", "to_lang": "es"}, "source_type"),
        ({"source_type": "youtube", "url": "https://x/y"}, "to_lang is required"),
        ({"source_type": "youtube", "url": "https://x/y", "to_lang": "zz"}, "cannot speak 'zz'"),
        (
            {"source_type": "youtube", "url": "https://x/y", "to_lang": "es", "from_lang": "es"},
            "same",
        ),
        (
            {"source_type": "youtube", "url": "https://x/y", "to_lang": "es", "from_lang": "xx"},
            "unknown source language",
        ),
        (
            {"source_type": "youtube", "url": "https://x/y", "to_lang": "es", "backend": "nope"},
            "unknown backend",
        ),
        (
            {"source_type": "youtube", "url": "https://x/y", "to_lang": "es", "burn_subtitles": "maybe"},
            "burn_subtitles",
        ),
        ({"source_type": "upload", "to_lang": "es"}, "file is required"),
    ],
)
def test_validation_errors_are_400_json_and_create_nothing(
    client: TestClient, settings: Settings, runner: StubRunner, form: dict[str, str], fragment: str
) -> None:
    response = client.post("/jobs", data=form)
    assert response.status_code == 400, response.text
    assert fragment in response.json()["error"]
    assert runner.submitted == []
    assert store_for(settings).list_ids() == []


def test_a_backend_that_is_not_installed_is_refused_with_its_own_reason(
    client: TestClient, settings: Settings, runner: StubRunner
) -> None:
    """Whatever the backend registry reports as missing is refused before the job exists (D-27)."""
    infos = available_backends(settings)
    missing = [info for info in infos.values() if not info.installed]
    if not missing:
        pytest.skip("every TTS backend is installed here")
    info = missing[0]
    response = client.post(
        "/jobs",
        data={**YOUTUBE_FORM, "backend": info.name, "to_lang": sorted(info.languages)[0] or "es"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == info.reason
    assert runner.submitted == []


def test_a_target_language_the_backend_cannot_speak_is_refused(
    client: TestClient, settings: Settings
) -> None:
    """The target list comes from the backend, so anything outside it is a 400 (D-24)."""
    kokoro = available_backends(settings)["kokoro"]
    outside = sorted(set(NAME_TO_CODE.values()) - kokoro.languages)
    if not outside:
        pytest.skip("kokoro speaks every source language")
    response = client.post("/jobs", data={**YOUTUBE_FORM, "to_lang": outside[0]})
    assert response.status_code == 400
    assert f"cannot speak '{outside[0]}'" in response.json()["error"]


def test_regional_codes_are_normalised(client: TestClient, settings: Settings) -> None:
    job_id = client.post("/jobs", data={**YOUTUBE_FORM, "to_lang": "ES-mx"}).json()["id"]
    assert read_status(settings, job_id)["options"]["to_lang"] == "es"


def test_uploads_can_be_disabled(
    settings: Settings, runner: StubRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOW_UPLOADS", "false")
    get_settings.cache_clear()
    client = TestClient(app)
    response = client.post(
        "/jobs",
        data={"source_type": "upload", "to_lang": "es"},
        files={"file": ("clip.mp4", b"x" * 10, "video/mp4")},
    )
    assert response.status_code == 400
    assert "uploads are disabled" in response.json()["error"]
    assert runner.submitted == []


def test_a_forged_content_length_is_413_before_the_body_is_parsed(
    client: TestClient, settings: Settings, runner: StubRunner
) -> None:
    """F8: the declared size is refused by the middleware, so nothing is spooled to disk."""
    forged = settings.max_upload_mb * 1024 * 1024 + 4 * 1024 * 1024
    response = client.post(
        "/jobs",
        data={"source_type": "upload", "to_lang": "es"},
        files={"file": ("big.mp4", b"tiny", "video/mp4")},
        headers={"Content-Length": str(forged)},
    )
    assert response.status_code == 413, response.text
    assert "larger than" in response.json()["error"]
    assert store_for(settings).list_ids() == []
    assert runner.submitted == []


def test_a_content_length_inside_the_limit_still_reaches_the_route(
    client: TestClient, settings: Settings
) -> None:
    """The guard adds one MB of form overhead, so a legitimate upload is never refused by it."""
    payload = b"\0" * (settings.max_upload_mb * 1024 * 1024 - 1024)
    response = client.post(
        "/jobs",
        data={"source_type": "upload", "to_lang": "es"},
        files={"file": ("clip.mp4", payload, "video/mp4")},
    )
    assert response.status_code == 201, response.text


def test_oversize_upload_is_413_and_removes_the_job_dir(
    client: TestClient, settings: Settings, runner: StubRunner
) -> None:
    too_big = b"\0" * (settings.max_upload_mb * 1024 * 1024 + 2048)
    response = client.post(
        "/jobs",
        data={"source_type": "upload", "to_lang": "es"},
        files={"file": ("big.mp4", too_big, "video/mp4")},
    )
    assert response.status_code == 413, response.text
    assert "larger than" in response.json()["error"]
    assert store_for(settings).list_ids() == []
    assert runner.submitted == []


# --------------------------------------------------------------------------- 3.3 voice picker


def test_backends_publishes_a_voice_table_per_language(client: TestClient) -> None:
    """The form fills its voice select straight from this (flow.md B6, plan.md 3.3)."""
    backends = {entry["name"]: entry for entry in client.get("/api/backends").json()["backends"]}
    kokoro = backends["kokoro"]["voices"]
    assert set(kokoro) <= set(backends["kokoro"]["languages"])
    assert kokoro["es"][0] == {"id": "ef_dora", "name": "Dora (female)"}
    assert all({"id", "name"} == set(voice) for voices in kokoro.values() for voice in voices)
    assert backends["chatterbox"]["voices"] == {}, "a cloning backend hides the select"


def test_a_picked_voice_is_stored_in_the_options(client: TestClient, settings: Settings) -> None:
    response = client.post("/jobs", data={**YOUTUBE_FORM, "voice": "em_alex"})
    assert response.status_code == 201, response.text
    assert read_status(settings, response.json()["id"])["options"]["voice"] == "em_alex"


def test_an_empty_voice_field_means_the_backend_default(client: TestClient, settings: Settings) -> None:
    """The form always posts the field; blank must mean "default", not "no such voice"."""
    response = client.post("/jobs", data={**YOUTUBE_FORM, "voice": ""})
    assert response.status_code == 201, response.text
    assert read_status(settings, response.json()["id"])["options"]["voice"] is None


@pytest.mark.parametrize("voice", ["bogus", "am_adam"])  # unknown, and one from another language
def test_a_voice_the_backend_cannot_use_is_400_with_the_list(
    client: TestClient, settings: Settings, runner: StubRunner, voice: str
) -> None:
    response = client.post("/jobs", data={**YOUTUBE_FORM, "voice": voice})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert voice in error and "ef_dora" in error
    assert runner.submitted == []
    assert store_for(settings).list_ids() == []


def test_a_voice_for_a_backend_that_has_none_is_refused(client: TestClient, settings: Settings) -> None:
    chatterbox = available_backends(settings)["chatterbox"]
    if not chatterbox.installed:
        pytest.skip("the `clone` extra is not installed here")
    response = client.post("/jobs", data={**YOUTUBE_FORM, "backend": "chatterbox", "voice": "ef_dora"})
    assert response.status_code == 400
    assert "no preset voices" in response.json()["error"]


# --------------------------------------------------------------------------- 3E: voice previews


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, str, str]]:
    """Stand in for the synthesiser: record the triple and hand back a real little WAV file."""
    asked: list[tuple[str, str, str]] = []

    def fake_ensure(_settings: Settings, backend: str, lang: str, voice: str) -> Path:
        asked.append((backend, lang, voice))
        path = tmp_path / f"{backend}-{lang}-{voice}.wav"
        path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        return path

    monkeypatch.setattr(api_module, "ensure_sample", fake_ensure)
    return asked


def test_a_voice_sample_is_served_as_a_cacheable_wav(
    client: TestClient, sampler: list[tuple[str, str, str]]
) -> None:
    response = client.get("/api/voices/kokoro/es/ef_dora")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert response.content.startswith(b"RIFF")
    assert sampler == [("kokoro", "es", "ef_dora")]


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("/api/voices/banana/es/ef_dora", "unknown backend"),
        ("/api/voices/kokoro/es/bogus", "no voice 'bogus'"),
        ("/api/voices/kokoro/es/am_adam", "no voice 'am_adam'"),  # a real id, but English
        ("/api/voices/kokoro/zz/ef_dora", "no voices for 'zz'"),
    ],
)
def test_a_bad_triple_is_400_and_never_reaches_the_filesystem(
    client: TestClient, settings: Settings, path: str, fragment: str
) -> None:
    response = client.get(path)
    assert response.status_code == 400, response.text
    assert fragment in response.json()["error"]
    assert not (settings.data_dir / "voice_samples").exists()


@pytest.mark.parametrize("raw", ["..", "../../etc/passwd", "%2e%2e%2f", "ef_dora.wav", "ef_dora%2F.."])
def test_a_traversing_voice_id_can_never_reach_the_filesystem(
    client: TestClient, settings: Settings, raw: str
) -> None:
    """The id is matched against the backend's own table, never sanitised into a path."""
    response = client.get(f"/api/voices/kokoro/es/{raw}")
    assert response.status_code in {400, 404}, response.text
    assert not (settings.data_dir / "voice_samples").exists()


def test_a_backend_with_no_preset_voices_is_404(client: TestClient) -> None:
    """Chatterbox clones the original speaker, so there is nothing here to preview — ever."""
    response = client.get("/api/voices/chatterbox/es/ef_dora")
    assert response.status_code == 404, response.text
    assert "no preset voices" in response.json()["error"]


def test_the_voice_sample_route_runs_in_the_threadpool_not_on_the_loop() -> None:
    """First play loads a model and synthesises; a coroutine route would do it on the loop (F1)."""
    assert not inspect.iscoroutinefunction(api_module.voice_sample)


# --------------------------------------------------------------------------- 3.5 rate limiting


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("   ", None),
        ("10/hour", (10, 3600.0)),
        ("3/minute", (3, 60.0)),
        ("100/day", (100, 86400.0)),
        ("30/second", (30, 1.0)),
        (" 5 / 2 HOURS ", (5, 7200.0)),
    ],
)
def test_parse_rate_limit_reads_the_documented_forms(raw: str, expected: tuple[int, float] | None) -> None:
    assert api_module.parse_rate_limit(raw) == expected


@pytest.mark.parametrize("raw", ["10", "ten/hour", "10/fortnight", "0/hour", "10 per hour", "/hour"])
def test_a_misspelt_rate_limit_raises_instead_of_silently_disabling_itself(raw: str) -> None:
    with pytest.raises(ValueError):
        api_module.parse_rate_limit(raw)


def test_the_bucket_starts_full_empties_and_refills() -> None:
    limiter = api_module.RateLimiter(api_module.RateLimit(count=3, seconds=60.0))
    assert [limiter.check("1.2.3.4", now=0.0) for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = limiter.check("1.2.3.4", now=0.0)
    assert wait == pytest.approx(20.0)  # one token every 20 s at 3/minute
    assert limiter.check("5.6.7.8", now=0.0) == 0.0, "another address has its own bucket"
    assert limiter.check("1.2.3.4", now=19.0) > 0.0
    assert limiter.check("1.2.3.4", now=20.0) == 0.0
    assert limiter.check("1.2.3.4", now=10_000.0) == 0.0, "a long-idle bucket is full again"


def test_the_bucket_table_is_capped_so_a_flood_cannot_grow_it_forever() -> None:
    """F5: one dict entry per address is itself the leak when addresses are free (an IPv6 /64).

    Forgetting a bucket only ever hands that address a *full* bucket back, so evicting the
    least-recently-seen entries costs one extra allowed submission, never an unbounded process.
    """
    limiter = api_module.RateLimiter(api_module.RateLimit(count=1, seconds=3600.0))
    for n in range(3000):
        assert limiter.check(f"2001:db8::{n:x}", now=float(n)) == 0.0

    assert limiter.check("203.0.113.1", now=3000.0) == 0.0
    assert len(limiter._buckets) <= api_module.BUCKET_PRUNE_AT
    assert "203.0.113.1" in limiter._buckets, "the newest bucket is never the one evicted"
    assert "2001:db8::0" not in limiter._buckets, "the oldest buckets go first"


def test_no_rate_limit_by_default(client: TestClient) -> None:
    for _ in range(5):
        assert client.post("/jobs", data=YOUTUBE_FORM).status_code == 201


def _limited_client(monkeypatch: pytest.MonkeyPatch, rate: str, trust_proxy: bool = False) -> TestClient:
    monkeypatch.setenv("RATE_LIMIT_JOBS", rate)
    monkeypatch.setenv("TRUST_PROXY", "true" if trust_proxy else "false")
    get_settings.cache_clear()
    api_module.reset_rate_limiter()
    return TestClient(app)


def test_the_fourth_job_in_a_minute_is_429_with_retry_after(
    settings: Settings, runner: StubRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _limited_client(monkeypatch, "3/minute")
    for attempt in range(3):
        assert client.post("/jobs", data=YOUTUBE_FORM).status_code == 201, f"attempt {attempt}"

    refused = client.post("/jobs", data=YOUTUBE_FORM)
    assert refused.status_code == 429
    assert refused.json()["error"] == "too many jobs from this address; try again in 1 minute"
    assert 1 <= int(refused.headers["Retry-After"]) <= 60
    assert len(runner.submitted) == 3, "the refused request created nothing"
    assert len(store_for(settings).list_ids()) == 3


def test_the_limit_is_checked_before_the_form_is_validated(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonsense must cost a flooder the same token a real job does (otherwise it is free)."""
    client = _limited_client(monkeypatch, "1/hour")
    assert client.post("/jobs", data={"source_type": "banana"}).status_code == 400
    assert client.post("/jobs", data=YOUTUBE_FORM).status_code == 429


def test_an_oversize_upload_is_refused_without_spending_a_token(
    settings: Settings, runner: StubRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F7: a body that can never become a job must not cost the browser its whole allowance.

    Picking one too-large file is a mistake a person makes in the file dialog, not a flood; the 413
    is free, and the next, correct, submission still goes through under a 1/minute limit.
    """
    client = _limited_client(monkeypatch, "1/minute")
    forged = (settings.max_upload_mb + 4) * 1024 * 1024
    refused = client.post(
        "/jobs",
        data={"source_type": "upload", "to_lang": "es"},
        files={"file": ("big.mp4", b"tiny", "video/mp4")},
        headers={"Content-Length": str(forged)},
    )
    assert refused.status_code == 413, refused.text
    assert client.post("/jobs", data=YOUTUBE_FORM).status_code == 201, "the 413 spent a token"
    assert len(runner.submitted) == 1


def test_the_proxy_header_is_ignored_unless_trust_proxy_is_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anyone can send X-Forwarded-For; believing it without a proxy is a free reset."""
    client = _limited_client(monkeypatch, "1/minute")
    assert client.post("/jobs", data=YOUTUBE_FORM, headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 201
    second = client.post("/jobs", data=YOUTUBE_FORM, headers={"X-Forwarded-For": "8.8.8.8"})
    assert second.status_code == 429, "both came from the same real address"


def test_with_trust_proxy_the_last_forwarded_entry_is_the_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: a proxy *appends* the peer it saw, so the rightmost entry is the only trustworthy one."""
    client = _limited_client(monkeypatch, "1/minute", trust_proxy=True)
    peer = {"X-Forwarded-For": "203.0.113.7, 9.9.9.9"}  # junk the client sent, then our proxy's peer
    assert client.post("/jobs", data=YOUTUBE_FORM, headers=peer).status_code == 201
    assert client.post("/jobs", data=YOUTUBE_FORM, headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 201
    assert client.post("/jobs", data=YOUTUBE_FORM, headers=peer).status_code == 429


def test_a_client_supplied_forwarded_prefix_cannot_buy_a_fresh_bucket(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: reading the leftmost entry meant one header line bought unlimited buckets."""
    client = _limited_client(monkeypatch, "1/minute", trust_proxy=True)
    invented = {"X-Forwarded-For": "1.1.1.1, 9.9.9.9"}
    again = {"X-Forwarded-For": "2.2.2.2, 9.9.9.9"}  # same real peer, a different invented prefix
    assert client.post("/jobs", data=YOUTUBE_FORM, headers=invented).status_code == 201
    assert client.post("/jobs", data=YOUTUBE_FORM, headers=again).status_code == 429


def test_a_forwarded_entry_that_is_not_an_address_falls_back_to_the_real_peer() -> None:
    """F1/F5: the bucket key is always a parsed IP, never whatever text arrived in the header."""

    class _Req:
        headers = {"x-forwarded-for": "10.0.0.1, unknown"}
        client = types.SimpleNamespace(host="203.0.113.5")

    assert api_module.client_address(_Req(), trust_proxy=True) == "203.0.113.5"  # type: ignore[arg-type]
    assert api_module.parse_ip("  [2001:db8::1] ") == "2001:db8::1"
    assert api_module.parse_ip("not-an-ip") is None
    assert api_module.parse_ip(None) is None


def test_a_broken_rate_limit_setting_refuses_the_job_rather_than_ignoring_the_limit(
    settings: Settings, runner: StubRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _limited_client(monkeypatch, "10 jobs per hour")
    response = client.post("/jobs", data=YOUTUBE_FORM)
    assert response.status_code == 500
    assert "RATE_LIMIT_JOBS" in response.json()["error"]
    assert runner.submitted == []


def test_client_address_falls_back_when_there_is_no_peer() -> None:
    """Starlette leaves `request.client` None for some transports; the limiter still needs a key."""

    class _Req:
        headers = {"x-forwarded-for": "9.9.9.9"}
        client = None

    assert api_module.client_address(_Req(), trust_proxy=False) == "unknown"  # type: ignore[arg-type]
    assert api_module.client_address(_Req(), trust_proxy=True) == "9.9.9.9"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- GET routes


def test_health_and_backends_still_answer(client: TestClient, settings: Settings) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["jobs_dir"] == str(settings.jobs_dir)
    backends = client.get("/api/backends").json()
    assert backends["default"] == "kokoro"


def test_unknown_job_is_404_on_both_read_routes(client: TestClient) -> None:
    for path in ("/api/jobs/deadbeef", "/api/jobs/deadbeef/download"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "error" in response.json()
    # a syntactically impossible id must not escape the jobs dir either
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404


def test_status_route_returns_the_file_on_disk(client: TestClient, settings: Settings) -> None:
    job_id = client.post("/jobs", data=YOUTUBE_FORM).json()["id"]
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json() == read_status(settings, job_id)


def test_download_404_before_done_409_when_missing_and_200_after(
    client: TestClient, settings: Settings
) -> None:
    job_id = client.post("/jobs", data=YOUTUBE_FORM).json()["id"]
    store = store_for(settings)

    assert client.get(f"/api/jobs/{job_id}/download").status_code == 404

    store.update(job_id, state="done", title="Hello / World", output="out.mp4", progress=1.0)
    assert read_status(settings, job_id)["download_url"] == f"/api/jobs/{job_id}/download"

    # done, but the file is not there (swept, or a crash between write and rename)
    conflict = client.get(f"/api/jobs/{job_id}/download")
    assert conflict.status_code == 409
    assert "gone" in conflict.json()["error"]

    (store.path(job_id) / "out.mp4").write_bytes(b"\x00\x01mp4")
    ok = client.get(f"/api/jobs/{job_id}/download")
    assert ok.status_code == 200
    assert ok.content == b"\x00\x01mp4"
    assert ok.headers["content-type"] == "video/mp4"
    # the title is sanitised (no "/") and the target language is appended
    assert "Hello _ World-es.mp4" in unquote(ok.headers["content-disposition"])


# --------------------------------------------------------------------------- JobStore


def test_store_update_merges_bumps_and_derives(settings: Settings) -> None:
    store = store_for(settings)
    job_id = store.create(
        source={"type": "youtube", "url": "https://y/1"},
        options={"to_lang": "es", "from_lang": None, "backend": "kokoro", "burn_subtitles": True},
        title="A Title",
    )
    created = store.get(job_id)
    assert created is not None
    assert created["title"] == "A Title"

    updated = store.update(job_id, options={"backend": "chatterbox"}, detected_language="en")
    assert updated["options"] == {
        "to_lang": "es",
        "from_lang": None,
        "backend": "chatterbox",
        "voice": None,
        "burn_subtitles": True,
    }
    assert updated["detected_language"] == "en"
    assert parse_iso(updated["updated_at"]) > parse_iso(created["updated_at"])

    assert store.update(job_id, output="out.mp4")["download_url"] == f"/api/jobs/{job_id}/download"
    assert store.update(job_id, output=None)["download_url"] is None

    marked = store.mark(job_id, "running", step="translate", progress=0.5)
    assert (marked["state"], marked["step"], marked["progress"]) == ("running", "translate", 0.5)

    failed = store.fail(job_id, "translate: no package")
    assert failed["state"] == "failed"
    assert failed["error"] == "translate: no package"

    assert store.list_ids() == [job_id]
    assert store.path(job_id) == settings.jobs_dir / job_id


def test_fail_clears_the_detail_line(settings: Settings) -> None:
    """F4: `detail` describes work happening now; a failed job must not still claim to be speaking."""
    store = store_for(settings)
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    running = store.update(job_id, state="running", step="speak", detail="speaking segment 4 of 12")
    assert running["detail"] == "speaking segment 4 of 12"

    failed = store.fail(job_id, "speak: kokoro ran out of voices")
    assert failed["state"] == "failed"
    assert failed["error"] == "speak: kokoro ran out of voices"
    assert failed["detail"] is None
    assert read_status(settings, job_id)["detail"] is None
    assert failed["step"] == "speak", "the step that died is still named"


def test_store_rejects_nonsense(settings: Settings) -> None:
    store = store_for(settings)
    assert store.get("nope") is None
    with pytest.raises(JobNotFoundError):
        store.update("nope", state="done")
    with pytest.raises(ValueError, match="invalid job id"):
        store.path("../escape")
    with pytest.raises(ValueError, match="source type"):
        store.create(source={"type": "torrent"}, options={"to_lang": "es"})
    with pytest.raises(ValueError, match="to_lang"):
        store.create(source={"type": "youtube", "url": "u"}, options={})
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    with pytest.raises(ValueError, match="unknown job state"):
        store.mark(job_id, "sideways")  # type: ignore[arg-type]


def test_status_json_is_replaced_atomically(settings: Settings) -> None:
    store = store_for(settings)
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    for step in ("probe", "fetch", "transcribe"):
        store.mark(job_id, "running", step=step)
    leftovers = [p.name for p in store.path(job_id).iterdir() if p.name != "status.json"]
    assert leftovers == []  # no temp files survive


# --------------------------------------------------------------------------- JobRunner


def test_runner_reports_the_placeholder_failure(settings: Settings) -> None:
    store = store_for(settings)
    runner = JobRunner(store, settings)
    assert runner.run_fn is placeholder_run
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    try:
        runner.submit(job_id).result(timeout=10)
    finally:
        runner.shutdown()
    status = store.get(job_id)
    assert status is not None
    assert status["state"] == "failed"
    assert status["error"] == "start: the dubbing pipeline is not wired up yet"


def test_runner_names_the_step_that_died(settings: Settings) -> None:
    store = store_for(settings)

    def boom(job_id: str, _settings: Settings, store_: JobStore) -> None:
        store_.mark(job_id, "running", step="speak", progress=0.6)
        raise RuntimeError("kokoro exploded")

    runner = JobRunner(store, settings, run_fn=boom)
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    try:
        runner.submit(job_id).result(timeout=10)
    finally:
        runner.shutdown()
    status = store.get(job_id)
    assert status is not None
    assert status["error"] == "speak: kokoro exploded"
    assert status["state"] == "failed"
    assert status["step"] == "speak"


def test_submit_returns_immediately_and_marks_running(settings: Settings) -> None:
    store = store_for(settings)
    started = threading.Event()
    release = threading.Event()

    def slow(job_id: str, _settings: Settings, store_: JobStore) -> None:
        started.set()
        release.wait(10)
        store_.mark(job_id, "done", step="finish", progress=1.0)

    runner = JobRunner(store, settings, run_fn=slow)
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    try:
        begin = time.perf_counter()
        future = runner.submit(job_id)
        assert time.perf_counter() - begin < 0.5
        assert started.wait(5)
        running = store.get(job_id)
        assert running is not None and running["state"] == "running"
        release.set()
        future.result(timeout=10)
    finally:
        release.set()
        runner.shutdown()
    done = store.get(job_id)
    assert done is not None and done["state"] == "done"


def test_singletons_are_shared_per_process(settings: Settings) -> None:
    assert jobs_module.get_store(settings) is jobs_module.get_store(settings)
    assert jobs_module.get_runner(settings) is jobs_module.get_runner(settings)
    assert jobs_module.get_runner(settings).store is jobs_module.get_store(settings)


# --------------------------------------------------------------------------- sweeper


def _aged(store: JobStore, state: str, minutes: float) -> str:
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    stamp = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat(timespec="microseconds")
    status = store.require(job_id)
    status.update(state=state, created_at=stamp, updated_at=stamp)
    (store.path(job_id) / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return job_id


def test_sweep_uses_state_and_age(settings: Settings) -> None:
    store = store_for(settings)
    fresh_done = _aged(store, "done", 1)
    young_done = _aged(store, "done", 30)
    old_done = _aged(store, "done", 90)
    old_failed = _aged(store, "failed", 120)
    running_now = _aged(store, "running", 90)
    running_stale = _aged(store, "running", 60 * 7)
    queued_stale = _aged(store, "queued", 60 * 8)

    removed = sweep(store, ttl_minutes=settings.job_ttl_minutes)

    assert sorted(removed) == sorted([old_done, old_failed, running_stale, queued_stale])
    assert sorted(store.list_ids()) == sorted([fresh_done, young_done, running_now])
    assert not (settings.jobs_dir / old_done).exists()


def test_sweep_never_touches_anything_younger_than_five_minutes(settings: Settings) -> None:
    store = store_for(settings)
    just_failed = _aged(store, "failed", 2)
    assert sweep(store, ttl_minutes=0) == []
    assert store.list_ids() == [just_failed]


def test_sweep_accepts_an_explicit_now(settings: Settings) -> None:
    store = store_for(settings)
    done = _aged(store, "done", 0)
    later = datetime.now(UTC) + timedelta(minutes=settings.job_ttl_minutes + 10)
    assert sweep(store, ttl_minutes=settings.job_ttl_minutes, now=later) == [done]


def test_sweep_removes_a_directory_without_status_json(settings: Settings) -> None:
    store = store_for(settings)
    orphan = store.jobs_dir / "orphaned"
    orphan.mkdir(parents=True)
    old = time.time() - 7 * 3600
    os.utime(orphan, (old, old))
    assert sweep(store, ttl_minutes=60) == ["orphaned"]


def test_start_sweeper_runs_and_stops(settings: Settings) -> None:
    store = store_for(settings)
    doomed = _aged(store, "done", 999)
    handle = start_sweeper(store, settings, interval_seconds=0.05)
    try:
        assert handle.thread.daemon
        deadline = time.time() + 5
        while time.time() < deadline and store.list_ids():
            time.sleep(0.05)
        assert store.list_ids() == [], f"{doomed} should have been swept"
    finally:
        handle.stop.set()
    handle.thread.join(timeout=5)
    assert not handle.thread.is_alive()


# --------------------------------------------------------------------------- F2: a broken job dir


def _write_status(store: JobStore, job_id: str, raw: str) -> None:
    (store.path(job_id) / "status.json").write_text(raw, encoding="utf-8")


def test_sweep_survives_unreadable_status_files(settings: Settings) -> None:
    """One job with `{}` and one holding a JSON list must not stop the pass (F2)."""
    store = store_for(settings)
    empty = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    _write_status(store, empty, "{}")
    listish = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    _write_status(store, listish, '["not", "a", "status"]')
    truncated = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    _write_status(store, truncated, '{"id": "x", "state": "do')
    stamped = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    _write_status(store, stamped, '{"id": "x", "state": "done", "updated_at": "not a date"}')
    old_done = _aged(store, "done", 90)

    removed = sweep(store, ttl_minutes=settings.job_ttl_minutes)

    assert removed == [old_done], "an expired job must still be swept in the same pass"
    assert sorted(store.list_ids()) == sorted([empty, listish, truncated, stamped])


def test_sweep_ages_an_unreadable_status_by_the_directory_mtime(settings: Settings) -> None:
    """No usable timestamp: fall back to the directory's mtime and the 6 h rule (F2)."""
    store = store_for(settings)
    young = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    _write_status(store, young, "{}")
    old = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    _write_status(store, old, "{}")
    stamp = time.time() - 7 * 3600
    os.utime(store.path(old), (stamp, stamp))

    assert sweep(store, ttl_minutes=settings.job_ttl_minutes) == [old]
    assert store.list_ids() == [young]


# --------------------------------------------------------------------------- F3: live jobs


def test_sweep_never_deletes_a_job_a_worker_is_inside(settings: Settings) -> None:
    store = store_for(settings)
    stale = _aged(store, "running", 60 * 7)  # older than the 6 h "the server died" rule
    assert sweep(store, ttl_minutes=settings.job_ttl_minutes, active={stale}) == []
    assert store.list_ids() == [stale]
    assert sweep(store, ttl_minutes=settings.job_ttl_minutes) == [stale]


def test_runner_publishes_the_jobs_it_is_running(settings: Settings) -> None:
    store = store_for(settings)
    started = threading.Event()
    release = threading.Event()
    seen: set[str] = set()

    def slow(job_id: str, _settings: Settings, store_: JobStore) -> None:
        seen.update(runner.active_ids)
        started.set()
        release.wait(10)

    runner = JobRunner(store, settings, run_fn=slow)
    job_id = store.create(source={"type": "youtube", "url": "u"}, options={"to_lang": "es"})
    try:
        future = runner.submit(job_id)
        assert started.wait(5)
        assert job_id in runner.active_ids
        release.set()
        future.result(timeout=10)
    finally:
        release.set()
        runner.shutdown()
    assert seen == {job_id}
    assert runner.active_ids == set(), "the id must be dropped even though the job ended"


def test_start_sweeper_asks_the_runner_what_is_live(settings: Settings) -> None:
    store = store_for(settings)
    doomed = _aged(store, "done", 999)
    runner = JobRunner(store, settings)
    runner.active_ids.add(doomed)
    handle = start_sweeper(store, settings, runner=runner, interval_seconds=0.05)
    try:
        time.sleep(0.4)  # several passes
        assert store.list_ids() == [doomed], "an active job must survive the sweeper"
        runner.active_ids.discard(doomed)
        deadline = time.time() + 5
        while time.time() < deadline and store.list_ids():
            time.sleep(0.05)
        assert store.list_ids() == []
    finally:
        handle.stop.set()
        runner.shutdown(wait=False)
    handle.thread.join(timeout=5)


# --------------------------------------------------------------------------- F1: off the event loop


@pytest.mark.parametrize("name", ["health", "backends", "create_job", "job_status", "job_download"])
def test_routes_run_in_the_threadpool_not_on_the_loop(name: str) -> None:
    """A coroutine route would do its disk writes and its torch import on the event loop (F1)."""
    endpoint = getattr(api_module, name)
    assert not inspect.iscoroutinefunction(endpoint), f"{name} must be a plain def"


def test_importing_the_api_never_loads_torch_or_chatterbox() -> None:
    """`/api/backends` must not be able to drag half a gigabyte of model code onto the loop (F1).

    Run in a subprocess: by this point in the session another test has almost certainly imported
    torch already, so only a fresh interpreter can answer the question.
    """
    code = (
        "import sys\n"
        "import respeak.api\n"
        "from respeak.config import Settings\n"
        "from respeak.pipeline.tts import available_backends\n"
        "available_backends(Settings())\n"
        "heavy = [m for m in sys.modules if m == 'torch' or m.startswith(('torch.', 'chatterbox'))]\n"
        "print(','.join(sorted(heavy)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"importing respeak.api pulled in {proc.stdout.strip()}"


def test_resolved_device_is_computed_once_per_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.cuda.is_available() costs about a second; /api/health must not pay it per request."""
    calls: list[int] = []

    def counted(self: Settings) -> str:
        calls.append(1)
        return "cpu"

    monkeypatch.setattr(Settings, "_detect_device", counted)
    settings = Settings(device="auto")
    assert settings.resolved_device() == "cpu"
    assert settings.resolved_device() == "cpu"
    assert len(calls) == 1
    assert Settings(device="auto").resolved_device() == "cpu"
    assert len(calls) == 2, "the cache is per instance, not global"


# --------------------------------------------------------------------------- F6: startup imports


def test_pipeline_run_fn_returns_the_real_entry_point() -> None:
    from respeak.pipeline.run import run_job

    assert main_module._pipeline_run_fn() is run_job


def test_pipeline_run_fn_tolerates_only_its_own_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "respeak.pipeline.run", None)
    assert main_module._pipeline_run_fn() is None


def test_pipeline_run_fn_reraises_a_broken_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing torch must fail at startup with its own traceback, not become a silent None (F6)."""
    broken = types.ModuleType("respeak.pipeline.run")

    def _raise(name: str) -> Any:
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    broken.__getattr__ = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "respeak.pipeline.run", broken)
    with pytest.raises(ModuleNotFoundError, match="torch"):
        main_module._pipeline_run_fn()
