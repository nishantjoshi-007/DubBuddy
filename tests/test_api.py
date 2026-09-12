"""WP-A: job store, job runner, routes and sweeper (flow.md B3, B5, B7).

Everything here is offline: no model, no network, no ffmpeg. The pipeline itself is stubbed, which is
the point of injecting ``run_fn`` into :class:`respeak.jobs.JobRunner`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from respeak import jobs as jobs_module
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

# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Real Settings pointed at tmp_path, with the singletons reset around each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOW_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "2")
    monkeypatch.setenv("JOB_TTL_MINUTES", "60")
    get_settings.cache_clear()
    jobs_module.reset_state()
    yield get_settings()
    jobs_module.reset_state()
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


def test_upload_job_streams_the_file_and_sanitises_its_name(
    client: TestClient, settings: Settings
) -> None:
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
    """Whatever WP-D reports as missing must be refused before the job exists (D-27)."""
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
