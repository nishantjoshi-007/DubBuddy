from fastapi.testclient import TestClient

from respeak.main import app


def test_health_reports_name_and_limits():
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "respeak"
    assert body["device"] in {"cpu", "cuda"}
    assert body["limits"]["max_video_seconds"] == 900


def test_backends_lists_kokoro_as_default():
    with TestClient(app) as client:
        r = client.get("/api/backends")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "kokoro"
    names = {b["name"] for b in body["backends"]}
    assert {"kokoro", "chatterbox"} <= names
    assert body["source_languages"][0]["code"]


def test_home_and_job_pages_render():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        r = client.get("/jobs/abc123")
    assert r.status_code == 200
    assert 'data-job-id="abc123"' in r.text
