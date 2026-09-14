"""Job state on disk, the worker pool and the TTL sweeper (flow.md B5, B7).

One directory per job under ``DATA_DIR/jobs/<id>/``; ``status.json`` is the only source of truth, so
any web process (``--workers 2``) can answer for any job. Writes are atomic (temp file + ``os.replace``),
reads never lock. Nothing here is async: the pipeline is a plain synchronous function that the
:class:`JobRunner` calls in a worker thread.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple

from .config import Settings

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "failed"]
STATES: frozenset[str] = frozenset({"queued", "running", "done", "failed"})
FINISHED_STATES: frozenset[str] = frozenset({"done", "failed"})

STATUS_FILENAME = "status.json"
OUTPUT_FILENAME = "out.mp4"

#: A job id must be usable as a single directory name; this also makes path traversal impossible.
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SWEEP_INTERVAL_SECONDS = 300.0
#: Never delete a job directory younger than this, whatever its state (flow.md B5).
MIN_AGE_MINUTES = 5
#: A `queued`/`running` job older than this belongs to a server that died mid-job (flow.md B5).
STALE_RUNNING_HOURS = 6

#: ``run_fn(job_id, settings, store)`` — the injected pipeline entry point (WP-F's ``pipeline/run.py``).
RunFn = Callable[[str, Settings, "JobStore"], None]


class JobNotFoundError(LookupError):
    """Raised when a job id has no directory under ``DATA_DIR/jobs``."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"unknown job id: {job_id}")
        self.job_id = job_id


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def parse_iso(value: str) -> datetime:
    """Parse an ISO timestamp from status.json; a naive timestamp is read as UTC."""
    moment = datetime.fromisoformat(value)
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def _check_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ValueError(f"invalid job id: {job_id!r}")
    return job_id


class JobStore:
    """Reads and writes ``DATA_DIR/jobs/<id>/status.json`` (schema: flow.md B5)."""

    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- paths

    def path(self, job_id: str) -> Path:
        """The job's directory (it may not exist yet)."""
        return self.jobs_dir / _check_job_id(job_id)

    def status_path(self, job_id: str) -> Path:
        return self.path(job_id) / STATUS_FILENAME

    def list_ids(self) -> list[str]:
        """Every job directory name currently on disk, sorted."""
        if not self.jobs_dir.is_dir():
            return []
        return sorted(
            entry.name for entry in self.jobs_dir.iterdir() if entry.is_dir() and JOB_ID_RE.match(entry.name)
        )

    # --------------------------------------------------------------- create

    def create(
        self,
        source: dict[str, Any],
        options: dict[str, Any],
        title: str | None = None,
    ) -> str:
        """Create ``<jobs_dir>/<uuid4 hex>/status.json`` in state ``queued`` and return the id."""
        source_type = source.get("type")
        if source_type not in {"youtube", "upload"}:
            raise ValueError(f"source type must be 'youtube' or 'upload', got {source_type!r}")
        if not options.get("to_lang"):
            raise ValueError("options must contain a non-empty 'to_lang'")

        job_id = uuid.uuid4().hex
        job_dir = self.path(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        now = _iso(_utcnow())
        status: dict[str, Any] = {
            "id": job_id,
            "state": "queued",
            "step": None,
            "progress": 0.0,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "title": title,
            "detected_language": None,
            "source": {
                "type": source_type,
                "url": source.get("url"),
                "filename": source.get("filename"),
            },
            "options": {
                "to_lang": options["to_lang"],
                "from_lang": options.get("from_lang"),
                "backend": options.get("backend"),
                "voice": options.get("voice"),  # a backend voice id, or None for its default
                "burn_subtitles": bool(options.get("burn_subtitles", True)),
            },
            "output": None,
            "download_url": None,
            "warnings": [],
            "detail": None,
        }
        self._write(job_id, _derive(status))
        log.info("job %s created (%s → %s)", job_id, source_type, status["options"]["to_lang"])
        return job_id

    # ----------------------------------------------------------- read/write

    def get(self, job_id: str) -> dict[str, Any] | None:
        """The status dict, or ``None`` when there is no such job. Corrupt JSON raises."""
        try:
            path = self.status_path(job_id)
        except ValueError:
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            status = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(status, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        return status

    def require(self, job_id: str) -> dict[str, Any]:
        status = self.get(job_id)
        if status is None:
            raise JobNotFoundError(job_id)
        return status

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into status.json, bump ``updated_at`` and write atomically.

        ``source`` and ``options`` merge one level deep, so ``update(id, options={"backend": "kokoro"})``
        keeps the other options.
        """
        with self._lock:
            status = self.require(job_id)
            for key, value in fields.items():
                current = status.get(key)
                if isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                else:
                    status[key] = value
            status["updated_at"] = _iso(_utcnow())
            status = _derive(status)
            self._write(job_id, status)
        return status

    def mark(
        self,
        job_id: str,
        state: JobState,
        step: str | None = None,
        progress: float | None = None,
    ) -> dict[str, Any]:
        """Set the state and, when given, the current step and progress."""
        if state not in STATES:
            raise ValueError(f"unknown job state: {state!r} (expected one of {sorted(STATES)})")
        fields: dict[str, Any] = {"state": state}
        if step is not None:
            fields["step"] = step
        if progress is not None:
            fields["progress"] = float(progress)
        return self.update(job_id, **fields)

    def fail(self, job_id: str, error: str) -> dict[str, Any]:
        """Move the job to ``failed`` with a visible message (flow.md B8: never a silent None).

        ``detail`` is cleared with it: it is the sentence for work that is *happening now*, and a
        failed job leaving "speaking segment 4 of 12" on the page next to the error reads as if the
        job were somehow still going. Only ``error`` speaks for a failed job.
        """
        log.error("job %s failed: %s", job_id, error)
        return self.update(job_id, state="failed", error=str(error), detail=None)

    def delete(self, job_id: str) -> None:
        """Remove the whole job directory. Missing is not an error."""
        shutil.rmtree(self.path(job_id), ignore_errors=True)

    # -------------------------------------------------------------- private

    def _write(self, job_id: str, status: dict[str, Any]) -> None:
        job_dir = self.path(job_id)
        if not job_dir.is_dir():
            raise JobNotFoundError(job_id)
        target = job_dir / STATUS_FILENAME
        tmp = job_dir / f".{STATUS_FILENAME}.{uuid.uuid4().hex}.tmp"
        payload = json.dumps(status, indent=2, ensure_ascii=False, sort_keys=False)
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def _derive(status: dict[str, Any]) -> dict[str, Any]:
    """``download_url`` is never stored by hand: it exists exactly while ``output`` does."""
    output = status.get("output")
    status["download_url"] = f"/api/jobs/{status['id']}/download" if output else None
    return status


def placeholder_run(job_id: str, settings: Settings, store: JobStore) -> None:
    """Default ``run_fn`` until WP-F lands ``respeak/pipeline/run.py``."""
    raise RuntimeError("the dubbing pipeline is not wired up yet")


class JobRunner:
    """A bounded thread pool that runs one synchronous pipeline call per job (flow.md B1, B3)."""

    def __init__(self, store: JobStore, settings: Settings, run_fn: RunFn | None = None) -> None:
        self.store = store
        self.settings = settings
        self.run_fn: RunFn = run_fn or placeholder_run
        #: Jobs a worker thread is inside right now. The sweeper reads it so that a job that is
        #: genuinely still running (a long video on a slow CPU) is never deleted under its own feet.
        self.active_ids: set[str] = set()
        self._active_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(settings.max_concurrent_jobs)),
            thread_name_prefix="respeak-job",
        )

    def submit(self, job_id: str) -> Future[None]:
        """Schedule the job and return at once; extra jobs wait in the pool queue in ``queued``."""
        _check_job_id(job_id)
        return self._pool.submit(self._run, job_id)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)

    def _run(self, job_id: str) -> None:
        with self._active_lock:
            self.active_ids.add(job_id)
        try:
            self.store.mark(job_id, "running")
            self.run_fn(job_id, self.settings, self.store)
        except Exception as exc:  # every failure has to end up visible on the job page
            step = self._current_step(job_id)
            log.exception("job %s died in %s", job_id, step)
            try:
                self.store.fail(job_id, f"{step}: {exc}")
            except Exception:  # pragma: no cover - the job dir vanished under us
                log.exception("could not record the failure of job %s", job_id)
        finally:
            with self._active_lock:
                self.active_ids.discard(job_id)

    def _current_step(self, job_id: str) -> str:
        try:
            status = self.store.get(job_id)
        except Exception:  # pragma: no cover - unreadable status.json
            return "start"
        if not status:
            return "start"
        return str(status.get("step") or "start")


def sweep(
    store: JobStore,
    ttl_minutes: int,
    now: datetime | None = None,
    active: Collection[str] = frozenset(),
) -> list[str]:
    """Delete expired job directories and return the ids removed (flow.md B5).

    * ``done`` / ``failed`` older than ``ttl_minutes``
    * ``queued`` / ``running`` older than 6 h (a server that died mid-job)
    * a directory without a readable status.json, or with unreadable timestamps, older than 6 h
      (measured from the directory's own mtime)
    * never anything younger than 5 minutes, and never an id in ``active``

    One unreadable job never stops the pass: every directory is examined on its own.
    """
    moment = now or _utcnow()
    floor = moment - timedelta(minutes=MIN_AGE_MINUTES)
    ttl_cutoff = moment - timedelta(minutes=max(0, int(ttl_minutes)))
    stale_cutoff = moment - timedelta(hours=STALE_RUNNING_HOURS)

    removed: list[str] = []
    for job_id in store.list_ids():
        if job_id in active:
            log.debug("sweeper skipped job %s: a worker is running it", job_id)
            continue
        try:
            if _sweep_one(store, job_id, moment, floor, ttl_cutoff, stale_cutoff):
                removed.append(job_id)
        except Exception:  # one broken job directory must never end the pass
            log.exception("sweeper skipped job %s: it could not be examined", job_id)
    return removed


def _sweep_one(
    store: JobStore,
    job_id: str,
    moment: datetime,
    floor: datetime,
    ttl_cutoff: datetime,
    stale_cutoff: datetime,
) -> bool:
    """Delete one job directory if it has expired; True when it was removed."""
    job_dir = store.path(job_id)
    touched, cutoff = _age_of(store, job_id, job_dir, moment, ttl_cutoff, stale_cutoff)
    if touched >= cutoff or touched > floor:
        return False
    try:
        shutil.rmtree(job_dir)
    except OSError as exc:  # pragma: no cover - permissions / races
        log.warning("sweeper could not remove %s: %s", job_dir, exc)
        return False
    log.info("sweeper removed job %s (last touched %s)", job_id, _iso(touched))
    return True


def _age_of(
    store: JobStore,
    job_id: str,
    job_dir: Path,
    moment: datetime,
    ttl_cutoff: datetime,
    stale_cutoff: datetime,
) -> tuple[datetime, datetime]:
    """``(last touched, the cutoff it must be older than)`` for one job directory."""
    try:
        status = store.get(job_id)  # raises on invalid JSON or a JSON list
    except ValueError as exc:
        log.warning("job %s has an unreadable status.json (%s); ageing it by its directory", job_id, exc)
        status = None
    if isinstance(status, dict):
        raw = status.get("updated_at") or status.get("created_at")
        if raw:
            try:
                touched = parse_iso(str(raw))
            except (TypeError, ValueError):
                log.warning("job %s has an unreadable timestamp %r; ageing it by its directory", job_id, raw)
            else:
                state = str(status.get("state", "queued"))
                return touched, (ttl_cutoff if state in FINISHED_STATES else stale_cutoff)
    # No status, no timestamps: the directory's own mtime and the 6 h rule for a crashed server.
    try:
        touched = datetime.fromtimestamp(job_dir.stat().st_mtime, UTC)
    except OSError:  # pragma: no cover - the directory vanished under us
        touched = moment
    return touched, stale_cutoff


class SweeperHandle(NamedTuple):
    """The sweeper thread and the event that stops it (unpacks as ``thread, stop``)."""

    thread: threading.Thread
    stop: threading.Event


def start_sweeper(
    store: JobStore,
    settings: Settings,
    runner: JobRunner | None = None,
    interval_seconds: float = SWEEP_INTERVAL_SECONDS,
) -> SweeperHandle:
    """Start the daemon thread that sweeps every 5 minutes; ``handle.stop.set()`` ends it.

    Pass the ``runner`` so that jobs its workers are inside right now are never swept.
    """
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval_seconds):
            try:
                active = runner.active_ids if runner is not None else frozenset()
                removed = sweep(store, settings.job_ttl_minutes, active=active)
            except Exception:  # pragma: no cover - the sweeper must never die
                log.exception("sweeper pass failed")
            else:
                if removed:
                    log.info("sweeper removed %d job dir(s)", len(removed))

    thread = threading.Thread(target=loop, name="respeak-sweeper", daemon=True)
    thread.start()
    return SweeperHandle(thread=thread, stop=stop)


# --------------------------------------------------------------------------
# One store and one runner per process, created on first use so that importing
# respeak.api costs nothing. main.py and api.py share these.
# --------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_store: JobStore | None = None
_runner: JobRunner | None = None


def get_store(settings: Settings) -> JobStore:
    """The process-wide :class:`JobStore` for ``settings.jobs_dir``."""
    global _store
    jobs_dir = Path(settings.jobs_dir).resolve()
    with _singleton_lock:
        if _store is None or _store.jobs_dir.resolve() != jobs_dir:
            _store = JobStore(jobs_dir)
        return _store


def get_runner(settings: Settings, run_fn: RunFn | None = None) -> JobRunner:
    """The process-wide :class:`JobRunner`. Rebuilt if the store or ``run_fn`` changed."""
    global _runner
    store = get_store(settings)
    with _singleton_lock:
        stale = _runner is not None and (
            _runner.store is not store or (run_fn is not None and _runner.run_fn is not run_fn)
        )
        if stale and _runner is not None:
            _runner.shutdown(wait=False)
            _runner = None
        if _runner is None:
            _runner = JobRunner(store, settings, run_fn)
        return _runner


def reset_state(wait: bool = False) -> None:
    """Drop the singletons (tests, and a lifespan that restarts in the same process)."""
    global _store, _runner
    with _singleton_lock:
        if _runner is not None:
            _runner.shutdown(wait=wait)
        _runner = None
        _store = None
