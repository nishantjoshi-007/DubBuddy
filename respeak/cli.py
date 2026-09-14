"""The ``respeak`` command (flow.md B2, B6; plan.md 3.2).

    respeak dub <url-or-file> --to es [--from en] [--backend …] [--voice ID] [--no-burn]
                              [--out PATH] [--data-dir DIR]
    respeak serve [--host H] [--port P] [--reload]
    respeak prewarm [--languages en,es …]

``dub`` is the web page without the web: the same :func:`respeak.pipeline.run.run_job`, the same job
directory, the same validation — only the browser is replaced by a terminal. It runs the pipeline in
*this* thread (there is no server to keep responsive), narrates every status change to stderr, prints
the finished file's path to stdout and exits 0, or prints the failure and exits 1.

Every import that costs real time (fastapi, uvicorn, torch by way of the pipeline) happens inside the
subcommand that needs it, so ``respeak --help`` answers instantly.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from . import __version__
from .config import Settings, get_settings
from .jobs import JobStore

log = logging.getLogger("respeak.cli")

#: Loopback, not 0.0.0.0: a laptop running `respeak serve` is not asking to be on the network.
#: The Docker image binds 0.0.0.0 in its own CMD, where that is the point.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

UPLOAD_FILENAME = "upload.bin"
#: What a dub is called when the source had no title: `<stem>-<lang>.mp4` needs a stem.
FALLBACK_STEM = "respeak"

#: Libraries that raise their *own* logger to INFO at import (argostranslate/utils.py does exactly
#: that), so records reach our root handler however quiet the root logger is. Without `-v` they
#: would bury the progress lines under a tokenised copy of every sentence being translated.
LOUD_LIBRARIES: tuple[str, ...] = ("argostranslate", "stanza")


# --------------------------------------------------------------------------------------- progress


class ProgressStore(JobStore):
    """A :class:`~respeak.jobs.JobStore` that narrates what it is asked to write.

    The browser polls ``status.json`` every three seconds; a terminal can do better. ``run_job``
    writes every step, progress value and detail through ``update()``, so subclassing it reports
    each change the moment it happens — no polling thread, and nothing for the pipeline to know
    about. Only real changes are printed: the pipeline writes far more often than it changes.
    """

    def __init__(self, jobs_dir: Path, stream: TextIO | None = None) -> None:
        super().__init__(jobs_dir)
        self.stream = stream if stream is not None else sys.stderr
        self._last: tuple[str, str, int] | None = None
        self._warned: set[str] = set()

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        status = super().update(job_id, **fields)
        self.report(status)
        return status

    def report(self, status: dict[str, Any]) -> None:
        """Print one line per visible change: ``[ 60%] speak: segment 4 of 12``."""
        step = str(status.get("step") or status.get("state") or "")
        detail = str(status.get("detail") or "")
        percent = int(round(float(status.get("progress") or 0.0) * 100))
        key = (step, detail, percent)
        if key != self._last:
            self._last = key
            self._emit(f"[{percent:3d}%] {step}" + (f": {detail}" if detail else ""))
        for warning in status.get("warnings") or []:
            text = str(warning)
            if text not in self._warned:
                self._warned.add(text)
                self._emit(f"warning: {text}")

    def _emit(self, line: str) -> None:
        print(line, file=self.stream, flush=True)


# ------------------------------------------------------------------------------------------- dub


def _settings_for(data_dir: str | None) -> Settings:
    """The process settings, with ``--data-dir`` overriding ``DATA_DIR`` when it was given."""
    settings = get_settings()
    if not data_dir:
        return settings
    return settings.model_copy(update={"data_dir": Path(data_dir).expanduser()})


def looks_like_url(text: str) -> bool:
    """True for something yt-dlp should be handed rather than something to open as a file."""
    return urlparse(text.strip()).scheme in {"http", "https"}


def _job_request(
    settings: Settings, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    """``(source, options, local file)`` for :meth:`JobStore.create`, or :class:`Rejected`.

    The backend, language, voice and public-URL checks are literally the ones ``POST /jobs`` runs
    (``respeak.api``), so a flag the web form would refuse is refused here with the same sentence.
    ``ALLOW_UPLOADS`` is not consulted: it locks down a *server*, and this is the operator's own
    shell.
    """
    from .api import Rejected, check_options, check_source_url, safe_name

    options = check_options(settings, args.to_lang, args.from_lang, args.backend, args.voice)
    options["burn_subtitles"] = not args.no_burn

    if looks_like_url(args.input):
        return {"type": "youtube", "url": check_source_url(args.input), "filename": None}, options, None

    path = Path(args.input).expanduser()
    if not path.is_file():
        raise Rejected(400, f"{args.input!r} is neither an http(s) URL nor a file that exists")
    source = {"type": "upload", "url": None, "filename": safe_name(path.name, fallback="upload")}
    return source, options, path


def _place_upload(source: Path, dest: Path) -> None:
    """Put the local video where the pipeline expects it, without copying a gigabyte if possible.

    A hard link costs nothing and no disk; the job's own cleanup unlinks it at the end, which never
    touches the user's file (it keeps its own link). Across filesystems that is impossible, so copy.
    """
    try:
        os.link(source, dest)
    except OSError as exc:
        log.debug("could not link %s (%s); copying instead", source, exc)
        shutil.copy2(source, dest)


def _output_name(status: dict[str, Any], to_lang: str) -> str:
    """``<title>-<lang>.mp4``, the name the download route offers — minus the browser's escaping."""
    from .api import safe_name

    stem = safe_name(status.get("title"))
    if not stem:  # a job whose source had no title at all: name it after the file that was dubbed
        filename = (status.get("source") or {}).get("filename")
        stem = safe_name(Path(str(filename)).stem) if filename else ""
    return f"{stem or FALLBACK_STEM}-{to_lang}.mp4"


def _destination(out: str | None, status: dict[str, Any], to_lang: str) -> Path:
    """Where ``--out`` says the video goes; a directory keeps the default name inside it."""
    if not out:
        return Path.cwd() / _output_name(status, to_lang)
    target = Path(out).expanduser()
    if target.is_dir():
        return target / _output_name(status, to_lang)
    return target


def _dub(args: argparse.Namespace) -> int:
    """Run one job to completion in this process (flow.md B6 "same run_job, progress on stderr")."""
    from .api import Rejected
    from .jobs import OUTPUT_FILENAME
    from .pipeline.ffmpeg import ensure_binaries
    from .pipeline.run import run_job

    settings = _settings_for(args.data_dir)
    try:
        source, options, upload = _job_request(settings, args)
    except Rejected as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    ensure_binaries()  # ffmpeg, ffprobe and deno on PATH, exactly as the server's lifespan does
    store = ProgressStore(settings.jobs_dir)
    job_id = store.create(source=source, options=options)
    job_dir = store.path(job_id)
    if upload is not None:
        _place_upload(upload, job_dir / UPLOAD_FILENAME)
    print(f"job {job_id} in {job_dir}", file=sys.stderr)

    try:
        run_job(job_id, settings, store)
    except Exception as exc:
        step = str((store.get(job_id) or {}).get("step") or "start")
        store.fail(job_id, f"{step}: {exc}")  # the same record JobRunner would leave on the web
        print(f"error: {step}: {exc}", file=sys.stderr)
        return 1

    status = store.require(job_id)
    produced = job_dir / Path(str(status.get("output") or OUTPUT_FILENAME)).name
    if not produced.is_file():
        print(f"error: the pipeline finished but {produced} is not there", file=sys.stderr)
        return 1

    destination = _destination(args.out, status, str(options["to_lang"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, destination)
    print(f"subtitles and status stayed in {job_dir}", file=sys.stderr)
    print(destination)  # stdout carries exactly one thing: the path, for a script to read
    return 0


# ----------------------------------------------------------------------------------- serve, prewarm


def _serve(args: argparse.Namespace) -> int:
    """``uvicorn respeak.main:app``, with the flags a person actually retypes."""
    import uvicorn

    # uvicorn trusts X-Forwarded-For only from these peers; mirror TRUST_PROXY so the rate limiter and
    # uvicorn agree on who the client is (uvicorn's own default is to trust 127.0.0.1).
    from .config import get_settings

    allow = "*" if get_settings().trust_proxy else "127.0.0.1"
    uvicorn.run(
        "respeak.main:app", host=args.host, port=args.port, reload=args.reload, forwarded_allow_ips=allow
    )
    return 0


def _prewarm(args: argparse.Namespace) -> int:
    """Hand everything after the subcommand to :func:`respeak.prewarm.main` unchanged."""
    from .prewarm import main as prewarm_main

    return prewarm_main(args.args)


# ---------------------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """The whole command line. ``--help`` on any subcommand documents every flag it takes."""
    parser = argparse.ArgumentParser(
        prog="respeak",
        description="Dub a video into another language on your own machine.",
    )
    parser.add_argument("--version", action="version", version=f"respeak {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show the pipeline's own log (at LOG_LEVEL, INFO unless you changed it)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    dub = subcommands.add_parser(
        "dub",
        help="dub one video and write the result to a file",
        description="Transcribe, translate, re-voice, subtitle and mux one video. Progress goes to "
        "stderr; the path of the finished file is printed to stdout.",
    )
    dub.add_argument("input", metavar="URL-OR-FILE", help="a YouTube URL, or a local video file")
    dub.add_argument(
        "--to",
        dest="to_lang",
        required=True,
        metavar="LANG",
        help="target language as an ISO-639-1 code, e.g. es (must be one the backend speaks)",
    )
    dub.add_argument(
        "--from",
        dest="from_lang",
        default=None,
        metavar="LANG",
        help="source language; omit it and Whisper detects one",
    )
    dub.add_argument(
        "--backend",
        default=None,
        metavar="NAME",
        help="TTS backend: kokoro or chatterbox (default: TTS_BACKEND)",
    )
    dub.add_argument(
        "--voice",
        default=None,
        metavar="ID",
        help="a voice id for the target language, e.g. am_adam; omit for the backend's default "
        "(GET /api/backends lists them, an unknown one is refused with the list)",
    )
    dub.add_argument(
        "--no-burn",
        action="store_true",
        help="keep the subtitles as a soft track instead of burning them in (no video re-encode)",
    )
    dub.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="where to write the mp4; a directory keeps the default name (default: ./<title>-<lang>.mp4)",
    )
    dub.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="where the job directory is created (default: DATA_DIR, ./data)",
    )
    dub.set_defaults(handler=_dub, configure_logging=True)

    serve = subcommands.add_parser(
        "serve",
        help="run the web server",
        description="Run the FastAPI app (the same one `uvicorn respeak.main:app` runs).",
    )
    serve.add_argument("--host", default=DEFAULT_HOST, help="address to bind (default: %(default)s)")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help="port (default: %(default)s)")
    serve.add_argument("--reload", action="store_true", help="restart when the source changes")
    serve.set_defaults(handler=_serve, configure_logging=False)

    # No flags of its own and `add_help=False`: everything after the word `prewarm` — `--help`
    # included — comes back from `parse_known_args` and is handed to `python -m respeak.prewarm`,
    # which documents and validates its own options. (`nargs=REMAINDER` cannot do this: argparse
    # refuses to let a REMAINDER positional swallow a leading `--flag`.)
    prewarm = subcommands.add_parser(
        "prewarm",
        help="download the models and packages the first job would otherwise wait for",
        description="Passes every flag through to `python -m respeak.prewarm` "
        "(try `respeak prewarm --help`).",
        add_help=False,
    )
    prewarm.set_defaults(handler=_prewarm, configure_logging=False, args=[])

    return parser


def _log_level(settings: Settings, verbose: bool) -> str:
    """``LOG_LEVEL`` with ``-v``, otherwise WARNING.

    ``LOG_LEVEL`` configures a long-running server, and ``.env.example`` ships it as INFO; here the
    progress lines *are* the output, and yt-dlp, ctranslate2 and kokoro would bury them. So the log
    is off by default and ``-v`` turns on exactly what the server would have shown.
    """
    return settings.log_level.upper() if verbose else "WARNING"


def main(argv: Sequence[str] | None = None) -> int:
    """The ``respeak`` entry point (``[project.scripts]``). Returns the process exit code."""
    parser = build_parser()
    args, extras = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.command == "prewarm":
        args.args = extras  # prewarm owns its own flags; see build_parser
    elif extras:
        parser.error("unrecognized arguments: " + " ".join(extras))
    if getattr(args, "configure_logging", False):
        logging.basicConfig(
            level=_log_level(get_settings(), args.verbose),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        if not args.verbose:
            for name in LOUD_LIBRARIES:
                logging.getLogger(name).setLevel(logging.WARNING)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised by the `respeak` script
    raise SystemExit(main())


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "ProgressStore", "build_parser", "looks_like_url", "main"]
