# Respeak — Flow

## Purpose

Use this document to answer one question:

> **When a user clicks Submit, what literally happens, in what order, and where does it break?**

It has two halves, kept strictly apart:

```text
PART A — OLD SYSTEM   DubBuddy as found on 2026-09-11 (branches `dev` and `docker`)
PART B — NEW SYSTEM   Respeak as it is being built on `main`
```

Part A is history and evidence. Part B is the contract the code implements. When Part B and the code disagree, the code is wrong or this file is out of date — fix one of them.

Read it top to bottom once. After that, use it as a map: find the piece, read only that piece.

---
---

# PART A — OLD SYSTEM (DubBuddy, as found)

---

# A1. The Old System Map

Five pieces. Most of the damage lived in #3 and #4.

```text
1. browser          templates/*.html + static/scripts/*.js
        ↓  HTML form POST
2. FastAPI app      main.py  (routes, one global "job")
        ↓  BackgroundTasks
3. pipeline         src/main.py::final_method  (6 stages, strictly sequential)
        ↓  writes files
4. job directory    ./static/process_videos/<uuid>/...
        ↓  served by StaticFiles
5. browser again    polls /process-status, then plays /static/.../<title>.mp4
```

Notice the loop: the browser read its result straight out of the public static mount.

That one fact explained three later problems: outputs were public, the status JSON carried a filesystem path, and cleanup could delete a file the browser was still playing.

---

# A2. Old Entry Point — What Actually Ran

Readme command: `fastapi dev main.py`

```text
fastapi-cli sees __init__.py next to main.py
        ↓
treats the repo folder as a package named "Dubbuddy"
        ↓
runs uvicorn Dubbuddy.main:app --reload  (parent dir on sys.path)
        ↓
main.py imports src/main.py
        ↓
src/main.py imports torch, whisper, argostranslate, moviepy, cv2 at import time
        ↓
~32 s later the first request could be answered
```

It relied on the folder name being a valid identifier, on starting from the repo root, and on nobody running `uvicorn main:app` (relative-import crash). The docker branch did the same trick with a folder named `app`.

---

# A3. One Old Request, End to End (measured)

```text
POST /success  (video_url, from_lang, to_lang, tos_check)
        ↓
Video.as_form validates → HttpUrl, two language names, bool
        ↓
globals: processing_status = "processing"
        ↓
background_task.add_task(final_method, ...)      ← final_method is `async def`
        ↓
TemplateResponse(success.html) is queued, not yet flushed
        ↓
Starlette awaits final_method on the SAME event loop
        ↓
every stage inside is blocking CPU / network work
        ↓
the loop cannot flush the response, serve /static, or answer /process-status
        ↓
first `await asyncio.sleep(180)` → loop free → response finally leaves
```

Measured: the POST response took **14.8 s on a 19 s clip**. For a 10-minute video that is minutes of a frozen site.

Mechanism, not rule:

> An `async def` handed to `BackgroundTasks` is awaited in-loop. Blocking code inside it freezes the whole server.

The docker branch noticed and pushed the work to a thread (`asyncio.to_thread` → `asyncio.run(...)`). Right instinct, clumsy shape, and it kept the globals (A5).

Old routes, one line each:

```text
GET  /                index.html; languages.json filled the two selects
GET  /about           about.html
GET  /contact-us      contact-us.html
POST /contact-us      saved uploads, POSTed to a Google Apps Script URL
POST /success         started the pipeline, rendered success.html
GET  /process-status  {"status","final_video"} when done, otherwise `null`
GET  /download-video  FileResponse of the global path; HTTP 500 when none
```

---

# A4. The Old Pipeline — Stage by Stage

Base dir: `./static/process_videos/<uuid4>/`

Every stage was a class with one method. A stage that failed returned `None`; every later stage was skipped; the 180 s sleep and the cleanup still ran.

### A4.1 Download — `src/youtube_download.py`

```text
video_download()   yt-dlp format="bestvideo/best"     → original_video/<title>.mp4   (video-only AV1, no height cap)
audio_download()   yt-dlp format="bestaudio/best"     → original_audio/<title>.webm  (opus)
title_extract()    yt-dlp format="best", no download  → FAILED without a JS runtime → "No title Found."
```

Three YouTube round-trips for what one `extract_info` already returns. Every later file was named `No title Found.*`. On an HD source the selector picked **2160p, 712 MB**.

### A4.2 Transcribe — `src/audio_process.py`

Whisper `base`. Language auto-detected; `from_lang` never passed. Output: text only, no timestamps. Worked on the sample (9 s for 19 s of audio).

### A4.3 Translate — `src/translation_process.py`

Argos, direct `<from>→<to>` package only.

```text
en → es      direct package exists     → worked
ja → es      no direct package         → raised → swallowed → None
```

Argos ships X↔en packages only. Every UI pair is reachable through English, but the code never pivoted.

### A4.4 Text-to-speech — `src/text_to_speech.py` (dev branch: Fish Audio)

```text
tos_check false → None
pydub: original audio → original_audio/<title>.wav
whole wav read into memory as the cloning reference   (no trimming)
output file opened for writing                         ← BEFORE the API call
Fish session.tts(...) → 402 Payment Required           (account credit was 0)
        ↓
0-byte translated_audio/<title>finalaudio.wav left behind, returned None
```

On success it would have written MP3 bytes into a `.wav` name.

### A4.5 Subtitles — `src/subtitles.py`

whisper_timestamped re-transcribed the TTS audio → `.srt` with timestamps truncated to whole seconds.

### A4.6 Merge — `src/video_process.py`

```text
speed_factor = tts.duration / video.duration
if != 1:
    sp(audio, factor)                               ← `sp` undefined → NameError
    rewrite srt times / speed_factor
for each subtitle:
    cv2.putText on a 100 px black strip             ← top of frame, no wrapping, Hindi → ??????
    ImageClip(...).with_duration().set_start()      ← set_start does not exist in moviepy 2 → AttributeError
    CompositeVideoClip + write_videofile            ← INSIDE the loop → returned after subtitle #1
```

Forced through with two patches: h264 + **mp3** audio in mp4, subtitle band at the top, text cut off at the right edge.

### A4.7 After the old pipeline

```text
callback(path) → globals: translated_video_download, processing_status = "completed"
        ↓
await asyncio.sleep(180)
        ↓
cleanup(udir) → rmtree     ← deleted the final video 3 minutes after completion
```

On any exception: `print(...)`. Status stayed "processing". Directory stayed on disk.

---

# A5. Where the Old State Lived (and why it broke)

```text
main.py
  translated_video_download = None      module global
  processing_status                     assigned on first POST, never declared
```

```text
one process → one set of globals → one job at a time
        ↓
second user POSTs → overwrites the first user's status
        ↓
server restarts → everything gone
        ↓
two worker processes → each has its own globals
        ↓
browser polls worker B while worker A runs the job → never "completed"
```

The docker branch ran `--workers 2`.

---

# A6. Old Browser Side

```text
index.html
  index.js     fetch static/languages.json → fill the selects
  popup.js     ToS modal; "tosAccepted" cookie or localStorage
  toggle.js    dark theme; cookie if ToS accepted, else localStorage
  <form method=post action=/success>  → full page navigation
success.html
  success.js   setInterval 5 s → GET /process-status; on "completed" show <video> + Download
```

Gotchas: relative `static/...` URLs, invalid `rel="DubBuddy icon"`, a `<head>` injected inside `<body>`, two submit handlers doing the same cookie migration, `null` polling swallowed by `.catch`, no failure state.

---

# A7. Old Job Directory

```text
static/process_videos/<uuid>/
  original_video/<yt title>.mp4            video-only stream
  original_audio/<yt title>.webm           audio-only stream
  original_audio/<app title>.wav           pydub conversion
  original_text/<app title>.txt
  translated_text/<app title>.txt
  translated_audio/<app title>finalaudio.wav
  subtitles/<app title>.srt
  translated_video/<app title>.mp4         final deliverable, publicly served
```

---

# A8. Old Docker Branch — Where It Differed

```text
same stages, same globals, plus:
  pipeline in a worker thread      ✔ right idea
  raise instead of return None     ✔ better; status still never became "failed"
  utf-8 on file writes, logging    ✔ kept in the new system
  Coqui XTTS-v2 + FreeVC           ✘ dependency tree unbuildable; checked for the output file before creating it
  moviepy v1 API                   ✘ PyPI ships moviepy 2 → import failed at startup
  plain `fastapi` dependency       ✘ no CLI, no form parsing, no email validation
  --workers 2                      ✘ see A5
  main.log committed               ✘
  uploads under /static/inquires   ✘ public
```

Verified: `docker build` failed at `pip install -r requirements.txt` (Coqui TTS backtracked to 0.17.5, pandas 1.4 from source).

The old pipeline files were deleted from `main` at the start of the rebuild. They remain readable with `git show 3651025:src/<file>.py` (the last commit before it).

---
---

# PART B — NEW SYSTEM (Respeak, target and contract)

---

# B1. The New System Map

```text
1. browser          respeak/templates/index.html + respeak/static/app.js   (one page)
        ↓  fetch POST /jobs (multipart)  → JSON {id}
2. FastAPI app      respeak/main.py, respeak/api.py, respeak/pages.py         (no job state in memory)
        ↓  JobRunner.submit(job_id)
3. worker pool      ThreadPoolExecutor(MAX_CONCURRENT_JOBS)       (never the event loop)
        ↓  respeak/pipeline/run.py → 8 stages, each writes status.json
4. job directory    DATA_DIR/jobs/<id>/  (status.json + files)    (never under /static)
        ↓  GET /api/jobs/<id>  reads status.json
5. browser again    polls every 3 s; on done plays /api/jobs/<id>/download
```

Contrast with A1: state lives on disk per job, outputs are served by a route, and the loop only ever reads small JSON files.

---

# B2. New Entry Points

```text
development     uv sync  →  uv run uvicorn respeak.main:app --reload           (:8000)
docker          docker compose up                                          (:8000)
cli             uv run respeak dub <url-or-file> --to es
```

`respeak/main.py` builds the app in `create_app()`; the lifespan starts the worker pool and the TTL sweeper and checks that ffmpeg, ffprobe and deno are reachable (`static_ffmpeg.add_paths()` first). Heavy models are loaded lazily by the worker on first use, so the server answers in under a second.

---

# B3. One New Request, End to End

```text
browser  ──POST /jobs (multipart: source_type, url | file, to_lang, from_lang?, backend?, burn_subtitles)──▶ api.py
                                                        │ 413 from Content-Length before the body is parsed when it exceeds MAX_UPLOAD_MB
                                                        │ validate (400 JSON on any problem, before any download)
                                                        │ JobStore.create() → DATA_DIR/jobs/<id>/status.json {state: queued}
                                                        │ JobRunner.submit(id)         (returns immediately)
browser  ◀── 201 {id, url: "/jobs/<id>"} ───────────────┘
browser  pushState /jobs/<id>; GET /api/jobs/<id> every 3 s ──▶ reads status.json, returns it
browser  on state == done: <video src=/api/jobs/<id>/download>, download link
browser  on state == failed: shows error text
```

Nothing in this path can block longer than reading one small file.

---

# B4. The New Pipeline — Stages, Modules, Interfaces

This is the contract. The modules follow this layout exactly.

```text
respeak/
  main.py            create_app(); lifespan: JobRunner, sweeper, binaries check
  config.py          Settings (pydantic-settings)  — every value in .env.example
  jobs.py            JobStore (status.json read/write), JobRunner (thread pool), sweeper()
  api.py             /api/health, /api/backends, POST /jobs, /api/jobs/{id}, /api/jobs/{id}/download
  pages.py           GET /, GET /jobs/{id}  → index.html
  lang_codes.py      NAME_TO_CODE, CODE_TO_NAME, SOURCE_LANGUAGES
  pipeline/
    run.py           run_job(job_id, settings) — orchestrates B4.1…B4.8, updates status, raises → failed
    ffmpeg.py        ensure_binaries(), run(args), probe(path) -> dict, duration(path) -> float
    inputs.py        probe_url(url) -> Probe; fetch_url(url, dest, max_height) -> Path
                     validate_upload(path, limits) -> Probe; remux(path, dest) -> Path
                     extract_audio(source_mp4, dest_dir) -> (source_wav, reference_wav)
    asr.py           transcribe(wav, language|None, settings) -> Transcript
    translate.py     Translator protocol; ArgosTranslator(settings).translate(segments, src, dst) -> list[str]
    tts/
      __init__.py    available_backends(settings) -> dict[name, BackendInfo]; get_backend(name, settings) -> TTSBackend
      base.py        TTSBackend protocol
      kokoro.py      KokoroBackend
      chatterbox.py  ChatterboxBackend (import guarded; reports "not installed")
    audio.py         fit(clip_wav, target_seconds, out) -> (Path, actual_seconds); assemble(placed_clips, total_seconds, out) -> Path
    subtitles.py     build_srt(cues, out) -> Path
    mux.py           mux(source_mp4, dubbed_wav, subs_srt|None, burn, lang, out) -> Path
```

Shared types (`respeak/pipeline/types.py`):

```text
Word        start: float, end: float, text: str
Segment     start: float, end: float, text: str, words: list[Word]
Transcript  language: str, segments: list[Segment]
Probe       duration: float, width: int, height: int, title: str|None, has_video: bool
Cue         start: float, end: float, text: str
Placed      path: Path, start: float, end: float, text: str
BackendInfo name: str, installed: bool, languages: set[str], cloning: bool, reason: str|None
```

`TTSBackend` protocol:

```text
name: str
languages() -> set[str]                       ISO-639-1 codes the backend can speak
synthesize(text, lang, reference_wav: Path|None, out: Path, voice: str|None = None) -> Path   writes 24 kHz mono wav
voices() -> dict[str, list[Voice]]       Voice(id, name) per language; empty for backends that clone
```

### B4.1 probe
Link (any site yt-dlp supports, not only YouTube): the host must resolve to a public address (loopback, private, link-local and reserved ranges are refused both in the API and here, because yt-dlp's generic extractor would otherwise fetch any URL the server can reach), then one `yt_dlp.extract_info(download=False)` → title, duration, id; playlists, live streams and audio-only links are refused. Upload: `ffprobe`. Reject `duration > MAX_VIDEO_SECONDS`, missing video stream, or unsupported language before anything else. Writes `status.title`.

### B4.2 fetch
Link: `bestvideo[height<=MAX_HEIGHT][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=MAX_HEIGHT]+bestaudio/best[height<=MAX_HEIGHT]/best` merged → `source.mp4` (plain `best` last, for sites that report no height); a download with no video stream is refused; `deno` on PATH; `YTDLP_COOKIES_FILE` when set. Upload: `upload.bin` → re-muxed `source.mp4`. The result is checked with ffprobe and re-encoded to h264 + aac if a WebM/VP9/Opus stream was copied through (so `-c:v copy` in the mux can never leak non-MP4 codecs into `out.mp4`). Then `ffmpeg` → `source.wav` (16 kHz mono) and `reference.wav` (loudest ≤ 30 s window, 24 kHz).

### B4.3 transcribe
faster-whisper `WhisperModel(WHISPER_MODEL, device, compute_type)`; `language = from_lang or None`; `beam_size=5`, `word_timestamps=True`, `vad_filter=True`. Returns `Transcript`. Writes `status.detected_language`.

### B4.4 translate
Argos. If `src != en` install `src→en`; if `dst != en` install `en→dst`; obtain `from_lang.get_translation(to_lang)` (Argos composes the pivot). Translate segment by segment; keep 1:1 with segments.

### B4.5 speak
`backend.synthesize(text_i, dst, reference_wav if backend.cloning else None, seg_i.wav, voice=options.voice)` per segment; `detail = "segment i of n"`.

Reference clip: chosen after transcription — the ≤ 30 s window containing the most speech by ASR segments, cut from `source.wav` at 24 kHz; falls back to the first 30 s when there is no speech.

### B4.6 fit
Per segment: slot = `seg.end - seg.start`; ratio = clip_seconds / slot; factor = min(ratio, MAX_SPEECH_SPEEDUP) when ratio > 1, otherwise 1.0 — a clip that already fits is never slowed down; `atempo` (chained when outside 0.5–2.0) → fitted clip.

Then the timeline (never cut speech; see docs/decisions.md):
```text
s = max(1, max_i  Σ_{j≥i} d_j / (V − start_i))     d_j fitted clip lengths, V video length, i over sentences
        ↓
s ≤ MAX_VIDEO_STRETCH  → the video is slowed by s (setpts, re-encoded even with burn off); sentences are
                          placed on the stretched timeline: start_i = max(s·seg.start, prev_end)
s >  MAX_VIDEO_STRETCH  → s = MAX_VIDEO_STRETCH, refit speech with a 1.5× cap, recompute; if it still does
                          not fit, trim the tail as the last resort and name the affected sentences in `warnings`
```
Assemble on silence of `s·V` seconds → `dubbed.wav` (24 kHz mono), exact sample count via numpy + soundfile; every other media operation is an ffmpeg subprocess. A stretch > 1 is reported in `warnings` as information ("the video was slowed by 12 % so all the speech fits").

### B4.7 subtitles
`Cue(start_i, end_i, translated_text_i)` → `subs.srt`, millisecond precision, via the `srt` library. No second transcription pass.

### B4.8 mux
```text
burn on:   ffmpeg -i source.mp4 -i dubbed.wav -i subs.srt
             -filter_complex "[0:v:0]subtitles=subs.srt:force_style='FontName=Noto Sans,Outline=1,MarginV=30'[v]"
             -map [v] -map 1:a -map 2:s -c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 160k
             -c:s mov_text -metadata:s:s:0 language=<iso639-2> -t <min(video, audio)> -movflags +faststart out.mp4
burn off:  same, but -map 0:v:0 -c:v copy   (no re-encode)

Why `-t` and not `-shortest`: measured — with a subtitle input, `-shortest` counts the sparse subtitle
stream and cut a 10 s video to 3.0 s (the end of the last cue). `-shortest` is only used when no
subtitle file is attached. `[0:v:0]` keeps cover-art streams from making the specifier ambiguous.
```
Then `status = done`, `output = out.mp4`, intermediates deleted (`upload.bin`, `source.*`, `reference.wav`, `seg_*.wav`, `dubbed.wav`; `subs.srt` kept).

Every stage: `JobStore.update(id, step=..., progress=...)` on entry; any exception → `JobStore.fail(id, f"{step}: {exc}")`.

---

# B5. Where the New State Lives

```text
DATA_DIR/jobs/<id>/status.json
{
  "id": "…", "state": "queued|running|done|failed",
  "step": "probe|fetch|transcribe|translate|speak|fit|subtitles|mux|finish",
  "progress": 0.0–1.0, "error": null | "step: message",
  "created_at": iso, "updated_at": iso,
  "title": str|null, "detected_language": str|null,
  "source": {"type": "url|upload", "url": str|null, "filename": str|null},
  "options": {"to_lang": "es", "from_lang": null, "backend": "kokoro", "burn_subtitles": true},
  "output": null | "out.mp4", "download_url": null | "/api/jobs/<id>/download",
  "warnings": [],         human sentences, e.g. speech that did not fit before the video ended (shown, not fatal)
  "detail": null | "downloading 3.1 MB of 7.4 MB" | "segment 4 of 12"   short text for the current step
}

`options` also carries `"voice": null | "<voice id>"`.
```

Writes are atomic (write temp, `os.replace`). Reads never lock. Any web process can answer for any job. `MAX_CONCURRENT_JOBS` bounds CPU use; extra jobs wait in the pool queue in `queued`.

Sweeper: every 5 minutes, delete job dirs whose `updated_at` is older than `JOB_TTL_MINUTES` and whose state is `done` or `failed`, plus `running`/`queued` dirs and dirs with no readable status.json older than 6 h (crashed server). Never touch a dir younger than 5 minutes.

Failure naming: the runner reads `step` from status.json when an exception escapes and records `"<step>: <message>"`; a job that dies before any stage ran is recorded as `"start: …"`.

Predict: with `--workers 2`, what changes for the browser? Nothing — both workers read the same file.

---

# B6. New Browser Side

```text
index.html  (served for / and /jobs/<id>; <body data-job-id="…">)
  app.js
    load   → GET /api/backends → fill backend selector (hidden when only one) and target-language <select>
             theme toggle from localStorage
    submit → fetch POST /jobs (FormData) → {id} → history.pushState("/jobs/<id>") → start polling
    poll   → GET /api/jobs/<id> every 3 s → render step list + progress bar + error
             done → <video src="/api/jobs/<id>/download" controls> + download link; stop polling
    reload on /jobs/<id> → start polling immediately
  app.css   system font stack, prefers-color-scheme defaults, toggle overrides
```

Form fields: source tabs (video link / upload, upload tab hidden when `ALLOW_UPLOADS=false`), target language, backend (conditional), voice (conditional: shown when the backend lists voices for the chosen language), "Burn subtitles into the video" (default on), Advanced fold → source language override, one-line rights notice. The job view shows `step`, `progress`, `detail`, `warnings`, `error`.

`GET /api/backends`: each backend also carries `voices: {lang: [{id, name}]}`; Chatterbox's is empty because it clones.

`GET /api/voices/{backend}/{lang}/{voice}`: a short sample sentence in that voice, `audio/wav`, generated by the backend on first request into `DATA_DIR/voice_samples/…` and served from there afterwards; 400 for an id not in the backend's table, 404 for a backend without presets. The picker's play button calls it.

Theme: `<html>` gets the stored theme before first paint; with nothing stored the page follows `prefers-color-scheme` and reacts to OS flips; the toggle writes `localStorage.theme` and wins from then on.

Rate limit: `RATE_LIMIT_JOBS="10/hour"` (off by default) — token bucket per client IP, applied in the middleware before the body is parsed (after the Content-Length check, so an oversize upload costs no token); 429 `{"error": …}` with `Retry-After`. With `TRUST_PROXY=true` the client is the **last** `X-Forwarded-For` entry (the one the trusted proxy appended; the leftmost is client-supplied), parsed as an IP or ignored; the bucket table is capped and evicts the oldest addresses. `respeak serve` passes the matching `forwarded_allow_ips` to uvicorn.

```text
why the last entry:   client sends  X-Forwarded-For: 1.1.1.1
                      nginx appends   X-Forwarded-For: 1.1.1.1, 203.0.113.9   ← 203.0.113.9 is what nginx saw
                      trusting the first entry would give every request a fresh bucket
```

CLI: `respeak dub <url-or-file> --to LANG [--from LANG] [--backend …] [--voice ID] [--no-burn] [--out PATH] [--data-dir DIR]`, `respeak serve`, `respeak prewarm`; same `run_job`, progress on stderr, exit 0/1.

---

# B7. New Job Directory

```text
DATA_DIR/jobs/<id>/
  status.json
  upload.bin           uploads only, deleted after remux
  source.mp4           deleted at finish
  source.wav           deleted at finish
  reference.wav        deleted at finish
  seg_0001.wav …       deleted at finish
  dubbed.wav           deleted at finish
  subs.srt             kept
  out.mp4              kept until the TTL sweeper
```

---

# B8. Where Each Old Failure Lands in the New System

```text
OLD FAILURE (Part A)              NEW MECHANISM (Part B)
------------------------------------------------------------------
event-loop freeze                 worker pool, sync pipeline function      B1, B3
one global job                    status.json per job id                   B5
silent None                       raise → JobStore.fail → visible error    B4, B6
"No title Found."                 single extract_info in probe             B4.1
4K downloads                      MAX_HEIGHT in the format string          B4.2
moviepy crashes                   ffmpeg only                              B4.8
top / cut-off / ?? subtitles      libass + Noto fonts, bottom, wrapped     B4.8
chipmunk audio                    atempo per segment                       B4.6
whole-second subtitle times       cues from placed times, ms precision     B4.7
deleted while watching            intermediates only; TTL sweeper          B5, B7
public outputs                    DATA_DIR + download route                B1, B7
paid TTS                          Kokoro / Chatterbox behind one interface B4.5
Python 3.11 only                  locked on 3.14, floor 3.12               B2
```

---

# B9. Check Your Model

Answer without scrolling up.

1. Why did the old POST response take 14.8 s on a 19 s clip when the template itself rendered instantly?
2. Two users submit ten seconds apart on the old code. What did the second one see? The first?
3. In the new system, what exactly makes `--workers 2` safe?
4. A segment's TTS clip is 2× longer than its slot. Walk through B4.6: what happens to it and to the next segment?
5. Name two things libass does for subtitles that OpenCV's `putText` could not, and why that matters for a translation tool.
