# Respeak — Decisions

## Purpose

The choices that shape Respeak: what I chose, why, and what I rejected. `flow.md` next to this file
describes how the system is put together; this file says why it is the way it is.

Two systems are named throughout:

```text
OLD SYSTEM   DubBuddy — the app this replaced (flow.md Part A)
NEW SYSTEM   Respeak  — this repository          (flow.md Part B)
```

Each entry reads **Chosen → Why → Rejected**. The measurements the entries lean on are at the end.

---

# 1. Fixed up front

These were settled before any code was written and are not up for debate.

```text
platform       Python 3.12 or newer; locked and tested on 3.14 first. Nobody downgrades, not even to deploy.
speech         free, open-source models only; nothing that bills per request
hardware       CPU by default; a GPU is optional and only makes things faster
distribution   a public repository anyone can clone and run: Docker for users, uv for development; no PyPI package
TTS            exactly two models — Kokoro (default) and Chatterbox (optional, voice cloning) — and no third engine
languages      the target list follows the selected model
inputs         a video link (any site yt-dlp supports) or an uploaded file
cap            15 minutes of video by default, one environment variable to change it
product        one page, one form; no accounts, no contact form, no about page
release        the first release is v2.0.0, because this is a rebuild of the old app, not an increment
```

---

# 2. Repo and runtime

**One branch**
Chosen: `main`.
Why: the old repository had two branches that disagreed with each other and neither produced a video. The rebuild started from the newer one; the other survives as the tag `archive/docker-coqui`.

**Docker as files, not a branch**
Chosen: the `Dockerfile` and `docker-compose.yml` live inside `main`. One codebase, two ways to run it.

**Python version**
Chosen: `requires-python >= 3.12`; the lockfile is resolved on 3.14; CI runs 3.12, 3.13 and 3.14.
Why: the whole stack installs and imports on 3.14. The packages that break on 3.13+ (pydub's `audioop`) or churn (numba and triton through openai-whisper) were removed anyway.
Rejected: pinning 3.11, as the old readme did.

**Dependencies with uv**
Chosen: `uv` + `pyproject.toml` + `uv.lock`; CPU torch through `[tool.uv.sources]`; Chatterbox behind the `clone` extra.
Why: deterministic installs and one-line index pinning.
Rejected: an unpinned `requirements.txt`, which resolved differently on every install.

**Routes**
Chosen: `POST /jobs`, `GET /jobs/<id>`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/download`, `GET /api/backends`, `GET /api/voices/{backend}/{lang}/{voice}`, `GET /api/health`.
Why: the old names (`/success`, `/process-status`, `/download-video`) encoded the one-global-job model, and nothing outside the page called them.

**Where outputs live**
Chosen: `DATA_DIR` (default `./data`, a Docker volume), served only through the download route.
Rejected: writing under `static/`, which is public, leaks paths, and races with deletion.

**Install path**
Chosen: `docker compose up` is the headline; `uv sync` + `uv run` right below it for development.
Why: the image bundles ffmpeg, deno, fonts and the model caches; developers never need Docker.

**No PyPI package**
Chosen: clone and run only. No `pip install respeak`, no pipx.
Why: publishing adds a name to hold, a release process and an expectation of updates, and changes nothing about how the thing runs. `pyproject.toml` stays because uv reads it to lock and install dependencies; `respeak/` is a folder Python imports, not a published package.

**The old code**
Chosen: deleted from `main` at the start of the rebuild rather than moved aside. Every old file is still readable with `git show 3651025:src/<name>.py`, flow.md Part A documents what it did, and nothing new imports it.
Why: dead modules on disk only tempt an import of moviepy or of a paid API.

**Which documents are public**
`docs/flow.md` (architecture, old system versus new) and this file. My working notes are not in the repository; everything a contributor needs is in these two.

---

# 3. Pipeline

**Media compositing**
Chosen: ffmpeg for every media step — libass for subtitles, `atempo` for speed, AAC for audio.
Why: a 60 s 720p job took ffmpeg **20.7 s** and moviepy **229.7 s**. libass renders Spanish (wrapped), Hindi, Arabic (shaped, right to left) and Chinese correctly at the bottom of the frame; the old OpenCV text could do none of that.
Rejected: fixing the half-migrated moviepy calls (keeps the slow path and the text problem); OpenCV.

**Speech recognition**
Chosen: faster-whisper, `small` by default, int8 on CPU and float16 on CUDA, with word timestamps and voice-activity filtering. `WHISPER_MODEL` selects another size.
Why: the same weights as openai-whisper; on a laptop CPU `small` with beam 5 went from 33.0 s to 8.7 s on a 19 s clip. The CUDA path, VAD and word timestamps are built in.
Rejected: openai-whisper (slower, numba and triton); whisper.cpp (on par on CPU — see the measurements — but its Metal and Core ML advantage needs a source build); Qwen3-ASR (29.5 s warm on the same CPU and missing several of the source languages); Parakeet v3 (European languages only).
A note on size: `base` misheard "trunks" as "punks"; `small` got it right; the larger models said "fronts". Bigger is not automatically better on 19 s of 2005 audio.

**Translation**
Chosen: Argos Translate behind a `Translator` interface, pivoting through English when no direct package exists.
Why: offline, free, and every pair the UI offers is reachable through English. Argos is OPUS-MT converted to CTranslate2, so the obvious alternatives are the same models.
Rejected: NLLB-200 (CC-BY-NC); an LLM translator.

**Argos runs at its shipped precision**
What broke: the first CI run translated Spanish to English as "mainmainmain…". Argos lets CTranslate2 pick the compute type, which is int8 on a CPU, and the es→en model decodes to garbage under int8 while en→es and en→hi happen to survive it.
Chosen: `ARGOS_COMPUTE_TYPE` defaults to `default`, the precision the package ships. Translation is a few seconds per job, so int8's speed is not worth silent garbage. The variable still overrides.

**Sync strategy**
Chosen: fit each sentence into its own slot — a bounded `atempo` per sentence, silence padding, and the exact placed times reused for the subtitles.
Rejected: one global speed change (the old code), which gave a chipmunk voice and drifting subtitles.

**When the dub is longer than the video**
What happens: English → Hindi expands 20–30 %. The per-sentence speed-up is capped, so later sentences cascade until the last one runs past the end of the picture. The first version trimmed the audio to the video length and cut the last words.
Chosen: never cut speech. The remedies, in order, each with a cap:

```text
1  use the natural gaps between sentences                               (the cascade already does this)
2  speed the speech up                                                   at most MAX_SPEECH_SPEEDUP = 1.3×
3  slow the VIDEO down uniformly by the smallest factor s that fits       at most MAX_VIDEO_STRETCH = 1.15
   everything; sentences are placed on the stretched timeline so picture and speech stay aligned;
   the video is re-encoded when stretched, even with burn-in off
4  only if still over: raise the speech cap to 1.5×, and as the very last resort trim, with a
   warning that names how much speech was cut

closed form for s:  s = max(1, max_i  Σ_{j≥i} d_j / (V − start_i))  over sentences i,
                    where d_j are the fitted clip lengths and V the video length
```

Why: a 3–10 % slowdown is invisible on talking-head or slide content; a cut sentence is not.
Rejected: cutting speech; raising the speech cap alone (rushed, unintelligible Hindi); freeze-frames only where speech overflows (works, but jerky and much harder to get right).

**Subtitles**
Chosen: burned in with libass when the checkbox is on (default), plus a `mov_text` soft track always. A slowed picture is muxed at a constant frame rate, because ffmpeg 6 and 8 otherwise disagree on the frame count.

**Limits**
Chosen defaults, all environment-tunable: `MAX_VIDEO_SECONDS=900`, `MAX_UPLOAD_MB=500`, `MAX_HEIGHT=720`, `MAX_CONCURRENT_JOBS=1`, `JOB_TTL_MINUTES=60`.

**Language list follows the model**
Chosen: `GET /api/backends` returns each backend with its languages (Chatterbox lists its 23 even when not installed, with `installed: false` and a reason); the form offers only installed backends and fills the target list from the selected one.
Rejected: a fixed list of 16 targets, which would promise languages the active backend cannot speak.

---

# 4. Text-to-speech

**Two backends**
Chosen: `KokoroBackend` by default (Apache-2.0, 82M parameters, about real time on a laptop CPU, 8 languages, 54 voices, no cloning) and `ChatterboxBackend` behind the `clone` extra (MIT, 500M parameters, 23 languages, zero-shot voice cloning, a GPU in practice). One `TTSBackend` interface.
Rejected: Fish Audio (paid); Coqui XTTS (non-commercial licence, dependency tree no longer resolves); Piper 1.3+ (GPL-3.0); Fish Speech / OpenAudio (CC-BY-NC); edge-tts (an unofficial endpoint, not a model); Qwen3-TTS (good, Apache, but GPU-only).

**The roster is closed**
Kokoro and Chatterbox are the only speech models Respeak will ship; faster-whisper is the only ASR; Argos the only translator. The interfaces stay because they keep the two backends honest, not because more are coming. A pull request adding a third engine is the one kind I will turn down.

**How the switch works**
```text
TTS_BACKEND in .env            server default (kokoro)
--backend on the CLI           overrides it
backend selector on the form   overrides per job; shown only when more than one backend is installed
```
`chatterbox` is offered only if the extra imports; otherwise `/api/health` says why.

**Voices**
Chosen: every voice the Kokoro repository bundles is offered, with one curated default per language used when a job picks none. Chatterbox clones the speaker of the source video and so has no list.

**Voice previews**
Problem: a list of voice names tells nobody what a voice sounds like.
Chosen: a play button beside the picker → `GET /api/voices/{backend}/{lang}/{voice}`. The sample is one short sentence in that language, synthesised by the backend on first request and cached under `DATA_DIR/voice_samples/`; later plays are instant. Ids are validated against the voice table.
Rejected: shipping pre-rendered clips in the repository (binaries, a build step); generating all 54 at startup (slow and wasteful).

**The reference clip for cloning**
Chosen: the ≤ 30 s window of the source audio that holds the most speech according to the transcript, cut at 24 kHz; the first 30 s when there is no speech.
Why: the loudest window is often music. Twenty seconds of loud intro followed by quieter speech must pick the speech.

---

# 5. Inputs and safety

**A link can point at any site yt-dlp supports**
Chosen: no restriction on the host. The form value is `url`, the label is "video link", and the hint under the field names YouTube, TikTok, Reddit, Dailymotion and Bilibili and links to yt-dlp's list.
Why: nothing in the pipeline was ever YouTube-specific; the label was. Checked on 2026-09-14 through the probe stage: Dailymotion, TikTok, Reddit, Streamable, Bilibili and archive.org worked; Vimeo asked for a login; Instagram reported no duration. Sites break at yt-dlp's pace and some need cookies, so the promise is "most", not "all", and YouTube is the one I test regularly.
What a link must pass: the host resolves to a public address; the metadata carries a duration (so live streams and premieres are refused); it is a single video, not a playlist or channel; it is not audio-only (every format reports `vcodec == "none"`, or the download holds no video stream). The format selector ends in plain `best`, because a site that reports no height matches none of the capped selectors.
Considered and accepted: a public server fetches URLs on behalf of strangers. The public-address rule, one job at a time, the rate limit, and the duration cap running before any bytes are downloaded bound what that can cost.

**Uploads too**
Chosen: a video link **or** a file upload.
Why: yt-dlp from datacenter addresses is often blocked by YouTube's bot wall; uploads make the tool work anywhere, and make the tests hermetic.

**Upload safety rules**
```text
size      MAX_UPLOAD_MB enforced from Content-Length before the body is read, and again while streaming to disk
name      the client filename is used only, sanitised, for the download name; stored as jobs/<id>/upload.bin
probe     ffprobe must succeed: exactly one video stream, duration ≤ MAX_VIDEO_SECONDS
remux     ffmpeg → source.mp4, re-encoded to h264 + aac when the container carried anything else
lifetime  job directories are deleted after JOB_TTL_MINUTES
switch    ALLOW_UPLOADS=false hides the tab and disables the route
```

**URL safety**
Chosen: the host of a link must resolve to a public address; loopback, private, link-local and reserved ranges are refused in the API and again in the pipeline.
Why: yt-dlp falls back to a generic extractor for anything it does not recognise, so without this a posted `http://169.254.169.254/…` would be fetched by the server — cloud metadata, the LAN, or a service on localhost.

**Rate limit**
Chosen: an optional token bucket per client address (`RATE_LIMIT_JOBS`, e.g. `10/hour`), applied before the body is parsed and after the size check, so an oversize upload costs no token. With `TRUST_PROXY=true` the client is the **last** `X-Forwarded-For` entry — the one the trusted proxy appended — parsed as an IP or ignored; the bucket table is capped.
Lesson: any header a proxy *appends* to must be read from the right. Reading the leftmost entry meant one header line bought unlimited buckets.

**Job state on disk**
Chosen: every job is a directory with a `status.json` in it; a bounded thread pool runs the pipeline off the event loop; a sweeper deletes directories older than the TTL and never touches a running job or one younger than a floor.
Why: the old app kept job state in two module globals — one job at a time, lost on restart, wrong the moment a second worker existed. A file any process can read fixes all three.

---

# 6. User interface

**Layout**
One page: the form, then progress and result in place; `/jobs/<id>` reloads the same page with that job; the footer holds a GitHub link.

**No terms-of-service modal**
Removed with its cookie and localStorage logic; replaced by one line under the submit button. That also removed a "commercial use needs approval" sentence that contradicted the MIT licence.

**Theme**
The toggle stays. With nothing stored the page follows the operating system, live; the toggle overrides and persists in localStorage. Checked in a headless browser across the six combinations of system scheme × stored value.

**Source language**
Auto-detected by Whisper and shown in the job view; an optional override sits in an Advanced fold.

**Burn subtitles**
A checkbox, default on. The soft track is always attached.

**No frontend framework**
One template, one stylesheet, one script, system fonts. No build step, no external resources. Bootstrap and the Google Font are gone.

**Intro text**
One short heading and one sentence. A longer hero read as too much text.

**Command line**
`respeak dub <url-or-file> --to LANG`, `respeak serve`, `respeak prewarm`, driving the same `run_job` the page does, with progress on stderr and the output path on stdout. The checks are literally the ones `POST /jobs` runs, so a flag the form would refuse is refused with the same sentence.

---

# 7. Deployment and hardware

**GPU support**
Chosen: CUDA is a post-step, not an extra: `uv sync`, then `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` (repeated after each `uv sync`); `DEVICE=auto|cpu|cuda`; `docker compose --profile gpu` builds with a `TORCH_INDEX` build argument.
Why: uv cannot express "CPU torch by default, CUDA on request" in one lockfile — a conditional source is shadowed by the unconditional CPU source, and group-conditioned sources do not reach the transitive torch that Kokoro requires. A GPU speeds Whisper 5–10× and Chatterbox about 20×; Kokoro barely needs one. CPU stays the default and the only path CI tests.

**What it needs**
About 2 GB of RAM (torch + Whisper `small` + Kokoro) and minutes of CPU per job. A 512 MB tier cannot run it.

**The Docker image**
Built on the official uv image for Python 3.14; ffmpeg and the Noto fonts (what libass draws non-Latin subtitles with) from apt; a compiler in the builder stage only, for the one dependency without a 3.14 wheel; runs as an unprivileged user; updates yt-dlp at every start because video sites change more often than the image is rebuilt; `PREWARM=1` bakes the models in at build time.

---

# 8. Quality and process

**Tests**
Fixture-based, offline: a checked-in 10 s clip and a generated speech sample. Anything that needs a download is marked `slow` or skips itself with a reason. CI runs ruff, the formatter, the suite on three Python versions, and the Docker build.

**A review pass before anything closes**
Rule: after a piece of work lands I go back over the whole diff against flow.md Part B and the old system's defect list, in a pass that changes nothing and reproduces every finding by running it. Every finding then gets a fix and a test.
What those passes have caught, so the same mistakes are not made twice:

```text
heavy imports reaching the event loop through async routes      → plain `def` routes for anything that loads a model
the sweeper dying on one bad status.json                          → one bad job never stops the pass
short clips stretched 25 % by the fit                             → lo=None: never slow a clip that already fits
dropped dub tails reported nowhere                                → status.warnings, named in seconds
VP9 + Opus copied into out.mp4                                    → re-encode anything mp4 cannot carry
the upload cap enforced only after the body was spooled           → 413 from Content-Length in middleware
the generic extractor reachable for internal URLs                 → the public-address rule, twice
X-Forwarded-For read from the client-controlled end               → the rightmost entry only
a pkg_resources shim leaking into jieba and breaking Chinese       → a context manager, not a sys.modules entry
fetch progress freezing after yt-dlp's first file                 → one writer for the whole download
failed jobs keeping a stale `detail`                              → detail cleared on failure
```

**Legal text**
MIT is the only licence. The README carries a short responsible-use note instead of terms.

---

# 9. Measurements

All on one laptop: Intel i7-8550U, 4 cores / 8 threads, AVX2, 22 GB RAM, no usable GPU.

**TTS candidates**

```text
MODEL                         LICENCE              LANGUAGES             CLONE   CPU?                 PY 3.14    VERDICT
--------------------------------------------------------------------------------------------------------------------------
Kokoro-82M                    Apache-2.0           8                     no      ≈ 1× real time       tested     default
Chatterbox Multilingual       MIT                  23                    yes     500M → GPU           resolves   `clone` extra
Qwen3-TTS 0.6B / 1.7B         Apache-2.0           10                    yes     GPU ≥ 4 GB           untested   rejected (roster closed)
Piper ≥ 1.3                   GPL-3.0              ~30, no ja/ko         no      very fast            ≤ 3.13     rejected (licence)
Coqui XTTS-v2                 CPML non-commercial  17                    yes     slow                 resolves   rejected (licence, deps)
Fish Speech / OpenAudio       CC-BY-NC-SA          many                  yes     GPU                  —          rejected
edge-tts                      not a model          all                   no      n/a                  yes        rejected (unofficial)
```

**Speech recognition** (19 s clip, 8 threads)

```text
ENGINE / MODEL                      GREEDY     BEAM 5    NOTE
--------------------------------------------------------------------------
openai-whisper base                 4.0 s      7.0 s     "punks"
openai-whisper small                14.0 s     33.0 s    "trunks" ✔
faster-whisper base int8            2.4 s      3.0 s
faster-whisper small int8           5.8 s      8.7 s     ✔ default
faster-whisper large-v3-turbo int8  17.9 s     —         "fronts"
whisper.cpp base-q8_0               —          2.3 s     78 MB file
whisper.cpp small-q8_0              —          6.9 s     253 MB file
Qwen3-ASR-0.6B (transformers, CPU)  29.5 s warm          "fronts"; 92 s to load
```

**Media**

```text
60 s 1280×720, replace audio, atempo 1.25, burn 4 Unicode cues, h264 veryfast
  ffmpeg + libass      20.7 s wall   (13.4 s without subtitles)
  moviepy 2            229.7 s wall  single-threaded
```

**Whole jobs**

```text
19 s YouTube clip, English → Spanish, Kokoro, burn-in       28 s
18 s upload, English → French, Kokoro, burn-in              18 s
one minute of video on 4 cores                              about two minutes
Chatterbox on this CPU                                      60–90 s per sentence — a GPU backend in practice
```
