# Respeak — Decisions

## Purpose

Every choice that shapes the rebuild: who made it, why, and what was rejected.

Two systems are named throughout:

```text
OLD SYSTEM   DubBuddy — branches `dev` and `docker` as found on 2026-09-11 (flow.md Part A)
NEW SYSTEM   Respeak  — branch `main`, built from Phase 0 onward          (flow.md Part B)
```

If a question comes up mid-build, the answer should be here. If it is not, add it here before acting.

How to read:

```text
N-xx   fixed by Nishant                      not up for debate
D-xx   design decision, recommended by Claude, accepted unless a row says otherwise
A-xx   how the analysis itself was done       method, not product
```

Superseded rows stay, marked. IDs never change; the plan and the flow refer to them.

---

# 1. Fixed by Nishant (all on 2026-09-11)

```text
N-01  Analysis and planning first; no fixes until explicitly approved.
N-02  End state is one branch.
N-03  That branch is `main`.
N-04  Must work on the latest Python (3.14); nobody downgrades, not even for deployment.
N-05  No Fish Audio. TTS must be free, open source, deployable.
N-06  Nishant's PC is CPU-only; GPU must be an option for others when it helps.
N-07  Not deployed today; if deployed: Render / Heroku class, minimise cost.
N-08  Public open-source repo; installing and running must be easy for anyone.
N-09  Both Kokoro and Chatterbox; Kokoro default; switch with a simple flag.
N-10  The language list follows the selected model.
N-11  File uploads accepted, with the safety rules in D-26.
N-12  Contact form removed.
N-13  15-minute video cap by default, trivially changeable.
N-14  Docker stays the distribution path; development happens without Docker.
N-15  ffmpeg for muxing, faster-whisper for ASR.
N-16  About page removed. Keep the product as simple as possible.
N-17  UI answers: D-29 yes · D-30 yes · D-31 NO, keep the theme toggle · D-32 yes · D-33 yes ·
      D-34 yes · D-35 yes but after the pipeline works · D-36 yes · D-37 yes, voice picker later,
      then voice cloning for other TTS models later.
N-18  "DubBuddy" is out; a new name is chosen from D-38. `learning-harness.md` stays local (.gitignore).
N-19  The name is **Respeak** (package `respeak`, CLI `respeak`, page title "Respeak").
N-20  `plan.md` is a private working document and is not pushed; `decisions.md` and `flow.md` are committed.
N-21  Go-ahead for Phase 0 and Phase 1: Claude orchestrates, subagents (Opus where it matters) execute tasks.
N-22  Clone-and-run only. No PyPI package. (2026-09-12)
N-23  Skip building the Docker image for now; the Dockerfile and compose file stay in the repo untested. Root disk freed (39 GB). Folder renamed to Respeak on disk. (2026-09-12)
N-24  Exactly two TTS models, Kokoro and Chatterbox, and no other engine of any kind (no Qwen3-ASR, no LLM translator, no whisper.cpp, no third TTS). The CLI is wanted. Phase 3 runs in full. (2026-09-13)
```

---

# 2. Design Decisions

Each record: **Problem** → **Chosen** → **Why** → **Rejected**. Evidence lives in §5.

## 2.1 Repo and runtime

**D-01 Final branch** — accepted (N-03)
Problem: two diverging branches, neither works.
Chosen: `main` created from `dev`; `dev` and `docker` deleted after consolidation; tag `archive/docker-coqui` at `3c1b8dc`.
Why: `dev` has the newer code and the moviepy-2 half-migration; `docker` is unbuildable and its keepers (thread offload, raise-not-None, utf-8, logging, Dockerfile skeleton) are small and get rebuilt anyway.

**D-02 Docker as a file, not a branch** — accepted (N-14)
Chosen: Dockerfile + compose live inside `main`. One codebase, two ways to run.

**D-03 Python version** — superseded by N-04
Chosen: `requires-python >= 3.12`, lock and CI on 3.14, matrix 3.12/3.13/3.14.
Why: the whole replacement stack installs and imports on 3.14.4 (verified, §4). Things that break on 3.13+ (pydub's `audioop`) or churn (numba/triton via openai-whisper) are being removed anyway.
Rejected: pinning 3.11 as the readme does.

**D-04 Dependency manager** — accepted
Chosen: `uv` + `pyproject.toml` + `uv.lock`; CPU torch through `[tool.uv.sources]`; extras `cuda`, `clone`.
Why: deterministic, already installed here, one-line index pinning.
Rejected: `requirements.txt` with no pins (today's state; both branches resolve differently every install).

**D-13 Routes** — accepted
Chosen: `POST /jobs`, `GET /jobs/<id>`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/download`, `GET /api/backends`, `GET /api/health`. Old `/success`, `/process-status`, `/download-video` removed.
Why: no external consumers; the old names encode the one-global-job model.

**D-14 Where outputs live** — accepted
Chosen: `DATA_DIR` (default `./data`, a Docker volume), served only through the download route.
Rejected: under `static/` (public, path leaks, deletion race).

**D-17 Docker disk on this machine** — assumed
Chosen: leave the Docker daemon alone; wrap every build in the free-space watchdog; prune after.
Why: root disk has ~13 GB free and Docker's data-root is on it; Nishant said "docker changes later".
Alternative if builds keep failing: move `data-root` to the 900 GB drive (one-line `daemon.json`).

**D-19 Names** — accepted
Chosen: `YouTubeDownloader`, `inquiries` (gone anyway), `separate_thread` gone with the old runner.

**D-22 Install path for users** — accepted (N-14)
Chosen: `docker compose up` is the headline; `uv sync` + `uv run` right below it for contributors and for day-to-day development.
Why: the image bundles ffmpeg, deno, fonts and model caches; developers never need Docker. No PyPI release (D-43).

## 2.2 Pipeline

**D-05 Media compositing** — accepted (N-15)
Problem: moviepy calls crash on v2; OpenCV subtitles cannot wrap or shape non-Latin text.
Chosen: direct ffmpeg — libass subtitles, `atempo`, AAC.
Why: 60 s 720p job: ffmpeg **20.7 s**, moviepy **229.7 s**; libass rendered Spanish (wrapped), Hindi, Arabic (shaped, RTL) and Chinese at the bottom of the frame.
Rejected: fixing the moviepy-2 calls (keeps the slow path and the OpenCV text problem).

**D-06 ASR engine** — accepted (N-15)
Chosen: faster-whisper, default `small`, `WHISPER_MODEL` selectable, int8 on CPU, float16 on CUDA.
Why: same weights as openai-whisper; on this CPU `small` beam-5 went 33.0 s → 8.7 s; CUDA path, VAD and word timestamps built in.
Rejected for the default: openai-whisper (slower, numba/triton); whisper.cpp (on par on CPU — base-q8 2.3 s, small-q8 6.9 s — but its Metal/Core ML advantage needs a source build; parked for Mac users); Qwen3-ASR (Apache, 30 languages, but 29.5 s warm on this CPU and misses 6 of the 29 UI source languages; GPU-tier later); Parakeet v3 (25 European languages only).
Model-size note: `base` misheard "trunks" as "punks"; `small` got it right; turbo and Qwen3-ASR said "fronts". Bigger is not automatically better on 19 s of 2005 audio.

**D-09 Sync strategy** — accepted
Chosen: per-segment fit — bounded `atempo` per sentence, silence padding, exact times reused for subtitles.
Rejected: global speed change (today's code; chipmunk voice, subtitles drift).

**D-10 Subtitles** — accepted
Chosen: burned-in via libass when the checkbox is on (D-33), plus a `mov_text` soft track always.

**D-11 Limits** — accepted (N-13)
Chosen defaults, all env-tunable: `MAX_VIDEO_SECONDS=900`, `MAX_UPLOAD_MB=500`, `MAX_HEIGHT=720`, `MAX_CONCURRENT_JOBS=1`, `JOB_TTL_MINUTES=60`.

**D-24 Language list vs TTS coverage** — accepted (N-10)
Chosen: `GET /api/backends` returns each backend with its languages (Chatterbox lists its 23 even when not installed, with `installed: false` and a `reason`); the form offers only installed backends, and the target `<select>` is filled from the selected one. Source languages stay at the 29 Whisper + Argos cover.
Rejected: keeping a fixed list of 16 (would promise languages the active backend cannot speak).

**D-36 Translation engine** — accepted
Chosen: Argos behind a `Translator` interface; pivot through `en`.
Why: offline, free, covers every UI pair via English (verified against the package index); Argos is OPUS-MT converted to CTranslate2, so the obvious alternatives are the same models.
Rejected: NLLB-200 (CC-BY-NC); an LLM translator (parked for GPU users, Phase 3).

## 2.3 Text-to-speech

**D-07 Backends** — accepted (N-09)
Chosen: `KokoroBackend` default (Apache-2.0, 82M, ≈1× real time on this CPU, 8 languages, no cloning) + `ChatterboxBackend` as the `clone` extra (MIT, 500M, 23 languages, zero-shot cloning, GPU in practice). One `TTSBackend` interface.
Rejected: Fish (paid, 402); Coqui XTTS (CPML non-commercial, dependency tree dead); Piper ≥ 1.3 (GPL-3.0); Fish Speech / OpenAudio (CC-BY-NC); edge-tts (not an open model, unofficial endpoint); Qwen3-TTS (Apache, 10 languages, good, GPU-only — possible third backend later).

**D-08 Fish account** — closed
Fish removed entirely; `FISH_API_KEY` and `fish-audio-sdk` go. Rotate the key in `.env` if it was ever shared.

**D-27 How the switch works** — accepted (N-09)
```text
TTS_BACKEND in .env            server default (kokoro)
--tts on the CLI               overrides it            (CLI lands in Phase 3)
backend selector on the form   overrides per job, shown only when > 1 backend is installed
```
`chatterbox` is offered only if the extra imports; otherwise `/api/health` says why.

**D-37 Voices** — accepted (N-17)
v1: one curated voice per language per backend. Phase 3: voice picker (gender / accent) from `/api/backends`. Later: voice cloning for additional TTS models when one earns a place.

## 2.4 Inputs and safety

**D-12 Contact form** — closed (N-12)
Removed: route, model, template, js, css, `inquires/`, the Google Apps Script URL. Nav gets a GitHub link.

**D-23 Uploads** — accepted (N-11)
Chosen: YouTube URL **or** file upload.
Why: yt-dlp from datacenter IPs is often blocked by YouTube's bot wall; uploads make the tool work anywhere and make tests hermetic.

**D-26 Upload safety rules** — Claude
```text
size      MAX_UPLOAD_MB enforced while streaming to disk
name      client filename only used, sanitised, for the download name; stored as jobs/<id>/upload.bin
probe     ffprobe must succeed: one video stream, duration ≤ MAX_VIDEO_SECONDS
remux     ffmpeg → source.mp4 (neutralises odd containers)
lifetime  job dirs TTL-deleted
switch    ALLOW_UPLOADS=false hides the tab and disables the route
later     per-IP rate limit for public deployments (Phase 3)
```

## 2.5 User interface

**D-28 When to change the UI** — accepted
Chosen: structure in Phase 1 (the pages are gutted anyway), visual redesign in Phase 3 once the pipeline is proven. Templates get rewritten twice at most: once for structure, once for looks.

**D-29 Layout** — accepted (N-17)
Single page: form, then progress and result in place; `/jobs/<id>` reloads the same page with that job; footer holds a GitHub link.

**D-30 ToS modal** — accepted (N-17)
Removed with its cookie/localStorage logic; replaced by one line under the submit button. Also removes the "commercial use needs approval" text that contradicted the MIT licence.

**D-31 Theme toggle** — **kept, by Nishant** (N-17)
The toggle stays. With the ToS gone it stores the preference in localStorage only; `cookies.js` is removed.

**D-32 Source language** — accepted (N-17)
Auto-detect (Whisper language ID, shown in the job view); optional override in an Advanced fold.

**D-33 Burn subtitles** — accepted (N-17)
Checkbox, default on. Soft track always attached.

**D-34 Frontend tooling** — accepted (N-17)
No framework, no build step: one template, one CSS file, one JS file, system font stack. Bootstrap and the Google Font go.

**D-35 CLI** — accepted, deferred to Phase 3 (N-17)
`dubbuddy dub <url-or-file> --to es [--tts chatterbox] [--out file]` calling the same pipeline function. Built after the web pipeline is proven.

## 2.6 Deployment and hardware

**D-15 Deployment target** — closed (N-07)
See D-21.

**D-16 GPU on this machine** — accepted
Stay CPU for the rebuild; `nvidia-smi` reports a driver/library mismatch here.

**D-21 Where it can actually run** — accepted
```text
needs        ≈ 1.5–2.5 GB RAM (torch + whisper small + Kokoro), minutes of CPU per job
Render free  512 MB, 0.1 CPU, 15-min spin-down         cannot run it
Heroku Basic 512 MB                                     cannot run it
HF Spaces    Docker/Gradio Spaces need PRO (~$9/mo); then 2 vCPU / 16 GB at no hourly cost
Oracle Free  ARM 2 OCPU / 12 GB since 2026-06-15        fits (aarch64 wheels exist)
2 GB VPS     ~$7–15/mo                                   fits
```
Chosen: ship Docker for self-hosting first; document the three deploy recipes; state the 2 GB floor in the README.

**D-25 GPU support** — accepted (N-06)
CUDA is a post-step, not an extra: `uv sync` then `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` (re-run after each `uv sync`); `DEVICE=auto|cpu|cuda`; `docker compose --profile gpu` builds with a `TORCH_INDEX` build-arg. Reason (verified 2026-09-11): uv cannot express "CPU torch by default, CUDA on request" in one lockfile — a conditional `extra = "cuda"` source is shadowed by the unconditional CPU source, and group-conditioned sources do not reach the transitive torch that kokoro requires. GPU speeds Whisper ~5–10× and Chatterbox ~20×; Kokoro barely needs it. CPU stays the default and the only path CI tests.

## 2.7 Quality and process

**D-18 Tests** — accepted
Fixture-based smoke test on a 10 s local clip, offline, Kokoro backend; unit tests for atempo chaining, srt building, pivot selection, upload validation; all in CI.

**D-20 Legal text** — accepted
The ToS is gone (D-30). README gets a short "responsible use" note. MIT is the only licence.

**D-38 Project name** — **Made (Nishant, N-19): Respeak**
Problem: "DubBuddy" is out (N-18). Nishant wants **one word, easy to pronounce, flowing**, not necessarily built on "dub". It must also be free on PyPI and GitHub and not collide with the crowded dubbing product space.
Checked 2026-09-11 against PyPI, GitHub exact-name repos, RDAP for .dev / .com / .io, and web search for products:
```text
NAME       SAY IT        MEANING / ROOT                                   PYPI   GITHUB          .DEV   .COM    .IO
-----------------------------------------------------------------------------------------------------------------------
dobla      DOH-bla       Spanish "dubs" (doblar = to dub, doblaje = dubbing)   free   none exact      free   taken   free
locuto     lo-KOO-to     from "locution", a way of speaking (invented)          free   none exact      free   taken   free
lalia      LAH-lee-a     Greek "speech" (as in echolalia)                       free   1 repo ★6       free   taken   free
loquela    lo-KWEH-la    Latin "speech, manner of speaking"                     free   1 tiny repo     free   taken   free
parlata    par-LAH-ta    Italian "spoken"                                       free   none exact      free   taken   free
lengua     LEN-gwa       Spanish "tongue / language"                            free   none exact      free   taken   free
respeak    ri-SPEEK      "respeaking" is the real name of subtitling by re-speaking into ASR   free   none exact   free   taken   free
sonoro     so-NO-ro      "with sound"; cine sonoro = sound film                 free   1 tiny repo     free   taken   free   ← Sonoro audio GmbH (hi-fi) + Sonoro Media exist
dublate    DUB-late      dub + translate (earlier round)                        free   none            free   free    free
```
Rejected after checking: loquo (voice-journal app), vocalo (radio station + English-learning app), verbo / sonara / talkie / aloud / recite / voxa / oratio / voco / sermo / sonora (PyPI taken), glotto / narro / relato / duetto / mimo (.dev taken or a starred repo), echoa ("Echo" apps everywhere), redub / polydub / dubtitle / revoice (products), loqui (Discord), synchro / parlo / idioma / glossa / voci / vocalize / revox (PyPI taken), subdub (Pandrator author's tool), otravoz / altavoz (two words fused).
Recommendation: **Dobla** — two syllables, smooth, literally "dubs" in Spanish, everything free except the .com. Second: **Locuto** — invented, zero baggage, flows well. Third: **Lalia**.

Finalists after Nishant's shortlist (**Respeak** vs **Sonoro**), domain irrelevant (subdomain of his personal domain):
```text
                     RESPEAK                                          SONORO
signal to a stranger "re-voice it" → says what the tool does         "sound" → needs a tagline to mean dubbing
as a CLI             `respeak video.mp4 --to es` reads like a verb    `sonoro video.mp4 --to es` reads like a brand
pronunciation        2 syllables, plain English                       3 syllables, rolling, pleasant
same-name products   none; poojasethi/respeak ★7 on GitHub            Sonoro audio GmbH (hi-fi + apps), Sonoro Media (podcasts), 24 GitHub repos
adjacent names       Respeecher (voice cloning), ReSpeaker (mics)     Sonos, Sonora, Sonoros
generic-word risk    "respeaking" is an industry term (subtitling)    common Spanish/Italian adjective
trademark            nothing surfaced in web search; databases not    Sonoro Audio GmbH files US marks → real risk in
                     reachable from here, check manually              the audio/software classes
npm / crates / Docker Hub / Homebrew   all free                       all free
GitHub org name      taken (a user)                                   taken (a user)
```
Recommendation: **Respeak**. It signals the purpose without a tagline, works as a command, and has no same-name product. Sonoro is the prettier word but sits on top of a hi-fi brand that files trademarks.

**D-44 Model roster is closed** — Made (Nishant, N-24)
Kokoro (default, CPU) and Chatterbox (cloning, GPU) are the only speech models Respeak will ever ship; faster-whisper is the only ASR; Argos the only translator. Every "possible third backend later" note in D-06 and D-07 is void. The interfaces stay, because they keep the two backends honest, not because more are coming.
Still to do by Nishant (2 minutes each): search "respeak" at https://tmsearch.uspto.gov and https://euipo.europa.eu/eSearch — automated lookups were blocked (HTTP 403).

**D-39 Which documents are public** — Made (Nishant, N-20)
```text
plan.md              private: tasks, estimates, our register        → .gitignore
learning-harness.md  private                                        → .gitignore
decisions.md         public: a decision log is what contributors want → committed (moves to docs/ in Phase 5)
flow.md              public: architecture, old vs new                 → committed (moves to docs/ in Phase 5)
```

**D-40 What happens to the old pipeline code** — Made (Claude)
Deleted from `main` in Phase 0 rather than moved. Every file stays readable with `git show dev:src/<name>.py`, flow.md Part A documents its behaviour, and nothing in the new system imports it. Keeping dead modules on disk would only tempt an import of moviepy or Fish.

**D-43 Distribution is clone-and-run only** — Made (Nishant, N-22)
```text
how people get it     git clone → `docker compose up`  or  `uv sync` + `uv run`
not done              no PyPI package, no `pip install respeak`, no pipx
why                   a hobby project with no maintenance promise; publishing adds a name to hold, a release
                      process and an expectation of updates, and changes nothing about security exposure
pyproject.toml stays  it is what uv reads to lock and install dependencies (the modern requirements.txt)
respeak/ stays        a folder Python imports; it has nothing to do with publishing
```

**D-42 Review before closing a phase** — Made (Claude)
After the Phase 1 work packages landed, a read-only reviewer agent examined the diff against flow.md Part B and the old system's defect list, confirming each finding by running it. It found ten issues the 127 tests did not: heavy imports reaching the event loop through `health`/`backends`/`resolved_device()`, the sweeper dying permanently on one bad status.json and able to delete a job running longer than 6 h, `fit()` stretching short clips by 25 %, dropped dub tails reported nowhere, `_pipeline_run_fn` swallowing every ImportError, VP9/Opus copied into `out.mp4`, the upload cap enforced only after Starlette spooled the body, yt-dlp's generic extractor reachable for internal URLs (SSRF), and blocking file I/O in an async route. All were fixed before the phase closed (FIX-1). Rule going forward: every phase ends with an independent review pass, and the reviewer must reproduce, not guess.

**D-41 How the build is run** — Made (Claude, per N-21)
```text
orchestrator   Claude (this session): defines contracts (flow.md Part B), assigns tasks, integrates, verifies
subagents      one per work package, Opus for pipeline / backend / frontend work, strict file ownership
ownership      an agent edits only the modules listed in its brief; shared files (pyproject, config) are owned by the orchestrator
verification   every "Done when" from plan.md is run on this machine by the orchestrator before a phase closes
git            commits land on local `main` at each phase boundary; nothing is pushed until Nishant says so
repo rename    `gh repo rename respeak` and the folder rename are Nishant's calls (Phase 5)
```
Not checked: trademarks. Do a quick USPTO/EUIPO search before any commercial use.

---

# 3. How the Analysis Was Done (A-xx)

```text
A-01  No source modified; only plan.md, flow.md, decisions.md added; gitignored venvs created for running.
A-02  Dev branch run in an isolated uv venv on Python 3.11 (readme's version), then the replacement stack proven on 3.14 in venv/py314.
A-03  CPU-only torch; ffmpeg/ffprobe from the static-ffmpeg pip package (no host ffmpeg, no passwordless sudo).
A-04  Docker branch tested through a temporary git worktree and a real `docker build` under a 4 GB free-space watchdog; failed build cache (~6 GB) pruned after.
A-05  In the stage harness the paid Fish stage was stubbed after the real 402, and two monkey-patches (`sp`, `set_start`) were applied only inside the harness to reveal the next bug each time.
A-06  Sample: "Me at the zoo" (19 s, English) → Spanish.
A-07  Test artifacts removed from the repo and from the root disk; whisper base/small and Kokoro caches kept for Phase 1.
```

---

# 4. Facts That Constrain Every Decision (verified 2026-09-11)

```text
machine      Intel i7-8550U, 4 cores / 8 threads, AVX2, no VNNI; 22 GB RAM; root disk ~13 GB free;
             project drive ~900 GB free; Docker data-root on the root disk; no usable GPU (driver mismatch)
python       system 3.14.4; whole replacement stack installs and imports on it (torch 2.14 CPU, 2.11+cu128 CUDA,
             faster-whisper 1.2.1, ctranslate2 4.8.2, argostranslate 1.11, stanza, sentencepiece 0.2.2, yt-dlp, fastapi 0.141,
             pydantic-settings, kokoro 0.9.4); chatterbox-tts 0.1.7 resolves (untested install)
youtube      yt-dlp works from this network; needs deno for combined formats; title fetch with format "best" fails without it;
             datacenter IPs frequently hit the bot wall
argos        X→en exists for all 29 UI source languages, en→X for all 16 former targets; no direct non-English pairs
coqui        `TTS` cannot be installed on Python 3.11 from PyPI today (pip and uv both fail); moviepy on PyPI is 2.x
opencv       v5 Hershey text: Latin/Cyrillic/CJK/Korean ok, Hindi → ??????, Arabic unshaped; v4 (apt): all non-ASCII → ?
hosting      Render free 512 MB / 0.1 CPU / 15-min spin-down; HF Docker Spaces need PRO; Oracle Always Free = 2 OCPU / 12 GB
kokoro       misaki[en] pip-installs en_core_web_sm at first import → pin the wheel in the lockfile
```

---

# 5. Research Record

## 5.1 TTS candidates

A table because the decision genuinely turns on four columns at once.

```text
MODEL                         LICENCE            LANGS (of 16)         CLONE   CPU?                 PY 3.14   VERDICT
------------------------------------------------------------------------------------------------------------------------------
Kokoro-82M                    Apache-2.0         8: en es fr hi it ja pt zh   no   ≈1× RT measured   tested    default
Chatterbox Multilingual V3    MIT                14: all but cs hu     yes     500M → GPU           resolves  `clone` extra
Chatterbox Nano / Turbo       MIT                English only          yes     Nano 3× RT / 8 cores resolves  maybe later
Qwen3-TTS 0.6B / 1.7B         Apache-2.0         10                    yes     GPU ≥ 4 GB           untested  possible 3rd backend
Piper ≥ 1.3                   GPL-3.0            ~30, no ja/ko         no      very fast            ≤ 3.13    rejected (licence)
Coqui XTTS-v2                 CPML non-commercial 17                   yes     slow                 resolves  rejected (licence, deps)
Fish Speech / OpenAudio       CC-BY-NC-SA        many                  yes     GPU                  —         rejected
edge-tts                      not a model        all                   no      n/a                  yes       rejected (unofficial)
MeloTTS / OpenVoice v2        MIT                6                     tone    yes                  git only  fallback idea only
```

## 5.2 ASR measurements (19 s clip, 8 threads, this CPU)

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
Qwen3-ASR-0.6B (transformers, CPU)  29.5 s warm          "fronts"; 92 s load
```

## 5.3 Media measurements

```text
60 s 1280×720, replace audio, atempo 1.25, burn 4 Unicode cues, h264 veryfast
  ffmpeg + libass      20.7 s wall   (13.4 s without subtitles)
  moviepy 2            229.7 s wall  single-threaded
```

---

# 6. Parking Lot

Ideas explicitly deferred. Each has a home in the plan's Phase 3 or IGNORE tier.

```text
Docker image build + multi-arch       when Nishant says build (N-23)
GPU profile verification              needs someone with an NVIDIA card
HF Spaces / Oracle / VPS recipes      README deploy section covers the essentials (D-21)
Docker data-root / uv cache move      only if disk becomes a problem again (D-17)
```
Closed by N-24: any further TTS, ASR or translation engine.

---

# 7. Open Items

None. Phase 0 is cleared to start.

---

# 8. Register

```text
DATE        WHAT CHANGED
------------------------------------------------------------------
2026-09-11  analysis of both branches; D-01…D-20 proposed
2026-09-11  N-03…N-08 fixed; D-21…D-25 added; Python 3.14 verified; TTS research
2026-09-11  N-09…N-14 fixed; D-26, D-27 added; contact form dropped; 15-min cap
2026-09-11  N-15 fixed; ASR, mux and whisper.cpp measured; D-05/D-06 closed
2026-09-11  N-16, N-17 fixed; D-28…D-37 settled; documents rewritten in harness style
2026-09-11  N-18: rename; D-38 shortlist checked against PyPI / GitHub / domains; learning-harness.md gitignored
2026-09-11  N-19 Respeak chosen; N-20 plan.md private; N-21 go-ahead; D-39…D-41; flow.md split into Part A (old) / Part B (new contract)
2026-09-11  Phase 0 closed: package `respeak/`, uv.lock on 3.14 with CPU torch, D-25 adjusted (CUDA post-step)
2026-09-12  Phase 1: WP-A…F landed (127 tests), live acceptance passed (YouTube en→es 28 s, upload en→fr 18 s, queue verified), review D-42, fixes F1–F9
2026-09-12  N-22 / D-43: clone-and-run only, no PyPI
2026-09-13  N-24 / D-44: model roster closed (Kokoro + Chatterbox only), CLI in; Phase 3 started in full
```
