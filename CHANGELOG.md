# Changelog

## v2.0.0 — 2026-09-14

### A rebuild, not an update

The old app was called DubBuddy and it had two branches that disagreed with each other. `dev` sent
every line of speech to Fish Audio, a paid service; my credit there was zero, so it answered 402 and
left a 0-byte file behind. `docker` had been rewritten around Coqui XTTS, whose dependency tree no
longer resolves at all. Neither branch produced a dubbed video on my machine.

The rest matched. The pipeline was awaited on the event loop, so one POST held the server for 14.8 s
and froze the site for everyone. Job state lived in two module globals: one job at a time, lost on
restart, wrong the moment a second worker existed. Exceptions were printed to the console and never
reached the page, so a failure looked like a spinner that never stopped. The merge step raised
`NameError`. Subtitles were drawn by OpenCV in a band across the top of the frame, unwrapped, with
Hindi as question marks and Arabic unshaped. The finished video was deleted 180 seconds later.

So I did not repair it. I rebuilt it as **Respeak**: the same idea — dub a video into another
language on your own machine — with a new pipeline, a new page and a new set of promises. The tag is
v2.0.0 because this replaces the old app rather than continuing it.

### Pipeline

- **Transcription** by faster-whisper (`small` by default, int8 on CPU and float16 on CUDA), with
  word timestamps and voice-activity filtering. 30 source languages, auto-detected or stated.
- **Translation** by Argos Translate, entirely offline, pivoting through English when no direct
  package exists — so pairs like Japanese to Spanish work instead of failing.
- **Speech** by Kokoro (Apache-2.0, 82M parameters, roughly real time on a laptop CPU, 8 languages,
  54 voices), or by Chatterbox behind the `clone` extra (MIT, 23 languages), which clones the
  speaker out of the source audio.
- **Fit per sentence**: each line is placed in its own slot and sped up with `atempo`, which
  preserves pitch, capped at 1.3×, then padded with silence. The old global speed change that turned
  everyone into a chipmunk and slid the subtitles out of sync is gone.
- **Never cut speech**: when the dub still runs past the end of the picture, the video is slowed
  uniformly by the smallest factor that makes everything fit, up to 1.15×. Only past that does
  anything get trimmed, and then the job says which sentences it touched.
- **Subtitles** built from the translated segments at their fitted times, to the millisecond, burned
  in with libass — wrapped, at the bottom, Arabic shaped and CJK drawn — and always also carried as
  a soft `mov_text` track.
- **ffmpeg only.** It does every media step now; moviepy, OpenCV, pydub and ffmpeg-python are gone.
  A 60 s 720p job that took moviepy 229.7 s takes ffmpeg 20.7 s.

### Web app

- One page: paste a link or upload a file, choose a target language, and the result appears where
  the form was. Reloading a job URL brings it back.
- A link can point at any site yt-dlp supports, not only YouTube: TikTok, Reddit, Dailymotion,
  Bilibili and most of the rest. Audio-only links and playlists are refused before anything is
  downloaded.
- Progress moves per stage with a detail line under it — bytes downloaded, seconds transcribed,
  sentence n of m, encoder position. Failures appear on the page, in words.
- A voice picker with a play button beside it; previews are synthesised on first request and cached.
- Uploads checked for size, duration and a real video stream before anything starts, and switchable
  off entirely.
- The theme follows the operating system, live, until you touch the toggle; then your choice sticks.

### Operations

- Every job is a directory on disk with a `status.json` in it, so any process can read it and a
  restart does not lose it.
- A thread pool runs the pipeline off the event loop; `MAX_CONCURRENT_JOBS` sets how many at once.
- A sweeper deletes job directories older than `JOB_TTL_MINUTES` (60) and never touches a running one.
- An optional per-IP rate limit (`RATE_LIMIT_JOBS`, off by default), with `TRUST_PROXY` deciding
  whether a forwarded address is believed.
- yt-dlp upgrades itself at startup when asked to, because video sites break more often than I
  rebuild an image.
- A `Dockerfile` and a `docker-compose.yml` with a CPU service and a GPU profile, volumes for the
  data and the model caches, and a prewarm step that bakes the weights into the image.
- Caps on everything, all settable in `.env`: 15 minutes of video, 500 MB of upload, 720p.

### Developer

- Runs on Python 3.12, 3.13 and 3.14, locked with uv; CPU torch by default and CUDA as a documented
  follow-up step, because one lockfile cannot hold both.
- 351 tests that pass offline against a checked-in 10-second clip. Anything needing a download is
  marked slow or skips itself with a reason.
- A `respeak` command line with `dub`, `serve` and `prewarm`, driving the same pipeline the page does.
- Continuous integration: the three Python versions, ruff, the test suite, and a Docker build.
- `docs/decisions.md` and `docs/flow.md` record the choices, what was rejected, and the mechanism
  each one protects.

### Known limits

- Chatterbox is a GPU feature in practice: on my CPU one sentence takes 60–90 seconds. That is why
  Kokoro is the default and Chatterbox is an opt-in extra.
- The Docker image has been written and read but not yet run in anger; CI builds it for the first
  time, and the GPU profile stays unverified until someone with a working driver tries it.
- No authentication. Anyone who can reach the port can start a job, so keep it on localhost or put
  something in front of it.
