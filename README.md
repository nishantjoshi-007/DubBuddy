# Respeak

[![CI](https://github.com/nishantjoshi-007/DubBuddy/actions/workflows/ci.yml/badge.svg)](https://github.com/nishantjoshi-007/DubBuddy/actions/workflows/ci.yml)

Dub any video into another language on your own machine.

Give Respeak a video link or a video file and a target language. It transcribes the speech, translates
it, re-voices it with an open-source text-to-speech model, builds subtitles, and hands back an MP4 with
the new audio and a subtitle track. Everything runs locally; there is no paid API anywhere.

Links are fetched by yt-dlp, so YouTube, TikTok, Reddit, Dailymotion, Bilibili and
[most other sites it supports](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) work.
YouTube is the one I test regularly; a site that asks you to sign in needs `YTDLP_COOKIES_FILE`.

```text
video ──▶ transcribe (faster-whisper) ──▶ translate (Argos) ──▶ speak (Kokoro | Chatterbox)
      ──▶ fit each sentence into its original slot ──▶ subtitles ──▶ mux (ffmpeg) ──▶ out.mp4
```

This is a personal open-source project of mine, offered as is under the MIT license. Clone it and
run it; there is no package to install. Contributions are welcome; see [Contributing](#contributing).

## Run with Docker (recommended)

```bash
git clone https://github.com/nishantjoshi-007/DubBuddy.git respeak
cd respeak
mkdir -p data          # so the bind mount is owned by you, not root
docker compose up
```

Open http://localhost:8000. The first job downloads the speech and voice models (about 1 GB) into the
`respeak-models` volume; later jobs start immediately. Finished videos live in `./data/jobs/` for an
hour, then get swept.

Optional:

```bash
docker compose build --build-arg PREWARM=1     # download models at build time instead of first run
docker compose --profile gpu up respeak-gpu    # NVIDIA GPU (needs the NVIDIA container toolkit)
docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)   # if your user is not uid 1000
```

The container runs as an unprivileged user, updates yt-dlp at every start, and keeps downloaded models
in the `respeak-models` volume. `docker compose down -v` removes them.

## Run from source

Requires Python 3.12 or newer (tested and locked on 3.14) and [uv](https://docs.astral.sh/uv/).
ffmpeg and a JavaScript runtime for yt-dlp are installed as pip packages; nothing else is needed.

```bash
uv sync                                     # creates .venv with CPU torch
cp .env.example .env                        # optional: change limits, models, backend
uv run uvicorn respeak.main:app --reload    # http://localhost:8000
uv run pytest                               # offline test suite
```

Optional:

```bash
uv sync --extra clone                                                       # Chatterbox voice-cloning backend
uv pip install torch --index-url https://download.pytorch.org/whl/cu128     # CUDA build of torch (after each uv sync)
uv run python -m unidic download                                            # Japanese dictionary (needed for the ja target)
uv run python -m respeak.prewarm --languages en,es,fr                       # download models ahead of time
```

## Command line

The same pipeline without the browser, for scripts and batch work:

```bash
uv run respeak dub video.mp4 --to es                        # a local file
uv run respeak dub https://youtu.be/... --to fr --voice ff_siwis --no-burn --out out.mp4
uv run respeak serve --port 8000                            # the web app
uv run respeak prewarm --languages en,es,ja                 # download models ahead of time
```

Progress prints to stderr, the output path to stdout, and the exit code is 0 on success.

## Backends and languages

| Backend | License | Voice cloning | Languages | Runs on |
|---|---|---|---|---|
| Kokoro (default) | Apache-2.0 | no, one curated voice per language | en, es, fr, hi, it, ja, pt, zh | CPU, about real time |
| Chatterbox Multilingual | MIT | yes, from the original speaker | ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh (23) | GPU recommended; about 80 s per sentence on CPU |

Select with `TTS_BACKEND=kokoro|chatterbox` in `.env`; when both are installed the page shows a selector.
The target-language list always follows the selected backend, and Kokoro offers a voice picker per
language (54 voices). Source language is auto-detected, with an override under Advanced.

Chatterbox has no voice list — it clones the speaker in the video — and it downloads a 3 GB checkpoint
the first time it runs. On CPU it is a demo, not a workflow: measured on a 4-core laptop, loading the
model takes 25–50 s and each sentence takes 70–90 s, about 25x slower than real time, so a one-minute
video is hours. Use Kokoro unless you have a GPU.

## Configuration

Every setting is an environment variable or a line in `.env`. Defaults in `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `TTS_BACKEND` | `kokoro` | `kokoro` or `chatterbox` |
| `DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `WHISPER_MODEL` | `small` | `base` (fast), `small`, `medium`, `large-v3-turbo`, `large-v3` |
| `MAX_VIDEO_SECONDS` | `900` | reject longer videos before downloading |
| `MAX_UPLOAD_MB` | `500` | upload size cap |
| `MAX_HEIGHT` | `720` | download resolution cap |
| `MAX_CONCURRENT_JOBS` | `1` | jobs are CPU-bound; raise only with cores to spare |
| `JOB_TTL_MINUTES` | `60` | finished jobs are deleted after this |
| `MAX_SPEECH_SPEEDUP` | `1.3` | how much faster a sentence may be spoken to fit its slot |
| `MAX_VIDEO_STRETCH` | `1.15` | how much the video may be slowed so all speech fits; speech is never cut before this is exhausted |
| `DATA_DIR` | `./data` | where jobs live (never served as static files) |
| `ALLOW_UPLOADS` | `true` | set `false` to accept links only |
| `YTDLP_COOKIES_FILE` | unset | Netscape cookies file for sites that ask you to sign in or to prove you're not a bot |
| `YTDLP_AUTO_UPDATE` | `false` (`true` in Docker) | update yt-dlp at startup |
| `RATE_LIMIT_JOBS` | unset | per-address submission limit for public servers, e.g. `10/hour` |
| `TRUST_PROXY` | `false` | honour `X-Forwarded-For` for the rate limit, only behind a reverse proxy (`respeak serve` passes the matching `--forwarded-allow-ips` to uvicorn; with plain uvicorn set it yourself) |
| `PREWARM_LANGUAGES` | `en,es` | languages `python -m respeak.prewarm` downloads models for |

## How long does it take

On a 4-core laptop CPU a one-minute video takes roughly two minutes: transcription runs at about a third
of real time with `small`, Kokoro speaks at about real time, and the 720p encode is fast. A GPU makes
transcription five to ten times faster and makes Chatterbox usable.

## How it works

`docs/flow.md` describes the request path, the pipeline stages and the on-disk job state.
`docs/decisions.md` records the design choices, what was rejected and why, and the measurements
behind them.

## Contributing

Contributions are appreciated. Open an issue for a bug or an idea, or send a pull request against
`main`. Before you open one, run `uv run ruff check`, `uv run ruff format --check` and
`uv run pytest -m "not slow"`; that is exactly what CI runs on Python 3.12, 3.13 and 3.14. Keep a
change small and add a test for what it changes. I keep Respeak to two TTS backends on purpose, so a
new backend is the one kind of change I will turn down; `docs/decisions.md` records why.

## Responsible use

I built Respeak for videos you have the right to use, and I ask you to keep it to those. It clones
nobody's voice unless you install the Chatterbox backend and choose it; even then, use it only for
content you are allowed to re-voice.

## License

MIT. See `LICENSE`.
