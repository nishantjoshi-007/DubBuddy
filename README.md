# Respeak

Dub any video into another language on your own machine.

Give Respeak a YouTube URL or a video file and a target language. It transcribes the speech, translates
it, re-voices it with an open-source text-to-speech model, builds subtitles, and hands back an MP4 with
the new audio and a subtitle track. Everything runs locally; there is no paid API anywhere.

```text
video ──▶ transcribe (faster-whisper) ──▶ translate (Argos) ──▶ speak (Kokoro | Chatterbox)
      ──▶ fit each sentence into its original slot ──▶ subtitles ──▶ mux (ffmpeg) ──▶ out.mp4
```

This is a personal open-source project offered as is. Clone it and run it; there is no package to
install and no promise of updates. See [Status and maintenance](#status-and-maintenance).

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

## Backends and languages

| Backend | License | Voice cloning | Languages | Runs on |
|---|---|---|---|---|
| Kokoro (default) | Apache-2.0 | no, one curated voice per language | en, es, fr, hi, it, ja, pt, zh | CPU, about real time |
| Chatterbox Multilingual | MIT | yes, from the original speaker | 23 languages | GPU recommended |

Select with `TTS_BACKEND=kokoro|chatterbox` in `.env`; when both are installed the page shows a selector.
The target-language list always follows the selected backend. Source language is auto-detected, with
an override under Advanced.

## Configuration

Every setting is an environment variable or a line in `.env`. Defaults in `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `TTS_BACKEND` | `kokoro` | `kokoro` or `chatterbox` |
| `DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `WHISPER_MODEL` | `small` | `base` (fast), `small`, `medium`, `large-v3-turbo`, `large-v3` |
| `MAX_VIDEO_SECONDS` | `900` | reject longer videos before downloading |
| `MAX_UPLOAD_MB` | `500` | upload size cap |
| `MAX_HEIGHT` | `720` | YouTube download resolution cap |
| `MAX_CONCURRENT_JOBS` | `1` | jobs are CPU-bound; raise only with cores to spare |
| `JOB_TTL_MINUTES` | `60` | finished jobs are deleted after this |
| `DATA_DIR` | `./data` | where jobs live (never served as static files) |
| `ALLOW_UPLOADS` | `true` | set `false` to accept YouTube URLs only |
| `YTDLP_COOKIES_FILE` | unset | Netscape cookies file for YouTube's "confirm you're not a bot" wall |
| `YTDLP_AUTO_UPDATE` | `false` (`true` in Docker) | update yt-dlp at startup |
| `PREWARM_LANGUAGES` | `en,es` | languages `python -m respeak.prewarm` downloads models for |

## How long does it take

On a 4-core laptop CPU a one-minute video takes roughly two minutes: transcription runs at about a third
of real time with `small`, Kokoro speaks at about real time, and the 720p encode is fast. A GPU makes
transcription five to ten times faster and makes Chatterbox usable.

## Deploying

Respeak needs about 2 GB of RAM and minutes of CPU per job, so it does not fit free web tiers
(Render free and Heroku Basic are 512 MB). It runs fine on any 2 GB VPS, on Oracle's Always Free ARM
instances, or on a Hugging Face Docker Space (paid plan). Cloud IPs are often blocked by YouTube's
bot wall; uploads always work, and `YTDLP_COOKIES_FILE` helps with URLs.

## How it works

`flow.md` describes the request path, the pipeline stages and the on-disk job state. `decisions.md`
records every design choice, what was rejected and why, and the measurements behind them.

## Status and maintenance

Respeak is a hobby project by one person. It is offered as is, under the MIT license, with no promise of
fixes or releases. The most likely thing to break is YouTube downloading; the Docker image updates yt-dlp
on every start for that reason, and from source `uv lock --upgrade-package yt-dlp && uv sync` or `YTDLP_AUTO_UPDATE=true` does the same.

## Responsible use

Only dub videos you have the right to use. The tool clones nobody's voice unless you install the
Chatterbox backend and choose it; even then, use it for content you are allowed to re-voice.

## License

MIT. See `LICENSE`.
