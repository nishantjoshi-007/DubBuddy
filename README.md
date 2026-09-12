# Respeak

Dub any video into another language on your own machine.

Respeak takes a YouTube URL or a video file, transcribes it, translates it, re-voices it with an
open-source text-to-speech model, builds subtitles, and hands back an MP4. No paid API anywhere.

> Status: under reconstruction (formerly DubBuddy). The pipeline is being rebuilt; this README will
> be completed when the first end-to-end release lands.

## Run from source

```bash
uv sync                                   # Python 3.12–3.14, CPU torch, ffmpeg + deno from pip
cp .env.example .env                      # optional overrides
uv run uvicorn respeak.main:app --reload  # http://localhost:8000
uv run pytest                             # offline tests
```

Optional extras:

```bash
uv sync --extra clone                     # Chatterbox voice-cloning backend (GPU recommended)
uv sync --no-group cpu --group cuda       # CUDA build of torch instead of the CPU build
```

## Run with Docker

Coming in the next phase.

## License

MIT. See `LICENSE`.
