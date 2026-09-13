# Respeak — plan.md 2.1, decisions D-22 (docker compose up is the headline), D-25 (CUDA is a post-step).
#
# Two stages on the same base: the builder resolves and installs the locked environment, the runtime
# keeps only /app (the venv and the source) plus ffmpeg and the fonts libass needs. uv is kept in the
# final image on purpose — YTDLP_AUTO_UPDATE uses it to upgrade yt-dlp at container start (plan.md 2.0).
#
# Build:   docker build -t respeak:local .
# CUDA:    docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t respeak:gpu .
# Prewarm: docker build --build-arg PREWARM=1 -t respeak:local .

# The image astral publishes with CPython 3.14 and uv already on PATH (/usr/local/bin/uv).
ARG BASE_IMAGE=ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# --------------------------------------------------------------------------------------- builder
FROM ${BASE_IMAGE} AS builder

# Where the CUDA post-step gets torch from; the default is the same CPU index the lockfile uses, and
# then the post-step is skipped entirely (D-25: uv cannot express both in one lockfile).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: they change only when the lockfile does, so editing
# respeak/ never re-resolves torch. --no-install-project because the source is not here yet.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now the project itself. README.md is copied because pyproject's `readme = "README.md"` makes it a
# build input: hatchling fails without it.
COPY README.md ./
COPY respeak ./respeak
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# D-25: CUDA is a post-step, never an extra. Only runs when the caller asked for a different index.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH_INDEX" != "https://download.pytorch.org/whl/cpu" ]; then \
        echo "installing torch from $TORCH_INDEX" && \
        uv pip install --python /app/.venv/bin/python --index-url "$TORCH_INDEX" torch; \
    fi

# The Japanese G2P dictionary (~250 MB) unpacks inside the installed `unidic` package, i.e. into the
# venv — so it travels with the venv to the runtime stage and is not a model-cache concern.
# Without it the `ja` target fails (respeak/pipeline/tts/kokoro.py names this command in the error).
RUN /app/.venv/bin/python -m unidic download

# --------------------------------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Respeak" \
      org.opencontainers.image.description="Dub any video into another language on your own machine." \
      org.opencontainers.image.source="https://github.com/nishantjoshi-007/DubBuddy" \
      org.opencontainers.image.licenses="MIT"

ARG PREWARM=0

# The uid the server runs as. 1000 is the first user on almost every Linux host, and it has to match
# the owner of the bind-mounted ./data — a container user that does not own that directory cannot
# write a job into it. Someone whose host uid differs rebuilds with `--build-arg UID=$(id -u)`.
ARG UID=1000
ARG GID=1000

# ffmpeg does every media step (D-05). The Noto fonts are what libass draws burned-in subtitles with:
# fonts-noto-core covers Latin/Cyrillic/Greek/Devanagari, fonts-noto-cjk covers Japanese and Chinese —
# without them those targets burn in as empty boxes. curl is only here for the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-core \
        fonts-noto-cjk \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root: the server only ever writes to /app/data, /app/models and its own venv.
RUN groupadd --gid "$GID" respeak \
    && useradd --uid "$UID" --gid "$GID" --create-home --home-dir /home/respeak \
       --shell /usr/sbin/nologin respeak

# `python` and `uvicorn` mean the venv's, without activating anything.
ENV HOME=/home/respeak \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# Jobs live here; the compose file bind-mounts ./data over it.
ENV DATA_DIR=/app/data

# Both caches point into the /app/models volume so the ~1 GB of weights survives a container
# recreation: HF_HOME holds the Whisper and Kokoro downloads, and argostranslate installs its
# packages under $XDG_DATA_HOME/argos-translate.
ENV HF_HOME=/app/models/hf \
    XDG_DATA_HOME=/app/models/xdg

# plan.md 2.0: YouTube breaks more often than this image is rebuilt, so ask uv for the newest yt-dlp
# at every start. main.py does it in a daemon thread; it never delays the first response.
ENV YTDLP_AUTO_UPDATE=true

# uv's cache for that upgrade: inside $HOME, which the respeak user owns — never inside the volume.
ENV UV_CACHE_DIR=/home/respeak/.cache/uv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# The venv, the source, pyproject.toml and uv.lock, all already built and owned by the runtime user
# (the venv has to stay writable: YTDLP_AUTO_UPDATE upgrades yt-dlp inside it).
COPY --from=builder --chown=respeak:respeak /app /app

RUN mkdir -p /app/data /app/models/hf /app/models/xdg \
    && chown -R respeak:respeak /app/data /app/models /home/respeak

USER respeak

# Optional: download the models at build time instead of on the first job (plan.md 2.3). Runs as the
# non-root user so the caches land with the right ownership, and before VOLUME so they are baked into
# the image — docker seeds an empty named volume from the image content at that path on first start.
# PREWARM_LANGUAGES (default en,es) chooses which languages are warmed.
RUN if [ "$PREWARM" = "1" ]; then /app/.venv/bin/python -m respeak.prewarm; fi

# Declared after every RUN that writes to them: writes to a VOLUME path in a later layer are discarded.
VOLUME ["/app/data", "/app/models"]

EXPOSE 8000

# /api/health is a plain dict read (respeak/api.py); it never loads a model, so it answers at once.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# One worker: a job is CPU-bound and MAX_CONCURRENT_JOBS inside the process is the real knob (D-11).
CMD ["uvicorn", "respeak.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
