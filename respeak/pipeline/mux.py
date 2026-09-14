"""One ffmpeg call turns the source video + `dubbed.wav` + `subs.srt` into `out.mp4`.

flow.md B4.8, plan.md 1.9, decisions.md D-05 (ffmpeg + libass, never OpenCV) and D-10 (burn when the
checkbox is on, and always attach a soft `mov_text` track).

    burn on   [0:v:0] → subtitles=…:force_style='FontName=Noto Sans,Outline=1,MarginV=30' → libx264
    burn off  -map 0:v:0 -c:v copy                                            (no video re-encode)
    stretch   [0:v:0] → setpts=s*PTS[,subtitles=…] → libx264                  (D-49; never a copy)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from respeak.lang_codes import ISO639_2
from respeak.pipeline import ffmpeg

log = logging.getLogger(__name__)

FORCE_STYLE = "FontName=Noto Sans,Outline=1,MarginV=30"
"""libass overrides: a font with wide script coverage, an outline, and room above the bottom edge."""

UNDETERMINED = "und"
"""ISO-639-2 for "language not known" — what an unmapped code becomes."""

STRETCH_EPSILON = 1e-4
"""Below this much slowdown the picture is left alone: re-encoding for 0.005 % is not worth it."""

ProgressCallback = Callable[[float], None]
"""Called with how much of the output is written, 0–1; always called once with 1.0 at the end."""


def mux(
    source_mp4: Path | str,
    dubbed_wav: Path | str,
    subs_srt: Path | str | None,
    burn: bool,
    lang: str,
    out: Path | str,
    *,
    stretch: float = 1.0,
    progress: ProgressCallback | None = None,
) -> Path:
    """Mux video + dubbed audio (+ subtitles) into `out` and return it.

    `stretch > 1` slows the picture down by that factor so the dub fits without cutting speech
    (D-49): `setpts=<stretch>*PTS` (chained before `subtitles` when burning, so the cues land on the
    stretched times they were written for) and the output runs `stretch ×` the source length. **A
    stretched video is always re-encoded**, with burn off too — a filter and `-c:v copy` cannot both
    apply, so the stream copy that burn=False normally buys is not available here. The frames
    themselves are not touched, only their timestamps: the output keeps every source frame and plays
    at a lower rate.

    With `progress`, ffmpeg is run with `-progress pipe:1` and the callback follows `out_time`
    through the encode (plan.md 3.1); without it the call is the plain, silent one it always was.
    """
    source = Path(source_mp4)
    audio = Path(dubbed_wav)
    subs = Path(subs_srt) if subs_srt is not None else None
    dest = Path(out)
    for path, what in ((source, "source video"), (audio, "dubbed audio")):
        if not path.exists():
            raise FileNotFoundError(f"cannot mux: the {what} {path} is missing")
    if subs is not None and not subs.exists():
        raise FileNotFoundError(f"cannot mux: the subtitle file {subs} is missing")
    if burn and subs is None:
        raise ValueError("mux(burn=True) needs a subtitle file to burn")
    if stretch <= 0:
        raise ValueError(f"mux() needs a positive stretch factor, got {stretch!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    slowed = stretch > STRETCH_EPSILON + 1.0

    args: list[str] = ["-i", str(source), "-i", str(audio)]
    if subs is not None:
        args += ["-i", str(subs)]
    chain: list[str] = []
    if slowed:
        chain.append(f"setpts={stretch:.6f}*PTS")
    if burn and subs is not None:
        chain.append(f"subtitles={escape_filter_path(subs)}:force_style='{FORCE_STYLE}'")
    if chain:
        args += ["-filter_complex", f"[0:v:0]{','.join(chain)}[v]", "-map", "[v]"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    else:
        # Safe to copy: `inputs.ensure_mp4_codecs()` has already turned anything mp4 cannot carry
        # (a VP9/Opus WebM, say) into h264 + aac before the pipeline gets here.
        args += ["-map", "0:v:0", "-c:v", "copy"]
    args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
    # The length of the output: what `-shortest` would pick, and what a progress fraction divides by.
    # A stretched picture is `stretch ×` the source, and `dubbed.wav` was assembled to that same length.
    output_seconds = 0.0
    if slowed or subs is not None or progress is not None:
        output_seconds = min(ffmpeg.duration(source) * stretch, ffmpeg.duration(audio))
    if subs is None and not slowed:
        args += ["-shortest"]
    else:
        # `-shortest` counts the subtitle track too, and the last cue normally ends before the video
        # does — it would cut the film off mid-scene. Cap the output explicitly instead.
        if subs is not None:
            args += ["-map", "2:s:0", "-c:s", "mov_text", "-metadata:s:s:0", f"language={iso639_2(lang)}"]
        args += ["-t", f"{output_seconds:.3f}"]
    args += ["-movflags", "+faststart", str(dest)]

    log.info("muxing %s (burn=%s, lang=%s, stretch=%.4f) → %s", source.name, burn, lang, stretch, dest.name)
    if progress is None:
        ffmpeg.run(args)
    else:
        ffmpeg.run_progress(args, _fraction_of(output_seconds, progress))
        progress(1.0)  # a stream copy can finish before ffmpeg prints a single progress block
    return dest


def _fraction_of(output_seconds: float, progress: ProgressCallback) -> ffmpeg.SecondsCallback:
    """Turn ffmpeg's `out_time` in seconds into the 0–1 fraction of the output it represents."""

    def report(seconds: float) -> None:
        if output_seconds <= 0:
            return
        progress(min(1.0, max(0.0, seconds / output_seconds)))

    return report


def iso639_2(lang: str) -> str:
    """'es' → 'spa'. Regional tags are folded ('zh-cn' → 'zho'); unknown codes become 'und'."""
    code = (lang or "").strip().lower().replace("_", "-")
    base = code.split("-")[0]
    if base in ISO639_2:
        return ISO639_2[base]
    if len(base) == 3 and base.isalpha():  # already ISO-639-2
        return base
    log.warning("no ISO-639-2 code for %r; tagging the subtitle stream as %s", lang, UNDETERMINED)
    return UNDETERMINED


def escape_filter_path(path: Path | str) -> str:
    """Quote a path for a filtergraph option value (verified against spaces, quotes, colons, brackets).

    The value is read twice — once by the filtergraph parser, once by the filter's option parser — so
    the quote is closed and reopened around an escaped `'`, and `:` stays escaped for the second pass.
    """
    text = str(path)
    text = text.replace("\\", "\\\\")
    text = text.replace("'", r"'\\\''")
    text = text.replace(":", r"\:")
    return f"'{text}'"
