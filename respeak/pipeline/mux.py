"""One ffmpeg call turns the source video + `dubbed.wav` + `subs.srt` into `out.mp4`.

flow.md B4.8, plan.md 1.9, decisions.md D-05 (ffmpeg + libass, never OpenCV) and D-10 (burn when the
checkbox is on, and always attach a soft `mov_text` track).

    burn on   [0:v:0] → subtitles=…:force_style='FontName=Noto Sans,Outline=1,MarginV=30' → libx264
    burn off  -map 0:v:0 -c:v copy                                            (no video re-encode)
"""

from __future__ import annotations

import logging
from pathlib import Path

from respeak.lang_codes import ISO639_2
from respeak.pipeline import ffmpeg

log = logging.getLogger(__name__)

FORCE_STYLE = "FontName=Noto Sans,Outline=1,MarginV=30"
"""libass overrides: a font with wide script coverage, an outline, and room above the bottom edge."""

UNDETERMINED = "und"
"""ISO-639-2 for "language not known" — what an unmapped code becomes."""


def mux(
    source_mp4: Path | str,
    dubbed_wav: Path | str,
    subs_srt: Path | str | None,
    burn: bool,
    lang: str,
    out: Path | str,
) -> Path:
    """Mux video + dubbed audio (+ subtitles) into `out` and return it."""
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
    dest.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["-i", str(source), "-i", str(audio)]
    if subs is not None:
        args += ["-i", str(subs)]
    if burn and subs is not None:
        graph = f"[0:v:0]subtitles={escape_filter_path(subs)}:force_style='{FORCE_STYLE}'[v]"
        args += ["-filter_complex", graph, "-map", "[v]"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    else:
        args += ["-map", "0:v:0", "-c:v", "copy"]
    args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
    if subs is None:
        args += ["-shortest"]
    else:
        # `-shortest` counts the subtitle track too, and the last cue normally ends before the video
        # does — it would cut the film off mid-scene. Cap the output explicitly instead.
        args += ["-map", "2:s:0", "-c:s", "mov_text", "-metadata:s:s:0", f"language={iso639_2(lang)}"]
        args += ["-t", f"{min(ffmpeg.duration(source), ffmpeg.duration(audio)):.3f}"]
    args += ["-movflags", "+faststart", str(dest)]

    log.info("muxing %s (burn=%s, lang=%s) → %s", source.name, burn, lang, dest.name)
    ffmpeg.run(args)
    return dest


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
