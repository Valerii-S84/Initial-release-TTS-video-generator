import os
import sys
import uuid
import time
import argparse
import subprocess
from pathlib import Path
import re

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Lazy import error shown later with guidance

try:
    from gtts import gTTS  # networked fallback when OpenAI key is missing
except Exception:
    gTTS = None


# Default target size (overridden by --aspect/--size)
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
TARGET_SECONDS = 12
TARGET_FPS = 30

# Text overlay defaults
TEXT_FONTSIZE =48
TEXT_FONTCOLOR = "white"
TEXT_BOXCOLOR = "black@0.6"
TEXT_BOXBORDERW = 15
TEXT_LINE_SPACING = 10
TEXT_MARGIN_BOTTOM = 160
TEXT_TOP_MARGIN = 80
TEXT_MAX_LINES = 10

# Audio/visual timing
VOICE_DELAY = 3  # seconds (delay voice relative to music/video)
TEXT_FADE_IN_AT = 3  # seconds (text appears with voice by default)

FFMPEG_BIN = "ffmpeg"


def shlex_join_win(args):
    # Simple join for logging on Windows
    return " ".join(f'"{a}"' if " " in a or any(ch in a for ch in ['\\', '"']) else a for a in args)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    # Force UTF-8 decoding of subprocess output and replace undecodable bytes
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}):\n{shlex_join_win(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def ensure_ffmpeg() -> None:
    try:
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, check=True)
    except Exception as e:
        raise RuntimeError(
            "ffmpeg is required but not found in PATH.\n"
            "Install it and ensure 'ffmpeg' is available in your terminal.\n"
            "Windows: https://www.gyan.dev/ffmpeg/builds/"
        ) from e


def _ffprobe_bin() -> str:
    # Try to use ffprobe from the same dir as ffmpeg when possible
    ff = Path(FFMPEG_BIN)
    name = ff.name.lower()
    if name.startswith("ffmpeg"):
        probe = ff.with_name("ffprobe" + (".exe" if ff.suffix.lower() == ".exe" or os.name == "nt" else ""))
        return str(probe) if probe.exists() else "ffprobe"
    return "ffprobe"


def probe_duration_seconds(media_path: Path) -> float | None:
    bin_path = _ffprobe_bin()
    try:
        proc = subprocess.run(
            [bin_path, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(media_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            return None
        s = proc.stdout.strip()
        return float(s)
    except Exception:
        return None


def list_media(dir_path: Path, exts: tuple[str, ...]) -> list[Path]:
    if not dir_path.exists():
        return []
    files = [p for p in sorted(dir_path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def read_quotes(quotes_file: Path, encoding: str = "utf-8") -> list[str]:
    if not quotes_file.exists():
        return []
    data: str
    try:
        data = quotes_file.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Fallback for BOM or legacy encodings
        for enc in ("utf-8-sig", "cp1251"):
            try:
                data = quotes_file.read_text(encoding=enc)
                break
            except Exception:
                continue
        else:
            raise
    lines: list[str] = []
    for line in data.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def tts_generate(quote: str, out_path: Path, model: str, voice: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install -r requirements.txt"
            )
        client = OpenAI()
        # Prefer streaming to write directly to file
        try:
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=quote,
                format="mp3",
            ) as response:
                response.stream_to_file(str(out_path))
        except Exception:
            # Fallback non-streaming (SDK/feature variability)
            resp = client.audio.speech.create(model=model, voice=voice, input=quote, format="mp3")
            audio_bytes = getattr(resp, "read", None)
            if callable(audio_bytes):
                data = audio_bytes()
            else:
                # Some SDK versions expose .content
                data = getattr(resp, "content", None)
            if not data:
                raise
            out_path.write_bytes(data)
    else:
        # Fallback to gTTS when no OpenAI key is set
        if gTTS is None:
            raise RuntimeError(
                "OPENAI_API_KEY not set and gTTS is not installed. Install it with: pip install gTTS"
            )
        # Note: gTTS uses Google TTS service and requires internet access; it is not offline.
        # Default to Ukrainian voice; can be overridden by CLI argument.
        raise RuntimeError("gTTS fallback requires language parameter via tts_generate_gtts().")


def tts_generate_gtts(quote: str, out_path: Path, lang: str = "uk") -> None:
    if gTTS is None:
        raise RuntimeError("gTTS is not installed. Run: pip install gTTS")
    tts = gTTS(text=quote, lang=lang)
    tts.save(str(out_path))


def _wrap_text(text: str, max_chars: int = 40) -> str:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) <= max_chars:
            cur.append(w)
            cur_len += len(w) + (1 if cur_len > 0 else 0)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _escape_drawtext(text: str) -> str:
    # Kept for potential inline usage, but we now prefer textfile= to avoid escaping headaches on Windows
    s = text.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    s = s.replace("\n", "\\n")
    return s


def _find_default_cyrillic_font() -> str | None:
    # Try common system fonts that include Cyrillic glyphs
    candidates: list[Path] = []
    win_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    mac_fonts = [
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Verdana.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    nix_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in win_fonts + mac_fonts + nix_fonts:
        path = Path(p)
        if path.exists():
            return path.as_posix()
    return None


def _dynamic_fontsize_for_quote(q: str, base_size: int = TEXT_FONTSIZE) -> int:
    n = len(q)
    size = base_size
    if n > 300:
        size = int(base_size * 0.75)
    elif n > 200:
        size = int(base_size * 0.85)
    return max(20, size)


def _wrap_chars_for_fontsize(fontsize: int, base_size: int = TEXT_FONTSIZE) -> int:
    # Scale wrap width roughly with font size, clamp to sane range
    base_chars = 36
    chars = int(base_chars * max(0.6, min(1.2, fontsize / base_size)))
    return max(24, min(60, chars))


def _estimate_text_height_px(line_count: int, fontsize: int) -> int:
    if line_count <= 0:
        return 0
    # Approximate text block height: lines*fontsize + gaps + box borders
    gaps = (line_count - 1) * TEXT_LINE_SPACING
    box = 2 * TEXT_BOXBORDERW
    return line_count * fontsize + gaps + box


def _fit_text_block(quote_text: str, base_fontsize: int, target_height: int) -> tuple[str, int, int]:
    # Returns (wrapped_text, fontsize_effective, line_count)
    fontsize_effective = _dynamic_fontsize_for_quote(quote_text, base_size=base_fontsize)
    allowed_h = target_height - TEXT_MARGIN_BOTTOM - TEXT_TOP_MARGIN
    while True:
        wrap_chars = _wrap_chars_for_fontsize(fontsize_effective, base_size=base_fontsize)
        wrapped = _wrap_text(quote_text, max_chars=wrap_chars)
        line_count = wrapped.count("\n") + 1 if wrapped else 0
        est_h = _estimate_text_height_px(line_count, fontsize_effective)
        if (line_count <= TEXT_MAX_LINES and est_h <= allowed_h) or fontsize_effective <= 20:
            return wrapped, fontsize_effective, line_count
        fontsize_effective = max(20, int(fontsize_effective * 0.9))


def preprocess_video_static(
    in_video: Path,
    out_video: Path,
    quote_text: str,
    fontfile: str | None = None,
    fontsize: int = TEXT_FONTSIZE,
    fontcolor: str = TEXT_FONTCOLOR,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
) -> None:
    # Scale to cover and center-crop to 720x1280, 10s, 30fps, add quote text overlay, h264 mp4, no audio
    scaled_crop = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"\
        f"crop={target_w}:{target_h}"
    )

    wrapped, fontsize_effective, _ = _fit_text_block(quote_text, fontsize, target_h)
    # Write wrapped text to a temp file (avoid inline escaping issues)
    tmp_text = out_video.parent / "quote.txt"
    tmp_text.write_text(wrapped, encoding="utf-8", newline="\n")
    draw_opts = []
    if fontfile:
        try:
            ff = Path(fontfile).as_posix()
        except Exception:
            ff = str(fontfile).replace("\\", "/")
        # Escape ':' for drawtext option parsing (Windows drive letter 'C:')
        ff = ff.replace(":", r"\:")
        draw_opts.append(f"fontfile='{ff}'")
    # Use textfile (relative path, no drive letter -> no ':' issues)
    draw_opts.append(f"textfile='{tmp_text.as_posix()}'")
    draw_opts.append(f"fontcolor={fontcolor}")
    draw_opts.append(f"fontsize={fontsize_effective}")
    draw_opts.append(f"line_spacing={TEXT_LINE_SPACING}")
    draw_opts.append("box=1")
    draw_opts.append(f"boxcolor={TEXT_BOXCOLOR}")
    draw_opts.append(f"boxborderw={TEXT_BOXBORDERW}")
    # Add subtle shadow for readability
    draw_opts.append("shadowcolor=black@0.7")
    draw_opts.append("shadowx=2")
    draw_opts.append("shadowy=2")
    # center horizontally, position near bottom within safe margins
    draw_opts.append("x=(w-text_w)/2")
    draw_opts.append(f"y=h-text_h-{TEXT_MARGIN_BOTTOM}")
    # Timed appearance of text from TEXT_FADE_IN_AT to end
    enable_expr = f"enable='gte(t,{TEXT_FADE_IN_AT})'"
    drawtext = "drawtext=" + ":".join(draw_opts + [enable_expr])

    vf = f"{scaled_crop},{drawtext},format=yuv420p"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(in_video),
        "-t", str(TARGET_SECONDS),
        "-vf", vf,
        "-r", str(TARGET_FPS),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        str(out_video),
    ]
    run(cmd)


def _ass_time(t: float) -> str:
    # ASS uses h:mm:ss.cs (centiseconds)
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def write_karaoke_ass(
    out_path: Path,
    full_text: str,
    start_time: float,
    end_time: float,
    fontname: str,
    fontsize: int,
    align: int = 2,
    margin_v: int = TEXT_MARGIN_BOTTOM,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    karaoke_color: str | None = None,
) -> None:
    # Prepare karaoke content: split by whitespace, durations proportional to length
    # Keep original wrapping line breaks (\n) mapped to ASS \N
    display_text = full_text.replace("\n", " ")  # timing by words overall
    words = [w for w in re.split(r"(\s+)", display_text) if w.strip()]
    total_chars = sum(len(w) for w in words)
    dur = max(0.5, end_time - start_time)
    total_cs = int(round(dur * 100))
    if total_chars <= 0:
        total_chars = 1
    ks: list[int] = []
    acc = 0
    for idx, w in enumerate(words):
        k = max(3, int(round(total_cs * len(w) / total_chars)))
        ks.append(k)
        acc += k
    if ks:
        ks[-1] += (total_cs - acc)

    # Build karaoke line: use wrapped text with \N for visual newlines
    # Escape braces that would be interpreted as ASS override starts
    safe_text = full_text.replace("{", "(").replace("}", ")")
    visual_text = safe_text.replace("\n", r"\N")
    # Inject \k tags before each word sequentially
    seq = []
    wi = 0
    for tok in re.split(r"(\s+)", visual_text):
        if tok.strip():
            k = ks[wi] if wi < len(ks) else 5
            seq.append(f"{{\\k{k}}}{tok}")
            wi += 1
        else:
            seq.append(tok)
    karaoke_line = "".join(seq)

    # Use different SecondaryColour to visualize karaoke highlight
    # ASS color format: &HAABBGGRR
    def _to_ass_color(c: str | None) -> str:
        if not c:
            return "&H0066CCFF"  # default: light blue (#66CCFF)
        c = c.strip().lower()
        named = {
            "lightblue": "#66ccff",
            "cyan": "#00ffff",
            "lime": "#00ff00",
            "springgreen": "#00ff7f",
            "salad": "#b8ff9f",
            "magenta": "#ff00ff",
            "yellow": "#ffff00",
        }
        if c in named:
            c = named[c]
        if c.startswith("&h"):
            return c.upper()
        if c.startswith("#"):
            h = c[1:]
            if len(h) == 6:
                rr = h[0:2]
                gg = h[2:4]
                bb = h[4:6]
                aa = "00"
            elif len(h) == 8:
                aa = h[0:2]
                rr = h[2:4]
                gg = h[4:6]
                bb = h[6:8]
            else:
                return "&H0066CCFF"
            return f"&H{aa}{bb}{gg}{rr}".upper()
        return "&H0066CCFF"

    secondary_colour = _to_ass_color(karaoke_color)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},&H00FFFFFF,{secondary_colour},&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,{align},30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    event = f"Dialogue: 0,{_ass_time(start_time)},{_ass_time(end_time)},Default,,0,0,0,,{karaoke_line}\n"
    out_path.write_text(header + event, encoding="utf-8")


def preprocess_video_karaoke(in_video: Path, out_video: Path, ass_path: Path, target_w: int = TARGET_WIDTH, target_h: int = TARGET_HEIGHT) -> None:
    scaled_crop = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"\
        f"crop={target_w}:{target_h}"
    )
    ass = ass_path.as_posix().replace(":", r"\:")
    vf = f"{scaled_crop},subtitles='{ass}',format=yuv420p"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(in_video),
        "-t", str(TARGET_SECONDS),
        "-vf", vf,
        "-r", str(TARGET_FPS),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        str(out_video),
    ]
    run(cmd)


def mix_audio(voice_path: Path, music_path: Path, out_audio: Path, voice_delay: float = VOICE_DELAY) -> None:
    # Mix TTS voice with background music (quieter) and make sure audio is exactly TARGET_SECONDS long.
    # Loop music as needed; normalize voice; pad/trim to length.
    # Add subtle compression and keep overall volume consistent.
    delay_ms = max(0, int(round(voice_delay * 1000)))
    # Delay only the voice; keep music from t=0
    filter_complex = (
        # Music: lower volume and compress
        "[1:a]volume=0.18,acompressor=threshold=-20dB:ratio=4:attack=5:release=200[m];"
        # Voice: delay start, normalize; then we'll trim after mix
        f"[0:a]adelay={delay_ms}|{delay_ms},dynaudnorm=f=150:g=15[v];"
        # Mix to longest, then trim to exact TARGET_SECONDS and reset PTS
        f"[v][m]amix=inputs=2:duration=longest:dropout_transition=2,volume=1,atrim=0:{TARGET_SECONDS},asetpts=N/SR/TB[aout]"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(voice_path),
        "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_audio),
    ]
    run(cmd)


def mux_video_audio(in_video: Path, in_audio: Path, out_video: Path, voice_delay: float = VOICE_DELAY) -> None:
    # Audio is already delayed in the mix. Just mux and constrain total duration.
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(in_video),
        "-i", str(in_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", str(TARGET_SECONDS),
        "-c:v", "copy",
        "-c:a", "aac",
        str(out_video),
    ]
    run(cmd)


def overlay_logo(in_video: Path, logo_path: Path, out_video: Path, pos: str = "top-right", scale_w: int = 160, margin: int = 20) -> None:
    # Overlay PNG/SVG logo at a corner. Re-encode video to keep things simple.
    # Positions: top-left, top-right, bottom-left, bottom-right
    pos = pos.lower()
    x_expr = {
        "top-left": f"{margin}",
        "bottom-left": f"{margin}",
        "top-right": f"W-w-{margin}",
        "bottom-right": f"W-w-{margin}",
    }.get(pos, f"W-w-{margin}")
    y_expr = {
        "top-left": f"{margin}",
        "top-right": f"{margin}",
        "bottom-left": f"H-h-{margin}",
        "bottom-right": f"H-h-{margin}",
    }.get(pos, f"{margin}")
    filter_complex = (
        f"[1]scale={scale_w}:-1[lg];[0][lg]overlay=x={x_expr}:y={y_expr}"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(in_video),
        "-i", str(logo_path),
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",
        str(out_video),
    ]
    run(cmd)


def concat_videos(segments: list[Path], out_video: Path) -> None:
    # Concatenate multiple MP4 segments with audio via filter_complex concat
    # Re-encodes to ensure compatibility
    if not segments:
        raise RuntimeError("No segments to concatenate")
    inputs = []
    maps = []
    for idx, seg in enumerate(segments):
        inputs += ["-i", str(seg)]
        maps.append(f"[{idx}:v:0][{idx}:a:0]")
    n = len(segments)
    fc = "".join(maps) + f"concat=n={n}:v=1:a=1[v][a]"
    cmd = [
        FFMPEG_BIN, "-y",
        *inputs,
        "-filter_complex", fc,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        str(out_video),
    ]
    run(cmd)


def unique_name(base: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{base}_{uuid.uuid4().hex[:6]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate short vertical videos with TTS and background music.")
    parser.add_argument("--input_videos", default="input_videos", help="Folder with source videos")
    parser.add_argument("--music", default="music", help="Folder with background music files")
    parser.add_argument("--quotes", default="quotes.txt", help="Text file with one quote per line")
    parser.add_argument("--output", default="output_videos", help="Folder for rendered videos")
    parser.add_argument("--model", default="gpt-4o-mini-tts", help="OpenAI TTS model (e.g., gpt-4o-mini-tts or tts-1)")
    parser.add_argument("--voice", default="alloy", help="OpenAI TTS voice (e.g., alloy)")
    parser.add_argument("--count", type=int, default=0, help="How many videos to produce (0=all)")
    parser.add_argument("--fontfile", default=None, help="Path to TTF/OTF font for Cyrillic (e.g., C:/Windows/Fonts/arial.ttf)")
    parser.add_argument("--fontsize", type=int, default=TEXT_FONTSIZE, help="Text font size")
    parser.add_argument("--fontcolor", default=TEXT_FONTCOLOR, help="Text color (e.g., white, #ffffff)")
    parser.add_argument("--tts_lang", default="uk", help="gTTS language code for fallback (e.g., uk, ru, en)")
    parser.add_argument("--ffmpeg_bin", default=None, help="Path or name of ffmpeg executable (e.g., C:/ffmpeg/bin/ffmpeg.exe)")
    parser.add_argument("--quotes_encoding", default="utf-8", help="Encoding for quotes.txt (utf-8, cp1251, etc.)")
    parser.add_argument("--text_mode", default="static", choices=["static", "karaoke"], help="Text overlay mode: static drawtext or karaoke ASS")
    parser.add_argument("--subtitle_fontname", default="Comic Sans MS", help="Font name for ASS karaoke subtitles")
    parser.add_argument("--karaoke_color", default="lightblue", help="Highlight color for karaoke (name or #RRGGBB)")
    parser.add_argument("--aspect", default="9:16", choices=["9:16","1:1","16:9"], help="Target aspect ratio preset")
    parser.add_argument("--logo", default=None, help="Path to PNG/SVG logo for watermark overlay")
    parser.add_argument("--logo_pos", default="top-right", choices=["top-left","top-right","bottom-left","bottom-right"], help="Logo corner position")
    parser.add_argument("--logo_scale", type=int, default=160, help="Logo width in pixels (height auto)")
    parser.add_argument("--logo_margin", type=int, default=20, help="Logo margin in pixels from edges")
    parser.add_argument("--intro", default=None, help="Optional intro video to prepend")
    parser.add_argument("--outro", default=None, help="Optional outro video to append")
    args = parser.parse_args()

    global FFMPEG_BIN
    if args.ffmpeg_bin:
        FFMPEG_BIN = args.ffmpeg_bin
    ensure_ffmpeg()

    in_dir = Path(args.input_videos)
    music_dir = Path(args.music)
    out_dir = Path(args.output)
    quotes_file = Path(args.quotes)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target size from aspect preset
    aspect_map = {
        "9:16": (720, 1280),
        "1:1": (1080, 1080),
        "16:9": (1920, 1080),
    }
    target_w, target_h = aspect_map.get(args.aspect, (TARGET_WIDTH, TARGET_HEIGHT))

    videos = list_media(in_dir, (".mp4", ".mov", ".mkv", ".avi", ".webm"))
    musics = list_media(music_dir, (".mp3", ".wav", ".m4a", ".aac", ".flac"))
    quotes = read_quotes(quotes_file, encoding=args.quotes_encoding)

    if not videos:
        print(f"No input videos in {in_dir}")
        return 1
    if not quotes:
        print(f"No quotes found in {quotes_file}")
        return 1
    if not musics:
        print(f"No music files in {music_dir}")
        return 1

    total = len(videos) if args.count == 0 else min(args.count, len(videos))
    print(f"Found {len(videos)} videos, {len(quotes)} quotes, {len(musics)} music tracks. Producing {total} video(s).")

    tmp_root = Path(".tmp_build")
    tmp_root.mkdir(exist_ok=True)

    start_ts = time.time()
    # pick a default Cyrillic-capable font if not provided
    default_font = args.fontfile or _find_default_cyrillic_font()
    if args.fontfile is None and default_font:
        print(f"Using default Cyrillic font: {default_font}")
    elif args.fontfile is None and not default_font:
        print("Warning: no Cyrillic font found; drawtext may miss glyphs. Consider --fontfile C:/Windows/Fonts/arial.ttf")

    for i in range(total):
        src_video = videos[i]
        quote = quotes[i % len(quotes)]
        music = musics[i % len(musics)]

        base = unique_name(src_video.stem)
        tmp_dir = tmp_root / base
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_video = tmp_dir / f"video.mp4"
        tts_mp3 = tmp_dir / f"voice.mp3"
        mixed_audio = tmp_dir / f"mix.m4a"
        out_video = out_dir / f"{base}.mp4"

        # Progress and ETA
        elapsed = time.time() - start_ts
        done = i
        eta = 0.0 if done == 0 else (elapsed / done) * (total - done)
        def _fmt(sec: float) -> str:
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        print(f"[{i+1}/{total}] Starting. Elapsed { _fmt(elapsed) }, ETA { _fmt(eta) }")

        print(f"[{i+1}/{total}] Generating TTS voice ({args.model}/{args.voice})")
        try:
            tts_generate(quote, tts_mp3, model=args.model, voice=args.voice)
        except RuntimeError as e:
            if "OPENAI_API_KEY not set" in str(e) or "OpenAI SDK" in str(e) or "gTTS fallback" in str(e):
                print(f"[{i+1}/{total}] Falling back to gTTS ({args.tts_lang})")
                tts_generate_gtts(quote, tts_mp3, lang=args.tts_lang)
            else:
                raise

        # Prepare video with text overlay
        if args.text_mode == "karaoke":
            print(f"[{i+1}/{total}] Building karaoke subtitles and preprocessing video")
            # Fit text to safe area for chosen base font size
            wrapped, fontsize_effective, _ = _fit_text_block(quote, args.fontsize, target_h)
            voice_len = probe_duration_seconds(tts_mp3) or (TARGET_SECONDS - VOICE_DELAY)
            start_t = VOICE_DELAY
            end_t = min(TARGET_SECONDS, start_t + voice_len)
            ass_path = tmp_dir / "quote.ass"
            write_karaoke_ass(
                ass_path,
                full_text=wrapped,
                start_time=start_t,
                end_time=end_t,
                fontname=args.subtitle_fontname,
                fontsize=fontsize_effective,
                align=2,
                margin_v=TEXT_MARGIN_BOTTOM,
                target_w=target_w,
                target_h=target_h,
                karaoke_color=args.karaoke_color,
            )
            preprocess_video_karaoke(src_video, tmp_video, ass_path, target_w=target_w, target_h=target_h)
        else:
            print(f"[{i+1}/{total}] Preprocessing video with text: {src_video.name}")
            preprocess_video_static(
                src_video,
                tmp_video,
                quote_text=quote,
                fontfile=default_font,
                fontsize=args.fontsize,
                fontcolor=args.fontcolor,
                target_w=target_w,
                target_h=target_h,
            )

        print(f"[{i+1}/{total}] Mixing audio with music: {music.name}")
        mix_audio(tts_mp3, music, mixed_audio)

        print(f"[{i+1}/{total}] Muxing final video: {out_video.name}")
        core_video = tmp_dir / "core.mp4"
        mux_video_audio(tmp_video, mixed_audio, core_video)

        final_video = core_video
        # Optional logo overlay
        if args.logo and Path(args.logo).exists():
            logoed = tmp_dir / "core_logo.mp4"
            overlay_logo(final_video, Path(args.logo), logoed, pos=args.logo_pos, scale_w=args.logo_scale, margin=args.logo_margin)
            final_video = logoed

        # Optional intro/outro concatenation
        segs: list[Path] = []
        if args.intro and Path(args.intro).exists():
            # Preprocess intro to target size, keep its own audio (no subtitles)
            scaled_intro = tmp_dir / "intro_scaled.mp4"
            cmd_intro = [
                FFMPEG_BIN, "-y",
                "-i", str(args.intro),
                "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}",
                "-r", str(TARGET_FPS),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                str(scaled_intro),
            ]
            run(cmd_intro)
            segs.append(scaled_intro)
        segs.append(final_video)
        if args.outro and Path(args.outro).exists():
            scaled_outro = tmp_dir / "outro_scaled.mp4"
            cmd_outro = [
                FFMPEG_BIN, "-y",
                "-i", str(args.outro),
                "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}",
                "-r", str(TARGET_FPS),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                str(scaled_outro),
            ]
            run(cmd_outro)
            segs.append(scaled_outro)

        if len(segs) > 1:
            concat_out = tmp_dir / "final_concat.mp4"
            concat_videos(segs, concat_out)
            # move to requested out path
            cmd_mv = [FFMPEG_BIN, "-y", "-i", str(concat_out), "-c", "copy", str(out_video)]
            run(cmd_mv)
        else:
            # move core (or logoed) to out path
            cmd_mv = [FFMPEG_BIN, "-y", "-i", str(final_video), "-c", "copy", str(out_video)]
            run(cmd_mv)

        # Clean per-video tmp files on success
        try:
            for p in tmp_dir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                tmp_dir.rmdir()
            except Exception:
                pass
        except Exception:
            pass

    # Attempt to remove global tmp folder if empty
    try:
        next(tmp_root.iterdir())
    except StopIteration:
        try:
            tmp_root.rmdir()
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
