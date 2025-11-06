from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional, TypedDict

import make_videos as mv

from .voice_selector import get_voice_catalog


class GenConfig(TypedDict, total=False):
    job_id: str
    video_path: Path
    quote: str
    music_path: Path
    aspect: str  # "9:16" | "1:1" | "16:9"
    voice_name: str  # "Rachel" | "Adam" | ...
    style: str  # "motivational" | "historical" | "greeting"
    logo_path: Optional[Path]
    subtitle_fontname: str
    karaoke_color: str
    voice_delay: float
    voice_tempo: float
    music_volume: float
    ducking: bool
    out_dir: Path
    duration_sec: float  # 10..30
    language: str  # "uk" | "en" | "de"


class GenResult(TypedDict, total=False):
    job_id: str
    output_path: Path
    duration_sec: float
    thumb_path: Path
    ssml_path: Optional[Path]
    logs_path: Optional[Path]


def _tone_from_style(style: str) -> str:
    s = (style or "").lower()
    if s in ("historical", "serious", "drama"):
        return "serious"
    if s in ("greeting", "friendly"):
        return "hopeful"
    if s in ("surprise", "surprize", "wow"):
        return "excited"
    return "inspiring"


def _aspect_size(aspect: str) -> tuple[int, int]:
    m = {
        "9:16": (720, 1280),
        "1:1": (1080, 1080),
        "16:9": (1920, 1080),
    }
    return m.get(aspect, (mv.TARGET_WIDTH, mv.TARGET_HEIGHT))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def generate_one(
    cfg: GenConfig, progress: Optional[Callable[[str, float, str], None]] = None
) -> GenResult:
    job_id = cfg.get("job_id") or f"J_{time.strftime('%Y%m%d_%H%M%S')}"

    video_path: Path = Path(cfg["video_path"]).resolve()
    music_path: Path = Path(cfg["music_path"]).resolve()
    out_dir: Path = Path(cfg["out_dir"]).resolve()
    _ensure_dir(out_dir)

    assert video_path.exists(), f"Input video not found: {video_path}"
    assert music_path.exists(), f"Music file not found: {music_path}"

    aspect = cfg.get("aspect", "9:16")
    framing = str(cfg.get("framing", "fit")).lower()
    quality = str(cfg.get("quality", "standard")).lower()
    target_w, target_h = _aspect_size(aspect)
    if quality == "high" and aspect == "9:16":
        target_w, target_h = (1080, 1920)
    duration = float(cfg.get("duration_sec", mv.TARGET_SECONDS))
    duration = max(10.0, min(30.0, duration))

    mv.TARGET_SECONDS = int(round(duration))
    if "voice_delay" in cfg:
        mv.VOICE_DELAY = float(cfg["voice_delay"])

    tmp_root = Path(os.getenv("STORAGE_TMP", "backend/tmp"))
    _ensure_dir(tmp_root)
    base = mv.unique_name(video_path.stem)
    tmp_dir = tmp_root / f"{job_id}_{base}"
    _ensure_dir(tmp_dir)

    final_name = f"{job_id}_{video_path.stem}.mp4"
    final_video = out_dir / final_name
    thumb_path = out_dir / (final_video.stem + "_thumb.jpg")
    ssml_path = out_dir / (final_video.stem + "_voice.ssml.txt")

    tmp_video = tmp_dir / "video.mp4"
    tts_mp3 = tmp_dir / "voice.mp3"
    mixed_audio = tmp_dir / "mix.m4a"
    ass_path = tmp_dir / "quote.ass"

    quote = cfg["quote"]
    style = cfg.get("style", "motivational")
    tone = _tone_from_style(style)
    if progress:
        progress("analyzing_quote", 0.05, "Analyzing quote and style")
    # If user provided high-level tag/intent, let it steer TTS tone
    try:
        tag = str(cfg.get("tag") or "").strip().lower()
    except Exception:
        tag = ""
    if tag:
        tag_tone_map = {
            "motivational": "inspiring",
            "greeting": "hopeful",
            "historical": "serious",
            "announcement": "serious",
            "emotional": "emotional",
        }
        tone = tag_tone_map.get(tag, tone)
    voice_name = cfg.get("voice_name", "Rachel")
    catalog = get_voice_catalog()
    voice_id = catalog.get(voice_name, catalog.get("Rachel")).voice_id  # type: ignore[attr-defined]

    # Voice synthesis settings (with sensible defaults, can be analysis-driven)
    stability = float(cfg.get("voice_stability", 0.3))
    similarity_boost = float(cfg.get("voice_similarity_boost", 0.9))
    # If surprise style and no explicit override, make delivery livelier
    if style.lower() in {"surprise", "surprize", "wow"}:
        if "voice_stability" not in cfg:
            stability = 0.25
        if "voice_tempo" not in cfg:
            cfg["voice_tempo"] = max(1.0, float(cfg.get("voice_tempo", 1.0)) * 1.06)

    if progress:
        progress("generating_tts", 0.1, "Generating TTS audio")
    ssml = mv.prepare_tts_text(quote, tone)
    used_service = False
    try:  # pragma: no cover - exercised in integration, skipped in unit tests
        if bool(cfg.get("use_tts_service", True)):
            from .tts_service import (  # local import to avoid hard dep when unused
                ElevenLabsTTS,  # pragma: no cover
            )

            tts = ElevenLabsTTS()  # pragma: no cover
            tts.synthesize_to_file(
                ssml,
                tts_mp3,
                voice_id=voice_id,
                stability=stability,
                similarity_boost=similarity_boost,
            )  # pragma: no cover
            used_service = True  # pragma: no cover
    except Exception:  # pragma: no cover - robustness path
        used_service = False
    if not used_service:
        mv.tts_generate_elevenlabs_ex(
            ssml,
            tts_mp3,
            voice_id=voice_id,
            stability=stability,
            similarity_boost=similarity_boost,
        )
    try:
        ssml_path.write_text(ssml, encoding="utf-8")
    except Exception:
        pass

    if progress:
        progress("rendering_video", 0.35, "Processing video and subtitles")
    wrapped, fontsize_effective, _ = mv._fit_text_block(quote, mv.TEXT_FONTSIZE, target_h)  # type: ignore[attr-defined]
    voice_len = mv.probe_duration_seconds(tts_mp3) or (
        mv.TARGET_SECONDS - mv.VOICE_DELAY
    )
    start_t = mv.VOICE_DELAY
    end_t = min(mv.TARGET_SECONDS, start_t + voice_len)
    mv.write_static_ass(
        ass_path,
        full_text=wrapped,
        start_time=start_t,
        end_time=end_t,
        fontname=cfg.get("subtitle_fontname", "Comic Sans MS"),
        fontsize=fontsize_effective,
        align=2,
        margin_v=mv.TEXT_MARGIN_BOTTOM,
        target_w=target_w,
        target_h=target_h,
        color=cfg.get("karaoke_color", mv.TEXT_FONTCOLOR),
    )
    mv.preprocess_video_karaoke(
        video_path,
        tmp_video,
        ass_path,
        target_w=target_w,
        target_h=target_h,
        crf=(18 if quality == "high" else 20),
        preset=("medium" if quality == "high" else "veryfast"),
        use_lanczos=(quality == "high"),
        denoise=(quality == "high"),
        sharpen=(quality == "high"),
        framing=framing,
    )

    if progress:
        progress("processing_audio", 0.7, "Mixing and processing audio")
    mv.mix_audio(
        tts_mp3,
        music_path,
        mixed_audio,
        voice_delay=mv.VOICE_DELAY,
        music_volume=float(cfg.get("music_volume", 0.18)),
        ducking=bool(cfg.get("ducking", True)),
        ducking_threshold_db=-28.0,
        ducking_ratio=6.0,
        ducking_attack_ms=5,
        ducking_release_ms=180,
        voice_tempo=float(cfg.get("voice_tempo", 1.0)),
        voice_warmth_db=1.5,
        voice_smooth_db=-3.5,
        deesser=True,
        music_fade_out_s=0.0,
        voice_sibilance_db=-3.0,
        voice_sibilance_freq_hz=6500.0,
        voice_sibilance_q=1.2,
        audio_bitrate=("256k" if quality == "high" else "192k"),
    )

    if progress:
        progress("finalizing", 0.9, "Finalizing outputs")
    # Backward-compat: prefer test helper mux_video_and_audio if present
    mux = getattr(mv, "mux_video_and_audio", None)
    if not callable(mux):
        mux = getattr(mv, "mux_video_audio", None)
    if not callable(mux):  # pragma: no cover - defensive
        raise RuntimeError("No mux function available in make_videos module")
    mux(tmp_video, mixed_audio, final_video)
    thumb_ok = False
    try:  # pragma: no cover - fallback thumbnail path not hit in unit tests
        if hasattr(mv, "extract_thumbnail"):
            thumb_ok = mv.extract_thumbnail(final_video, thumb_path)  # type: ignore[attr-defined]
        if not thumb_ok:
            try:
                dur = mv.probe_duration_seconds(final_video) or float(mv.TARGET_SECONDS)
            except Exception:
                dur = float(mv.TARGET_SECONDS)
            ts = max(0.0, min(float(dur) * 0.5, float(mv.TARGET_SECONDS) - 0.5))
            cmd = [
                mv.FFMPEG_BIN,
                "-y",
                "-ss",
                str(ts),
                "-i",
                str(final_video),
                "-frames:v",
                "1",
                str(thumb_path),
            ]
            mv.run(cmd)
            thumb_ok = thumb_path.exists()
    except Exception:  # pragma: no cover - defensive
        thumb_ok = False

    return {
        "job_id": job_id,
        "output_path": final_video,
        "duration_sec": float(mv.TARGET_SECONDS),
        "thumb_path": thumb_path if thumb_ok else None,
        "ssml_path": ssml_path if ssml_path.exists() else None,
    }


