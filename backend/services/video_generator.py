from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from ..core.logging import bind_job
from ..core.metrics import JOBS_QUEUED
from ..core.security import ensure_under, sanitize_text
from ..core.tracing import inject_trace_to_dict
from ..job_storage import get_job_storage


def _is_inline() -> bool:
    val = str(os.getenv("TASKS_INLINE", "0")).strip().lower()
    return val in ("1", "true", "yes", "on")


# Celery task handle is kept at module level for easy monkeypatching in tests.
# It is resolved lazily in enqueue_generate to avoid importing Celery when not needed.
task_generate = None  # type: ignore[assignment]


def build_generate_cfg(
    payload: Dict[str, Any],
    user_id: int | None,
    storage_input: Path,
    storage_output: Path,
    music_dir: Path,
) -> Dict[str, Any]:
    try:
        video_path = Path(payload["video_path"]).resolve()
        video_path = ensure_under(storage_input, video_path)
        quote = sanitize_text(str(payload["quote"]), 500)
        aspect = payload.get("aspect", "9:16")
        from pathlib import Path as _P

        music_file = payload.get("music")
        if not music_file:
            raise KeyError("music")
        # Accept either a plain filename or a path; normalize to filename
        music_path = (music_dir / _P(str(music_file)).name).resolve()
        voice_name = payload.get("voice")
        style = payload.get("style", "motivational")
        duration_sec = float(payload.get("duration_sec", 15))
        language = payload.get("language", "uk")
        quality = str(payload.get("quality", "standard")).lower()
        framing = str(payload.get("framing", "fit")).lower()
        tag = payload.get("tag")
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"code": "VALIDATION_ERROR", "message": f"Missing field: {e}"}
            },
        )

    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"code": "VALIDATION_ERROR", "message": "video_path not found"}
            },
        )
    if not music_path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"code": "VALIDATION_ERROR", "message": "music not found"}
            },
        )
    if len(quote.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "VALIDATION_ERROR", "message": "Quote is empty"}},
        )
    if not (10 <= duration_sec <= 30):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "duration_sec must be between 10 and 30",
                }
            },
        )
    if language not in {"uk", "en", "de"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "language must be uk|en|de",
                }
            },
        )

    # Auto-select voice using QuoteAnalyzer if not provided or marked as auto
    analysis = None
    if not voice_name or str(voice_name).strip().lower() in {"auto", "", "(auto)"}:
        try:
            from .voice_selector import select_voice_auto
            from ..quote_analyzer_v2 import analyze_quote  # type: ignore

            # Analyze once to drive both voice and timing/color hints
            analysis = analyze_quote(quote, language=language)
            auto_name = select_voice_auto(quote, language=language, tag=tag)
            if isinstance(auto_name, str) and auto_name:
                voice_name = auto_name
        except Exception:
            voice_name = voice_name or "Rachel"
    else:
        voice_name = str(voice_name)

    job_id = f"J_{uuid.uuid4().hex[:8]}"
    # Derive voice parameters from analysis if available (allow payload overrides)
    _tempo = (analysis.get("tempo") if isinstance(analysis, dict) else 1.0)
    _stability = (analysis.get("stability") if isinstance(analysis, dict) else 0.3)
    _similarity = (
        analysis.get("similarity_boost") if isinstance(analysis, dict) else 0.9
    )

    cfg: Dict[str, Any] = {
        "job_id": job_id,
        "video_path": video_path,
        "quote": quote,
        "music_path": music_path,
        "aspect": aspect,
        "voice_name": voice_name or "Rachel",
        "style": style,
        "subtitle_fontname": payload.get("subtitle_fontname", "Comic Sans MS"),
        "karaoke_color": payload.get(
            "karaoke_color", (analysis.get("color") if isinstance(analysis, dict) else "#66CCFF")
        ),
        "voice_delay": float(payload.get("voice_delay", 2.5)),
        # If analyzer suggested a tempo via select_voice_auto path, the caller can pass it;
        # keep payload override if provided, else analysis-driven/default
        "voice_tempo": float(payload.get("voice_tempo", _tempo)),
        "voice_stability": float(payload.get("voice_stability", _stability)),
        "voice_similarity_boost": float(
            payload.get("voice_similarity_boost", _similarity)
        ),
        "music_volume": float(payload.get("music_volume", 0.18)),
        "ducking": bool(payload.get("ducking", True)),
        "out_dir": storage_output,
        "duration_sec": duration_sec,
        "language": language,
        "quality": ("high" if quality == "high" else "standard"),
        "framing": (framing if framing in {"fill", "fit", "reframe"} else "fit"),
        "tag": tag,
        "user_id": user_id,
    }
    return cfg


def enqueue_generate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global task_generate
    job_id = cfg.get("job_id") or f"J_{uuid.uuid4().hex[:8]}"
    cfg["job_id"] = job_id

    Jobs = get_job_storage()

    def _run_inline() -> Dict[str, Any]:
        try:
            from ..core.logging import setup_logging

            setup_logging()
        except Exception:
            pass
        try:
            bind_job(job_id)
        except Exception:
            pass
        try:
            inject_trace_to_dict(cfg)  # type: ignore[arg-type]
        except Exception:
            pass
        Jobs.set(
            job_id,
            {
                "status": "processing",
                "step": "inline",
                "progress": 0.0,
                "message": "Processing inline",
                "result": None,
            },
        )
        from .video_engine import generate_one

        result = generate_one(cfg, progress=lambda s, p, m: Jobs.update(job_id, {"status": "processing", "step": s, "progress": float(p), "message": m}))  # type: ignore[arg-type]
        payload = {
            "output": str(result.get("output_path")),
            "thumb": str(result.get("thumb_path")),
            "duration_sec": result.get("duration_sec"),
            "ssml": str(result.get("ssml_path")) if result.get("ssml_path") else None,
        }
        Jobs.set(job_id, {"status": "completed", "result": payload})
        return {"job_id": job_id, "status": "completed", "result": payload}

    # Inline (synchronous) execution for demos or fallback
    # If a Celery task handle is already available (e.g., monkeypatched in tests),
    # prefer enqueueing even when TASKS_INLINE is set.
    if _is_inline() and (task_generate is None):  # type: ignore[truthy-bool]
        return _run_inline()

    # Default path: enqueue via Celery
    Jobs.set(
        job_id,
        {
            "status": "queued",
            "step": None,
            "progress": 0.0,
            "message": "In queue",
            "result": None,
            # Store a JSON-serializable snapshot for retry
            "cfg": {
                k: (str(v) if isinstance(v, __import__("pathlib").Path) else v)
                for k, v in cfg.items()
            },
        },
    )
    try:
        JOBS_QUEUED.inc()
    except Exception:
        pass
    try:
        bind_job(job_id)
    except Exception:
        pass
    try:
        inject_trace_to_dict(cfg)  # type: ignore[arg-type]
    except Exception:
        pass
    if task_generate is None:  # type: ignore[truthy-bool]
        try:
            from ..workers.celery_app import (
                task_generate as _task_generate,  # type: ignore
            )

            task_generate = _task_generate  # type: ignore[assignment]
        except Exception:
            from ..core.config import settings as _s

            if (getattr(_s, "APP_ENV", "dev") or "").lower() != "prod":
                return _run_inline()
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "Task queue is unavailable",
                    }
                },
            )

    try:
        task = task_generate.delay(cfg)  # type: ignore[attr-defined]
        Jobs.update(job_id, {"celery_id": task.id})
        return {"job_id": job_id, "status": "queued", "task_id": task.id}
    except Exception:
        from ..core.config import settings as _s2

        if (getattr(_s2, "APP_ENV", "dev") or "").lower() != "prod":
            return _run_inline()
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Task queue publish failed",
                }
            },
        )
