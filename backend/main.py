from __future__ import annotations

import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request

from .auth import AuthService, router as auth_router
from .services.voice_selector import get_voice_catalog
from .job_storage import get_job_storage
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from .utils.validators import FileValidator
from .core.logging import setup_logging, RequestIDMiddleware, bind_job
from sqlalchemy import text
from .db import get_db
from sqlalchemy.orm import Session
from .services.quota_service import check_quota, increment_usage
from .models import Video
from .core.metrics import PrometheusMiddleware, JOBS_QUEUED
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from .core.tracing import setup_tracing, inject_trace_to_dict
from .core.observability import setup_sentry
from .utils.cache import get_cache, get_redis_client
from .utils.etag import ETagStaticFiles
from .core.security import sanitize_text, ensure_under, sanitize_filename, require_ip_whitelist
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from .core.config import settings
from .api.dependencies import limiter


from .api.dependencies import STORAGE_INPUT, STORAGE_OUTPUT, STORAGE_TMP, MUSIC_DIR, MUSIC_PREVIEWS
BACKEND_HOST = settings.HOST
BACKEND_PORT = settings.PORT
LOGS_DIR = Path(os.getenv("LOGS_DIR", "backend/logs"))

for d in (STORAGE_INPUT, STORAGE_OUTPUT, STORAGE_TMP, MUSIC_DIR, MUSIC_PREVIEWS, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

origins = settings.ALLOWED_ORIGINS

setup_logging()
setup_tracing("video-api")
setup_sentry("video-api")
# Security sanity-checks for production
if (settings.APP_ENV.lower() == "prod") and (settings.JWT_SECRET_KEY in ("", "change_me", "dev-secret-change-me")):
    raise RuntimeError("Insecure JWT_SECRET_KEY in production. Please set a strong secret.")
app = FastAPI(title="Video Generator API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse({"error": {"code": "RATE_LIMIT", "message": str(exc)}}, status_code=429))
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)
if os.getenv("FORCE_HTTPS", "0") == "1":
    app.add_middleware(HTTPSRedirectMiddleware)


# Додаткові security headers (HSTS/CSP/Clickjacking)
from starlette.types import ASGIApp, Receive, Scope, Send


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        async def _send(message):
            if message.get("type") == "http.response.start":
                headers = message.setdefault("headers", [])
                def set_header(key: str, value: str):
                    headers.append((key.encode(), value.encode()))

                # HSTS тільки коли HTTPS
                if os.getenv("HSTS_ENABLED", "1") == "1":
                    set_header("strict-transport-security", "max-age=31536000; includeSubDomains; preload")
                # X-Frame-Options
                set_header("x-frame-options", "DENY")
                # X-Content-Type-Options
                set_header("x-content-type-options", "nosniff")
                # CSP (можна змінити через ENV)
                csp = os.getenv("CSP", "default-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'")
                set_header("content-security-policy", csp)
            await send(message)

        await self.app(scope, receive, _send)


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
app.mount("/media/music_previews", ETagStaticFiles(directory=str(MUSIC_PREVIEWS), check_dir=False, cache_seconds=86400), name="music_previews")
app.mount("/media/output_videos", ETagStaticFiles(directory=str(STORAGE_OUTPUT), check_dir=False, cache_seconds=86400), name="output_videos")

# Auth routes
app.include_router(auth_router)

# Підключення нових v1 роутерів (каркас для міграції)
from .api.v1 import auth as v1_auth
from .api.v1 import videos as v1_videos
from .api.v1 import jobs as v1_jobs
from .api.v1 import admin as v1_admin
from .api import api_router

api_router.include_router(v1_auth.router)
api_router.include_router(v1_videos.router)
api_router.include_router(v1_jobs.router)
api_router.include_router(v1_admin.router)
app.include_router(api_router)


# Jobs store (Redis with in-memory fallback)
Jobs = get_job_storage()


def _progress_cb(job_id: str):
    def _cb(step: str, progress: float, message: str):
        Jobs.update(job_id, {"status": "processing", "step": step, "progress": float(progress), "message": message})
    return _cb


@app.get("/api/config")
def get_config(current_user = Depends(AuthService.get_current_user)):
    # Кеш готової відповіді (голоси + музика)
    try:
        cache = get_cache()
        _music = cache.get_json("cfg:music")
        _voices = cache.get_json("cfg:voices")
        if _music is not None and _voices is not None:
            return {
                "aspects": ["9:16", "1:1", "16:9"],
                "styles": ["motivational", "historical", "greeting"],
                "voices": _voices,
                "music": _music,
                "limits": {"maxQuoteLen": 500, "durationSec": [10, 30], "languages": ["uk", "en", "de"]},
            }
    except Exception:
        pass
    # Music listing + previews
    music_items = []
    for p in sorted(MUSIC_DIR.glob("*.mp3")):
        prev = MUSIC_PREVIEWS / (p.stem + "_preview.mp3")
        music_items.append({"file": p.name, "preview": prev.name if prev.exists() else None, "label": p.stem})
    catalog = get_voice_catalog()
    voices = []
    for name in ("Rachel", "Adam", "Elli", "Antoni"):
        p = catalog.get(name)
        if p:
            desc = {
                "Rachel": "теплий спокійний",
                "Adam": "молодий енергійний",
                "Elli": "емоційна для історій",
                "Antoni": "серйозний авторитетний",
            }.get(name, "")
            voices.append({"name": name, "id": p.voice_id, "desc": desc})
    # Запис у кеш (1 година)
    try:
        cache.set_json("cfg:music", music_items, ttl=3600)
        cache.set_json("cfg:voices", voices, ttl=3600)
    except Exception:
        pass
    return {
        "aspects": ["9:16", "1:1", "16:9"],
        "styles": ["motivational", "historical", "greeting"],
        "voices": voices,
        "music": music_items,
        "limits": {"maxQuoteLen": 500, "durationSec": [10, 30], "languages": ["uk", "en", "de"]},
    }


@limiter.limit("20/hour")
@app.post("/api/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    current_user = Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    # Квота на завантаження
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/upload"}}, status_code=410)
    ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    vid = f"{uuid.uuid4().hex}{ext}"
    out_path = STORAGE_INPUT / vid
    # Validate and save
    fv = FileValidator(max_size_bytes=500 * 1024 * 1024)
    try:
        meta = await fv.avalidate_and_save_upload(file, out_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": str(e)}})
    # Write sidecar metadata with user_id for traceability
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    try:
        sidecar.write_text(json.dumps({"user_id": getattr(current_user, "id", None), **meta}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    # Зараховуємо використання квоти лише після успішного збереження
    increment_usage(db, current_user, "upload")
    return {"video_path": str(out_path), **meta}


@limiter.limit("5/minute")
@app.post("/api/generate")
async def generate(request: Request, payload: Dict[str, Any], current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/generate"}}, status_code=410)
    try:
        # Безпечний шлях: тільки з каталогу вводу
        video_path = Path(payload["video_path"]).resolve()
        video_path = ensure_under(STORAGE_INPUT, video_path)
        # Санітизуємо текст цитати
        quote = sanitize_text(str(payload["quote"]), 500)
        aspect = payload.get("aspect", "9:16")
        music_file = payload.get("music")
        if not music_file:
            raise KeyError("music")
        music_path = (MUSIC_DIR / music_file).resolve()
        voice_name = payload.get("voice", "Rachel")
        style = payload.get("style", "motivational")
        duration_sec = float(payload.get("duration_sec", 15))
        language = payload.get("language", "uk")
    except KeyError as e:
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": f"Missing field: {e}"}})

    if not video_path.exists():
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": "video_path not found"}})
    if not music_path.exists():
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": "music not found"}})
    if len(quote.strip()) == 0:
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": "Quote is empty"}})
    if not (10 <= duration_sec <= 30):
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": "duration_sec must be between 10 and 30"}})
    if language not in {"uk", "en", "de"}:
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": "language must be uk|en|de"}})

    job_id = f"J_{uuid.uuid4().hex[:8]}"
    Jobs.set(job_id, {"status": "queued", "step": None, "progress": 0.0, "message": "In queue", "result": None})
    try:
        JOBS_QUEUED.inc()
    except Exception:
        pass

    cfg = {
        "job_id": job_id,
        "video_path": video_path,
        "quote": quote,
        "music_path": music_path,
        "aspect": aspect,
        "voice_name": voice_name,
        "style": style,
        "subtitle_fontname": payload.get("subtitle_fontname", "Comic Sans MS"),
        "karaoke_color": payload.get("karaoke_color", "#66CCFF"),
        "voice_delay": float(payload.get("voice_delay", 2.5)),
        "voice_tempo": float(payload.get("voice_tempo", 1.0)),
        "music_volume": float(payload.get("music_volume", 0.18)),
        "ducking": bool(payload.get("ducking", True)),
        "out_dir": STORAGE_OUTPUT,
        "duration_sec": duration_sec,
        "language": language,
        "user_id": getattr(current_user, "id", None),
    }

    # Submit Celery task
    try:
        bind_job(job_id)
    except Exception:
        pass
    # Перевіряємо квоту генерацій перед постановкою в чергу
    check_quota(db, current_user, "generate")
    # Прокидуємо trace‑контекст у Celery через словник
    try:
        inject_trace_to_dict(cfg)  # type: ignore[arg-type]
    except Exception:
        pass
    task = task_generate.delay(cfg)  # type: ignore[arg-type]
    Jobs.update(job_id, {"celery_id": task.id})
    # Збільшуємо лічильник квот після успішної постановки задачі
    increment_usage(db, current_user, "generate")
    return {"job_id": job_id, "status": "queued", "task_id": task.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, current_user = Depends(AuthService.get_current_user)):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/jobs/*"}}, status_code=410)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, current_user = Depends(AuthService.get_current_user)):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/jobs/*"}}, status_code=410)


@app.post("/api/music/refresh_previews")
def refresh_music_previews(current_user = Depends(AuthService.get_current_user), request: Request = None):
    # IP whitelist (якщо задано)
    try:
        require_ip_whitelist(request)
    except Exception as e:
        raise
    # Create ~10s previews for all mp3 in MUSIC_DIR at +10s offset
    from make_videos import run, FFMPEG_BIN
    created = []
    for p in sorted(MUSIC_DIR.glob("*.mp3")):
        out = MUSIC_PREVIEWS / f"{p.stem}_preview.mp3"
        cmd = [FFMPEG_BIN, "-y", "-ss", "10", "-t", "10", "-i", str(p), "-acodec", "libmp3lame", str(out)]
        try:
            run(cmd)
            created.append(out.name)
        except Exception:
            continue
    try:
        get_cache().delete("cfg:music")
    except Exception:
        pass
    return {"created": created}


# Health checks
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/detailed")
def health_detailed(request: Request = None):
    try:
        require_ip_whitelist(request)
    except Exception:
        pass
    details: Dict[str, Any] = {"status": "ok"}
    # DB check
    try:
        from .db import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        details["database"] = "ok"
    except Exception as e:
        details["database"] = f"error: {e}"
        details["status"] = "degraded"

    # Redis check
    try:
        import redis  # type: ignore
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.Redis.from_url(url, decode_responses=True)
        r.ping()
        details["redis"] = "ok"
    except Exception as e:
        details["redis"] = f"error: {e}"
        details["status"] = "degraded"

    # ffmpeg/ffprobe check
    try:
        from make_videos import FFMPEG_BIN
        import subprocess
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, check=True)
        details["ffmpeg"] = "ok"
    except Exception as e:
        details["ffmpeg"] = f"error: {e}"
        details["status"] = "degraded"

    return details


# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# Chunked upload support (ініціалізація/чанки/завершення)
## Legacy chunk upload helpers removed; v1 handles chunk state internally


## Legacy chunk upload endpoint removed; use /api/v1/videos/upload/*


## Legacy chunk upload endpoint removed; use /api/v1/videos/upload/*


## Legacy chunk upload endpoint removed; use /api/v1/videos/upload/*


# Відео: історія та керування
@app.get("/api/videos")
def list_videos(
    current_user = Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
    page: int = 1,
    size: int = 20,
    sort: str = "-created_at",
    include_deleted: bool = False,
    q: str | None = None,
):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}}, status_code=410)
    if page < 1:
        page = 1
    if size < 1 or size > 100:
        size = 20
    query = db.query(Video).filter(Video.user_id == current_user.id)
    if not include_deleted:
        query = query.filter(Video.deleted_at.is_(None))
    if q:
        like = f"%{q}%"
        query = query.filter(Video.video_path.ilike(like))
    if sort.lstrip("-") == "created_at":
        if sort.startswith("-"):
            query = query.order_by(Video.created_at.desc())
        else:
            query = query.order_by(Video.created_at.asc())
    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    data = [
        {
            "id": v.id,
            "video_path": v.video_path,
            "thumb_path": v.thumb_path,
            "duration_sec": v.duration_sec,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "deleted_at": v.deleted_at.isoformat() if v.deleted_at else None,
        }
        for v in items
    ]
    return {"page": page, "size": size, "total": total, "items": data}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: int, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}}, status_code=410)
    v = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not v:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Відео не знайдено"}})
    if v.deleted_at is None:
        from datetime import datetime

        v.deleted_at = datetime.utcnow()
        db.add(v)
        db.commit()
    from .core.logging import audit_log
    audit_log("video.delete", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}


@app.post("/api/videos/bulk_delete")
def bulk_delete(ids: list[int], current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}}, status_code=410)
    if not ids:
        return {"updated": 0}
    from datetime import datetime

    now = datetime.utcnow()
    q = (
        db.query(Video)
        .filter(Video.user_id == current_user.id, Video.id.in_(ids))
        .all()
    )
    updated = 0
    for v in q:
        if v.deleted_at is None:
            v.deleted_at = now
            db.add(v)
            updated += 1
    db.commit()
    from .core.logging import audit_log
    audit_log("video.bulk_delete", getattr(current_user, "id", None), count=updated)
    return {"updated": updated}


@app.post("/api/videos/{video_id}/restore")
def restore_video(video_id: int, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    return JSONResponse({"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}}, status_code=410)
    v = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not v:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Відео не знайдено"}})
    if v.deleted_at is not None:
        v.deleted_at = None
        db.add(v)
        db.commit()
    from .core.logging import audit_log
    audit_log("video.restore", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}
