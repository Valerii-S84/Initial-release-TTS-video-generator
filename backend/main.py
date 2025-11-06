from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from .api.dependencies import (
    MUSIC_DIR,
    MUSIC_PREVIEWS,
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    STORAGE_TMP,
    limiter,
)
from .auth import AuthService
from .auth import router as auth_router
from .core.config import settings
from .core.logging import RequestIDMiddleware, bind_job, setup_logging
from .core.metrics import JOBS_QUEUED, PrometheusMiddleware
from .core.observability import setup_sentry
from .core.security import (
    ensure_under,
    require_ip_whitelist,
    sanitize_text,
)
from .core.tracing import inject_trace_to_dict, setup_tracing
from .db import get_db
from .job_storage import get_job_storage
from .models import Video
from .services.quota_service import check_quota, increment_usage
from .services.voice_selector import get_voice_catalog
from .utils.cache import get_cache, get_redis_client
from .utils.etag import ETagStaticFiles
from .utils.validators import FileValidator

BACKEND_HOST = settings.HOST
BACKEND_PORT = settings.PORT
LOGS_DIR = Path(os.getenv("LOGS_DIR", "backend/logs"))

for d in (
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    STORAGE_TMP,
    MUSIC_DIR,
    MUSIC_PREVIEWS,
    LOGS_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

origins = settings.ALLOWED_ORIGINS

setup_logging()
setup_tracing("video-api")
setup_sentry("video-api")
# Security sanity-checks for production
if (settings.APP_ENV.lower() == "prod") and (
    settings.JWT_SECRET_KEY in ("", "change_me", "dev-secret-change-me")
):
    raise RuntimeError(
        "Insecure JWT_SECRET_KEY in production. Please set a strong secret."
    )
if (settings.APP_ENV.lower() == "prod") and (
    not settings.ALLOWED_ORIGINS
    or any(o.strip() == "*" for o in settings.ALLOWED_ORIGINS)
):
    raise RuntimeError("ALLOWED_ORIGINS must be explicitly set (no '*') in production")
app = FastAPI(title="Video Generator API")
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        {"error": {"code": "RATE_LIMIT", "message": str(exc)}}, status_code=429
    ),
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)
if os.getenv("FORCE_HTTPS", "0") == "1":
    app.add_middleware(HTTPSRedirectMiddleware)

# Prod startup checks: ensure Redis reachable (fail-fast)
try:
    if (settings.APP_ENV or "").lower() == "prod":
        _rc = get_redis_client()
        # get_redis_client() raises in prod if ping fails
        if _rc is None:
            raise RuntimeError("Redis client not initialized in production")
except Exception as _e:
    # Raise immediately to stop boot
    raise

# Centralized error handlers
from fastapi.exceptions import RequestValidationError
from starlette import status


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    # Preserve FastAPI shape: {"detail": ...}
    detail = (
        exc.detail
        if isinstance(exc.detail, dict)
        else {"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}}
    )
    return JSONResponse({"detail": detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        {"detail": {"error": {"code": "VALIDATION_ERROR", "message": str(exc)}}},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    # Avoid leaking internals; rely on logs for details
    return JSONResponse(
        {
            "detail": {
                "error": {"code": "INTERNAL_ERROR", "message": "Unexpected error"}
            }
        },
        status_code=500,
    )


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
                    set_header(
                        "strict-transport-security",
                        "max-age=31536000; includeSubDomains; preload",
                    )
                # X-Frame-Options
                set_header("x-frame-options", "DENY")
                # X-Content-Type-Options
                set_header("x-content-type-options", "nosniff")
                # Referrer-Policy
                set_header(
                    "referrer-policy", os.getenv("REFERRER_POLICY", "no-referrer")
                )
                # Cross-Origin policies
                set_header(
                    "cross-origin-opener-policy", os.getenv("COOP", "same-origin")
                )
                set_header(
                    "cross-origin-resource-policy", os.getenv("CORP", "same-origin")
                )
                # Permissions-Policy
                set_header(
                    "permissions-policy",
                    os.getenv(
                        "PERMISSIONS_POLICY", "camera=(), microphone=(), geolocation=()"
                    ),
                )
                # CSP (можна змінити через ENV)
                csp = os.getenv(
                    "CSP",
                    "default-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'",
                )
                set_header("content-security-policy", csp)
            await send(message)

        await self.app(scope, receive, _send)


app.add_middleware(SecurityHeadersMiddleware)

# QuoteAnalyzer demo API under /qa
try:
    from quote_analyzer.config import AnalyzerConfig
    from quote_analyzer.fastapi_integration import get_router as get_qa_router

    qa_cfg = AnalyzerConfig.from_file("configs/analyzer_config.yml")
    app.include_router(get_qa_router(qa_cfg), prefix="/qa")
except Exception as _e:
    # Keep demo running even if analyzer extras aren't installed
    pass
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
app.mount(
    "/media/music_previews",
    ETagStaticFiles(
        directory=str(MUSIC_PREVIEWS), check_dir=False, cache_seconds=86400
    ),
    name="music_previews",
)
app.mount(
    "/media/output_videos",
    ETagStaticFiles(
        directory=str(STORAGE_OUTPUT), check_dir=False, cache_seconds=86400
    ),
    name="output_videos",
)
app.mount(
    "/demo",
    ETagStaticFiles(directory="assets/demo", check_dir=False, cache_seconds=0),
    name="demo",
)

# Auth routes
app.include_router(auth_router)

from .api import api_router

# Підключення нових v1 роутерів (каркас для міграції)
from .api.v1 import admin as v1_admin
from .api.v1 import auth as v1_auth
from .api.v1 import jobs as v1_jobs
from .api.v1 import tts as v1_tts
from .api.v1 import videos as v1_videos

api_router.include_router(v1_auth.router)
api_router.include_router(v1_videos.router)
api_router.include_router(v1_jobs.router)
api_router.include_router(v1_tts.router)
api_router.include_router(v1_admin.router)
app.include_router(api_router)


# OpenAPI customization: add shared error components and common responses
from fastapi.openapi.utils import get_openapi


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        routes=app.routes,
    )
    comp = schema.setdefault("components", {})
    responses = comp.setdefault("responses", {})

    def add_resp(code: str, desc: str):
        responses[f"Error{code}"] = {
            "description": desc,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorEnvelope"}
                }
            },
        }

    add_resp("400", "Bad Request")
    add_resp("401", "Unauthorized")
    add_resp("403", "Forbidden")
    add_resp("404", "Not Found")
    add_resp("409", "Conflict")
    add_resp("422", "Unprocessable Entity")
    add_resp("429", "Too Many Requests")
    add_resp("500", "Internal Server Error")

    # Attach common error responses to operations if not explicitly provided
    for path_item in schema.get("paths", {}).values():
        for method, op in list(path_item.items()):
            if method.lower() not in (
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "options",
                "head",
            ):
                continue
            r = op.setdefault("responses", {})
            for code in ("400", "401", "403", "404", "422"):
                if code not in r:
                    r[code] = {"$ref": f"#/components/responses/Error{code}"}

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


# Jobs store (Redis with in-memory fallback)
Jobs = get_job_storage()


def _progress_cb(job_id: str):
    def _cb(step: str, progress: float, message: str):
        Jobs.update(
            job_id,
            {
                "status": "processing",
                "step": step,
                "progress": float(progress),
                "message": message,
            },
        )

    return _cb


@app.get("/api/config")
def get_config(
    current_user=Depends(AuthService.get_current_user), request: Request = None
):
    # Кеш готової відповіді (голоси + музика)
    try:
        cache = get_cache()
        force_refresh = False
        try:
            force_refresh = (
                str(request.query_params.get("refresh", "0")) == "1"
                if request
                else False
            )
        except Exception:
            force_refresh = False
        _music = None if force_refresh else cache.get_json("cfg:music")
        _voices = None if force_refresh else cache.get_json("cfg:voices")
        if (not force_refresh) and (_music is not None and _voices is not None):
            return {
                "aspects": ["9:16", "1:1", "16:9"],
                "styles": ["motivational", "historical", "greeting"],
                "voices": _voices,
                "music": _music,
                "limits": {
                    "maxQuoteLen": 500,
                    "durationSec": [10, 30],
                    "languages": ["uk", "en", "de"],
                },
            }
    except Exception:
        pass
    # Music listing + previews
    music_items = []
    for p in sorted(MUSIC_DIR.glob("*.mp3")):
        prev = MUSIC_PREVIEWS / (p.stem + "_preview.mp3")
        music_items.append(
            {
                "file": p.name,
                "preview": prev.name if prev.exists() else None,
                "label": p.stem,
            }
        )
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
        "limits": {
            "maxQuoteLen": 500,
            "durationSec": [10, 30],
            "languages": ["uk", "en", "de"],
        },
    }


@limiter.limit("20/hour")
@app.post("/api/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    # Квота на завантаження
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/upload"}},
        status_code=410,
    )
    ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    vid = f"{uuid.uuid4().hex}{ext}"
    out_path = STORAGE_INPUT / vid
    # Validate and save
    fv = FileValidator(max_size_bytes=500 * 1024 * 1024)
    try:
        meta = await fv.avalidate_and_save_upload(file, out_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "VALIDATION_ERROR", "message": str(e)}},
        )
    # Write sidecar metadata with user_id for traceability
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    try:
        sidecar.write_text(
            json.dumps(
                {"user_id": getattr(current_user, "id", None), **meta},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    # Зараховуємо використання квоти лише після успішного збереження
    increment_usage(db, current_user, "upload")
    return {"video_path": str(out_path), **meta}


@limiter.limit("5/minute")
@app.post("/api/generate")
async def generate(
    request: Request,
    payload: Dict[str, Any],
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/generate"}},
        status_code=410,
    )
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

    job_id = f"J_{uuid.uuid4().hex[:8]}"
    Jobs.set(
        job_id,
        {
            "status": "queued",
            "step": None,
            "progress": 0.0,
            "message": "In queue",
            "result": None,
        },
    )
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
    from .workers.celery_app import task_generate
    task = task_generate.delay(cfg)  # type: ignore[arg-type]
    Jobs.update(job_id, {"celery_id": task.id})
    # Збільшуємо лічильник квот після успішної постановки задачі
    increment_usage(db, current_user, "generate")
    return {"job_id": job_id, "status": "queued", "task_id": task.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, current_user=Depends(AuthService.get_current_user)):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/jobs/*"}},
        status_code=410,
    )


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, current_user=Depends(AuthService.get_current_user)):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/jobs/*"}},
        status_code=410,
    )


@app.post("/api/music/refresh_previews")
def refresh_music_previews(
    current_user=Depends(AuthService.get_current_user), request: Request = None
):
    # IP whitelist (якщо задано)
    try:
        require_ip_whitelist(request)
    except Exception:
        raise
    # Create ~10s previews for all mp3 in MUSIC_DIR at +10s offset
    from make_videos import FFMPEG_BIN, run

    created = []
    for p in sorted(MUSIC_DIR.glob("*.mp3")):
        out = MUSIC_PREVIEWS / f"{p.stem}_preview.mp3"
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-ss",
            "10",
            "-t",
            "10",
            "-i",
            str(p),
            "-acodec",
            "libmp3lame",
            str(out),
        ]
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
@limiter.exempt
@app.get("/health")
def health():
    return {"status": "ok"}


@limiter.exempt
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
        import subprocess

        from make_videos import FFMPEG_BIN

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
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
    page: int = 1,
    size: int = 20,
    sort: str = "-created_at",
    include_deleted: bool = False,
    q: str | None = None,
):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}},
        status_code=410,
    )
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
def delete_video(
    video_id: int,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}},
        status_code=410,
    )
    v = (
        db.query(Video)
        .filter(Video.id == video_id, Video.user_id == current_user.id)
        .first()
    )
    if not v:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Відео не знайдено"}},
        )
    if v.deleted_at is None:
        from datetime import datetime

        v.deleted_at = datetime.utcnow()
        db.add(v)
        db.commit()
    from .core.logging import audit_log

    audit_log("video.delete", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}


@app.post("/api/videos/bulk_delete")
def bulk_delete(
    ids: list[int],
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}},
        status_code=410,
    )
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
def restore_video(
    video_id: int,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    return JSONResponse(
        {"error": {"code": "DEPRECATED", "message": "Use /api/v1/videos/*"}},
        status_code=410,
    )
    v = (
        db.query(Video)
        .filter(Video.id == video_id, Video.user_id == current_user.id)
        .first()
    )
    if not v:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Відео не знайдено"}},
        )
    if v.deleted_at is not None:
        v.deleted_at = None
        db.add(v)
        db.commit()
    from .core.logging import audit_log

    audit_log("video.restore", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}
