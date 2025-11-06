from __future__ import annotations

import inspect

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ...api.errors import (
    FORBIDDEN,
    MSG_CHUNK_TOO_LARGE,
    MSG_UPLOAD_FORBIDDEN,
    VALIDATION_ERROR,
    err,
)
from ...auth import AuthService
from ...core.config import settings
from ...core.logging import audit_log
from ...core.security import sanitize_filename
from ...db import get_db
from ...models import Video
from ...schemas.errors import ErrorEnvelope
from ...schemas.video import GeneratePayload, UploadInitPayload, VideoListResponse
from ...services.quota_service import check_quota, increment_usage
from ...utils.cache import get_redis_client
from ...services.storage_service import append_chunk as _append_chunk_storage
from ...services.storage_service import (
    finish_chunk_upload,
    get_upload_info,
    init_chunk_upload,
    save_direct_upload,
)
from ...services.video_generator import build_generate_cfg, enqueue_generate
import hashlib
import json
from ..dependencies import MUSIC_DIR, STORAGE_INPUT, STORAGE_OUTPUT, limiter


async def append_chunk(upload_id: str, offset: int, data: bytes, user_id):
    _res = _append_chunk_storage(upload_id, offset, data, user_id)
    return await _res if inspect.isawaitable(_res) else _res


router = APIRouter(prefix="/videos", tags=["videos"])

Jobs = None  # no direct use; kept for backward compatibility if referenced


@limiter.limit("100/minute")
@router.get(
    "",
    response_model=VideoListResponse,
    responses={
        200: {"description": "Ð¡Ð¿Ð¸ÑÐ¾Ðº Ð²Ñ–Ð´ÐµÐ¾"},
        401: {"model": ErrorEnvelope, "description": "ÐÐµÐ°Ð²Ñ‚Ð¾Ñ€Ð¸Ð·Ð¾Ð²Ð°Ð½Ð¾"},
    },
)
def list_videos(
    request: Request,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
    page: int = 1,
    size: int = 20,
    sort: str = "-created_at",
    include_deleted: bool = False,
    q: str | None = None,
):
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


@router.delete(
    "/{video_id}",
    responses={
        200: {"description": "Ð’Ð¸Ð´Ð°Ð»ÐµÐ½Ð¾ Ð°Ð±Ð¾ Ð²Ð¶Ðµ Ð²Ð¸Ð´Ð°Ð»ÐµÐ½Ð¾"},
        404: {"model": ErrorEnvelope, "description": "Ð’Ñ–Ð´ÐµÐ¾ Ð½Ðµ Ð·Ð½Ð°Ð¹Ð´ÐµÐ½Ð¾"},
    },
)
def delete_video(
    video_id: int,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    v = (
        db.query(Video)
        .filter(Video.id == video_id, Video.user_id == current_user.id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail=err("NOT_FOUND"))
    if v.deleted_at is None:
        from datetime import datetime

        v.deleted_at = datetime.utcnow()
        db.add(v)
        db.commit()
    audit_log("video.delete", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}


@router.post(
    "/bulk_delete",
    responses={
        200: {"description": "ÐšÑ–Ð»ÑŒÐºÑ–ÑÑ‚ÑŒ Ð¾Ð½Ð¾Ð²Ð»ÐµÐ½Ð¸Ñ… Ð·Ð°Ð¿Ð¸ÑÑ–Ð²"},
        401: {"model": ErrorEnvelope},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "array", "items": {"type": "integer"}},
                    "example": [1, 2, 3],
                }
            },
        }
    },
)
def bulk_delete(
    ids: list[int],
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    if not ids:
        return {"updated": 0}
    from datetime import datetime

    now = datetime.utcnow()
    qset = (
        db.query(Video)
        .filter(Video.user_id == current_user.id, Video.id.in_(ids))
        .all()
    )
    updated = 0
    for v in qset:
        if v.deleted_at is None:
            v.deleted_at = now
            db.add(v)
            updated += 1
    db.commit()
    audit_log("video.bulk_delete", getattr(current_user, "id", None), count=updated)
    return {"updated": updated}


@router.post(
    "/{video_id}/restore",
    responses={
        200: {"description": "Ð—Ð°Ð¿Ð¸Ñ Ð²Ñ–Ð´Ð½Ð¾Ð²Ð»ÐµÐ½Ð¾"},
        404: {"model": ErrorEnvelope, "description": "Ð’Ñ–Ð´ÐµÐ¾ Ð½Ðµ Ð·Ð½Ð°Ð¹Ð´ÐµÐ½Ð¾"},
    },
)
def restore_video(
    video_id: int,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    v = (
        db.query(Video)
        .filter(Video.id == video_id, Video.user_id == current_user.id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail=err("NOT_FOUND"))
    if v.deleted_at is not None:
        v.deleted_at = None
        db.add(v)
        db.commit()
    audit_log("video.restore", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}


# ---------- Upload (single, async) ----------


@limiter.limit(settings.RATE_LIMIT_UPLOAD)
@router.post(
    "/upload",
    responses={
        200: {"description": "ÐœÐµÑ‚Ð° Ð·Ð°Ð²Ð°Ð½Ñ‚Ð°Ð¶ÐµÐ½Ð¾Ð³Ð¾ Ñ„Ð°Ð¹Ð»Ñƒ"},
        400: {"model": ErrorEnvelope, "description": "ÐŸÐ¾Ð¼Ð¸Ð»ÐºÐ° Ð²Ð°Ð»Ñ–Ð´Ð°Ñ†Ñ–Ñ— Ñ„Ð°Ð¹Ð»Ñƒ"},
        413: {"model": ErrorEnvelope, "description": "Ð¤Ð°Ð¹Ð» Ð½Ð°Ð´Ñ‚Ð¾ Ð²ÐµÐ»Ð¸ÐºÐ¸Ð¹"},
    },
)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    check_quota(db, current_user, "upload")
    try:
        meta = await save_direct_upload(file, getattr(current_user, "id", None))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=err(VALIDATION_ERROR, None, hint="Очікується додатний розмір файлу"),
        )
    increment_usage(db, current_user, "upload")
    return meta


# ---------- Chunked upload (init/chunk/finish) ----------


@router.post(
    "/upload/init",
    responses={
        200: {
            "description": "Ð¡ÐµÑÑ–Ñ Ð·Ð°Ð²Ð°Ð½Ñ‚Ð°Ð¶ÐµÐ½Ð½Ñ ÑÑ‚Ð²Ð¾Ñ€ÐµÐ½Ð°",
            "content": {
                "application/json": {
                    "example": {
                        "upload_id": "U_abcd1234",
                        "token": "t_123",
                        "size": 12345,
                    }
                }
            },
        },
        400: {"model": ErrorEnvelope, "description": "ÐÐµÐ²Ð°Ð»Ñ–Ð´Ð½Ð¸Ð¹ Ñ€Ð¾Ð·Ð¼Ñ–Ñ€ Ñ„Ð°Ð¹Ð»Ñƒ"},
        401: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope, "description": "ÐŸÐ¾Ð¼Ð¸Ð»ÐºÐ° Pydantic"},
    },
)
def upload_init(
    payload: UploadInitPayload,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    check_quota(db, current_user, "upload")
    filename = sanitize_filename(str(payload.filename or "video.mp4"))
    size = int(payload.size or 0)
    if size <= 0:
        raise HTTPException(
            status_code=400,
            detail=err(VALIDATION_ERROR, None, hint="?????????? ???????? ?????? ?????"),
        )
    res = init_chunk_upload(filename, size, getattr(current_user, "id", None))
    audit_log(
        "upload.init",
        getattr(current_user, "id", None),
        upload_id=res["upload_id"],
        size=size,
    )
    return res


@router.patch(
    "/upload/chunk",
    responses={
        200: {"description": "Ð§Ð°ÑÑ‚Ð¸Ð½Ð° Ð¿Ñ€Ð¸Ð¹Ð½ÑÑ‚Ð°"},
        400: {"model": ErrorEnvelope, "description": "Chunk Ð½Ð°Ð´Ñ‚Ð¾ Ð²ÐµÐ»Ð¸ÐºÐ¸Ð¹"},
        403: {"model": ErrorEnvelope, "description": "Ð¢Ð¾ÐºÐµÐ½ Ð²Ñ–Ð´ÑÑƒÑ‚Ð½Ñ–Ð¹/Ð½ÐµÐ²Ñ–Ñ€Ð½Ð¸Ð¹"},
        404: {"model": ErrorEnvelope, "description": "Ð¡ÐµÑÑ–ÑŽ Ð½Ðµ Ð·Ð½Ð°Ð¹Ð´ÐµÐ½Ð¾"},
        409: {
            "description": "ÐÐµÐ²Ñ–Ñ€Ð½Ð¸Ð¹ offset",
            "content": {
                "application/json": {
                    "example": {
                        "error": {"code": "BAD_OFFSET", "message": "Bad offset"},
                        "received": 0,
                    }
                }
            },
        },
        422: {"model": ErrorEnvelope},
    },
)
async def upload_chunk(
    upload_id: str,
    offset: int,
    request: Request,
    current_user=Depends(AuthService.get_current_user),
):
    body = await request.body()
    if len(body) > settings.MAX_CHUNK_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=err(
                VALIDATION_ERROR,
                MSG_CHUNK_TOO_LARGE,
                hint=f"Максимальний розмір: {settings.MAX_CHUNK_SIZE_MB} MB",
            ),
        )
    if settings.ENFORCE_UPLOAD_TOKEN:
        token = request.query_params.get("token")
        info = get_upload_info(upload_id) or {}
        if not token or token != str(info.get("token")):
            raise HTTPException(
                status_code=403,
                detail=err(
                    FORBIDDEN,
                    MSG_UPLOAD_FORBIDDEN,
                    hint="Додайте параметр token= з відповіді upload_init",
                ),
            )
    try:
        _res = append_chunk(upload_id, offset, body, getattr(current_user, "id", None))
        res = await _res if inspect.isawaitable(_res) else _res
        if res.get("conflict"):
            return JSONResponse(
                {
                    "error": {
                        "code": "BAD_OFFSET",
                        "??????????? offset": "ÐÐµÐ²Ñ–Ñ€Ð½Ð¸Ð¹ offset",
                    },
                    "received": res.get("received", 0),
                },
                status_code=409,
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "NOT_FOUND",
                    "message": "Ð¡ÐµÑÑ–ÑŽ Ð·Ð°Ð²Ð°Ð½Ñ‚Ð°Ð¶ÐµÐ½Ð½Ñ Ð½Ðµ Ð·Ð½Ð°Ð¹Ð´ÐµÐ½Ð¾",
                }
            },
        )
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail={"error": {"code": "FORBIDDEN", "message": "ÐÐµÐ¼Ð° Ð´Ð¾ÑÑ‚ÑƒÐ¿Ñƒ Ð´Ð¾ ÑÐµÑÑ–Ñ—"}},
        )
    audit_log(
        "upload.chunk",
        getattr(current_user, "id", None),
        upload_id=upload_id,
        received=res["received"],
    )
    return res


@router.post(
    "/upload/finish",
    responses={
        200: {"description": "Ð¤Ð°Ð¹Ð» Ð·Ñ–Ð±Ñ€Ð°Ð½Ð¾"},
        400: {"model": ErrorEnvelope, "description": "ÐŸÐ¾Ð¼Ð¸Ð»ÐºÐ° Ð²Ð°Ð»Ñ–Ð´Ð°Ñ†Ñ–Ñ—/ffprobe"},
        403: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
    },
)
def upload_finish(
    upload_id: str,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    try:
        res = finish_chunk_upload(upload_id, getattr(current_user, "id", None))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "NOT_FOUND",
                    "message": "Ð¡ÐµÑÑ–ÑŽ Ð·Ð°Ð²Ð°Ð½Ñ‚Ð°Ð¶ÐµÐ½Ð½Ñ Ð½Ðµ Ð·Ð½Ð°Ð¹Ð´ÐµÐ½Ð¾",
                }
            },
        )
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail={"error": {"code": "FORBIDDEN", "message": "ÐÐµÐ¼Ð° Ð´Ð¾ÑÑ‚ÑƒÐ¿Ñƒ Ð´Ð¾ ÑÐµÑÑ–Ñ—"}},
        )
    except ValueError as e:
        msg = (
            "???????????? ?? ?????????"
            if str(e) == "incomplete"
            else "????????? ffprobe ?? ????????"
        )
        raise HTTPException(
            status_code=400,
            detail=err(VALIDATION_ERROR, None, hint="Очікується додатний розмір файлу"),
        )
    increment_usage(db, current_user, "upload")
    audit_log(
        "upload.finish",
        getattr(current_user, "id", None),
        video_path=str(res.get("video_path")),
    )
    return res


# ---------- Generate ----------


@limiter.limit(settings.RATE_LIMIT_GENERATE)
@router.post(
    "/generate",
    responses={
        200: {"description": "Ð—Ð°Ð´Ð°Ñ‡Ñƒ Ð¿Ð¾ÑÑ‚Ð°Ð²Ð»ÐµÐ½Ð¾ Ñƒ Ñ‡ÐµÑ€Ð³Ñƒ"},
        400: {"model": ErrorEnvelope, "description": "ÐÐµÐ²Ð°Ð»Ñ–Ð´Ð½Ñ– Ð´Ð°Ð½Ñ–"},
        401: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
    },
)
async def generate(
    request: Request,
    payload: GeneratePayload,
    current_user=Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    # Idempotency support (scoped by user + payload fingerprint)
    idem_key = request.headers.get("Idempotency-Key")
    user_id = getattr(current_user, "id", None) or "anon"
    # We compute fingerprint after we normalize payload below
    # Log ignored (unknown) fields for migration to `extra='forbid'` later
    try:
        raw = await request.json()
        known = set(payload.__class__.model_fields.keys())
        extras = sorted(
            k for k in (raw.keys() if isinstance(raw, dict) else []) if k not in known
        )
        if extras:
            audit_log(
                "videos.generate.ignored_fields",
                getattr(current_user, "id", None),
                extras=extras,
            )
    except Exception:
        pass
    data = payload.model_dump(exclude_none=True)
    # Compute stable fingerprint of request payload
    try:
        fingerprint = hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    except Exception:
        fingerprint = None  # fall back; idempotency will key only by header
    if idem_key:
        try:
            rc = get_redis_client()
            if rc is not None:
                idem_rc_key = f"idemp:{user_id}:{idem_key}"
                cached = rc.get(idem_rc_key)
                if cached:
                    try:
                        blob = json.loads(cached)
                        # If fingerprint matches, return cached response
                        if not fingerprint or (blob.get("fp") == fingerprint):
                            res_cached = blob.get("res")
                            if isinstance(res_cached, dict):
                                return res_cached
                    except Exception:
                        pass
        except Exception:
            pass
    cfg = build_generate_cfg(
        data,
        getattr(current_user, "id", None),
        STORAGE_INPUT,
        STORAGE_OUTPUT,
        MUSIC_DIR,
    )
    if idem_key:
        cfg["idempotency_key"] = idem_key  # type: ignore[index]
    check_quota(db, current_user, "generate")
    try:
        res = enqueue_generate(cfg)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        is_prod = (settings.APP_ENV or "").lower() == "prod"
        code = "BROKER_UNAVAILABLE" if is_prod else "GENERATION_FAILED"
        msg = "Task broker is unavailable" if is_prod else str(exc)
        raise HTTPException(
            status_code=(503 if is_prod else 500),
            detail={"error": {"code": code, "message": msg}},
        )
    # Cache job result mapping for idempotency (store response + fingerprint)
    try:
        if idem_key and isinstance(res, dict):
            rc = get_redis_client()
            if rc is not None:
                idem_rc_key = f"idemp:{user_id}:{idem_key}"
                blob = {"fp": fingerprint, "res": res}
                rc.set(idem_rc_key, json.dumps(blob, ensure_ascii=False), ex=24 * 3600)
    except Exception:
        pass
    increment_usage(db, current_user, "generate")
    return res
