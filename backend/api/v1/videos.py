from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from slowapi.errors import RateLimitExceeded

from ...auth import AuthService
from ...db import get_db
from ...models import Video
from ...core.logging import audit_log
from ...schemas.video import VideoListResponse
from ...services.quota_service import check_quota, increment_usage
from ...core.security import sanitize_filename
from ...core.config import settings
from ..dependencies import STORAGE_INPUT, STORAGE_OUTPUT, MUSIC_DIR, limiter
from ...services.storage_service import save_direct_upload, init_chunk_upload, append_chunk as _append_chunk_storage, finish_chunk_upload
from ...services.video_generator import enqueue_generate, build_generate_cfg
import inspect


async def append_chunk(upload_id: str, offset: int, data: bytes, user_id):
    _res = _append_chunk_storage(upload_id, offset, data, user_id)
    return await _res if inspect.isawaitable(_res) else _res

router = APIRouter(prefix="/videos", tags=["videos"])

Jobs = None  # no direct use; kept for backward compatibility if referenced


@router.get("", response_model=VideoListResponse)
def list_videos(
    current_user = Depends(AuthService.get_current_user),
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


@router.delete("/{video_id}")
def delete_video(video_id: int, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not v:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Відео не знайдено"}})
    if v.deleted_at is None:
        from datetime import datetime

        v.deleted_at = datetime.utcnow()
        db.add(v)
        db.commit()
    audit_log("video.delete", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}


@router.post("/bulk_delete")
def bulk_delete(ids: list[int], current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
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


@router.post("/{video_id}/restore")
def restore_video(video_id: int, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not v:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Відео не знайдено"}})
    if v.deleted_at is not None:
        v.deleted_at = None
        db.add(v)
        db.commit()
    audit_log("video.restore", getattr(current_user, "id", None), video_id=video_id)
    return {"status": "ok"}


# ---------- Upload (single, async) ----------

@limiter.limit(settings.RATE_LIMIT_UPLOAD)
@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    current_user = Depends(AuthService.get_current_user),
    db: Session = Depends(get_db),
):
    check_quota(db, current_user, "upload")
    try:
        meta = await save_direct_upload(file, getattr(current_user, "id", None))
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": str(e)}})
    increment_usage(db, current_user, "upload")
    return meta


# ---------- Chunked upload (init/chunk/finish) ----------

@router.post("/upload/init")
def upload_init(payload: dict, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    check_quota(db, current_user, "upload")
    filename = sanitize_filename(str(payload.get("filename") or "video.mp4"))
    size = int(payload.get("size") or 0)
    if size <= 0:
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": "Некоректний розмір файлу"}})
    res = init_chunk_upload(filename, size, getattr(current_user, "id", None))
    audit_log("upload.init", getattr(current_user, "id", None), upload_id=res["upload_id"], size=size)
    return res


@router.patch("/upload/chunk")
async def upload_chunk(upload_id: str, offset: int, request: Request, current_user = Depends(AuthService.get_current_user)):
    body = await request.body()
    try:
        _res = append_chunk(upload_id, offset, body, getattr(current_user, "id", None))
        res = await _res if inspect.isawaitable(_res) else _res
        if res.get("conflict"):
            return JSONResponse({"error": {"code": "BAD_OFFSET", "message": "Невірний offset"}, "received": res.get("received", 0)}, status_code=409)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Сесію завантаження не знайдено"}})
    except PermissionError:
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Нема доступу до сесії"}})
    audit_log("upload.chunk", getattr(current_user, "id", None), upload_id=upload_id, received=res["received"]) 
    return res


@router.post("/upload/finish")
def upload_finish(upload_id: str, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    try:
        res = finish_chunk_upload(upload_id, getattr(current_user, "id", None))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Сесію завантаження не знайдено"}})
    except PermissionError:
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Нема доступу до сесії"}})
    except ValueError as e:
        msg = "Файл завантажено не повністю" if str(e) == "incomplete" else "ffprobe перевірка не пройшла"
        raise HTTPException(status_code=400, detail={"error": {"code": "VALIDATION_ERROR", "message": msg}})
    increment_usage(db, current_user, "upload")
    audit_log("upload.finish", getattr(current_user, "id", None), video_path=str(res.get("video_path")))
    return res


# ---------- Generate ----------

@limiter.limit(settings.RATE_LIMIT_GENERATE)
@router.post("/generate")
async def generate(request: Request, payload: dict, current_user = Depends(AuthService.get_current_user), db: Session = Depends(get_db)):
    cfg = build_generate_cfg(payload, getattr(current_user, "id", None), STORAGE_INPUT, STORAGE_OUTPUT, MUSIC_DIR)
    check_quota(db, current_user, "generate")
    res = enqueue_generate(cfg)
    increment_usage(db, current_user, "generate")
    return res
