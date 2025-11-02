from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import aiofiles  # type: ignore

from ..core.config import settings
from ..utils.cache import get_redis_client
from ..utils.validators import FileValidator, ffprobe_ok


INPUT_DIR = Path(settings.STORAGE_INPUT)
TMP_DIR = Path(settings.STORAGE_TMP)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


_uploads_mem: Dict[str, Dict[str, Any]] = {}
_redis = None
try:
    _redis = get_redis_client()
except Exception:
    _redis = None


def _uploads_key(upload_id: str) -> str:
    return f"upload:{upload_id}"


def _uploads_get(upload_id: str) -> Optional[Dict[str, Any]]:
    if _redis is not None:
        raw = _redis.get(_uploads_key(upload_id))
        return json.loads(raw) if raw else None
    return _uploads_mem.get(upload_id)


def _uploads_set(upload_id: str, data: Dict[str, Any], ttl: int = 24 * 3600) -> None:
    if _redis is not None:
        _redis.set(_uploads_key(upload_id), json.dumps(data, ensure_ascii=False), ex=ttl)
    else:
        _uploads_mem[upload_id] = data


def _uploads_del(upload_id: str) -> None:
    if _redis is not None:
        _redis.delete(_uploads_key(upload_id))
    else:
        _uploads_mem.pop(upload_id, None)


async def save_direct_upload(file, user_id: Optional[int]) -> Dict[str, Any]:
    ext = Path(file.filename).suffix.lower() if getattr(file, "filename", None) else ".mp4"
    out_path = INPUT_DIR / f"{uuid.uuid4().hex}{ext}"
    fv = FileValidator(max_size_bytes=settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    meta = await fv.avalidate_and_save_upload(file, out_path)
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    try:
        sidecar.write_text(json.dumps({"user_id": user_id, **meta}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return {"video_path": str(out_path), **meta}


def init_chunk_upload(filename: str, size: int, user_id: Optional[int]) -> Dict[str, Any]:
    upload_id = f"U_{uuid.uuid4().hex[:10]}"
    token = uuid.uuid4().hex
    tmp_path = TMP_DIR / f"{upload_id}_{Path(filename).name}"
    info = {"filename": filename, "size": size, "received": 0, "tmp_path": str(tmp_path), "user_id": user_id, "token": token, "created_at": __import__('time').time()}
    _uploads_set(upload_id, info, ttl=24 * 3600)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    Path(str(tmp_path)).touch()
    return {"upload_id": upload_id, "tmp_path": str(tmp_path), "token": token}


async def append_chunk(upload_id: str, offset: int, data: bytes, user_id: Optional[int]) -> Dict[str, Any]:
    info = _uploads_get(upload_id)
    if not info:
        raise FileNotFoundError("upload session not found")
    if info.get("user_id") != user_id:
        raise PermissionError("not owner")
    if offset != int(info.get("received", 0)):
        return {"conflict": True, "received": info.get("received", 0)}
    tmp_path = Path(info["tmp_path"])
    async with aiofiles.open(tmp_path, "rb+") as f:
        await f.seek(offset)
        await f.write(data)
    info["received"] = offset + len(data)
    _uploads_set(upload_id, info)
    return {"received": info["received"], "size": info["size"]}


def finish_chunk_upload(upload_id: str, user_id: Optional[int]) -> Dict[str, Any]:
    info = _uploads_get(upload_id)
    if not info:
        raise FileNotFoundError("upload session not found")
    if info.get("user_id") != user_id:
        raise PermissionError("not owner")
    if int(info.get("received", 0)) != int(info.get("size", -1)):
        raise ValueError("incomplete")
    tmp_path = Path(info["tmp_path"]).resolve()
    if not tmp_path.exists():
        raise FileNotFoundError("tmp file missing")
    if not ffprobe_ok(tmp_path):
        raise ValueError("ffprobe failed")
    h = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    ext = Path(info.get("filename") or "video.mp4").suffix or ".mp4"
    final_path = INPUT_DIR / f"{uuid.uuid4().hex}{ext}"
    tmp_path.replace(final_path)
    try:
        (final_path.with_suffix(final_path.suffix + ".json")).write_text(json.dumps({
            "user_id": user_id,
            "size": info.get("size"),
            "sha256": h.hexdigest(),
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    _uploads_del(upload_id)
    return {"video_path": str(final_path), "size": info.get("size"), "sha256": h.hexdigest()}


def get_upload_info(upload_id: str) -> Optional[Dict[str, Any]]:
    return _uploads_get(upload_id)


def cleanup_expired_uploads(max_age_seconds: int = 24 * 3600) -> Dict[str, int]:
    import time
    removed_files = 0
    removed_sessions = 0
    now = time.time()
    # Clean temp files by mtime
    for p in TMP_DIR.glob("U_*_*"):
        try:
            if now - p.stat().st_mtime > max_age_seconds:
                p.unlink(missing_ok=True)
                removed_files += 1
        except Exception:
            continue
    # Clean in-memory sessions older than max_age
    try:
        for key, info in list(_uploads_mem.items()):
            try:
                if now - float(info.get("created_at", now)) > max_age_seconds:
                    _uploads_mem.pop(key, None)
                    removed_sessions += 1
            except Exception:
                continue
    except Exception:
        pass
    return {"files": removed_files, "sessions": removed_sessions}
