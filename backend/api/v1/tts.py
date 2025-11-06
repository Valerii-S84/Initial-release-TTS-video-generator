from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ...api.errors import RATE_LIMIT, SERVICE_UNAVAILABLE, VALIDATION_ERROR, err
from ...auth import AuthService
from ...core.config import settings
from ...services.tts_service import ElevenLabsTTS
from ...services.voice_selector import get_voice_catalog
from ..dependencies import limiter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/tts", tags=["tts"])


class PreviewPayload(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    voice_name: Optional[str] = None
    language: Optional[str] = None


@limiter.limit("30/minute")
@router.post("/preview")
def preview_tts(
    payload: PreviewPayload,
    request,  # SlowAPI limiter requires request arg
    current_user=Depends(AuthService.get_current_user),
):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail=err(VALIDATION_ERROR, None, hint="Порожній текст"))

    # Heuristic: cap text to avoid long preview (~1–1.5s)
    snippet = text[:180]

    # Resolve voice_id
    catalog = get_voice_catalog()
    voice_name = payload.voice_name or "Rachel"
    voice = catalog.get(voice_name) or catalog.get("Rachel")
    voice_id = voice.voice_id  # type: ignore[assignment]

    # Cache key (per user+voice+lang+text)
    fp = hashlib.sha256(f"{getattr(current_user,'id', 'anon')}|{voice_id}|{payload.language or 'en'}|{snippet}".encode("utf-8")).hexdigest()

    # Synthesize to temp file
    try:
        tts = ElevenLabsTTS()
    except Exception:
        raise HTTPException(status_code=503, detail=err(SERVICE_UNAVAILABLE, "TTS сервіс недоступний"))

    out_dir = Path("backend/tmp/previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"prev_{fp}.mp3"
    try:
        if not out_file.exists():
            tts.synthesize_to_file(snippet, out_file, voice_id=voice_id)
        data = out_file.read_bytes()
        audio_b64 = base64.b64encode(data).decode("ascii")
        return {"voice": voice.name, "voice_id": voice_id, "audio_base64": audio_b64}
    except Exception as e:
        raise HTTPException(status_code=503, detail=err(SERVICE_UNAVAILABLE, str(e)))
