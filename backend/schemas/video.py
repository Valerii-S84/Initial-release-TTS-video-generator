from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class VideoItem(BaseModel):
    id: int
    video_path: str
    thumb_path: Optional[str] = None
    duration_sec: Optional[float] = None
    created_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None


class VideoListResponse(BaseModel):
    page: int
    size: int
    total: int
    items: List[VideoItem]


class UploadInitPayload(BaseModel):
    filename: str
    size: int


class GeneratePayload(BaseModel):
    video_path: str | None = None
    quote: str | None = None
    music: str | None = None
    aspect: str | None = None
    voice: str | None = None
    style: str | None = None
    duration_sec: float | None = None
    language: str | None = None
