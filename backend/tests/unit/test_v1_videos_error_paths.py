from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app


def _override_auth(uid=1):
    from backend.auth import AuthService
    app.dependency_overrides[AuthService.get_current_user] = lambda: SimpleNamespace(id=uid)


def test_upload_init_size_validation_error(monkeypatch):
    _override_auth(42)
    client = TestClient(app)
    r = client.post("/api/v1/videos/upload/init", json={"filename": "x.mp4", "size": 0})
    assert r.status_code == 400
    assert r.json().get("detail", {}).get("error", {}).get("code") == "VALIDATION_ERROR"


def test_upload_chunk_not_found_and_forbidden(monkeypatch):
    import backend.api.v1.videos as v1_vid
    _override_auth(9)
    client = TestClient(app)

    # Not found branch
    monkeypatch.setattr(v1_vid, "append_chunk", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    r404 = client.patch("/api/v1/videos/upload/chunk", params={"upload_id": "U_missing", "offset": 0}, data=b"abc")
    assert r404.status_code == 404
    assert r404.json().get("detail", {}).get("error", {}).get("code") == "NOT_FOUND"

    # Forbidden branch
    monkeypatch.setattr(v1_vid, "append_chunk", lambda *a, **k: (_ for _ in ()).throw(PermissionError()))
    r403 = client.patch("/api/v1/videos/upload/chunk", params={"upload_id": "U_forbid", "offset": 0}, data=b"abc")
    assert r403.status_code == 403
    assert r403.json().get("detail", {}).get("error", {}).get("code") == "FORBIDDEN"


def test_upload_finish_exceptions_map_to_status(monkeypatch):
    import backend.api.v1.videos as v1_vid
    _override_auth(13)
    client = TestClient(app)

    # 404
    monkeypatch.setattr(v1_vid, "finish_chunk_upload", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    r404 = client.post("/api/v1/videos/upload/finish", params={"upload_id": "U_nf"})
    assert r404.status_code == 404
    assert r404.json().get("detail", {}).get("error", {}).get("code") == "NOT_FOUND"

    # 403
    monkeypatch.setattr(v1_vid, "finish_chunk_upload", lambda *a, **k: (_ for _ in ()).throw(PermissionError()))
    r403 = client.post("/api/v1/videos/upload/finish", params={"upload_id": "U_fbd"})
    assert r403.status_code == 403
    assert r403.json().get("detail", {}).get("error", {}).get("code") == "FORBIDDEN"

    # 400 ValueError("incomplete")
    monkeypatch.setattr(v1_vid, "finish_chunk_upload", lambda *a, **k: (_ for _ in ()).throw(ValueError("incomplete")))
    r400_incomplete = client.post("/api/v1/videos/upload/finish", params={"upload_id": "U_inc"})
    assert r400_incomplete.status_code == 400
    assert r400_incomplete.json().get("detail", {}).get("error", {}).get("code") == "VALIDATION_ERROR"

    # 400 other ValueError
    monkeypatch.setattr(v1_vid, "finish_chunk_upload", lambda *a, **k: (_ for _ in ()).throw(ValueError("ffprobe failed")))
    r400_other = client.post("/api/v1/videos/upload/finish", params={"upload_id": "U_val"})
    assert r400_other.status_code == 400
    assert r400_other.json().get("detail", {}).get("error", {}).get("code") == "VALIDATION_ERROR"


def test_list_videos_clamps_page_and_size(monkeypatch):
    # page < 1 -> 1; size > 100 -> 20
    _override_auth(21)
    client = TestClient(app)
    r = client.get("/api/v1/videos", params={"page": 0, "size": 1000})
    assert r.status_code == 200
    js = r.json()
    assert js.get("page") == 1 and js.get("size") == 20


def test_upload_video_validation_error(monkeypatch):
    import backend.api.v1.videos as v1_vid
    _override_auth(22)
    client = TestClient(app)
    # Force save_direct_upload to raise
    monkeypatch.setattr(v1_vid, "save_direct_upload", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    r = client.post("/api/v1/videos/upload", files={"file": ("x.mp4", b"abc", "video/mp4")})
    assert r.status_code == 400
    assert r.json().get("detail", {}).get("error", {}).get("code") == "VALIDATION_ERROR"


def test_upload_chunk_success_happy_path(monkeypatch):
    import backend.api.v1.videos as v1_vid
    _override_auth(23)
    client = TestClient(app)
    # Return successful append result
    monkeypatch.setattr(v1_vid, "append_chunk", lambda *a, **k: {"received": 3, "size": 3})
    r = client.patch("/api/v1/videos/upload/chunk", params={"upload_id": "U_ok", "offset": 0}, data=b"abc")
    assert r.status_code == 200
    assert r.json().get("received") == 3
