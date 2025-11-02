from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import backend.services.storage_service as ss


def test_uploads_with_fake_redis(monkeypatch):
    # Fake redis client to exercise get/set/delete branches
    store = {}
    class FakeRedis:
        def get(self, k):
            return store.get(k)
        def set(self, k, v, ex=None):
            store[k] = v
        def delete(self, k):
            store.pop(k, None)
    monkeypatch.setattr(ss, "_redis", FakeRedis(), raising=False)
    # Roundtrip
    up = ss.init_chunk_upload("v.mp4", 3, user_id=1)
    uid = up["upload_id"]
    info = ss._uploads_get(uid)
    assert info and info["received"] == 0
    ss._uploads_set(uid, {**info, "received": 1})
    info2 = ss._uploads_get(uid)
    assert info2["received"] == 1
    ss._uploads_del(uid)
    assert ss._uploads_get(uid) is None


def test_finish_chunk_missing_session_and_file(monkeypatch, tmp_path):
    # Missing session
    with pytest.raises(FileNotFoundError):
        ss.finish_chunk_upload("U_missing", user_id=1)

    # Append not found
    import asyncio
    with pytest.raises(FileNotFoundError):
        asyncio.get_event_loop().run_until_complete(ss.append_chunk("U_missing", 0, b"x", user_id=1))

    # Prepare a real session then remove tmp file before finish to hit missing tmp path
    monkeypatch.setattr(ss, "_redis", None, raising=False)
    up = ss.init_chunk_upload("v.mp4", 3, user_id=1)
    uid = up["upload_id"]
    # write one chunk (simulate completion)
    import asyncio
    asyncio.get_event_loop().run_until_complete(ss.append_chunk(uid, 0, b"abc", user_id=1))
    # Remove tmp file
    info = ss._uploads_get(uid)
    Path(info["tmp_path"]).unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        ss.finish_chunk_upload(uid, user_id=1)


def test_finish_chunk_metadata_write_failure(monkeypatch, tmp_path):
    # Make ffprobe_ok True to reach metadata write
    monkeypatch.setattr(ss, "ffprobe_ok", lambda p: True)
    monkeypatch.setattr(ss, "_redis", None, raising=False)
    up = ss.init_chunk_upload("v.mp4", 3, user_id=1)
    uid = up["upload_id"]
    import asyncio
    asyncio.get_event_loop().run_until_complete(ss.append_chunk(uid, 0, b"abc", user_id=1))
    # Monkeypatch Path.write_text to raise to cover except path
    from pathlib import Path as P
    orig_write_text = P.write_text
    def bad_write(self, *a, **k):
        raise RuntimeError("fail write")
    monkeypatch.setattr(P, "write_text", bad_write)
    try:
        res = ss.finish_chunk_upload(uid, user_id=1)
        assert "video_path" in res
    finally:
        monkeypatch.setattr(P, "write_text", orig_write_text)
