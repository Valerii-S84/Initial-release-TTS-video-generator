from __future__ import annotations

import types


def test_save_direct_upload_ext_default_and_sidecar_failure(monkeypatch, tmp_path):
    import backend.services.storage_service as ss
    # Fake validator to avoid real IO
    class FV:
        async def avalidate_and_save_upload(self, upload, out_path):
            return {"mime": "video/mp4", "size": 1, "sha256": "x"}
    monkeypatch.setattr(ss, "FileValidator", lambda max_size_bytes: FV())
    # Fake file without filename attribute
    class File:
        filename = None
        async def read(self, n):
            return b"x"
    # Make write_text raise to cover except path
    from pathlib import Path as P
    orig_write_text = P.write_text
    def bad_write(self, *a, **k):
        raise RuntimeError("fail write")
    monkeypatch.setattr(P, "write_text", bad_write)
    try:
        res = ss.save_direct_upload(File(), user_id=1)
        # Await coroutine result
        import asyncio
        res = asyncio.get_event_loop().run_until_complete(res)
        assert res["video_path"].endswith(".mp4")
    finally:
        monkeypatch.setattr(P, "write_text", orig_write_text)

