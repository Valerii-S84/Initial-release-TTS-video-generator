from __future__ import annotations

import os
import time
from pathlib import Path

import backend.services.storage_service as ss


def test_cleanup_expired_uploads_removes_old_files_and_sessions(monkeypatch, tmp_path: Path):
    # Point TMP_DIR to tmp_path for test
    monkeypatch.setattr(ss, "TMP_DIR", tmp_path)
    # Create an old temp file
    old = tmp_path / "U_abc_test.mp4"
    old.write_text("x")
    old_mtime = time.time() - 10_000
    os.utime(old, (old_mtime, old_mtime))
    # Create in-memory stale session
    monkeypatch.setattr(ss, "_redis", None, raising=False)
    ss._uploads_mem["U_sess"] = {"created_at": time.time() - 10_000}
    stats = ss.cleanup_expired_uploads(max_age_seconds=3600)
    assert stats["files"] >= 1 and stats["sessions"] >= 1
    assert not old.exists()


def test_cleanup_handles_exceptions(monkeypatch, tmp_path: Path):
    import types
    # Monkeypatch TMP_DIR.glob to yield a fake path whose stat raises
    import backend.services.storage_service as ss
    class FakePath:
        def stat(self):
            raise OSError("stat failed")
    class FakeTMP:
        def glob(self, pattern: str):
            return [FakePath()]
    monkeypatch.setattr(ss, "TMP_DIR", FakeTMP())
    # Inject bad created_at value to trigger float() ValueError
    monkeypatch.setattr(ss, "_redis", None, raising=False)
    ss._uploads_mem["bad"] = {"created_at": "not-a-float"}
    stats = ss.cleanup_expired_uploads(max_age_seconds=1)
    assert isinstance(stats, dict)


def test_cleanup_outer_except_items_raises(monkeypatch):
    import backend.services.storage_service as ss
    class BadMap:
        def items(self):
            raise RuntimeError("boom")
    monkeypatch.setattr(ss, "_uploads_mem", BadMap(), raising=False)
    stats = ss.cleanup_expired_uploads(max_age_seconds=1)
    assert isinstance(stats, dict)
