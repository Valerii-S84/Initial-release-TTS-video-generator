from __future__ import annotations

from pathlib import Path

import backend.utils.validators as v


def test_ffprobe_ok_true(monkeypatch, tmp_path: Path):
    p = tmp_path / "v.mp4"; p.write_text("x")
    class Res:
        returncode = 0
        stdout = "1.23\n"
    monkeypatch.setattr(v.subprocess, "run", lambda *a, **k: Res())
    assert v.ffprobe_ok(p) is True


def test_ffprobe_ok_false(monkeypatch, tmp_path: Path):
    p = tmp_path / "v.mp4"; p.write_text("x")
    class Res2:
        returncode = 1
        stdout = ""
    monkeypatch.setattr(v.subprocess, "run", lambda *a, **k: Res2())
    assert v.ffprobe_ok(p) is False


def test_ffprobe_exception_returns_false(monkeypatch, tmp_path: Path):
    p = tmp_path / "v.mp4"; p.write_text("x")
    def boom(*a, **k):
        raise OSError("no ffprobe")
    monkeypatch.setattr(v.subprocess, "run", boom)
    assert v.ffprobe_ok(p) is False
