from __future__ import annotations

import io
from types import SimpleNamespace
from pathlib import Path

import pytest

import backend.utils.validators as validators


class FakeUpload:
    def __init__(self, data: bytes, filename: str = "v.mp4"):
        self.file = io.BytesIO(data)
        self.filename = filename


def test_sync_magic_exception_fallback_and_unsupported(monkeypatch, tmp_path: Path):
    class BadMagic:
        def __init__(self, mime=True):
            pass
        def from_buffer(self, b: bytes) -> str:
            raise RuntimeError("boom")
    monkeypatch.setattr(validators, "magic", SimpleNamespace(Magic=BadMagic, from_buffer=BadMagic().from_buffer))
    fv = validators.FileValidator(max_size_bytes=100)
    with pytest.raises(ValueError):
        fv.validate_and_save_upload(FakeUpload(b"0" * 10), tmp_path / "x.mp4")


def test_sync_too_large_triggers_error(monkeypatch, tmp_path: Path):
    # Ensure MIME accepted
    class OkMagic:
        def __init__(self, mime=True):
            pass
        def from_buffer(self, b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", SimpleNamespace(Magic=OkMagic, from_buffer=OkMagic().from_buffer))
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: True)
    data = b"0" * 8300  # 8192 first + 108 remainder
    fv = validators.FileValidator(max_size_bytes=8200)
    with pytest.raises(ValueError):
        fv.validate_and_save_upload(FakeUpload(data), tmp_path / "x2.mp4")


def test_sync_seek_raises_is_ignored(monkeypatch, tmp_path: Path):
    class OkMagic:
        def __init__(self, mime=True):
            pass
        def from_buffer(self, b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", SimpleNamespace(Magic=OkMagic, from_buffer=OkMagic().from_buffer))
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: True)

    class FileLike:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        def read(self, n: int):
            return self._b.read(n)
        def seek(self, pos: int):
            raise RuntimeError("no seek")
    upload = SimpleNamespace(file=FileLike(b"1234"))
    out = tmp_path / "z.mp4"
    meta = validators.FileValidator(max_size_bytes=100).validate_and_save_upload(upload, out)
    assert meta["size"] > 0 and out.exists()


def test_sync_ffprobe_failure_raises(monkeypatch, tmp_path: Path):
    class OkMagic:
        def __init__(self, mime=True):
            pass
        def from_buffer(self, b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", SimpleNamespace(Magic=OkMagic, from_buffer=OkMagic().from_buffer))
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: False)
    with pytest.raises(ValueError):
        validators.FileValidator(max_size_bytes=100).validate_and_save_upload(FakeUpload(b"1234"), tmp_path / "f.mp4")


def test_sync_magic_none_unsupported(monkeypatch, tmp_path: Path):
    # Force no magic module
    monkeypatch.setattr(validators, "magic", None)
    with pytest.raises(ValueError):
        validators.FileValidator(max_size_bytes=100).validate_and_save_upload(FakeUpload(b"1234"), tmp_path / "g.mp4")


def test_sync_loop_writes_additional_chunks(monkeypatch, tmp_path: Path):
    class OkMagic:
        def __init__(self, mime=True):
            pass
        def from_buffer(self, b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", SimpleNamespace(Magic=OkMagic, from_buffer=OkMagic().from_buffer))
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: True)
    data = b"0" * 9000  # ensures while-loop runs
    out = tmp_path / "loop.mp4"
    meta = validators.FileValidator(max_size_bytes=100000).validate_and_save_upload(FakeUpload(data), out)
    assert meta["size"] == len(data)


@pytest.mark.asyncio
async def test_async_magic_exception_fallback_unsupported(monkeypatch, tmp_path: Path):
    class FakeMod:
        @staticmethod
        def from_buffer(b: bytes) -> str:
            raise RuntimeError("boom")
    monkeypatch.setattr(validators, "magic", FakeMod)
    class AUpload:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        async def read(self, n: int):
            return self._b.read(n)
        async def seek(self, pos: int):
            self._b.seek(pos)
    with pytest.raises(ValueError):
        await validators.FileValidator().avalidate_and_save_upload(AUpload(b"1" * 10), tmp_path / "a.mp4")


@pytest.mark.asyncio
async def test_async_too_large(monkeypatch, tmp_path: Path):
    class FakeMod:
        @staticmethod
        def from_buffer(b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", FakeMod)
    # Build async upload that yields > max
    class AUpload:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        async def read(self, n: int):
            return self._b.read(n)
        async def seek(self, pos: int):
            self._b.seek(pos)
    data = b"0" * 8300
    with pytest.raises(ValueError):
        await validators.FileValidator(max_size_bytes=8200).avalidate_and_save_upload(AUpload(data), tmp_path / "b.mp4")


@pytest.mark.asyncio
async def test_async_seek_error_is_ignored(monkeypatch, tmp_path: Path):
    class FakeMod:
        @staticmethod
        def from_buffer(b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", FakeMod)
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: True)
    class AUpload:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        async def read(self, n: int):
            return self._b.read(n)
        async def seek(self, pos: int):
            raise RuntimeError("no seek")
    out = tmp_path / "c.mp4"
    meta = await validators.FileValidator(max_size_bytes=100).avalidate_and_save_upload(AUpload(b"abcdef"), out)
    assert meta["size"] > 0 and out.exists()


@pytest.mark.asyncio
async def test_async_ffprobe_failure_raises(monkeypatch, tmp_path: Path):
    class FakeMod:
        @staticmethod
        def from_buffer(b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", FakeMod)
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: False)
    class AUpload:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        async def read(self, n: int):
            return self._b.read(n)
        async def seek(self, pos: int):
            self._b.seek(pos)
    with pytest.raises(ValueError):
        await validators.FileValidator(max_size_bytes=100).avalidate_and_save_upload(AUpload(b"1" * 10), tmp_path / "d.mp4")


@pytest.mark.asyncio
async def test_async_magic_none_unsupported(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(validators, "magic", None)
    class AUpload:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        async def read(self, n: int):
            return self._b.read(n)
        async def seek(self, pos: int):
            self._b.seek(pos)
    with pytest.raises(ValueError):
        await validators.FileValidator(max_size_bytes=100).avalidate_and_save_upload(AUpload(b"1234"), tmp_path / "h.mp4")


@pytest.mark.asyncio
async def test_async_loop_writes_additional_chunks(monkeypatch, tmp_path: Path):
    class FakeMod:
        @staticmethod
        def from_buffer(b: bytes) -> str:
            return "video/mp4"
    monkeypatch.setattr(validators, "magic", FakeMod)
    monkeypatch.setattr(validators, "ffprobe_ok", lambda p: True)
    class AUpload:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)
        async def read(self, n: int):
            return self._b.read(n)
        async def seek(self, pos: int):
            self._b.seek(pos)
    data = b"0" * 9000
    out = tmp_path / "aloop.mp4"
    meta = await validators.FileValidator(max_size_bytes=100000).avalidate_and_save_upload(AUpload(data), out)
    assert meta["size"] == len(data)
