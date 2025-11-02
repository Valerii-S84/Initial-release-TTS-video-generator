from __future__ import annotations

import os
import types
import pytest

import backend.services.tts_service as tts


def test_tts_service_missing_api_key(monkeypatch):
    # Ensure no key in settings or env
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    monkeypatch.setattr(tts.settings, "ELEVENLABS_API_KEY", "")
    # Keep SDK present to hit missing-key branch
    if tts.ElevenLabs is None:
        monkeypatch.setattr(tts, "ElevenLabs", types.SimpleNamespace)
    with pytest.raises(RuntimeError):
        tts.ElevenLabsTTS()


def test_tts_service_missing_sdk(monkeypatch):
    # Key present, but SDK missing
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    monkeypatch.setattr(tts, "ElevenLabs", None)
    with pytest.raises(RuntimeError):
        tts.ElevenLabsTTS()


def test_tts_service_invalid_client_type(monkeypatch):
    import backend.services.tts_service as tts
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    # Neither callable nor has .client
    monkeypatch.setattr(tts, "ElevenLabs", object())
    with pytest.raises(RuntimeError):
        tts.ElevenLabsTTS()


@pytest.mark.isolated
def test_tts_import_failure_sets_none(monkeypatch):
    import builtins, importlib
    import backend.services.tts_service as tts
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("elevenlabs.client"):
            raise ImportError("boom")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.reload(tts)
    assert tts.ElevenLabs is None
    # reload back to normal (monkeypatch will restore import)
    importlib.reload(tts)
