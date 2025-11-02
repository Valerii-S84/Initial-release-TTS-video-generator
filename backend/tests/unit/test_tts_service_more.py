from __future__ import annotations

from pathlib import Path
import types

import pytest


def test_tts_client_init_failure(monkeypatch):
    import backend.services.tts_service as tts
    # Provide API key
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    # Callable that raises during init
    class BadClient:
        def __call__(self, *a, **k):
            raise RuntimeError("bad init")
    monkeypatch.setattr(tts, "ElevenLabs", BadClient())
    with pytest.raises(RuntimeError):
        tts.ElevenLabsTTS()


def test_tts_synthesize_with_settings_and_weird_chunk(monkeypatch, tmp_path: Path):
    import backend.services.tts_service as tts_mod

    class Weirdo:
        def __bytes__(self):
            raise TypeError("no bytes")

    class FakeStream:
        def __iter__(self):
            yield Weirdo()

    class FakeClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
        class text_to_speech:  # type: ignore
            @staticmethod
            def convert_as_stream(**kwargs):
                return FakeStream()

    # Use callable class path
    monkeypatch.setattr(tts_mod, "ElevenLabs", FakeClient)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    tts = tts_mod.ElevenLabsTTS()
    out = tmp_path / "weird.mp3"
    # Pass extra settings to cover voice_settings building
    tts.synthesize_to_file(
        "hi",
        out,
        voice_id="VID",
        model_id="M",
        stability=0.1,
        similarity_boost=0.2,
        style=0.3,
        use_speaker_boost=True,
    )
    assert out.exists()
