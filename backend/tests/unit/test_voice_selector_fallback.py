from __future__ import annotations

import types
import importlib


def test_voice_selector_fallback_import(monkeypatch):
    # Remove package-relative module to trigger fallback
    monkeypatch.setitem(__import__('sys').modules, 'backend.services.voice_map', None)
    # Provide a top-level voice_map module with get_voice_catalog
    vm = types.SimpleNamespace(get_voice_catalog=lambda: {"Rachel": types.SimpleNamespace(voice_id="X")})
    monkeypatch.setitem(__import__('sys').modules, 'voice_map', vm)
    import backend.services.voice_selector as vs
    importlib.reload(vs)
    # Should import from top-level fallback and work
    assert vs.select_voice_by_name("Rachel") == "X"

