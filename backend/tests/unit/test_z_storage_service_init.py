from __future__ import annotations

import importlib
import pytest


@pytest.mark.isolated
def test_storage_service_redis_init_error(monkeypatch):
    import backend.services.storage_service as ss
    import backend.utils.cache as cache
    # Make get_redis_client raise to hit import-time except path
    monkeypatch.setattr(cache, "get_redis_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no redis")))
    importlib.reload(ss)
    assert ss._redis is None
    # Restore by reloading with original function (monkeypatch will auto-undo on test end)
    importlib.reload(ss)
