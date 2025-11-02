from __future__ import annotations

import sys
import types

import backend.utils.cache as c


def test_inmemorycache_get_none_and_bad_json():
    im = c.InMemoryCache()
    assert im.get_json("nope") is None  # line 16
    # Inject bad json
    im._data["k"] = (0, "{bad json}")
    assert im.get_json("k") is None  # lines 23-24


def test_rediscache_get_none_and_bad_json_and_get_cache_redis(monkeypatch):
    # Fake redis client
    class FakeClient:
        def __init__(self):
            self.store = {}
        def get(self, k):
            return None
        def set(self, k, v, ex=None):
            self.store[k] = v
        def delete(self, k):
            self.store.pop(k, None)
        def ping(self):
            return True

    class FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(url, decode_responses=True):
                return FakeClient()

    # Inject fake redis module
    monkeypatch.setitem(sys.modules, 'redis', FakeRedisModule())
    # Ensure cache globals reset
    monkeypatch.setattr(c, "_REDIS", None, raising=False)
    monkeypatch.setattr(c, "_CACHE", None, raising=False)
    # get_redis_client first time: go through import/construct, then return (line 68)
    client = c.get_redis_client()
    assert client is not None
    # get_redis_client second time: return cached (line 61)
    assert c.get_redis_client() is client
    # get_cache should return RedisCache (line 80)
    cache = c.get_cache()
    assert isinstance(cache, c.RedisCache)
    # Exercise RedisCache.get_json None path (line 41)
    assert cache.get_json("missing") is None
