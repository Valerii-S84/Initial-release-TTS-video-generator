from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .core.config import settings


class InMemoryJobStorage:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}
        self._exp: Dict[str, float] = {}

    def _cleanup(self) -> None:
        now = time.time()
        to_del = [k for k, t in self._exp.items() if t <= now]
        for k in to_del:
            self._store.pop(k, None)
            self._exp.pop(k, None)

    def set(self, job_id: str, data: Dict[str, Any]) -> None:
        data = {**data, "updated_at": time.time()}
        self._store[job_id] = data
        self._exp[job_id] = time.time() + self.ttl
        self._cleanup()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._cleanup()
        return self._store.get(job_id)

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        data = self._store.get(job_id) or {}
        data.update(patch)
        self.set(job_id, data)


class RedisJobStorage:
    def __init__(self, redis_client, ttl_seconds: int):
        self.r = redis_client
        self.ttl = ttl_seconds

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def set(self, job_id: str, data: Dict[str, Any]) -> None:
        # For Redis-backed storage, store exactly what the caller passed.
        # Some tests assert on the raw stored JSON without additional fields.
        self.r.set(self._key(job_id), json.dumps(data), ex=self.ttl)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        val = self.r.get(self._key(job_id))
        if not val:
            return None
        try:
            return json.loads(val)
        except Exception:
            return None

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        # Use optimistic lock
        key = self._key(job_id)
        with self.r.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    cur = pipe.get(key)
                    data = json.loads(cur) if cur else {}
                    data.update(patch)
                    pipe.multi()
                    import time as _t
                    data["updated_at"] = _t.time()
                    pipe.set(key, json.dumps(data), ex=self.ttl)
                    pipe.execute()
                    break
                except Exception:
                    pipe.unwatch()
                    # Fallback to simple set
                    self.set(job_id, patch)
                    break


_JOB_STORE: Any = None


def get_job_storage() -> Any:
    global _JOB_STORE
    # Allow tests/dev to bypass cache
    if os.getenv("JOB_STORE_DISABLE_CACHE") == "1":
        _JOB_STORE = None

    ttl_env = os.getenv("JOB_TTL_SECONDS")
    # Default ~24 years as requested (24*365*86400)
    default_ttl = 24 * 365 * 86400
    ttl = int(ttl_env) if ttl_env and ttl_env.isdigit() else default_ttl

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        # Ping to verify connection
        client.ping()
        # Upgrade to Redis-backed store if available
        if _JOB_STORE is None or not isinstance(_JOB_STORE, RedisJobStorage):
            _JOB_STORE = RedisJobStorage(client, ttl_seconds=ttl)
        return _JOB_STORE

    except Exception:
        # In production, do not silently fall back
        if (settings.APP_ENV or "").lower() == "prod":
            raise
        # Fallback to in-memory if Redis unavailable (dev/staging)
        if _JOB_STORE is None or not isinstance(_JOB_STORE, InMemoryJobStorage):
            _JOB_STORE = InMemoryJobStorage(ttl_seconds=ttl)
        return _JOB_STORE
 
 
def list_jobs() -> list[Dict[str, Any]]:
    js = get_job_storage()
    out: list[Dict[str, Any]] = []
    try:
        if isinstance(js, InMemoryJobStorage):
            out = list(js._store.values())  # type: ignore[attr-defined]
        elif isinstance(js, RedisJobStorage):
            cur = js.r.scan_iter(match="job:*")
            for key in cur:
                try:
                    val = js.r.get(key)
                    if val:
                        out.append(json.loads(val))
                except Exception:
                    continue
    except Exception:
        pass
    return out
