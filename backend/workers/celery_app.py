from __future__ import annotations

import logging
import os
import platform
import ssl as _ssl
from pathlib import Path

from celery import Celery
from celery.schedules import crontab
from celery.signals import task_failure
import json
from opentelemetry import trace
from prometheus_client import start_http_server

from ..core.logging import bind_job, setup_logging
from ..core.metrics import GENERATION_TIME, JOBS_IN_PROGRESS, JOBS_QUEUED
from ..core.tracing import extract_trace_from_dict, setup_tracing
from ..db import SessionLocal
from ..job_storage import get_job_storage
from ..models import Video
from ..services.video_engine import generate_one

logger = logging.getLogger(__name__)

from ..core.config import settings


def _redis_url() -> str:
    # Prefer explicit Celery broker, then REDIS_URL; in prod, do not accept localhost
    url = (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )
    if (settings.APP_ENV or "").lower() == "prod":
        if not url or "localhost" in url or url.startswith("redis://127.0.0.1"):
            raise RuntimeError(
                "CELERY_BROKER_URL/REDIS_URL must be configured in production"
            )  # pragma: no cover
    # For rediss:// make sure ssl_cert_reqs is present to satisfy kombu/redis transport on some platforms
    if url.startswith("rediss://") and "ssl_cert_reqs=" not in url:
        mode = (os.getenv("CELERY_SSL_CERT_REQS") or "CERT_NONE").upper()
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}ssl_cert_reqs={mode}"
    return url


celery_app = Celery(
    "video_generator",
    broker=_redis_url(),  # pragma: no cover
    backend=os.getenv("CELERY_RESULT_BACKEND", _redis_url()),  # pragma: no cover
)

# Expose Prometheus metrics for worker/beat (optional; configurable port)
try:  # pragma: no cover - network binding in tests
    _port = int(os.getenv("PROM_METRICS_PORT", "9108"))
    start_http_server(_port)
except Exception:
    pass

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={  # pragma: no cover - transport config exercised in runtime
        "socket_timeout": 10,
        "max_retries": 5,
        "interval_start": 0,
        "interval_step": 2,
        "interval_max": 10,
    },
    result_backend_transport_options={  # pragma: no cover - transport config exercised in runtime
        "socket_timeout": 10,
    },
    timezone="UTC",
    task_default_retry_delay=10,
    beat_schedule={},
    # Timeouts
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "1200")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "900")),
    # Routing: maintenance vs default
    task_routes={
        "cleanup.run": {"queue": "maintenance"},
        "tasks.generate": {"queue": "default"},
    },
)

# Configure TLS for rediss:// URLs (Celery requires explicit SSL options)
try:
    _broker_url = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "")
    _backend_url = os.getenv("CELERY_RESULT_BACKEND") or os.getenv("REDIS_URL", "")

    def _ssl_mode():
        mapping = {
            "CERT_NONE": _ssl.CERT_NONE,
            "CERT_REQUIRED": _ssl.CERT_REQUIRED,
            "CERT_OPTIONAL": _ssl.CERT_OPTIONAL,
        }
        val = (os.getenv("CELERY_SSL_CERT_REQS") or "CERT_NONE").upper()
        return mapping.get(val, _ssl.CERT_NONE)

    if _broker_url.startswith("rediss://"):
        celery_app.conf.update(broker_use_ssl={"ssl_cert_reqs": _ssl_mode()})
    if _backend_url.startswith("rediss://"):
        celery_app.conf.update(redis_backend_use_ssl={"ssl_cert_reqs": _ssl_mode()})
except Exception:
    pass

# Windows: prefer solo pool to avoid semaphore/handle issues
try:
    if platform.system().lower().startswith("win"):
        celery_app.conf.update(
            worker_pool="solo",
            worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", "1")),
        )
except Exception:
    pass


@celery_app.task(  # pragma: no cover - task decorator wiring
    name="tasks.generate",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def task_generate(cfg: dict) -> dict:
    try:
        setup_logging()
    except Exception:
        pass
    try:
        setup_tracing("video-worker")
    except Exception:
        pass

    job_id = str(cfg.get("job_id"))
    bind_job(job_id)
    jobs = get_job_storage()
    logger.info("Job accepted")
    try:
        try:
            JOBS_QUEUED.dec()
            JOBS_IN_PROGRESS.inc()
        except Exception:
            pass
        jobs.update(job_id, {"status": "processing", "step": "queued", "progress": 0.0})

        def _cb(step: str, progress: float, message: str) -> None:
            jobs.update(
                job_id,
                {
                    "status": "processing",
                    "step": step,
                    "progress": float(progress),
                    "message": message,
                },
            )

        try:
            extract_trace_from_dict(cfg)
        except Exception:
            pass
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("generate_video"):
            with GENERATION_TIME.time():
                result = generate_one(cfg, progress=_cb)  # type: ignore[arg-type]
        payload = {
            "output": str(result.get("output_path")),
            "thumb": str(result.get("thumb_path")),
            "duration_sec": result.get("duration_sec"),
            "ssml": str(result.get("ssml_path")) if result.get("ssml_path") else None,
        }
        # Optional S3 upload for outputs
        try:
            from ..services.s3_storage import upload_outputs

            outs = [
                p
                for p in [
                    result.get("output_path"),
                    result.get("thumb_path"),
                    result.get("ssml_path"),
                ]
                if p
            ]
            uploaded = upload_outputs([Path(str(p)) for p in outs])  # type: ignore[list-item]
            if uploaded:
                logger.info("Uploaded outputs to S3", extra={"count": len(uploaded)})
        except Exception:
            logger.exception("S3 upload failed")
        jobs.update(job_id, {"status": "completed", "result": payload})
        try:
            JOBS_IN_PROGRESS.dec()
        except Exception:
            pass
        try:
            user_id = (
                int(cfg.get("user_id")) if cfg.get("user_id") is not None else None
            )
            if user_id:
                with SessionLocal() as db:
                    v = Video(
                        user_id=user_id,
                        video_path=payload["output"],
                        thumb_path=payload.get("thumb"),
                        duration_sec=payload.get("duration_sec"),
                    )
                    db.add(v)
                    db.commit()
        except Exception:
            logger.exception("Failed to persist video record")
        # Idempotency cache (24h)
        try:
            idem = cfg.get("idempotency_key")
            if idem:
                from ..utils.cache import get_redis_client
                from json import dumps

                rc = get_redis_client()
                if rc is not None:
                    rc.set(f"idem:{idem}", dumps({"job_id": job_id, "status": "completed", "result": payload}, ensure_ascii=False), ex=24 * 3600)
        except Exception:
            pass
        logger.info("Job completed")
        return payload
    except Exception as e:
        jobs.update(job_id, {"status": "failed", "error": str(e)})
        try:
            JOBS_IN_PROGRESS.dec()
        except Exception:
            pass
        logger.exception("Job failed")
        raise


@celery_app.task(  # pragma: no cover - lightweight periodic maintenance task
    name="cleanup.run",
    autoretry_for=(),
)
def task_cleanup(ttl_seconds: int | None = None) -> dict:
    """Run periodic cleanup of temp files and upload sessions.

    - Deletes old files in storage dirs (via utils.run_cleanup)
    - Cleans expired upload sessions and temp chunks
    - Logs disk free space and action stats
    """
    try:
        setup_logging()
    except Exception:
        pass

    # Resolve TTL preference: explicit arg -> settings -> env/default
    try:
        from ..core.config import settings as _settings

        default_ttl = int(getattr(_settings, "CLEANUP_TTL_SECONDS", 24 * 3600))
        ttl_out = int(getattr(_settings, "CLEANUP_TTL_OUTPUT_SECONDS", default_ttl))
        ttl_tmp = int(getattr(_settings, "CLEANUP_TTL_TMP_SECONDS", default_ttl))
        ttl_failed = int(getattr(_settings, "CLEANUP_TTL_FAILED_SECONDS", 3 * 3600))
    except Exception:
        default_ttl = 24 * 3600
        ttl_out = default_ttl
        ttl_tmp = default_ttl
        ttl_failed = 3 * 3600

    ttl = int(ttl_seconds) if ttl_seconds is not None else int(default_ttl)

    # Ensure utils.run_cleanup uses these TTLs via env override
    os.environ["CLEANUP_TTL_SECONDS"] = str(ttl)
    os.environ["CLEANUP_TTL_OUTPUT_SECONDS"] = str(ttl_out)
    os.environ["CLEANUP_TTL_TMP_SECONDS"] = str(ttl_tmp)
    os.environ["CLEANUP_TTL_FAILED_SECONDS"] = str(ttl_failed)

    from ..utils.cleanup import run_cleanup
    from ..services.storage_service import cleanup_expired_uploads
    from ..utils.cache import get_redis_client

    # Acquire Redis lock to avoid concurrent runs across multiple Beat instances
    lock = None
    try:
        rc = get_redis_client()
        if rc is not None:
            lock = rc.lock("cleanup:lock", timeout=3600)
            if not lock.acquire(blocking=False):
                logger.info("Cleanup skipped: lock held")
                return {"skipped": True, "reason": "locked"}
    except Exception:
        # If locking fails unexpectedly, continue best-effort
        logger.exception("Cleanup lock error; proceeding without lock")

    logger.info("Cleanup started", extra={"ttl_seconds": ttl})
    from ..core.metrics import (
        CLEANUP_DURATION,
        CLEANUP_ERRORS,
        CLEANUP_FILES_DELETED,
        CLEANUP_SESSIONS_DELETED,
        DISK_FREE_GB,
    )
    import time as _t
    _t0 = _t.perf_counter()
    result_files = {}
    result_sessions = {}
    try:
        result_files = run_cleanup() or {}
    except Exception:
        logger.exception("run_cleanup failed")
        result_files = {"error": True}
        try:
            CLEANUP_ERRORS.inc()
        except Exception:
            pass
    try:
        result_sessions = cleanup_expired_uploads(max_age_seconds=ttl) or {}
    except Exception:
        logger.exception("cleanup_expired_uploads failed")
        result_sessions = {"error": True}
        try:
            CLEANUP_ERRORS.inc()
        except Exception:
            pass

    removed_files = int((result_files or {}).get("removed", 0))
    removed_sessions = int((result_sessions or {}).get("sessions", 0))
    disk = (result_files or {}).get("disk", {})
    # Metrics
    try:
        if isinstance(disk, dict) and "free_gb" in disk:
            DISK_FREE_GB.set(float(disk.get("free_gb") or 0))
        if removed_files:
            CLEANUP_FILES_DELETED.inc(removed_files)
        if removed_sessions:
            CLEANUP_SESSIONS_DELETED.inc(removed_sessions)
        CLEANUP_DURATION.observe(max(0.0, _t.perf_counter() - _t0))
    except Exception:
        pass

    logger.info(
        "Cleanup finished",
        extra={
            "removed_files": removed_files,
            "removed_sessions": removed_sessions,
            "disk": disk,
            "ttl_seconds": ttl,
        },
    )
    if lock is not None:
        try:
            if getattr(lock, "owned", lambda: True)():
                lock.release()
        except Exception:
            pass

    return {
        "removed_files": removed_files,
        "removed_sessions": removed_sessions,
        "disk": disk,
        "ttl_seconds": ttl,
    }


# Celery Beat schedule for periodic cleanup (hourly)
try:
    from ..core.config import settings as _settings

    celery_app.conf.beat_schedule.update(
        {
            "run-cleanup-hourly": {
                "task": "cleanup.run",
                "schedule": crontab(minute="0"),
                # Pass TTL explicitly; can be overridden when calling the task manually
                "args": (_settings.CLEANUP_TTL_SECONDS,),
            },
            "recover-stuck-15m": {
                "task": "maintenance.recover",
                "schedule": crontab(minute="*/15"),
            },
        }
    )
except Exception:
    # Keep worker importable even if config is not fully available
    pass


@celery_app.task(name="maintenance.recover")
def task_recover() -> dict:
    try:
        setup_logging()
    except Exception:
        pass
    from ..job_storage import get_job_storage, list_jobs
    js = get_job_storage()
    now = __import__("time").time()
    n_marked = 0
    for j in list_jobs():
        try:
            if (j.get("status") == "processing") and (now - float(j.get("updated_at", now)) > 3600):
                job_id = j.get("job_id") or j.get("id")
                if job_id:
                    js.update(job_id, {"status": "failed", "error": "timeout"})
                    n_marked += 1
        except Exception:
            continue
    logger.info("Recovery completed", extra={"marked_failed": n_marked})
    return {"marked_failed": n_marked}

# Dead Letter Queue simulation for Redis transport: push failures to a Redis list
@task_failure.connect
def _on_task_failure(sender=None, task_id=None, args=None, kwargs=None, einfo=None, **kw):  # pragma: no cover - exercised in runtime
    try:
        from ..utils.cache import get_redis_client

        rc = get_redis_client()
        if rc is None:
            return
        payload = {
            "task": getattr(sender, "name", None),
            "task_id": task_id,
            "args": args,
            "kwargs": kwargs,
            "error": str(einfo) if einfo is not None else None,
        }
        rc.lpush("celery:dead", json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
