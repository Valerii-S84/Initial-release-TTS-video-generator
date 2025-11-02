from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_jobs_404():
    client = TestClient(app)
    r = client.get("/api/v1/jobs/NO_SUCH")
    assert r.status_code == 404


def test_jobs_result_202_and_200(monkeypatch):
    client = TestClient(app)
    from backend.api.v1 import jobs as jobs_api

    try:
        jobs_api.Jobs.set("J_A", {"status": "running"})
        jobs_api.Jobs.set("J_B", {"status": "completed", "result": {"ok": True}})
    except Exception:
        # Redis-backed store not available; skip
        return

    r1 = client.get("/api/v1/jobs/J_A/result")
    assert r1.status_code == 202 and r1.json()["status"] == "running"

    r2 = client.get("/api/v1/jobs/J_B/result")
    assert r2.status_code == 200 and r2.json() == {"ok": True}


def test_jobs_result_404():
    client = TestClient(app)
    r = client.get("/api/v1/jobs/NO_SUCH/result")
    assert r.status_code == 404
