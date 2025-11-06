from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ...job_storage import get_job_storage
from ...job_storage import list_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])

Jobs = get_job_storage()


@router.get("/{job_id}")
def job_status(job_id: str):
    j = Jobs.get(job_id)
    if not j:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Job not found"}},
        )
    return j


@router.get("/{job_id}/result")
def job_result(job_id: str):
    j = Jobs.get(job_id)
    if not j:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Job not found"}},
        )
    if j.get("status") != "completed":
        return JSONResponse({"status": j.get("status")}, status_code=202)
    return j.get("result")


@router.get("/{job_id}/queue_position")
def job_queue_position(job_id: str):
    j = Jobs.get(job_id)
    if not j:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Job not found"}},
        )
    status = str(j.get("status") or "").lower()
    # If already not in queue, report position 0
    if status != "queued":
        return {"job_id": job_id, "status": status, "position": 0}
    try:
        all_jobs = list_jobs()
    except Exception:
        all_jobs = []
    try:
        t0 = float(j.get("updated_at") or 0)
    except Exception:
        t0 = 0.0
    ahead = 0
    for it in all_jobs:
        try:
            if str(it.get("status") or "").lower() == "queued":
                ti = float(it.get("updated_at") or 0)
                # Count jobs enqueued earlier as ahead in the queue
                if ti < t0:
                    ahead += 1
        except Exception:
            continue
    # 1-based position for friendlier UX
    position = ahead + 1 if ahead >= 0 else 0
    return {"job_id": job_id, "status": status, "position": position}
