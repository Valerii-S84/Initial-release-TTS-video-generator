from __future__ import annotations

from types import SimpleNamespace

from backend.services.video_generator import enqueue_generate


def test_enqueue_generate_handles_optional_failures(monkeypatch):
    class DummyJobs:
        def __init__(self):
            self.store = {}
        def set(self, k, v): self.store[k] = v
        def update(self, k, patch): self.store.setdefault(k, {}).update(patch)

    import backend.services.video_generator as vg
    monkeypatch.setattr(vg, "get_job_storage", lambda: DummyJobs())
    # Make these raise to hit except-pass blocks
    monkeypatch.setattr(vg, "JOBS_QUEUED", SimpleNamespace(inc=lambda: (_ for _ in ()).throw(RuntimeError())))
    monkeypatch.setattr(vg, "bind_job", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(vg, "inject_trace_to_dict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    # Fake celery task
    class DummyTask:
        def __init__(self, id="TTX"): self.id = id
    monkeypatch.setattr(vg, "task_generate", SimpleNamespace(delay=lambda cfg: DummyTask("TTX")))

    res = enqueue_generate({})
    assert res["status"] == "queued" and res["task_id"] == "TTX"

