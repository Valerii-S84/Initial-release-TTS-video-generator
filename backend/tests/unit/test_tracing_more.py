from __future__ import annotations

import os

import backend.core.tracing as tr


def test_setup_tracing_with_exporter(monkeypatch):
    # Force endpoint branch to add exporter/processor
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    # Reset provider: simulate non-SDK provider to avoid early return
    from opentelemetry import trace
    class DummyProvider: ...
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: DummyProvider())
    tr.setup_tracing("test-svc")


def test_extract_missing_header_uses_empty_list(monkeypatch):
    # Ensure attach is callable without real context and force propagate.extract to invoke our getter
    import types, sys
    monkeypatch.setitem(sys.modules, "opentelemetry.context", types.SimpleNamespace(attach=lambda ctx: None))
    def fake_extract(carrier, getter=None):
        # invoke getter for a missing key to trigger [] path
        if getter is not None:
            getter(carrier, "traceparent")
        return object()
    monkeypatch.setattr(tr.propagate, "extract", fake_extract)
    tr.extract_trace_from_dict({})
