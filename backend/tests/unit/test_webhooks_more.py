from __future__ import annotations

from backend.webhooks import sign_payload, verify_signature


def test_webhook_verify_malformed_header(monkeypatch):
    assert verify_signature("s", b"p", "malformed", tolerance=300) is False


def test_webhook_verify_tolerance_exceeded(monkeypatch):
    secret = "s"
    payload = b"hello"
    ts = 100
    header = sign_payload(secret, payload, ts=ts)
    # Move time far beyond tolerance
    monkeypatch.setattr("backend.webhooks.time.time", lambda: ts + 10000)
    assert verify_signature(secret, payload, header, tolerance=300) is False

