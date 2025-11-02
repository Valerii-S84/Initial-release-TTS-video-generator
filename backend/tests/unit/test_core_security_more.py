from __future__ import annotations

from types import SimpleNamespace

import pytest

import backend.core.security as sec


def test_require_ip_whitelist_no_client(monkeypatch):
    monkeypatch.setenv("ADMIN_IP_WHITELIST", "1.2.3.4")
    req = SimpleNamespace(client=None)
    with pytest.raises(Exception):
        sec.require_ip_whitelist(req)


def test_require_ip_whitelist_bad_client_ip(monkeypatch):
    monkeypatch.setenv("ADMIN_IP_WHITELIST", "1.2.3.4")
    req = SimpleNamespace(client=SimpleNamespace(host="not-an-ip"))
    with pytest.raises(Exception):
        sec.require_ip_whitelist(req)


def test_require_ip_whitelist_exact_match(monkeypatch):
    monkeypatch.setenv("ADMIN_IP_WHITELIST", "10.9.8.7")
    req = SimpleNamespace(client=SimpleNamespace(host="10.9.8.7"))
    assert sec.require_ip_whitelist(req) is None


def test_parse_whitelist_skips_invalid_entries():
    items = sec._parse_whitelist("bad,also-bad,10.0.0.0/24, 192.168.1.1")
    # Only valid network and address should be returned
    assert len(items) == 2


def test_is_url_safe_no_host():
    assert sec.is_url_safe("http:///path-only") is False


def test_is_url_safe_urlparse_exception(monkeypatch):
    monkeypatch.setattr(sec, "urlparse", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert sec.is_url_safe("http://whatever") is False


def test_require_ip_whitelist_exception_inside_loop(monkeypatch):
    # Prepare a container that raises in __contains__
    class BadContainer:
        def __contains__(self, item):
            raise RuntimeError("bad contains")

    monkeypatch.setenv("ADMIN_IP_WHITELIST", "1.2.3.4")
    # Force _parse_whitelist to include a bad item to hit exception path
    monkeypatch.setattr(sec, "_parse_whitelist", lambda wl: [BadContainer()])
    req = SimpleNamespace(client=SimpleNamespace(host="8.8.8.8"))
    with pytest.raises(Exception):
        sec.require_ip_whitelist(req)
