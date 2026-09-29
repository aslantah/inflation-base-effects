"""Tests for snapshot loading and verified network behavior."""

import json
import urllib.error
import urllib.request

import pandas as pd
import pytest

from inflation_base_effects.data import DataAccessError, _urlopen, load_cpi, validate_snapshot


def test_snapshot_load_is_deterministic():
    validate_snapshot()
    first = load_cpi()
    second = load_cpi()
    pd.testing.assert_frame_equal(first, second)
    assert not first.empty


def test_snapshot_manifest_declares_gs10():
    manifest = json.loads(open("data/snapshots/2026-09-28/manifest.json", encoding="utf-8").read())
    assert manifest["sources"]["fred_yields"]["B_USA_BD"] == "GS10"


def test_tls_verification_failure_is_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(DataAccessError, match="SSL_CERT_FILE"):
        _urlopen(urllib.request.Request("https://example.com"))
