"""API-key authentication tests for the SAR FastAPI service.

Exercises the ``require_api_key`` dependency wired onto the protected
endpoints. The dependency reads the expected key LIVE from the ``API_KEY``
environment variable, so these tests drive behavior purely via
``monkeypatch.setenv`` / ``delenv`` with no import-order concerns.

Auth model under test:
  * ``API_KEY`` unset  → auth disabled, requests pass with no header.
  * ``API_KEY`` set    → ``X-API-Key`` header must match exactly (401 otherwise).
  * ``GET /health``    → always open, never requires a key.

To stay fully offline, the cases validate auth through ``POST /classify``
(transaction path, no AWS) and ``GET /health`` only.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

API_KEY = "test-secret-123"


def _valid_tx(amount: str, date: str, tx_type: str = "CASH_DEPOSIT") -> Dict[str, Any]:
    return {"amount": amount, "date": date, "transactionType": tx_type}


def _structuring_transactions() -> List[Dict[str, Any]]:
    """Two $9,999.99 cash deposits one day apart → aggregates to Structuring (200)."""
    return [
        _valid_tx("9999.99", "2024-01-01"),
        _valid_tx("9999.99", "2024-01-02"),
    ]


def _classify_body() -> Dict[str, Any]:
    return {"transactions": _structuring_transactions()}


# ---------------------------------------------------------------------------
# Auth disabled when API_KEY is not configured
# ---------------------------------------------------------------------------

def test_classify_no_api_key_env_allows_without_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """No API_KEY env set → POST /classify with no X-API-Key header → 200."""
    monkeypatch.delenv("API_KEY", raising=False)

    resp = client.post("/classify", json=_classify_body())

    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Auth enforced when API_KEY is configured
# ---------------------------------------------------------------------------

def test_classify_with_correct_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """API_KEY set + correct X-API-Key header → 200."""
    monkeypatch.setenv("API_KEY", API_KEY)

    resp = client.post(
        "/classify",
        json=_classify_body(),
        headers={"X-API-Key": API_KEY},
    )

    assert resp.status_code == 200, resp.text


def test_classify_missing_api_key_header_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """API_KEY set + missing X-API-Key header → 401."""
    monkeypatch.setenv("API_KEY", API_KEY)

    resp = client.post("/classify", json=_classify_body())

    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid or missing API key"


def test_classify_wrong_api_key_header_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """API_KEY set + wrong X-API-Key header → 401."""
    monkeypatch.setenv("API_KEY", API_KEY)

    resp = client.post(
        "/classify",
        json=_classify_body(),
        headers={"X-API-Key": "wrong-key"},
    )

    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid or missing API key"


# ---------------------------------------------------------------------------
# Health is always open
# ---------------------------------------------------------------------------

def test_health_open_even_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health → 200 even when API_KEY is set and no header is sent."""
    monkeypatch.setenv("API_KEY", API_KEY)

    resp = client.get("/health")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "healthy"
