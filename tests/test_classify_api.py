# Feature: sar-classification-agent, Task 6.5: FastAPI TestClient API/validation tests
"""End-to-end API/validation tests for ``POST /classify`` via FastAPI TestClient.

Validates: Requirements 1.4, 2.1, 2.3, 2.4, 2.5, 2.7, 8.1, 8.2, 8.3, 8.4

These tests exercise the wired ``POST /classify`` endpoint (``app/main.py``)
through ``fastapi.testclient.TestClient`` so the full request-binding +
semantic-validation + Rule_Engine + response-assembly stack is covered.

Validation layering (important):
    * Pydantic-level failures (negative amount, missing required field) surface at
      request binding as **HTTP 422**, with the offending field + transaction index
      in ``detail[].loc``.
    * Semantic-pass failures (missing both inputs, empty set, oversize set,
      unparseable date, ``rollingWindowDays < 1``) surface as **HTTP 400** with the
      field + index named in the ``detail`` message.
    * Malformed JSON body is a client error (4xx) — FastAPI returns 422 (or 400);
      we only assert it is a 4xx and not a 200.

The transaction path is offline-safe to import (``boto3.resource`` is lazy), and
these tests never send a bsaId-only request (which hits the temporary 501
placeholder) EXCEPT the precedence case, which sends BOTH transactions and a
non-existent bsaId and expects 200 from the transaction path.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_tx(amount: str, date: str, tx_type: str = "CASH_DEPOSIT") -> Dict[str, Any]:
    """A minimal valid transaction payload."""
    return {"amount": amount, "date": date, "transactionType": tx_type}


def _structuring_transactions() -> List[Dict[str, Any]]:
    """Two sub-threshold cash transactions one day apart that aggregate above the
    $10,000.00 CTR threshold within the default 30-day window → Structuring.

    $9,999.99 + $9,999.99 = $19,999.98 (span of 1 day).
    """
    return [
        _valid_tx("9999.99", "2024-01-01"),
        _valid_tx("9999.99", "2024-01-02"),
    ]


def _detail_text(body: Dict[str, Any]) -> str:
    """Flatten a FastAPI error body to a searchable string.

    Works for both the semantic-pass shape (``detail`` is a string) and the
    Pydantic shape (``detail`` is a list of ``{loc, msg, type}`` dicts).
    """
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Req 2.3 — missing both transactions and bsaId → 400
# ---------------------------------------------------------------------------


def test_missing_both_inputs_returns_400():
    resp = client.post("/classify", json={})
    assert resp.status_code == 400
    detail = _detail_text(resp.json())
    assert "transactions" in detail and "bsaId" in detail


# ---------------------------------------------------------------------------
# Req 2.4 — empty transaction set → 400 "at least one"
# ---------------------------------------------------------------------------


def test_empty_transaction_set_returns_400():
    resp = client.post("/classify", json={"transactions": []})
    assert resp.status_code == 400
    assert "at least one" in _detail_text(resp.json()).lower()


# ---------------------------------------------------------------------------
# Req 2.5 — oversize set: 10,000 OK (200) / 10,001 → 400
# ---------------------------------------------------------------------------


def test_oversize_boundary_10000_ok_and_10001_rejected():
    # Build tiny valid, non-triggering cash transactions programmatically. Small
    # amounts on the same date keep it fast and avoid any structuring detection;
    # the case only needs a 200 at the upper allowed boundary.
    tx = _valid_tx("1.00", "2024-01-01")

    at_limit = {"transactions": [tx] * 10_000}
    resp_ok = client.post("/classify", json=at_limit)
    assert resp_ok.status_code == 200

    over_limit = {"transactions": [tx] * 10_001}
    resp_over = client.post("/classify", json=over_limit)
    assert resp_over.status_code == 400
    assert "10,000" in _detail_text(resp_over.json())


# ---------------------------------------------------------------------------
# Req 2.7 — missing required field → 422 referencing the index
# ---------------------------------------------------------------------------


def test_missing_field_returns_422_with_index():
    # Omit transactionType on the second transaction (index 1). Pydantic rejects
    # at request binding → 422 with the field + index in detail[].loc.
    payload = {
        "transactions": [
            _valid_tx("5000.00", "2024-01-01"),
            {"amount": "5000.00", "date": "2024-01-02"},  # missing transactionType
        ]
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 422
    detail = _detail_text(resp.json())
    assert "transactionType" in detail
    # The loc array includes the offending transaction index (1).
    assert "1" in detail


# ---------------------------------------------------------------------------
# Req 8.2 — negative amount → 422 referencing the index
# ---------------------------------------------------------------------------


def test_negative_amount_returns_422_with_index():
    # Negative amount on the second transaction (index 1). Pydantic rejects at
    # request binding → 422 with the field + index in detail[].loc.
    payload = {
        "transactions": [
            _valid_tx("5000.00", "2024-01-01"),
            _valid_tx("-250.00", "2024-01-02"),
        ]
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 422
    detail = _detail_text(resp.json())
    assert "amount" in detail
    assert "1" in detail


# ---------------------------------------------------------------------------
# Req 8.3 — unparseable date → 400 naming 'date' + index
# ---------------------------------------------------------------------------


def test_unparseable_date_returns_400_with_field_and_index():
    # 'date' is a free-form string on the model, so binding succeeds and the
    # semantic pass rejects it → 400 naming 'date' and the index (1).
    payload = {
        "transactions": [
            _valid_tx("5000.00", "2024-01-01"),
            _valid_tx("5000.00", "not-a-date"),
        ]
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 400
    detail = _detail_text(resp.json())
    assert "date" in detail
    assert "1" in detail


# ---------------------------------------------------------------------------
# Req 8.4 — rollingWindowDays < 1 → 400
# ---------------------------------------------------------------------------


def test_rolling_window_below_one_returns_400():
    payload = {
        "transactions": [_valid_tx("5000.00", "2024-01-01")],
        "rollingWindowDays": 0,
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 400
    assert "rollingWindowDays" in _detail_text(resp.json())


# ---------------------------------------------------------------------------
# Req 8.1 — malformed JSON body → client error (4xx), never 200
# ---------------------------------------------------------------------------


def test_malformed_json_returns_client_error():
    resp = client.post(
        "/classify",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert 400 <= resp.status_code < 500
    assert resp.status_code != 200


# ---------------------------------------------------------------------------
# Req 1.4 — X-Request-Id echo
# ---------------------------------------------------------------------------


def test_x_request_id_is_echoed_into_result():
    request_id = "req-abc-123"
    resp = client.post(
        "/classify",
        json={"transactions": _structuring_transactions()},
        headers={"X-Request-Id": request_id},
    )
    assert resp.status_code == 200
    assert resp.json()["requestId"] == request_id


def test_request_id_null_when_header_absent():
    resp = client.post(
        "/classify",
        json={"transactions": _structuring_transactions()},
    )
    assert resp.status_code == 200
    assert resp.json()["requestId"] is None


# ---------------------------------------------------------------------------
# Req 2.1 — transaction-set precedence over a (non-existent) bsaId
# ---------------------------------------------------------------------------


def test_transactions_take_precedence_over_nonexistent_bsaid():
    # Both a valid, detecting transaction set AND a non-existent bsaId are present.
    # The transaction path must win → 200 with Structuring detected, and crucially
    # NO 404 (the bsaId is never resolved).
    payload = {
        "transactions": _structuring_transactions(),
        "bsaId": "NON-EXISTENT",
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["typology"] == "Structuring"
