# Feature: sar-classification-agent, Task 8.2: Mocked-DynamoDB integration tests for the BSAID retrieval path
"""Mocked-DynamoDB integration tests for the ``POST /classify`` BSAID path.

Validates: Requirements 2.2, 2.6, 4.4, 6.3, 6.4, 8.5

These tests exercise the BSAID-only classification path in ``app/main.py``
(``_classify_from_bsaid``), which resolves a ``bsaId`` via the module-level
``table.get_item(...)`` call. Rather than standing up a real DynamoDB (moto) we
stub the single ``table.get_item`` call with ``monkeypatch`` — this is the
simplest robust way to avoid any real AWS access while covering the three
behaviours the task requires:

    1. BSAID resolves → 200, classified from derived aggregate/narrative signals
       with ``bsaId`` cited and a synthetic-data disclosure present. No itemized
       line items are fabricated, so ``typology`` is ``None`` and
       ``riskScore == 1`` (Req 2.2, 4.4, 6.3, 6.4).
    2. Unknown BSAID (no ``Item``) → 404 naming the unresolved bsaId (Req 2.6).
    3. ``get_item`` raises (client/throttling failure) → 503 "temporarily
       unavailable" (Req 8.5).

Each test installs its own ``get_item`` behaviour via ``monkeypatch`` so the
cases are fully isolated and never touch the network.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STORED_BSA_ID = "31000012345678"


def _stored_item() -> Dict[str, Any]:
    """A stored SAR record shaped like the synthetic data model.

    Carries only aggregate + narrative signals (no itemized line items), with
    ``SyntheticData`` at the top level of the record.
    """
    return {
        "Item": {
            "Activity": {
                "BSAID": STORED_BSA_ID,
                "SuspiciousActivity": [
                    {
                        "TotalSuspiciousAmountText": "45000.00",
                        "SuspiciousActivityFromDateText": "2024-01-01",
                        "SuspiciousActivityToDateText": "2024-01-20",
                        "SuspiciousActivityClassification": [
                            {"SuspiciousActivityTypeCodeDescription": "Structuring"}
                        ],
                    }
                ],
                "ActivityNarrativeInformation": [
                    {
                        "ActivityNarrativeText": (
                            "Multiple cash deposits just under the reporting "
                            "threshold over a three-week period."
                        )
                    }
                ],
            },
            "SyntheticData": True,
        }
    }


def _install_get_item(monkeypatch: pytest.MonkeyPatch, fake):
    """Monkeypatch ``app.main.table.get_item`` with ``fake`` for one test."""
    monkeypatch.setattr(main.table, "get_item", fake)


# ---------------------------------------------------------------------------
# Case 1: BSAID resolves → 200 with derived signals, citation + disclosure
# ---------------------------------------------------------------------------


def test_bsaid_resolves_to_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolvable bsaId is classified from aggregate/narrative signals.

    Validates: Requirements 2.2, 4.4, 6.3, 6.4
    """
    captured: Dict[str, Any] = {}

    def fake_get_item(Key=None, **kwargs):  # noqa: N803 - boto3 uses Key kwarg
        captured["Key"] = Key
        return _stored_item()

    _install_get_item(monkeypatch, fake_get_item)

    resp = client.post("/classify", json={"bsaId": STORED_BSA_ID})

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Resolution reused the SAR#{bsaId} / METADATA key pattern (Req 2.2).
    assert captured["Key"] == {"PK": f"SAR#{STORED_BSA_ID}", "SK": "METADATA"}

    # bsaId cited in the result (Req 4.4, 6.3).
    assert body["bsaId"] == STORED_BSA_ID

    # Synthetic-data disclosure present because the record carried
    # SyntheticData: true (Req 6.4).
    assert body["syntheticDataDisclosure"] is not None
    assert body["syntheticDataDisclosure"] != ""

    # No itemized line items were available: the rationale states the
    # limitation and no structuring typology is asserted (Req 4.4).
    rationale = body["rationale"].lower()
    assert "itemized" in rationale
    assert "unavailable" in rationale
    assert body["typology"] is None
    assert body["riskScore"] == 1

    # Ambiguity about the missing itemized data is surfaced, not resolved.
    assert body["ambiguityNotes"]


def test_bsaid_citation_falls_back_to_stored_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stored record's BSAID is cited even when it differs from the request.

    Validates: Requirements 4.4, 6.3
    """

    def fake_get_item(Key=None, **kwargs):  # noqa: N803
        return _stored_item()

    _install_get_item(monkeypatch, fake_get_item)

    # Request a differently-cased/aliased id; the stored BSAID should be cited.
    resp = client.post("/classify", json={"bsaId": "request-side-id"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["bsaId"] == STORED_BSA_ID


# ---------------------------------------------------------------------------
# Case 2: Unknown BSAID → 404 naming the bsaId
# ---------------------------------------------------------------------------


def test_unknown_bsaid_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bsaId with no stored Item returns 404 identifying the bsaId.

    Validates: Requirements 2.6
    """

    def fake_get_item(Key=None, **kwargs):  # noqa: N803
        return {}  # no "Item" key → unresolved

    _install_get_item(monkeypatch, fake_get_item)

    unknown = "UNKNOWN-1"
    resp = client.post("/classify", json={"bsaId": unknown})

    assert resp.status_code == 404, resp.text
    assert unknown in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Case 3: Client exception → 503 "temporarily unavailable"
# ---------------------------------------------------------------------------


def test_client_exception_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """A boto3/client failure maps to a 503 temporary-unavailability response.

    Validates: Requirements 8.5
    """

    def fake_get_item(Key=None, **kwargs):  # noqa: N803
        raise Exception("throttled")

    _install_get_item(monkeypatch, fake_get_item)

    resp = client.post("/classify", json={"bsaId": "ANY"})

    assert resp.status_code == 503, resp.text
    assert "temporarily unavailable" in resp.json()["detail"].lower()
