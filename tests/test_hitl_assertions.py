# Feature: sar-classification-agent, Task 10.2: Human-in-the-loop assertion tests
"""Human-in-the-loop (HITL) assertion tests for ``POST /classify``.

Validates: Requirements 4.6, 7.1, 7.3, 7.4

The SAR Classification Agent flags and classifies; it never makes legal
determinations and never acts on the Analyst's behalf. These tests assert the
human-in-the-loop guarantees that must hold on every classification result:

    * Req 4.6 / 7.1 — every result carries a non-empty ``modelDisclosure`` that
      frames the output as AI-produced and requiring Analyst review (the text
      mentions "analyst" and "review", case-insensitive).
    * Req 7.3 — recommended actions require Analyst approval. The current
      ``ClassificationResult`` model carries the approval requirement in
      ``modelDisclosure`` rather than as a structured ``recommendedActions``
      list, so the disclosure text must mention "approval". (If a structured
      list were present each entry would need marking; here we assert the
      disclosure-level guarantee.)
    * Req 7.4 — ambiguous input is reported via ``ambiguityNotes`` rather than
      resolved. The BSAID path (stored record lacking itemized line items) must
      surface the ambiguity in ``ambiguityNotes`` and must NOT fabricate a
      typology (``typology is None``).

The transaction path is offline-safe to import (``boto3.resource`` is lazy). For
the BSAID ambiguity case we monkeypatch ``app.main.table.get_item`` so no real
AWS access occurs. We also assert the ``MODEL_DISCLOSURE`` constant directly
(belt-and-suspenders constant-level check).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.models import MODEL_DISCLOSURE

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_tx(amount: str, date: str, tx_type: str = "CASH_DEPOSIT") -> Dict[str, Any]:
    """A minimal valid transaction payload."""
    return {"amount": amount, "date": date, "transactionType": tx_type}


def _structuring_transactions() -> List[Dict[str, Any]]:
    """Two sub-threshold cash transactions one day apart aggregating above the
    $10,000.00 CTR threshold within the default 30-day window → detected."""
    return [
        _valid_tx("9999.99", "2024-01-01"),
        _valid_tx("9999.99", "2024-01-02"),
    ]


def _non_detecting_transactions() -> List[Dict[str, Any]]:
    """A single small cash transaction → no structuring detected."""
    return [_valid_tx("100.00", "2024-01-01")]


def _assert_disclosure_requires_review_and_approval(disclosure: str) -> None:
    """Assert a disclosure string frames the result for Analyst review +
    approval (Req 4.6, 7.1, 7.3)."""
    assert isinstance(disclosure, str)
    assert disclosure.strip() != ""
    lowered = disclosure.lower()
    # Req 4.6 / 7.1 — requires Analyst review.
    assert "analyst" in lowered
    assert "review" in lowered
    # Req 7.3 — recommended actions require Analyst approval.
    assert "approval" in lowered


STORED_BSA_ID = "31000098765432"


def _stored_item_without_line_items() -> Dict[str, Any]:
    """A stored SAR record carrying only aggregate + narrative signals (no
    itemized per-transaction line items)."""
    return {
        "Item": {
            "Activity": {
                "BSAID": STORED_BSA_ID,
                "SuspiciousActivity": [
                    {
                        "TotalSuspiciousAmountText": "45000.00",
                        "SuspiciousActivityFromDateText": "2024-02-01",
                        "SuspiciousActivityToDateText": "2024-02-20",
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


# ---------------------------------------------------------------------------
# Constant-level check (belt-and-suspenders) — Req 4.6, 7.1, 7.3
# ---------------------------------------------------------------------------


def test_model_disclosure_constant_mentions_review_and_approval():
    """The standing MODEL_DISCLOSURE constant itself frames the result for
    Analyst review and marks recommended actions as requiring approval.

    Validates: Requirements 4.6, 7.1, 7.3
    """
    _assert_disclosure_requires_review_and_approval(MODEL_DISCLOSURE)


# ---------------------------------------------------------------------------
# Req 4.6 / 7.1 / 7.3 — every transaction-path result carries the disclosure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transactions, expect_detected",
    [
        (_structuring_transactions(), True),
        (_non_detecting_transactions(), False),
    ],
    ids=["detected", "not-detected"],
)
def test_every_result_includes_analyst_review_disclosure(
    transactions: List[Dict[str, Any]], expect_detected: bool
):
    """Both detected and not-detected transaction-path results include a
    non-empty modelDisclosure requiring Analyst review and marking recommended
    actions as requiring approval.

    Validates: Requirements 4.6, 7.1, 7.3
    """
    resp = client.post("/classify", json={"transactions": transactions})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Sanity: the two parametrized cases exercise the detected / not-detected
    # branches so the disclosure guarantee is covered for both.
    if expect_detected:
        assert body["typology"] == "Structuring"
    else:
        assert body["typology"] is None

    _assert_disclosure_requires_review_and_approval(body["modelDisclosure"])


# ---------------------------------------------------------------------------
# Req 7.4 — ambiguous BSAID input is reported, not resolved
# ---------------------------------------------------------------------------


def test_bsaid_ambiguity_is_reported_not_resolved(monkeypatch: pytest.MonkeyPatch):
    """When the stored record lacks itemized line items the ambiguity is
    surfaced via ambiguityNotes and no typology is fabricated.

    Validates: Requirements 7.4
    """

    def fake_get_item(Key=None, **kwargs):  # noqa: N803 - boto3 uses Key kwarg
        return _stored_item_without_line_items()

    monkeypatch.setattr(main.table, "get_item", fake_get_item)

    resp = client.post("/classify", json={"bsaId": STORED_BSA_ID})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Ambiguity reported, not resolved (Req 7.4): ambiguityNotes is a non-empty
    # list and its text indicates itemized data was unavailable.
    notes = body["ambiguityNotes"]
    assert isinstance(notes, list)
    assert len(notes) >= 1
    joined = " ".join(notes).lower()
    assert "itemized" in joined
    assert "unavailable" in joined

    # Ambiguity was NOT resolved into a detection: no typology is fabricated.
    assert body["typology"] is None

    # The disclosure guarantee still holds on the BSAID path.
    _assert_disclosure_requires_review_and_approval(body["modelDisclosure"])
