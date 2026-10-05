# Feature: sar-classification-agent, Property 8: every completed classification contains typology, riskScore, confidenceScore, triggering transactions, a non-empty rule rationale, and a non-empty Analyst-review model disclosure
"""Property 8 for the SAR Classification Agent — result completeness & disclosure.


Property statement: every completed classification (an
HTTP 200 from ``POST /classify``) returns a Classification_Result that contains
the detected typology key (which may be null when nothing is detected), a
Risk_Score, a Confidence_Score, the triggering-transactions collection, a
non-empty rule rationale, and a non-empty model disclosure stating the result
requires Analyst review before any action is taken.

Approach: drive the real endpoint end-to-end with a FastAPI ``TestClient``
(``from app.main import app`` is offline-safe for the transaction path — no
DynamoDB or Bedrock is touched when ``transactions[]`` is supplied). A Hypothesis
generator builds *valid* request payloads — a mix of cash and non-cash types,
amounts straddling the $10,000 / $9,000-$9,999.99 bands, and dates within and
around the rolling window, with an optional ``rollingWindowDays >= 1`` — so both
the detected and not-detected branches are exercised. Amounts are kept
non-negative with at most two decimal places and dates are ISO, so no request is
rejected with a 400; every generated request is a *completed* classification.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from app.main import app

client = TestClient(app)

# A base date for building concrete ISO calendar dates from integer day-offsets.
_BASE_DATE = date(2023, 1, 1)

# Transaction-type tokens mixing cash and non-cash movements so generated sets
# exercise both the cash partition and the non-cash exclusion path.
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT"]
_NON_CASH_TYPES = ["WIRE", "ACH", "CHECK", "CARD"]
_ALL_TYPES = _CASH_TYPES + _NON_CASH_TYPES


# ---------------------------------------------------------------------------
# Generators (valid inputs only — no 400s)
# ---------------------------------------------------------------------------


@st.composite
def _transaction_payload(draw) -> dict:
    """Generate one valid transaction request payload dict.

    Amounts are drawn in cents (so always <= 2 decimal places and non-negative)
    and straddle the sub-threshold band, the exact $10,000.00 boundary, and the
    $9,000-$9,999.99 near-threshold band. Dates cluster in a 0-60 day span so a
    rolling window of 1-60 days lands on both sides of real spans — producing
    both detected and not-detected classifications.
    """
    cents = draw(st.integers(min_value=0, max_value=1_500_000))  # $0.00 .. $15,000.00
    amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))

    day_offset = draw(st.integers(min_value=0, max_value=60))
    tx_date = _BASE_DATE + timedelta(days=day_offset)

    payload: dict = {
        "amount": str(amount),  # exact decimal string; preserves 2 dp
        "date": tx_date.isoformat(),  # ISO calendar date — always parseable
        "transactionType": draw(st.sampled_from(_ALL_TYPES)),
    }

    # Optionally set the explicit cash flag; otherwise leave it to the type.
    is_cash_flag = draw(st.sampled_from([None, True, False]))
    if is_cash_flag is not None:
        payload["isCash"] = is_cash_flag

    return payload


@st.composite
def _classify_request(draw) -> dict:
    """Generate a valid ``POST /classify`` request body.

    At least one transaction is always present (so the request is never an empty
    set and always takes the offline transaction path). ``rollingWindowDays`` is
    optionally included and, when present, is always ``>= 1`` so the window
    lower-bound check never fires.
    """
    size = draw(st.integers(min_value=1, max_value=8))
    body: dict = {"transactions": [draw(_transaction_payload()) for _ in range(size)]}

    if draw(st.booleans()):
        body["rollingWindowDays"] = draw(st.integers(min_value=1, max_value=90))

    return body


# ---------------------------------------------------------------------------
# Property 8
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(body=_classify_request())
def test_completed_classification_is_complete_and_discloses(body: dict) -> None:
    """Every 200 result carries the required fields and an Analyst-review disclosure."""
    response = client.post("/classify", json=body)

    # Valid inputs ⇒ a completed classification (no 400/404/5xx).
    assert response.status_code == 200, (
        f"expected 200 but got {response.status_code}: {response.text}"
    )

    result = response.json()

    #the typology key is present (its value may be null when nothing
    # was detected).
    assert "typology" in result

    # Risk_Score is an integer in [1, 10].
    assert "riskScore" in result
    risk = result["riskScore"]
    assert isinstance(risk, int) and not isinstance(risk, bool)
    assert 1 <= risk <= 10

    #Confidence_Score is a number in [0.0, 1.0].
    assert "confidenceScore" in result
    confidence = result["confidenceScore"]
    assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    assert 0.0 <= float(confidence) <= 1.0

    # the triggering-transactions collection is present and is a list.
    assert "triggeringTransactions" in result
    assert isinstance(result["triggeringTransactions"], list)

    # a non-empty rule rationale string.
    rationale = result.get("rationale")
    assert isinstance(rationale, str) and rationale.strip() != ""

    #  a non-empty model disclosure that frames the result as
    # requiring Analyst review before any action.
    disclosure = result.get("modelDisclosure")
    assert isinstance(disclosure, str) and disclosure.strip() != ""
    assert "analyst" in disclosure.lower()
