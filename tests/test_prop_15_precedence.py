# Feature: sar-classification-agent, Property 15: a non-empty transaction set is always classified from the submitted transactions regardless of a present bsaId
"""Property-based test for transaction-set precedence over BSAID (design Property 15).

Property 15:

    For any request containing a non-empty transaction set, the classification is
    derived from the submitted transactions regardless of whether a ``bsaId`` is
    also present (no stored-record substitution occurs).

Strategy
--------
This property is exercised end-to-end through ``POST /classify`` with FastAPI's
``TestClient``. For each generated transaction payload we issue two requests:

1. ``WITH`` a ``bsaId`` present alongside the ``transactions`` set — including
   ``bsaId`` values that do not resolve to any stored record (e.g.
   ``"DOES-NOT-EXIST-123"``).
2. ``WITHOUT`` any ``bsaId`` — the pure transaction path.

Transaction-set precedence requires that the presence of ``bsaId`` is
irrelevant when a non-empty ``transactions`` set is supplied. Concretely:

- The ``WITH-bsaId`` request MUST return ``200`` — never ``404`` (which would mean
  the handler attempted a stored-record lookup for the non-existent ``bsaId``)
  and never ``501`` (which is the not-yet-implemented BSAID-only path).
- The classification MUST be identical to the pure transaction path: the
  ``typology``, ``riskScore`` and ``triggeringTransactions`` produced ``WITH`` a
  ``bsaId`` must equal those produced ``WITHOUT`` one. Equal results prove the
  submitted transactions — not a substituted stored record — drove the verdict.

The transaction path never touches DynamoDB, so ``from app.main import app`` and
``TestClient(app)`` work fully offline.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from app.main import app

client = TestClient(app)

# A base date for building concrete calendar dates from integer day-offsets.
_BASE_DATE = date(2023, 1, 1)

# Mix of cash and non-cash transaction types so both the structuring-detected
# and no-detection paths are exercised; precedence must hold either way.
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT"]
_NON_CASH_TYPES = ["WIRE", "ACH", "CHECK", "CARD"]
_ALL_TYPES = _CASH_TYPES + _NON_CASH_TYPES

# bsaId values to pair with the transactions. These include clearly
# non-existent identifiers; precedence means none of them trigger a lookup.
_BSA_IDS = st.sampled_from(
    [
        "DOES-NOT-EXIST-123",
        "SAR-0000000000",
        "9999999999",
        "unknown-bsa-id",
        "31000012345678",
    ]
)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@st.composite
def _transaction_payload(draw, index: int) -> dict:
    """Generate one JSON transaction object for the request body.

    Amounts (in cents) straddle the sub-threshold band, the exact $10,000.00
    boundary, and the $9,000-$9,999.99 near-threshold band so detection and
    scoring boundaries are exercised. Dates cluster in a 0-45 day span so the
    default 30-day rolling window lands on both sides of real spans.
    """
    cents = draw(st.integers(min_value=0, max_value=1_500_000))  # $0.00 .. $15,000.00
    amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))

    day_offset = draw(st.integers(min_value=0, max_value=45))
    tx_date = _BASE_DATE + timedelta(days=day_offset)

    transaction_type = draw(st.sampled_from(_ALL_TYPES))
    is_cash_flag = draw(st.sampled_from([None, True, False]))

    payload: dict = {
        "amount": str(amount),  # serialize Decimal as a string to preserve scale
        "date": tx_date.isoformat(),
        "transactionType": transaction_type,
    }
    if is_cash_flag is not None:
        payload["isCash"] = is_cash_flag
    return payload


@st.composite
def _transactions(draw) -> list[dict]:
    """Generate a non-empty list of transaction payloads (1..8 records)."""
    size = draw(st.integers(min_value=1, max_value=8))
    return [draw(_transaction_payload(i)) for i in range(size)]


# ---------------------------------------------------------------------------
# Property 15
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    transactions=_transactions(),
    bsa_id=_BSA_IDS,
    rolling_window_days=st.one_of(st.none(), st.integers(min_value=1, max_value=60)),
)
def test_transaction_set_precedence_over_bsaid(
    transactions: list[dict],
    bsa_id: str,
    rolling_window_days: int | None,
) -> None:
    """A non-empty transaction set is classified from the submitted transactions
    regardless of a present (even non-existent) bsaId (Property 15)."""

    base_body: dict = {"transactions": transactions}
    if rolling_window_days is not None:
        base_body["rollingWindowDays"] = rolling_window_days

    # Request WITH a bsaId present alongside the transaction set.
    with_body = dict(base_body)
    with_body["bsaId"] = bsa_id
    resp_with = client.post("/classify", json=with_body)

    # Request WITHOUT any bsaId — the pure transaction path.
    resp_without = client.post("/classify", json=base_body)

    # Precedence: the bsaId must not trigger a stored-record lookup. The
    # with-bsaId request must succeed (200), never 404 (unresolved BSAID) and
    # never 501 (BSAID-only path not implemented).
    assert resp_with.status_code == 200, (
        "expected 200 from the transaction path with a present bsaId=%r, got %d: %s"
        % (bsa_id, resp_with.status_code, resp_with.text)
    )
    assert resp_with.status_code not in (404, 501)
    assert resp_without.status_code == 200, resp_without.text

    result_with = resp_with.json()
    result_without = resp_without.json()

    # The classification must be derived solely from the submitted transactions:
    # the verdict is identical whether or not a bsaId was supplied. Equal
    # typology, riskScore and triggeringTransactions prove no stored-record
    # substitution occurred.
    assert result_with["typology"] == result_without["typology"]
    assert result_with["riskScore"] == result_without["riskScore"]
    assert (
        result_with["triggeringTransactions"]
        == result_without["triggeringTransactions"]
    )
