# Feature: sar-classification-agent, Property 1: Structuring detected iff a subset of 2+ sub-threshold cash transactions within an inclusive span <= w sums to >= $10,000.00
"""Property-based test for Rule_Engine structuring detection (design Property 1).

Property 1:

    For any set of transactions and any rolling-window value ``w >= 1``, the
    Rule_Engine reports Structuring *detected* if and only if there exists a
    window (over the cash sub-threshold transactions sorted by date) whose
    inclusive date span is ``<= w`` and whose contained sub-threshold cash
    transactions number two or more and sum to ``>= $10,000.00``.

The oracle below is an *independent* reimplementation of the existence
condition. It deliberately does not import or reuse any of the engine's
private detection helpers; it re-derives the sliding-window existence check
from the documented semantics so that the test genuinely cross-checks the
engine rather than restating it.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import (
    CTR_THRESHOLD,
    Transaction,
    detect_structuring,
    make_transaction,
    resolve_is_cash,
)

# A base date for building concrete calendar dates from integer day-offsets.
_BASE_DATE = date(2023, 1, 1)

# A small set of transaction-type tokens mixing cash and non-cash movements so
# the generated sets exercise both the cash partition and the non-cash
# exclusion path. Cash resolution follows the engine's own ``resolve_is_cash``.
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT"]
_NON_CASH_TYPES = ["WIRE", "ACH", "CHECK", "CARD"]
_ALL_TYPES = _CASH_TYPES + _NON_CASH_TYPES


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------


def _oracle_detects(transactions: list[Transaction], window_days: int) -> bool:
    """Independently decide whether structuring should be detected.

    Mirrors the engine's documented semantics (sliding window over the cash
    sub-threshold transactions sorted by date) without touching engine
    internals:

    1. Keep only cash-type transactions strictly below the CTR threshold.
       Non-cash transactions and cash transactions at/above the threshold are
       excluded from the aggregation.
    2. Sort the survivors by ``(date, index)``.
    3. For each start position, extend forward while the inclusive span
       ``(current - start).days`` stays ``<= window_days``, accumulating the
       running sum. If at any point two or more transactions have been
       accumulated and the running sum reaches ``>= CTR_THRESHOLD``, the
       pattern exists.
    """
    candidates = [
        tx
        for tx in transactions
        if tx.isCash and tx.amount < CTR_THRESHOLD
    ]
    candidates.sort(key=lambda t: (t.date, t.index))

    n = len(candidates)
    for start in range(n):
        start_date = candidates[start].date
        running_sum = Decimal("0.00")
        count = 0
        for end in range(start, n):
            span = (candidates[end].date - start_date).days
            if span > window_days:
                break
            running_sum += candidates[end].amount
            count += 1
            if count >= 2 and running_sum >= CTR_THRESHOLD:
                return True
    return False


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@st.composite
def _transaction(draw, index: int) -> Transaction:
    """Generate one normalized :class:`Transaction` via the engine's helper.

    Amounts are drawn in cents and straddle the sub-threshold band, the exact
    $10,000.00 boundary, and the $9,000-$9,999.99 near-threshold band so the
    detection boundary is exercised. Dates cluster in a 0-60 day span so window
    values of 1-60 land on both sides of real spans.
    """
    cents = draw(
        st.integers(min_value=0, max_value=1_500_000)  # $0.00 .. $15,000.00
    )
    amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))

    day_offset = draw(st.integers(min_value=0, max_value=60))
    tx_date = _BASE_DATE + timedelta(days=day_offset)

    transaction_type = draw(st.sampled_from(_ALL_TYPES))
    # Occasionally set the explicit cash flag; otherwise leave it to the type.
    is_cash_flag = draw(st.sampled_from([None, True, False]))

    return make_transaction(
        amount=amount,
        date=tx_date,
        transactionType=transaction_type,
        index=index,
        isCash=is_cash_flag,
    )


@st.composite
def _transaction_set(draw) -> list[Transaction]:
    """Generate a list of normalized transactions with correct original indices."""
    size = draw(st.integers(min_value=0, max_value=8))
    return [draw(_transaction(i)) for i in range(size)]


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------


@settings(max_examples=300)
@given(
    transactions=_transaction_set(),
    window_days=st.integers(min_value=1, max_value=60),
)
def test_detection_matches_threshold_and_window_condition(
    transactions: list[Transaction],
    window_days: int,
) -> None:
    """Engine detection matches the independent existence oracle (Property 1)."""
    verdict = detect_structuring(transactions, rolling_window_days=window_days)
    expected = _oracle_detects(transactions, window_days)

    assert verdict.detected == expected, (
        "engine.detected=%r but oracle=%r for window=%d; "
        "candidates=%r"
        % (
            verdict.detected,
            expected,
            window_days,
            [
                (str(t.amount), t.date.isoformat(), t.isCash, t.index)
                for t in transactions
            ],
        )
    )

    # When detected, the engine must report the structuring typology and a
    # non-empty triggering set; when not, it must report no typology (Req 3.8).
    if verdict.detected:
        assert verdict.typology == "Structuring"
        assert len(verdict.triggeringTransactions) >= 2
    else:
        assert verdict.typology is None
        assert verdict.triggeringTransactions == []


def test_resolve_is_cash_sanity() -> None:
    """Guard: the oracle relies on the same cash resolution as the engine."""
    assert resolve_is_cash("CASH_DEPOSIT", None) is True
    assert resolve_is_cash("WIRE", None) is False
    assert resolve_is_cash("WIRE", True) is True
