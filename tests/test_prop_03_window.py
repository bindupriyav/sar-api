# Feature: sar-classification-agent, Property 3: condition satisfied exactly when inclusive span <= w; default w = 30 when unspecified
"""Property-based test for the Rule_Engine window-span boundary (Property 3).

Property 3 (design.md): For any set of cash sub-threshold transactions summing to
``>= $10,000.00`` and any window value ``w``, the Structuring condition is satisfied
exactly when the contributing subset's inclusive date span is ``<= w``; when no
window is specified, ``w = 30`` is used.


Tested against the real engine (``app.rule_engine.detect_structuring``) — the engine
is never re-implemented here. Minimum 100 iterations via ``@settings(max_examples=100)``.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import (
    CTR_THRESHOLD,
    DEFAULT_ROLLING_WINDOW_DAYS,
    detect_structuring,
    make_transaction,
)

# A fixed anchor date keeps the generators deterministic and avoids any clock use.
_ANCHOR_DATE = date(2023, 1, 1)

# Two sub-threshold cash amounts that together always reach the CTR threshold.
# Each is strictly < $10,000.00 so both are structuring candidates (never
# reportable); their sum is >= $10,000.00 so a single satisfying window would
# detect structuring.
_SUB_THRESHOLD_AMOUNT = Decimal("9999.99")  # strictly < 10000.00
# 9999.99 + 9999.99 = 19999.98 >= 10000.00, so two of these always qualify.


def _two_sub_threshold_cash(span_days: int):
    """Build exactly two sub-threshold cash transactions separated by ``span_days``.

    Both amounts are ``$9,999.99`` (strictly below the $10,000.00 threshold) and
    their sum (``$19,999.98``) is well above the threshold, so the only thing that
    governs detection is whether the inclusive date span fits inside the window.
    """
    earliest = make_transaction(
        amount=_SUB_THRESHOLD_AMOUNT,
        date=_ANCHOR_DATE,
        transactionType="CASH_DEPOSIT",
        index=0,
    )
    latest = make_transaction(
        amount=_SUB_THRESHOLD_AMOUNT,
        date=_ANCHOR_DATE + timedelta(days=span_days),
        transactionType="CASH_DEPOSIT",
        index=1,
    )
    return [earliest, latest]


# Spans and windows are generated across a range that straddles the default (30)
# and produces both span <= w and span > w cases frequently.
_span_strategy = st.integers(min_value=0, max_value=60)
_window_strategy = st.integers(min_value=1, max_value=60)


@settings(max_examples=100)
@given(span_days=_span_strategy, window_days=_window_strategy)
def test_window_span_boundary_specified_window(span_days: int, window_days: int) -> None:
    """Detection holds iff the inclusive span is within the specified window (Req 3.3, 3.4)."""
    transactions = _two_sub_threshold_cash(span_days)

    verdict = detect_structuring(transactions, rolling_window_days=window_days)

    # Precondition sanity: the pair always aggregates to >= the threshold, so the
    # ONLY reason to not detect is the span exceeding the window.
    assert _SUB_THRESHOLD_AMOUNT * 2 >= CTR_THRESHOLD

    expected = span_days <= window_days
    assert verdict.detected is expected, (
        f"span={span_days}, window={window_days}: expected detected={expected}, "
        f"got {verdict.detected}"
    )

    if verdict.detected:
        # The engine reports the inclusive span of the contributing subset (Req 3.3).
        assert verdict.windowSpanDays == span_days
        assert verdict.windowSpanDays <= window_days
        assert verdict.windowDays == window_days


@settings(max_examples=100)
@given(span_days=_span_strategy)
def test_default_window_is_30_when_unspecified(span_days: int) -> None:
    """Omitting the window (passing None) uses the default of 30 days (Req 3.5).

    Passing ``rolling_window_days=None`` must behave identically to passing the
    explicit default of 30 for the same transactions.
    """
    none_verdict = detect_structuring(
        _two_sub_threshold_cash(span_days), rolling_window_days=None
    )
    explicit_30_verdict = detect_structuring(
        _two_sub_threshold_cash(span_days),
        rolling_window_days=DEFAULT_ROLLING_WINDOW_DAYS,
    )

    # The default is exactly 30.
    assert DEFAULT_ROLLING_WINDOW_DAYS == 30

    # None behaves identically to the explicit default.
    assert none_verdict.detected is explicit_30_verdict.detected
    assert none_verdict.windowDays == explicit_30_verdict.windowDays == 30
    assert none_verdict.windowSpanDays == explicit_30_verdict.windowSpanDays

    # And the boundary condition holds against the default window of 30.
    expected = span_days <= DEFAULT_ROLLING_WINDOW_DAYS
    assert none_verdict.detected is expected
