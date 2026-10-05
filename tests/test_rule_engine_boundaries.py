"""Boundary and example unit tests for the deterministic Rule_Engine.

These example-based tests pin the Rule_Engine's behavior at the exact numeric and
temporal boundaries called out by the structuring requirements. They complement
the Hypothesis property tests by nailing down specific edge cases:

- Exactly $10,000.00 cash is reportable, never a contributor.
- Two $9,999.99 cash transactions one day apart aggregate to $19,999.98 and are
  detected.
- Window-span edges: span == window detects, span == window + 1 does not .
- Non-cash transactions never create or alter a verdict.
- A near-threshold contributor strictly raises confidence.
- The default window of 30 days is applied when none is supplied.
- The no-detection path yields typology=None and riskScore=1.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from app.rule_engine import (
    CTR_THRESHOLD,
    DEFAULT_ROLLING_WINDOW_DAYS,
    STRUCTURING_TYPOLOGY,
    detect_structuring,
    make_transaction,
)

BASE_DATE = date(2024, 1, 1)


def _cash(amount: str, *, index: int, day_offset: int = 0) -> "object":
    """Build a cash transaction at ``BASE_DATE + day_offset`` with an explicit index."""
    return make_transaction(
        amount=Decimal(amount),
        date=BASE_DATE + timedelta(days=day_offset),
        transactionType="CASH_DEPOSIT",
        index=index,
    )


def _noncash(amount: str, *, index: int, day_offset: int = 0) -> "object":
    """Build a non-cash transaction at ``BASE_DATE + day_offset`` with an explicit index."""
    return make_transaction(
        amount=Decimal(amount),
        date=BASE_DATE + timedelta(days=day_offset),
        transactionType="WIRE_TRANSFER",
        index=index,
        isCash=False,
    )


# ---------------------------------------------------------------------------
# exactly $10,000.00 cash is reportable, not a contributor
# ---------------------------------------------------------------------------


def test_exactly_threshold_cash_is_reportable_not_contributor():
    """A single $10,000.00 cash transaction is reportable, not a structuring contributor."""
    txs = [_cash("10000.00", index=0)]

    verdict = detect_structuring(txs)

    # The at-threshold transaction is reportable...
    assert len(verdict.reportableTransactions) == 1
    assert verdict.reportableTransactions[0].index == 0
    # ...and never a structuring contributor; with no sub-threshold pool there is
    # nothing to aggregate, so structuring is not detected.
    assert verdict.detected is False
    assert verdict.typology is None
    assert verdict.triggeringTransactions == []


def test_at_threshold_cash_excluded_even_when_subthreshold_set_structures():
    """An at-threshold cash tx stays reportable while a separate sub-threshold set structures."""
    txs = [
        _cash("10000.00", index=0, day_offset=0),  # reportable, excluded as contributor
        _cash("5000.00", index=1, day_offset=1),
        _cash("5000.00", index=2, day_offset=2),
    ]

    verdict = detect_structuring(txs)

    assert verdict.detected is True
    assert verdict.aggregatedAmount == Decimal("10000.00")
    triggering_indexes = {t.index for t in verdict.triggeringTransactions}
    assert triggering_indexes == {1, 2}
    # The $10,000.00 tx is reportable and NOT among the contributors.
    assert [t.index for t in verdict.reportableTransactions] == [0]
    assert 0 not in triggering_indexes


# ---------------------------------------------------------------------------
# two $9,999.99 cash one day apart → detected, aggregate $19,999.98
# ---------------------------------------------------------------------------


def test_two_near_threshold_cash_one_day_apart_detected():
    """Two $9,999.99 cash transactions one day apart aggregate to $19,999.98 and detect."""
    txs = [
        _cash("9999.99", index=0, day_offset=0),
        _cash("9999.99", index=1, day_offset=1),
    ]

    verdict = detect_structuring(txs)

    assert verdict.detected is True
    assert verdict.typology == STRUCTURING_TYPOLOGY
    assert verdict.aggregatedAmount == Decimal("19999.98")
    assert len(verdict.triggeringTransactions) == 2
    assert verdict.windowSpanDays == 1


# ---------------------------------------------------------------------------
#  window-span edges: span == window detects; span == window + 1 does not
# ---------------------------------------------------------------------------


def test_span_equal_to_window_is_detected():
    """When the inclusive span equals the window exactly, structuring is detected."""
    window = 10
    txs = [
        _cash("5000.00", index=0, day_offset=0),
        _cash("5000.00", index=1, day_offset=window),  # span == window
    ]

    verdict = detect_structuring(txs, rolling_window_days=window)

    assert verdict.detected is True
    assert verdict.windowSpanDays == window
    assert verdict.aggregatedAmount == Decimal("10000.00")


def test_span_one_beyond_window_is_not_detected():
    """When the inclusive span is window + 1, the transactions are out of range."""
    window = 10
    txs = [
        _cash("5000.00", index=0, day_offset=0),
        _cash("5000.00", index=1, day_offset=window + 1),  # span == window + 1
    ]

    verdict = detect_structuring(txs, rolling_window_days=window)

    assert verdict.detected is False
    assert verdict.typology is None
    assert verdict.triggeringTransactions == []


# ---------------------------------------------------------------------------
#a large non-cash transaction does not create or alter a verdict
# ---------------------------------------------------------------------------


def test_large_noncash_does_not_create_detection():
    """A large non-cash transaction alongside one sub-threshold cash tx never structures."""
    txs = [
        _cash("5000.00", index=0, day_offset=0),
        _noncash("50000.00", index=1, day_offset=1),  # excluded from aggregation
    ]

    verdict = detect_structuring(txs)

    assert verdict.detected is False
    assert verdict.triggeringTransactions == []
    # Non-cash is never recorded as reportable cash either.
    assert verdict.reportableTransactions == []


def test_large_noncash_does_not_alter_existing_verdict():
    """Adding a large non-cash tx leaves an existing cash-driven verdict unchanged."""
    cash_only = [
        _cash("6000.00", index=0, day_offset=0),
        _cash("6000.00", index=1, day_offset=1),
    ]
    with_noncash = cash_only + [_noncash("500000.00", index=2, day_offset=1)]

    verdict_cash = detect_structuring(cash_only)
    verdict_mixed = detect_structuring(with_noncash)

    assert verdict_mixed.detected == verdict_cash.detected is True
    assert verdict_mixed.aggregatedAmount == verdict_cash.aggregatedAmount
    assert {t.index for t in verdict_mixed.triggeringTransactions} == {
        t.index for t in verdict_cash.triggeringTransactions
    }
    assert verdict_mixed.reportableTransactions == verdict_cash.reportableTransactions


# ---------------------------------------------------------------------------
# near-threshold contributors yield strictly higher confidence
# ---------------------------------------------------------------------------


def test_near_threshold_set_has_higher_confidence():
    """$9,500 + $9,500 (near-threshold) beats $5,000 x3 in confidence."""
    near_threshold = [
        _cash("9500.00", index=0, day_offset=0),
        _cash("9500.00", index=1, day_offset=1),
    ]
    no_near_threshold = [
        _cash("5000.00", index=0, day_offset=0),
        _cash("5000.00", index=1, day_offset=1),
        _cash("5000.00", index=2, day_offset=2),
    ]

    verdict_near = detect_structuring(near_threshold)
    verdict_plain = detect_structuring(no_near_threshold)

    assert verdict_near.detected is True
    assert verdict_plain.detected is True
    assert verdict_near.confidenceScore > verdict_plain.confidenceScore
    # Confidence stays within the valid [0.0, 1.0] range (Req 4.3).
    assert 0.0 <= verdict_plain.confidenceScore <= 1.0
    assert 0.0 <= verdict_near.confidenceScore <= 1.0


# ---------------------------------------------------------------------------
# default window applied when omitted equals 30
# ---------------------------------------------------------------------------


def test_default_window_applied_when_omitted_equals_30():
    """Omitting rolling_window_days applies the 30-day default."""
    txs = [
        _cash("5000.00", index=0, day_offset=0),
        _cash("5000.00", index=1, day_offset=1),
    ]

    verdict = detect_structuring(txs)

    assert verdict.windowDays == DEFAULT_ROLLING_WINDOW_DAYS == 30


def test_default_window_boundary_detects_at_30_days_span():
    """Under the default window, a 30-day span detects but a 31-day span does not."""
    at_edge = [
        _cash("5000.00", index=0, day_offset=0),
        _cash("5000.00", index=1, day_offset=30),  # span == default window
    ]
    beyond_edge = [
        _cash("5000.00", index=0, day_offset=0),
        _cash("5000.00", index=1, day_offset=31),  # span == default window + 1
    ]

    assert detect_structuring(at_edge).detected is True
    assert detect_structuring(beyond_edge).detected is False


# ---------------------------------------------------------------------------
# no-detection path → typology None, riskScore 1
# ---------------------------------------------------------------------------


def test_no_detection_path_typology_none_and_risk_one():
    """A set that cannot structure reports no typology and risk score of exactly 1."""
    txs = [
        _cash("1000.00", index=0, day_offset=0),
        _cash("2000.00", index=1, day_offset=1),
    ]

    verdict = detect_structuring(txs)

    assert verdict.detected is False
    assert verdict.typology is None
    assert verdict.riskScore == 1


def test_no_detection_path_empty_input():
    """An empty transaction list is a clean no-detection with risk score 1."""
    verdict = detect_structuring([])

    assert verdict.detected is False
    assert verdict.typology is None
    assert verdict.riskScore == 1
    assert verdict.triggeringTransactions == []
    assert verdict.reportableTransactions == []
