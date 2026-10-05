# Feature: sar-classification-agent, Property 4: every triggering transaction is cash, strictly < $10,000.00, within the window, and their sum is >= $10,000.00
"""Property 4 for the SAR Classification Agent Rule_Engine.


Property statement (design.md Property 4): for any transaction set classified as
Structuring, every transaction reported in ``triggeringTransactions`` is cash-type,
strictly less than $10,000.00, falls within the satisfying Rolling_Window span, and
the sum of those triggering amounts is ``>= $10,000.00``.

This test generates arbitrary transaction sets (mixing cash/non-cash, sub- and
over-threshold amounts, varied dates) plus an arbitrary window ``w`` and asserts the
property against the real :func:`app.rule_engine.detect_structuring` only on the
``detected`` branch.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import (
    CTR_THRESHOLD,
    detect_structuring,
    make_transaction,
)

# A fixed epoch so generated day-offsets map to real calendar dates deterministically.
_EPOCH = date(2024, 1, 1)

# Transaction-type tokens: a mix of cash-denoting and clearly non-cash tokens so the
# engine exercises both the cash-inclusion and non-cash-exclusion paths.
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT", "CASH"]
_NONCASH_TYPES = ["WIRE", "ACH", "CHECK", "CARD_PAYMENT"]


@st.composite
def _transactions(draw):
    """Generate a list of normalized transactions spanning a realistic input space.

    Each transaction gets an amount in cents (0.00 .. ~15000.00 so both
    sub-threshold and reportable amounts occur), a day offset within a wide range
    (so spans both inside and outside typical windows arise), and either a cash or
    non-cash transaction type. ``isCash`` is sometimes set explicitly and sometimes
    left to type inference, exercising both resolution paths.
    """
    n = draw(st.integers(min_value=0, max_value=12))
    txs = []
    for i in range(n):
        cents = draw(st.integers(min_value=0, max_value=1_500_000))
        amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
        day_offset = draw(st.integers(min_value=0, max_value=120))
        is_cash_tx = draw(st.booleans())
        if is_cash_tx:
            ttype = draw(st.sampled_from(_CASH_TYPES))
            explicit = draw(st.sampled_from([None, True]))
        else:
            ttype = draw(st.sampled_from(_NONCASH_TYPES))
            explicit = draw(st.sampled_from([None, False]))
        txs.append(
            make_transaction(
                amount=amount,
                date=_EPOCH + timedelta(days=day_offset),
                transactionType=ttype,
                index=i,
                isCash=explicit,
            )
        )
    return txs


@settings(max_examples=200)
@given(
    transactions=_transactions(),
    window=st.integers(min_value=1, max_value=120),
)
def test_triggering_transactions_are_valid_contributors(transactions, window):
    verdict = detect_structuring(transactions, rolling_window_days=window)

    if not verdict.detected:
        # Property 4 constrains only the detected branch; nothing to assert here.
        return

    triggering = verdict.triggeringTransactions

    # There must be at least two contributors for a structuring detection.
    assert len(triggering) >= 2

    # Every triggering transaction is cash-type and strictly below the CTR threshold.
    for tx in triggering:
        assert tx.isCash is True
        assert tx.amount < CTR_THRESHOLD

    # The triggering subset falls within the window: inclusive span <= w.
    span_days = (max(tx.date for tx in triggering) - min(tx.date for tx in triggering)).days
    assert span_days <= window
    # And the engine's reported span agrees with the actual contributor span.
    assert verdict.windowSpanDays == span_days

    # The sum of the triggering amounts reaches or exceeds the CTR threshold.
    total = sum((tx.amount for tx in triggering), Decimal("0.00"))
    assert total >= CTR_THRESHOLD
    # And matches the engine's reported aggregated amount.
    assert verdict.aggregatedAmount == total
