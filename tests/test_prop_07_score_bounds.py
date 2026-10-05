# Feature: sar-classification-agent, Property 7: riskScore integer in [1,10], confidenceScore decimal in [0.0,1.0], and riskScore == 1 when no typology detected
"""Property 7 for the SAR Classification Agent Rule_Engine.


Property statement: for any classification result, the
Risk_Score is an integer in ``[1, 10]`` and the Confidence_Score is a decimal in
``[0.0, 1.0]``; and when no typology is detected, the Risk_Score equals 1.

This test generates arbitrary transaction sets (mixing cash/non-cash, sub- and
over-threshold amounts, varied dates) plus an arbitrary Rolling_Window and asserts
the property against the real :func:`app.rule_engine.detect_structuring`.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import (
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
    left to type inference, exercising both resolution paths. The empty set is
    allowed so the no-detection path is also exercised.
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
def test_score_bounds_hold_for_every_classification(transactions, window):
    verdict = detect_structuring(transactions, rolling_window_days=window)

    # Risk_Score is an integer in [1, 10] inclusive (Req 4.2). Guard against bool,
    # which is an int subclass, to assert a true integer score.
    assert isinstance(verdict.riskScore, int)
    assert not isinstance(verdict.riskScore, bool)
    assert 1 <= verdict.riskScore <= 10

    # Confidence_Score is a decimal value in [0.0, 1.0] inclusive (Req 4.3).
    assert isinstance(verdict.confidenceScore, float)
    assert 0.0 <= verdict.confidenceScore <= 1.0

    # When no typology is detected, the Risk_Score equals 1 (Req 4.5).
    if verdict.typology is None:
        assert verdict.detected is False
        assert verdict.riskScore == 1
