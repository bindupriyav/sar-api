# Feature: sar-classification-agent, Property 5: every cash tx >= $10,000.00 appears in reportableTransactions and never in triggeringTransactions; no sub-threshold tx appears in reportableTransactions
"""Property 5 for the SAR Classification Agent Rule_Engine.


Property statement (design.md Property 5): for any transaction set, every cash
transaction whose amount is ``>= $10,000.00`` is partitioned out of the structuring
aggregation and recorded in ``reportableTransactions`` ; such a reportable
transaction never appears in ``triggeringTransactions`` ; conversely, no
sub-threshold transaction (``amount < $10,000.00``) and no non-cash transaction ever
appears in ``reportableTransactions``.

This test generates arbitrary mixed transaction sets (cash/non-cash, sub- and
over-threshold amounts, varied dates) plus an arbitrary window ``w`` and asserts the
partition against the real :func:`app.rule_engine.detect_structuring`. Membership is
compared by the original transaction ``index`` so identity is unambiguous.
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
    sub-threshold and reportable amounts occur), a day offset within a wide range,
    and either a cash or non-cash transaction type. ``isCash`` is sometimes set
    explicitly and sometimes left to type inference, exercising both resolution
    paths. The amount range straddles the $10,000.00 CTR threshold so both the
    reportable and candidate-contributor partitions are populated.
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
def test_reportable_cash_partitioned_out_of_structuring(transactions, window):
    verdict = detect_structuring(transactions, rolling_window_days=window)

    # Index sets for unambiguous membership comparison.
    reportable_indices = {tx.index for tx in verdict.reportableTransactions}
    triggering_indices = {tx.index for tx in verdict.triggeringTransactions}

    # The set of indices that OUGHT to be reportable: cash transactions whose
    # amount is at or above the CTR threshold (Req 3.9).
    expected_reportable_indices = {
        tx.index
        for tx in transactions
        if tx.isCash and tx.amount >= CTR_THRESHOLD
    }

    # Req 3.9: every cash tx >= $10,000.00 appears in reportableTransactions.
    assert reportable_indices == expected_reportable_indices

    # Every entry in reportableTransactions is genuinely a cash tx at/above threshold.
    for tx in verdict.reportableTransactions:
        assert tx.isCash is True
        assert tx.amount >= CTR_THRESHOLD

    # Req 3.10: a reportable transaction never appears in triggeringTransactions.
    assert reportable_indices.isdisjoint(triggering_indices)

    # No sub-threshold transaction (< $10,000.00) ever appears in reportable.
    by_index = {tx.index: tx for tx in transactions}
    for idx in reportable_indices:
        assert by_index[idx].amount >= CTR_THRESHOLD

    # No non-cash transaction ever appears in reportableTransactions.
    for idx in reportable_indices:
        assert by_index[idx].isCash is True
