# Feature: sar-classification-agent, Property 2: verdict and aggregated contributing amount for S equal those for S union N (N non-cash)
"""Property 2 — non-cash transactions do not affect the structuring verdict.

**Property 2: verdict and aggregated contributing amount for S equal those for
S union N (N non-cash)**


For any transaction set ``S`` and any collection of non-cash transactions ``N``,
the structuring verdict and the aggregated contributing amount computed for ``S``
equal those computed for ``S ∪ N``. The test asserts this against the real
``detect_structuring`` engine (it does not re-implement the detection logic),
using Hypothesis with a minimum of 100 iterations.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import detect_structuring, make_transaction

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# A fixed epoch so generated dates are deterministic calendar dates. Transaction
# dates are expressed as an integer day-offset from this epoch.
_EPOCH = date(2020, 1, 1)

# Cash transaction types the engine recognizes as cash movements.
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT"]

# Non-cash transaction types: must NOT contain the CASH/CURRENCY tokens so the
# engine resolves them to non-cash (and we also force isCash=False explicitly).
_NONCASH_TYPES = ["WIRE", "ACH", "CHECK", "CARD", "TRANSFER"]


def _amounts() -> st.SearchStrategy[Decimal]:
    """Amounts in cents, straddling the near-threshold band and the CTR edge.

    Covers values well below, within the $9,000-$9,999.99 band, right at
    $9,999.99 / $10,000.00, and above the threshold so both contributor and
    reportable paths are exercised.
    """
    return st.integers(min_value=0, max_value=1_500_000).map(
        lambda cents: (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
    )


def _day_offsets() -> st.SearchStrategy[int]:
    """Day offsets clustered around typical rolling-window spans (0..60 days)."""
    return st.integers(min_value=0, max_value=60)


# A base transaction descriptor: (amount, day_offset, type, isCash_flag).
# The base set S is an arbitrary mix of cash and non-cash transactions.
_base_tx = st.tuples(
    _amounts(),
    _day_offsets(),
    st.sampled_from(_CASH_TYPES + _NONCASH_TYPES),
    st.sampled_from([True, False, None]),
)

# A non-cash transaction descriptor: (amount, day_offset, type). These are forced
# non-cash: a non-cash type AND an explicit isCash=False, so the engine excludes
# them from the structuring aggregation regardless of inference.
_noncash_tx = st.tuples(
    _amounts(),
    _day_offsets(),
    st.sampled_from(_NONCASH_TYPES),
)


def _build_base(descriptors):
    """Build the base transaction list S with unique sequential indices."""
    txs = []
    for i, (amount, offset, ttype, is_cash) in enumerate(descriptors):
        txs.append(
            make_transaction(
                amount=amount,
                date=_EPOCH + timedelta(days=offset),
                transactionType=ttype,
                index=i,
                isCash=is_cash,
            )
        )
    return txs


def _build_noncash(descriptors, start_index):
    """Build the non-cash list N, reindexed to continue after the base set."""
    txs = []
    for j, (amount, offset, ttype) in enumerate(descriptors):
        txs.append(
            make_transaction(
                amount=amount,
                date=_EPOCH + timedelta(days=offset),
                transactionType=ttype,
                index=start_index + j,
                isCash=False,  # force non-cash
            )
        )
    return txs


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    base=st.lists(_base_tx, min_size=0, max_size=12),
    noncash=st.lists(_noncash_tx, min_size=0, max_size=12),
    window=st.one_of(st.none(), st.integers(min_value=1, max_value=90)),
)
def test_noncash_transactions_do_not_affect_verdict(base, noncash, window):
    """Adding non-cash transactions leaves the structuring verdict unchanged.

    detect_structuring(S) and detect_structuring(S ∪ N) must agree on ``detected``,
    ``aggregatedAmount``, and ``typology`` for any non-cash set N (Req 3.2).
    Indices are kept unique across the merged set by reindexing N after S.
    """
    s = _build_base(base)
    n = _build_noncash(noncash, start_index=len(s))
    merged = s + n

    verdict_s = detect_structuring(s, rolling_window_days=window)
    verdict_merged = detect_structuring(merged, rolling_window_days=window)

    # Non-cash additions must not change whether structuring is detected.
    assert verdict_s.detected == verdict_merged.detected
    # ...nor the typology label.
    assert verdict_s.typology == verdict_merged.typology
    # ...nor the aggregated contributing amount.
    assert verdict_s.aggregatedAmount == verdict_merged.aggregatedAmount
