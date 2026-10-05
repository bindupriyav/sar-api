# Feature: sar-classification-agent, Property 6: a detected set with a $9,000.00-$9,999.99 contributor yields strictly higher confidence than one without (both clamped <= 1.0)
"""Property 6 for the SAR Classification Agent Rule_Engine.


Property statement (design.md Property 6): a detected Transaction_Set that contains
at least one near-threshold contributing cash transaction (an amount in the inclusive
band ``$9,000.00``-``$9,999.99``) yields a strictly higher ``confidenceScore`` than an
otherwise-analogous detected set with no near-threshold contributor, with both
confidence scores clamped to ``<= 1.0``.

This test constructs, for each example, two comparable detected sets:

- ``near_set``: contains at least one near-threshold contributor (plus a
  complementary sub-threshold cash contributor so the pair detects), and
- ``plain_set``: contains only sub-$9,000.00 cash contributors (all strictly below
  the near-threshold band) that still aggregate to ``>= $10,000.00`` within the
  window.

Both sets are generated so they always detect (two or more sub-threshold cash
contributors summing to ``>= $10,000.00`` within an inclusive span ``<= w``). The
test asserts both are detected, both confidences are clamped ``<= 1.0``, and the
near-threshold set's confidence is strictly greater.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import (
    CTR_THRESHOLD,
    NEAR_THRESHOLD_LOWER,
    NEAR_THRESHOLD_UPPER,
    detect_structuring,
    is_near_threshold,
    make_transaction,
)

# A fixed epoch so generated day-offsets map to real calendar dates deterministically.
_EPOCH = date(2024, 1, 1)

_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT", "CASH"]


def _cents(amount: Decimal) -> int:
    """Return the integer cents for a quantized Decimal amount."""
    return int((amount * 100).to_integral_value())


def _amount(cents: int) -> Decimal:
    """Build a 2-dp Decimal dollar amount from integer cents."""
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


@st.composite
def _comparable_detected_sets(draw):
    """Generate two comparable, always-detecting structuring sets.

    Returns ``(near_set, plain_set, window)`` where:

    - ``near_set`` contains exactly one near-threshold contributor ``near`` in
      ``[$9,000.00, $9,999.99]`` plus a complementary sub-threshold cash
      contributor ``comp`` chosen so ``near + comp >= $10,000.00`` and both are
      strictly below the CTR threshold.
    - ``plain_set`` contains two or more cash contributors each strictly below
      ``$9,000.00`` (so none is near-threshold) that still sum to
      ``>= $10,000.00``.

    Both sets place all contributors within an inclusive span ``<= window`` so the
    engine always detects, exercising the detected branch where the near-threshold
    boost applies.
    """
    window = draw(st.integers(min_value=2, max_value=120))

    # --- near_set: one near-threshold contributor + a complementary contributor ---
    near_cents = draw(
        st.integers(
            min_value=_cents(NEAR_THRESHOLD_LOWER),
            max_value=_cents(NEAR_THRESHOLD_UPPER),
        )
    )
    near = _amount(near_cents)
    # Complementary contributor: enough to reach the threshold together with ``near``,
    # yet strictly below the CTR threshold itself. Lower bound guarantees the pair
    # aggregates to >= $10,000.00; the complementary amount stays sub-threshold.
    comp_min = _cents(CTR_THRESHOLD) - near_cents  # near + comp >= threshold
    comp_cents = draw(
        st.integers(min_value=comp_min, max_value=_cents(CTR_THRESHOLD) - 1)
    )
    comp = _amount(comp_cents)

    # Day offsets for the near_set pair, kept within the window span.
    near_day = draw(st.integers(min_value=0, max_value=window))
    comp_day = draw(st.integers(min_value=0, max_value=window))

    near_set = [
        make_transaction(
            amount=near,
            date=_EPOCH + timedelta(days=near_day),
            transactionType=draw(st.sampled_from(_CASH_TYPES)),
            index=0,
            isCash=True,
        ),
        make_transaction(
            amount=comp,
            date=_EPOCH + timedelta(days=comp_day),
            transactionType=draw(st.sampled_from(_CASH_TYPES)),
            index=1,
            isCash=True,
        ),
    ]

    # --- plain_set: contributors each strictly below $9,000.00, summing to >= 10k ---
    # Use k contributors each in [some floor, $8,999.99] so none is near-threshold.
    k = draw(st.integers(min_value=2, max_value=5))
    # Each contributor strictly below the near-threshold lower bound.
    plain_upper = _cents(NEAR_THRESHOLD_LOWER) - 1  # <= $8,999.99
    plain_lower = draw(st.integers(min_value=1, max_value=plain_upper))
    plain_amounts = [
        _amount(draw(st.integers(min_value=plain_lower, max_value=plain_upper)))
        for _ in range(k)
    ]
    # Guarantee aggregate reaches the threshold: if short, top up the last amount's
    # count is not possible (each is capped sub-$9,000), so instead add more
    # contributors until the running sum reaches the threshold.
    while sum(plain_amounts, Decimal("0.00")) < CTR_THRESHOLD:
        plain_amounts.append(_amount(plain_upper))

    plain_set = []
    for i, amt in enumerate(plain_amounts):
        day = draw(st.integers(min_value=0, max_value=window))
        plain_set.append(
            make_transaction(
                amount=amt,
                date=_EPOCH + timedelta(days=day),
                transactionType=draw(st.sampled_from(_CASH_TYPES)),
                index=i,
                isCash=True,
            )
        )

    return near_set, plain_set, window


@settings(max_examples=200)
@given(sets=_comparable_detected_sets())
def test_near_threshold_contributor_strictly_increases_confidence(sets):
    near_set, plain_set, window = sets

    near_verdict = detect_structuring(near_set, rolling_window_days=window)
    plain_verdict = detect_structuring(plain_set, rolling_window_days=window)

    # Both sets are constructed to always detect within the window.
    assert near_verdict.detected is True
    assert plain_verdict.detected is True

    # The near set actually has a near-threshold contributor; the plain set has none.
    assert any(is_near_threshold(tx.amount) for tx in near_verdict.triggeringTransactions)
    assert not any(
        is_near_threshold(tx.amount) for tx in plain_verdict.triggeringTransactions
    )

    # Both confidence scores are clamped to the valid [0.0, 1.0] range (Req 4.3).
    assert 0.0 <= near_verdict.confidenceScore <= 1.0
    assert 0.0 <= plain_verdict.confidenceScore <= 1.0

    # The near-threshold set yields strictly higher confidence (Req 3.7).
    assert near_verdict.confidenceScore > plain_verdict.confidenceScore
