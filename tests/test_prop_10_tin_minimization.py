# Feature: sar-classification-agent, Property 10: output includes only the last four TIN digits and the serialized result never contains the full TIN
"""Property 10 for the SAR Classification Agent output minimization.


Property statement: the assembled Classification_Result
includes only the last four digits of any taxpayer identification number, and
the serialized result never contains the full TIN.

The engine's internal ``Transaction`` only ever carries ``tinLast4`` (the last
four digits), never a full TIN. To meaningfully exercise the "full TIN never
leaves the service" guarantee, this test generates a *full* TIN as a digit
string of length 5-11, reduces it with :func:`app.models.tin_last4` when building
each triggering transaction, drives a detected :class:`StructuringVerdict` through
the real :func:`app.rule_engine.detect_structuring`, assembles the result via
:func:`app.models.assemble_classification_result`, and serializes it with
``model_dump_json()``. It then asserts:

  (a) the full TIN string does NOT appear anywhere in the serialized JSON
      (the generated full TIN has >4 digits, so its last-4 suffix is strict);
  (b) the last-4 suffix DOES appear for each cited (triggering) transaction;
  (c) every ``TransactionRef.tinLast4`` is at most four characters.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.models import assemble_classification_result, tin_last4
from app.rule_engine import CTR_THRESHOLD, detect_structuring, make_transaction

# A fixed epoch so generated day-offsets map to real calendar dates deterministically.
_EPOCH = date(2024, 1, 1)

# Cash-type tokens so the generated contributors participate in the structuring
# aggregation (non-cash is excluded by the engine).
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT", "CASH"]


@st.composite
def _full_tin(draw) -> str:
    """Generate a full TIN as a digit string of length 5-11.

    Length >= 5 guarantees the last-4 suffix is strict (the full TIN differs from
    its own last four digits), so asserting the full TIN is absent from the output
    is a meaningful check rather than trivially satisfied.
    """
    length = draw(st.integers(min_value=5, max_value=11))
    digits = draw(
        st.lists(st.sampled_from("0123456789"), min_size=length, max_size=length)
    )
    return "".join(digits)


@st.composite
def _triggering_set(draw):
    """Build a cash-only transaction set that reliably triggers structuring.

    Two or more sub-threshold cash transactions within a tight date span whose
    amounts sum to >= the CTR threshold. Each transaction carries the *reduced*
    tinLast4 (never a full TIN), mirroring how the normalization layer feeds the
    engine. Returns ``(transactions, full_tins)`` where ``full_tins`` is the list
    of the original full TINs used (one per transaction).
    """
    n = draw(st.integers(min_value=2, max_value=6))

    # Choose per-transaction amounts strictly below the threshold whose sum
    # reaches/exceeds it. Amounts in [5000.00, 9999.99] with n >= 2 guarantee a
    # detection while keeping each contributor sub-threshold.
    amounts = []
    for _ in range(n):
        cents = draw(st.integers(min_value=500_000, max_value=999_999))
        amounts.append((Decimal(cents) / Decimal(100)).quantize(Decimal("0.01")))

    # Keep all dates within a small inclusive span so a 30-day window detects.
    txs = []
    full_tins = []
    for i, amount in enumerate(amounts):
        full_tin = draw(_full_tin())
        full_tins.append(full_tin)
        day_offset = draw(st.integers(min_value=0, max_value=5))
        txs.append(
            make_transaction(
                amount=amount,
                date=_EPOCH + timedelta(days=day_offset),
                transactionType=draw(st.sampled_from(_CASH_TYPES)),
                index=i,
                isCash=True,
                # The engine only ever receives the reduced last-4, never the full TIN.
                tinLast4=tin_last4(full_tin),
            )
        )
    return txs, full_tins


@settings(max_examples=100)
@given(data=_triggering_set())
def test_tin_minimization_serialized_result_never_contains_full_tin(data):
    transactions, full_tins = data

    verdict = detect_structuring(transactions, rolling_window_days=30)
    # The generator is constructed so detection always occurs; guard the invariant.
    assert verdict.detected is True
    assert verdict.aggregatedAmount >= CTR_THRESHOLD

    result = assemble_classification_result(verdict)
    serialized = result.model_dump_json()

    # (c) Every cited TransactionRef carries at most a 4-char last-4 (never a full TIN).
    for ref in result.triggeringTransactions:
        if ref.tinLast4 is not None:
            assert len(ref.tinLast4) <= 4

    # Map each triggering transaction's index to its original full TIN so we can
    # assert the exact last-4 suffix is present and the full TIN is absent.
    index_to_full = {tx.index: full_tins[tx.index] for tx in verdict.triggeringTransactions}

    for ref in result.triggeringTransactions:
        full_tin = index_to_full[ref.index]
        last4 = full_tin[-4:]

        # (b) The last-4 digits appear in the serialized output for each cited tx.
        assert last4 in serialized
        assert ref.tinLast4 == last4

        # (a) The full TIN (strictly longer than its last-4) never appears.
        assert len(full_tin) > 4
        assert full_tin not in serialized
