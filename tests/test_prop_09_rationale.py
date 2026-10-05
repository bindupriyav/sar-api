# Feature: sar-classification-agent, Property 9: for a detected set, rationale text includes aggregated amount, contributor count, and Rolling_Window used
"""Property-based test for the rule-based rationale text (design Property 9).

Property 9

    For any transaction set that the Rule_Engine *detects* as Structuring, the
    rule-based rationale produced by ``build_rationale`` is non-empty and
    explicitly states the aggregated amount, the number of contributing
    (triggering) transactions, and the Rolling_Window value used.

To exercise the detected path deterministically, the generator constructs sets
that are guaranteed to trigger structuring: two or more sub-threshold cash
transactions whose amounts aggregate to at least the CTR threshold and whose
dates fall inside a single rolling window. The assertions derive the expected
substrings from the engine's own verdict fields using the same formatting the
engine uses (``f"{amount:,.2f}"`` for the amount), so the test cross-checks the
rationale against the authoritative verdict rather than restating constants.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.rule_engine import (
    CTR_THRESHOLD,
    Transaction,
    build_rationale,
    detect_structuring,
    make_transaction,
)

# A base date used to turn integer day-offsets into concrete calendar dates.
_BASE_DATE = date(2023, 1, 1)

# Cash transaction-type tokens so the generated contributors resolve to cash.
_CASH_TYPES = ["CASH_DEPOSIT", "CASH_WITHDRAWAL", "CURRENCY_DEPOSIT"]


@st.composite
def _detected_transaction_set(draw) -> tuple[list[Transaction], int]:
    """Generate a transaction set guaranteed to be detected as Structuring.

    Strategy:
    - Draw a window value ``window_days`` in ``[1, 60]``.
    - Draw a contributor count ``n`` in ``[2, 6]``.
    - Give each contributor a sub-threshold cash amount in ``$1.00``..``$9,999.99``
      and bump the total until it reaches at least the CTR threshold, so the
      subset is guaranteed to aggregate to ``>= $10,000.00`` while every single
      amount stays strictly below the threshold.
    - Place every contributor's date inside a span of ``<= window_days`` so the
      whole subset lands in one rolling window.
    - Optionally sprinkle in some non-contributing noise (non-cash and/or
      at-threshold reportable cash) that does not break detection.
    """
    window_days = draw(st.integers(min_value=1, max_value=60))
    n = draw(st.integers(min_value=2, max_value=6))

    # Draw sub-threshold cents for each contributor: $1.00 .. $9,999.99.
    cents_list = draw(
        st.lists(
            st.integers(min_value=100, max_value=999_999),
            min_size=n,
            max_size=n,
        )
    )

    threshold_cents = int(CTR_THRESHOLD * 100)  # 1_000_000 cents
    # Ensure the aggregate reaches the threshold by topping up the first
    # contributor while keeping every amount strictly below the threshold.
    total = sum(cents_list)
    if total < threshold_cents:
        deficit = threshold_cents - total
        # Raise the first contributor, capping it just below the threshold so it
        # stays a sub-threshold contributor rather than a reportable transaction.
        headroom = (threshold_cents - 1) - cents_list[0]
        cents_list[0] += min(deficit, headroom)
        total = sum(cents_list)
        # If a single top-up could not close the gap, distribute across others.
        idx = 1
        while total < threshold_cents and idx < n:
            room = (threshold_cents - 1) - cents_list[idx]
            bump = min(threshold_cents - total, room)
            cents_list[idx] += bump
            total = sum(cents_list)
            idx += 1

    # Place contributors on dates spanning at most ``window_days`` days.
    max_span = window_days
    day_offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=max_span),
            min_size=n,
            max_size=n,
        )
    )

    transactions: list[Transaction] = []
    index = 0
    for cents, offset in zip(cents_list, day_offsets):
        amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
        transactions.append(
            make_transaction(
                amount=amount,
                date=_BASE_DATE + timedelta(days=offset),
                transactionType=draw(st.sampled_from(_CASH_TYPES)),
                index=index,
            )
        )
        index += 1

    # Optional non-contributing noise: non-cash and at/above-threshold cash.
    noise_count = draw(st.integers(min_value=0, max_value=3))
    for _ in range(noise_count):
        kind = draw(st.sampled_from(["noncash", "reportable"]))
        offset = draw(st.integers(min_value=0, max_value=90))
        if kind == "noncash":
            amount = (
                Decimal(draw(st.integers(min_value=0, max_value=2_000_000)))
                / Decimal(100)
            ).quantize(Decimal("0.01"))
            tx_type = draw(st.sampled_from(["WIRE", "ACH", "CHECK"]))
        else:  # reportable cash at/above the threshold
            amount = (
                Decimal(draw(st.integers(min_value=1_000_000, max_value=5_000_000)))
                / Decimal(100)
            ).quantize(Decimal("0.01"))
            tx_type = draw(st.sampled_from(_CASH_TYPES))
        transactions.append(
            make_transaction(
                amount=amount,
                date=_BASE_DATE + timedelta(days=offset),
                transactionType=tx_type,
                index=index,
            )
        )
        index += 1

    return transactions, window_days


@settings(max_examples=100)
@given(data=_detected_transaction_set())
def test_detected_rationale_states_the_structuring_facts(
    data: tuple[list[Transaction], int],
) -> None:
    """Detected rationale states aggregated amount, count, and window (Property 9)."""
    transactions, window_days = data
    verdict = detect_structuring(transactions, rolling_window_days=window_days)

    # Guard: the generator is designed to always trigger detection.
    assert verdict.detected is True, (
        "generator failed to produce a detected set: %r"
        % [(str(t.amount), t.date.isoformat(), t.isCash) for t in transactions]
    )

    rationale = build_rationale(verdict)

    # The rationale must be a non-empty string.
    assert isinstance(rationale, str)
    assert rationale.strip() != ""

    # Derive the expected substrings from the verdict using the engine's own
    # formatting so the test tracks the engine rather than hard-coded constants.
    expected_amount = f"${verdict.aggregatedAmount:,.2f}"
    expected_count = str(len(verdict.triggeringTransactions))
    expected_window = str(verdict.windowDays)

    assert expected_amount in rationale, (
        "aggregated amount %r not found in rationale: %r"
        % (expected_amount, rationale)
    )
    assert expected_count in rationale, (
        "contributor count %r not found in rationale: %r"
        % (expected_count, rationale)
    )
    # The window value is stated as "{windowDays}-day Rolling_Window".
    assert f"{expected_window}-day Rolling_Window" in rationale, (
        "Rolling_Window value %r not found in rationale: %r"
        % (expected_window, rationale)
    )
