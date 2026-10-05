# Feature: sar-classification-agent, Property 14: HTTP 400 returned iff the specified Rolling_Window value is less than 1
"""Property 14 for the SAR Classification Agent rolling-window validation.


Property statement: a specified ``rollingWindowDays``
value is rejected with HTTP 400 if and only if it is not ``None`` and is less
than 1 (``MIN_ROLLING_WINDOW_DAYS``). An omitted window (``None``) and any value
``>= 1`` are accepted (no error) so the Rule_Engine applies its default or the
specified value.

The test generates integer window values across a range spanning negatives,
zero, and positives, plus the ``None`` (omitted) case. For each value it computes
the oracle expectation directly from the specification condition
(``window is not None and window < 1``) and asserts that:

- :func:`app.validation.validate_rolling_window` raises an ``HTTPException`` with
  ``status_code == 400`` exactly when the oracle says it should, and otherwise
  returns ``None`` without raising; and
- the same behavior holds end-to-end through
  :func:`app.validation.normalize_and_validate` with a single valid transaction
  present, so the window check is actually reached (it runs last, after the
  per-transaction checks succeed).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pytest
from fastapi import HTTPException
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models import ClassifyRequest, TransactionInput
from app.validation import (
    MIN_ROLLING_WINDOW_DAYS,
    normalize_and_validate,
    validate_rolling_window,
)


def _oracle_rejects(window: Optional[int]) -> bool:
    """Specification oracle: reject (400) iff a value is specified and < 1."""
    return window is not None and window < MIN_ROLLING_WINDOW_DAYS


def _build_request(window: Optional[int]) -> ClassifyRequest:
    """A request with one valid transaction so the window check is reached.

    The single transaction is well-formed (non-negative 2-dp amount, parseable
    ISO date, non-empty type) so that :func:`normalize_and_validate` passes the
    presence, set-size, and per-transaction checks and proceeds to the final
    rolling-window step.
    """
    return ClassifyRequest(
        transactions=[
            TransactionInput(
                amount=Decimal("5000.00"),
                date="2024-01-01",
                transactionType="CASH_DEPOSIT",
            )
        ],
        rollingWindowDays=window,
    )


def _assert_matches_oracle(window: Optional[int], call) -> None:
    """Run ``call`` and assert raise/no-raise matches the oracle for ``window``."""
    if _oracle_rejects(window):
        with pytest.raises(HTTPException) as exc_info:
            call()
        assert exc_info.value.status_code == 400
    else:
        # No exception expected; the call must complete.
        call()


# Integer window values spanning negatives, zero, and positives(
# bound is 1), plus the omitted (None) case which must be accepted.
_WINDOWS = st.one_of(
    st.none(),
    st.integers(min_value=-50, max_value=365),
)


@settings(max_examples=100)
@given(window=_WINDOWS)
def test_validate_rolling_window_rejects_iff_below_one(window: Optional[int]):
    request = _build_request(window)
    _assert_matches_oracle(window, lambda: validate_rolling_window(request))


@settings(max_examples=100)
@given(window=_WINDOWS)
def test_normalize_and_validate_window_rejects_iff_below_one(window: Optional[int]):
    request = _build_request(window)
    _assert_matches_oracle(window, lambda: normalize_and_validate(request))


@pytest.mark.parametrize(
    "window, should_reject",
    [
        (None, False),
        (-50, True),
        (-1, True),
        (0, True),
        (1, False),
        (30, False),
        (365, False),
    ],
)
def test_window_representative_examples(window: Optional[int], should_reject: bool):
    # The oracle agrees with the hand-picked expectation.
    assert _oracle_rejects(window) is should_reject

    request = _build_request(window)

    if should_reject:
        with pytest.raises(HTTPException) as exc_info:
            validate_rolling_window(request)
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info2:
            normalize_and_validate(request)
        assert exc_info2.value.status_code == 400
    else:
        assert validate_rolling_window(request) is None
        # A valid request normalizes to exactly the one transaction supplied.
        normalized = normalize_and_validate(request)
        assert len(normalized) == 1
