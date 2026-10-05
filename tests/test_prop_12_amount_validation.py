# Feature: sar-classification-agent, Property 12: a request is accepted only when every amount has at most two decimals and is non-negative, rejected otherwise
"""Property 12 for the SAR Classification Agent request models.


Property statement: a ``TransactionInput`` is accepted
only when its ``amount`` is non-negative AND has at most two decimal places;
otherwise construction raises a Pydantic ``ValidationError``.

This test generates ``Decimal`` amounts (including fractional, negative, and
high-scale values) and computes the oracle expectation directly from the
Decimal's own sign and scale (``as_tuple().exponent``). It then asserts that
:class:`app.models.TransactionInput` is constructable exactly when the oracle
says the amount is valid, and raises :class:`pydantic.ValidationError` otherwise.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models import TransactionInput


def _scale(value: Decimal) -> int:
    """Number of fractional digits implied by the Decimal's exponent.

    A finite Decimal's exponent is the negative count of fractional digits, so
    the scale is ``-exponent`` when the exponent is negative, else 0 (trailing
    integer zeros give a non-negative exponent, e.g. ``Decimal('1E+2')``).
    """
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)  # finite values only in this test
    return -exponent if exponent < 0 else 0


def _oracle_valid(value: Decimal) -> bool:
    """The specification oracle: non-negative AND at most two decimal places."""
    return value.is_finite() and value >= 0 and _scale(value) <= 2


def _build(amount: Decimal) -> TransactionInput:
    return TransactionInput(
        amount=amount,
        date="2024-01-01",
        transactionType="CASH_DEPOSIT",
    )


# A broad space of finite Decimals: whole, 1-dp, 2-dp, and 3+dp values, both
# positive and negative, plus zero, so both the accept (<=2 dp, non-negative)
# and reject (>2 dp or negative) sides of the property are exercised.
_AMOUNTS = st.decimals(
    min_value=Decimal("-20000"),
    max_value=Decimal("20000"),
    allow_nan=False,
    allow_infinity=False,
)


@settings(max_examples=200)
@given(amount=_AMOUNTS)
def test_amount_accepted_iff_nonnegative_and_at_most_two_decimals(amount: Decimal):
    expected_valid = _oracle_valid(amount)

    if expected_valid:
        model = _build(amount)
        # The exact Decimal value is preserved (no float drift / rounding).
        assert model.amount == amount
    else:
        with pytest.raises(ValidationError):
            _build(amount)


@pytest.mark.parametrize(
    "raw, should_accept",
    [
        ("0", True),
        ("100.00", True),
        ("9999.99", True),
        ("10000.00", True),
        ("0.1", True),
        ("0.01", True),
        ("100.123", False),
        ("-5.00", False),
        ("0.001", False),
    ],
)
def test_amount_representative_examples(raw: str, should_accept: bool):
    amount = Decimal(raw)
    # The oracle agrees with the hand-picked expectation.
    assert _oracle_valid(amount) is should_accept

    if should_accept:
        model = _build(amount)
        assert model.amount == amount
    else:
        with pytest.raises(ValidationError):
            _build(amount)
