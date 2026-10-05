# Feature: sar-classification-agent, Property 13: missing required field, negative amount, or unparseable date -> HTTP 400 identifying the offending field and the index
"""Property 13 for the SAR Classification Agent semantic validation pass.


Property statement: a per-transaction problem — a missing
required field, a negative amount, or an unparseable date — is rejected with HTTP
400 whose message identifies the offending field and the index of the affected
transaction.

Where the rejection happens depends on where the invariant is expressed:

- **Unparseable date:** ``date`` is a free-form ``str`` on
  :class:`~app.models.TransactionInput`, so construction succeeds and the
  rejection surfaces from :func:`app.validation.normalize_and_validate` as a
  :class:`fastapi.HTTPException` with ``status_code == 400`` whose detail names
  ``'date'`` and the index. This is the core property exercised with Hypothesis
  (>= 100 examples): a randomly chosen index ``k`` is seeded with garbage while
  every other transaction is valid, and the raised 400 must point at ``'date'``
  and index ``k``.

- **Negative amount** and **missing required field (Req 2.7):** these
  invariants are enforced by :class:`~app.models.TransactionInput` /
  :class:`~app.models.ClassifyRequest` at bind time, so they raise a Pydantic
  :class:`~pydantic.ValidationError` before ``normalize_and_validate`` is ever
  reached. These are covered by the example-based portion below.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from fastapi import HTTPException

from app.models import ClassifyRequest, TransactionInput
from app.validation import normalize_and_validate


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# A valid, parseable ISO calendar date string.
_VALID_DATES = st.dates(
    min_value=__import__("datetime").date(2000, 1, 1),
    max_value=__import__("datetime").date(2030, 12, 31),
).map(lambda d: d.isoformat())

# Garbage date strings that cannot be parsed as a calendar date. These include
# empty / whitespace-only values, non-date text, and out-of-range components.
_UNPARSEABLE_DATES = st.sampled_from(
    [
        "not-a-date",
        "2024-13-99",   # month 13, day 99 — out of range
        "2024-02-30",   # Feb never has 30 days
        "",             # empty
        "   ",          # whitespace only
        "garbage",
        "99/99/9999",
        "2024-00-00",
        "yesterday",
        "2024-13-01T00:00:00",
    ]
)


def _valid_input(date_str: str) -> TransactionInput:
    """A valid TransactionInput except that its date is supplied by the caller.

    The amount/type are always valid so that only the date can be the offending
    field in the generated transaction.
    """
    return TransactionInput(
        amount=Decimal("5000.00"),
        date=date_str,
        transactionType="CASH_DEPOSIT",
    )


# ---------------------------------------------------------------------------
# Core property: unparseable date -> HTTP 400 naming 'date' and the index
# ---------------------------------------------------------------------------


@settings(max_examples=150)
@given(data=st.data())
def test_unparseable_date_reports_field_and_index(data):
    # Build a transaction list of size n with a single offending index k.
    n = data.draw(st.integers(min_value=1, max_value=8), label="n")
    k = data.draw(st.integers(min_value=0, max_value=n - 1), label="k")
    bad_date = data.draw(_UNPARSEABLE_DATES, label="bad_date")

    inputs: List[TransactionInput] = []
    for i in range(n):
        if i == k:
            inputs.append(_valid_input(bad_date))
        else:
            good = data.draw(_VALID_DATES, label=f"good_date_{i}")
            inputs.append(_valid_input(good))

    request = ClassifyRequest(transactions=inputs)

    with pytest.raises(HTTPException) as exc_info:
        normalize_and_validate(request)

    exc = exc_info.value
    assert exc.status_code == 400
    detail = str(exc.detail)
    # The message must identify the offending field 'date' and the index k.
    assert "date" in detail
    assert str(k) in detail


# ---------------------------------------------------------------------------
# Negative amount: rejected at TransactionInput construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["-0.01", "-5000.00", "-10000.00", "-1"])
def test_negative_amount_rejected_at_construction(raw: str):
    with pytest.raises(ValidationError):
        TransactionInput(
            amount=Decimal(raw),
            date="2024-01-01",
            transactionType="CASH_DEPOSIT",
        )


def test_negative_amount_rejected_in_request_binding():
    # A negative amount inside a ClassifyRequest also fails at bind time, before
    # normalize_and_validate is reached.
    with pytest.raises(ValidationError):
        ClassifyRequest(
            transactions=[
                {
                    "amount": Decimal("5000.00"),
                    "date": "2024-01-01",
                    "transactionType": "CASH_DEPOSIT",
                },
                {
                    "amount": Decimal("-250.00"),
                    "date": "2024-01-02",
                    "transactionType": "CASH_DEPOSIT",
                },
            ]
        )


# ---------------------------------------------------------------------------
# Missing required field (Req 2.7): rejected at TransactionInput construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # Missing transactionType.
        {"amount": Decimal("5000.00"), "date": "2024-01-01"},
        # Missing amount.
        {"date": "2024-01-01", "transactionType": "CASH_DEPOSIT"},
        # Missing date.
        {"amount": Decimal("5000.00"), "transactionType": "CASH_DEPOSIT"},
    ],
)
def test_missing_required_field_rejected_at_construction(payload: dict):
    with pytest.raises(ValidationError):
        TransactionInput(**payload)


def test_missing_transaction_type_rejected_in_request_binding():
    # Omitting transactionType on one record of a ClassifyRequest raises a
    # Pydantic ValidationError at bind time (Req 2.7).
    with pytest.raises(ValidationError):
        ClassifyRequest(
            transactions=[
                {
                    "amount": Decimal("5000.00"),
                    "date": "2024-01-01",
                    # transactionType omitted
                }
            ]
        )
