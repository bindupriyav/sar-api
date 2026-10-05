"""Semantic validation pass for the SAR Classification Agent.

This module holds the domain-level validation that Pydantic binding does not
express structurally, plus the normalization that turns validated
:class:`~app.models.TransactionInput` records into the Rule_Engine's internal
:class:`~app.rule_engine.Transaction` objects.

It is imported by the ``POST /classify`` handler; it does not touch
``app/main.py`` or ``app/models.py``. All error conditions raise
:class:`fastapi.HTTPException` with the status codes and message shapes defined
in the design's "Error Handling" table.

Validation ordering (design "Error Handling" — precedence and ordering notes):

1. Input presence — neither ``transactions`` nor ``bsaId`` present → 400.
2. Set-size checks — empty ``transactions`` → 400; length > 10,000 →
   400.
3. Per-transaction field/value checks — missing required field, negative amount,
   or unparseable date → 400 naming the field and the transaction index.
4. Rolling-window check — a specified ``rollingWindowDays < 1`` → 400.

The per-transaction amount scale/sign is already enforced by the
``TransactionInput`` model at bind time; this pass re-affirms the sign defensively
with index-specific messaging so it also works on an already-bound model, and
focuses on date parsing, which Pydantic keeps as a free-form string.

This module performs no I/O and makes no model calls.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import List

from fastapi import HTTPException

from app.models import ClassifyRequest, TransactionInput
from app.rule_engine import Transaction, make_transaction

# The maximum number of transactions a single request may carry.
MAX_TRANSACTIONS: int = 10_000

# The minimum valid Rolling_Window value in days. Values below this
# are rejected as HTTP 400.
MIN_ROLLING_WINDOW_DAYS: int = 1

# Date formats accepted by :func:`parse_transaction_date` beyond ISO-8601, which
# is handled first via ``date.fromisoformat`` / ``datetime.fromisoformat``. These
# cover the common calendar-date and timestamp shapes a caller is likely to send.
_SUPPORTED_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",            # 2024-01-31 (also handled by fromisoformat)
    "%Y/%m/%d",            # 2024/01/31
    "%m/%d/%Y",            # 01/31/2024
    "%m-%d-%Y",            # 01-31-2024
    "%d-%m-%Y",            # 31-01-2024
    "%d/%m/%Y",            # 31/01/2024
    "%Y%m%d",              # 20240131
    "%Y-%m-%dT%H:%M:%S",   # 2024-01-31T12:30:00
    "%Y-%m-%d %H:%M:%S",   # 2024-01-31 12:30:00
    "%m/%d/%Y %H:%M:%S",   # 01/31/2024 12:30:00
)


def _bad_request(detail: str) -> HTTPException:
    """Build an HTTP 400 :class:`HTTPException` with ``detail``.

    Centralizes the 400 construction so every semantic-validation failure uses a
    consistent shape (matching the existing endpoints' ``HTTPException`` usage).
    """
    return HTTPException(status_code=400, detail=detail)


def parse_transaction_date(value: str, index: int) -> date:
    """Parse a transaction ``date`` string into a calendar :class:`~datetime.date`.

    Accepts common formats, trying ISO-8601 first (``YYYY-MM-DD`` and full ISO
    timestamps via :meth:`datetime.date.fromisoformat` / ``datetime.fromisoformat``)
    and then a small set of common calendar and timestamp layouts
    (:data:`_SUPPORTED_DATE_FORMATS`). A trailing ``Z`` (UTC designator) on an ISO
    timestamp is tolerated.

    Raises HTTP 400 identifying the invalid ``date`` field and the ``index`` of the
    affected transaction when the value cannot be parsed.
    """
    if not isinstance(value, str) or not value.strip():
        raise _bad_request(
            f"Transaction at index {index} has an unparseable 'date' value: "
            f"{value!r}."
        )

    raw = value.strip()

    # 1) ISO-8601 calendar date (YYYY-MM-DD) — the primary expected shape.
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass

    # 2) ISO-8601 timestamp (tolerate a trailing 'Z' UTC designator).
    iso_candidate = raw[:-1] if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(iso_candidate).date()
    except ValueError:
        pass

    # 3) A small set of common explicit formats.
    for fmt in _SUPPORTED_DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue

    raise _bad_request(f"Transaction at index {index} has an unparseable 'date' value: {value!r}.")


def validate_rolling_window(request: ClassifyRequest) -> None:
    """Validate a specified ``rollingWindowDays`` value.

    When the request specifies a Rolling_Window value that is less than 1, raise
    HTTP 400 identifying the invalid window value. When ``rollingWindowDays``
    is omitted (``None``), no error is raised — the Rule_Engine applies its default
    of 30 days.
    """
    window = request.rollingWindowDays
    if window is not None and window < MIN_ROLLING_WINDOW_DAYS:
        raise _bad_request(
            f"rollingWindowDays must be at least {MIN_ROLLING_WINDOW_DAYS}; "
            f"received {window}."
        )


def _normalize_transaction(item: TransactionInput, index: int) -> Transaction:
    """Validate and normalize a single :class:`TransactionInput` at ``index``.

    Performs defensive per-transaction field/value checks with index-specific
    messaging, then builds the Rule_Engine
    :class:`~app.rule_engine.Transaction`:

    - Re-affirms the required fields are present (``amount``, ``date``,
      ``transactionType``) — a missing field maps to 400 naming the field and the
      index.
    - Re-affirms the amount is non-negative with index-specific messaging.
    - Parses the ``date`` string, raising 400 on failure.
    - Resolves ``isCash`` from the explicit flag or the transaction type and carries
      the original index and TIN last-4 through (full TINs are never carried).
    """
    # Required fields present. Pydantic enforces this at bind time, but
    # re-affirm defensively so this pass works on an already-bound model and
    # produces an index-specific message.
    if item.amount is None:
        raise _bad_request(
            f"Transaction at index {index} is missing required field 'amount'."
        )
    if item.date is None or (isinstance(item.date, str) and not item.date.strip()):
        raise _bad_request(
            f"Transaction at index {index} is missing required field 'date'."
        )
    if item.transactionType is None or not str(item.transactionType).strip():
        raise _bad_request(
            f"Transaction at index {index} is missing required field "
            f"'transactionType'."
        )

    # Non-negative amount, index-specific messaging.
    if item.amount < 0:
        raise _bad_request(
            f"Transaction at index {index} has an invalid negative 'amount': "
            f"{item.amount}."
        )

    # Parse the date (raises 400 with field + index on failure).
    parsed_date = parse_transaction_date(item.date, index)

    # Reduce any supplied TIN to its last four digits; never carry the full TIN.
    tin_last4 = item.tin[-4:] if item.tin else None

    return make_transaction(
        amount=item.amount,
        date=parsed_date,
        transactionType=item.transactionType,
        index=index,
        isCash=item.isCash,
        subject=item.subject,
        tinLast4=tin_last4,
    )


def normalize_and_validate(request: ClassifyRequest) -> List[Transaction]:
    """Run the full semantic validation pass over the transaction-set path.

    Applies the design's validation ordering and returns the normalized
    :class:`~app.rule_engine.Transaction` list ready for ``detect_structuring``:

    1. Presence check — neither ``transactions`` nor ``bsaId`` → 400.
    2. Empty set → 400 "must contain at least one Transaction".
    3. Oversize set (> 10,000) → 400 "exceeds the maximum of 10,000".
    4. Per-transaction field/value checks — missing field, negative amount,
       unparseable date → 400 naming the field and the index.
    5. Rolling-window check — specified ``rollingWindowDays < 1`` → 400.

    This handles the transaction path only. When the request carries a ``bsaId``
    and no ``transactions``, this returns an empty list without error so the caller
    can run the BSAID retrieval path; the window check still applies.
    """
    transactions = request.transactions

    # Distinguish "transactions key absent" (None) from "present but empty" ([]).
    # An absent key with a bsaId is the BSAID-only path; an explicitly empty set
    # is a 400 regardless of bsaId, so it must not fall through to the
    # BSAID path.
    transactions_supplied = transactions is not None

    # Step 1: input presence. Neither a transactions key nor a bsaId.
    if not transactions_supplied and not request.bsaId:
        raise _bad_request(
            "Request must include either a non-empty 'transactions' set or a "
            "'bsaId' reference; neither was provided."
        )

    # BSAID-only path: no transactions key supplied. Nothing to normalize here;
    # still enforce the window lower bound.
    if not transactions_supplied:
        validate_rolling_window(request)
        return []

    # Step 2: empty transaction set. The key was supplied but carries
    # zero records — a 400 even when a bsaId is also present.
    if len(transactions) == 0:
        raise _bad_request("The 'transactions' set must contain at least one Transaction.")

    # Step 3: oversize set.
    if len(transactions) > MAX_TRANSACTIONS:
        raise _bad_request(
            f"The 'transactions' set exceeds the maximum of "
            f"{MAX_TRANSACTIONS:,} Transaction records."
        )

    # Step 4: per-transaction field/value checks + normalization.
    normalized: List[Transaction] = [
        _normalize_transaction(item, index)
        for index, item in enumerate(transactions)
    ]

    # Step 5: rolling-window lower bound.
    validate_rolling_window(request)

    return normalized
