"""Deterministic Rule_Engine for the SAR Classification Agent.

This module contains the pure, deterministic structuring-detection logic for the
``POST /classify`` endpoint. It performs the CTR threshold and rolling-window math
with no external dependencies: no I/O, no boto3, no model calls, and no
non-determinism. It produces the authoritative structuring verdict, the triggering
and reportable transactions, and the base risk/confidence scores.

Interfaces (filled in by later tasks):
    - ``Transaction`` normalized dataclass
    - ``StructuringVerdict`` result container
    - ``detect_structuring(transactions, rolling_window_days=30) -> StructuringVerdict``

Purity contract: this module performs no I/O, uses no boto3, makes no model calls,
and contains no non-determinism (no clocks, no randomness). Every function is a
pure transformation of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

# ---------------------------------------------------------------------------
# Cash-type resolution
# ---------------------------------------------------------------------------

# Transaction-type tokens that denote a physical cash (currency) movement.
# Matching is case-insensitive and based on the presence of the "CASH" token so
# that both ``CASH_DEPOSIT`` and ``CASH_WITHDRAWAL`` (and similar variants) are
# recognized as cash movements for structuring aggregation.
_CASH_TRANSACTION_TYPES: frozenset[str] = frozenset({
    "CASH",
    "CASH_DEPOSIT",
    "CASH_WITHDRAWAL",
    "CURRENCY_DEPOSIT",
    "CURRENCY_WITHDRAWAL",
})


def transaction_type_is_cash(transaction_type: str) -> bool:
    """Return ``True`` when a transaction type denotes a cash (currency) movement.

    The check is case-insensitive and whitespace-tolerant. A type is treated as
    cash when it exactly matches a known cash token (e.g. ``CASH_DEPOSIT``,
    ``CASH_WITHDRAWAL``) or when it contains the ``CASH`` or ``CURRENCY`` token.
    This keeps the engine robust to minor naming variants while remaining a pure,
    deterministic string classification with no I/O.
    """
    if not transaction_type:
        return False
    normalized = transaction_type.strip().upper()
    if normalized in _CASH_TRANSACTION_TYPES:
        return True
    return "CASH" in normalized or "CURRENCY" in normalized


def resolve_is_cash(transaction_type: str, is_cash: Optional[bool]) -> bool:
    """Resolve the effective cash indicator for a transaction.

    ``isCash`` is ``True`` when the explicit flag is ``True``; otherwise it is
    inferred from ``transactionType`` denoting a cash movement. An
    explicit ``False`` does not force non-cash: a type that clearly denotes a cash
    movement still resolves to cash, since the transaction type is authoritative
    about the nature of the movement. ``None`` means "not specified", so the type
    is used.
    """
    if is_cash is True:
        return True
    return transaction_type_is_cash(transaction_type)


# ---------------------------------------------------------------------------
# Normalized Transaction
# ---------------------------------------------------------------------------


@dataclass
class Transaction:
    """A normalized, validated transaction used by the Rule_Engine.

    This is the internal representation produced after request validation and
    normalization. All fields are already validated and parsed:

    - ``amount``: USD as a :class:`~decimal.Decimal` with at most two decimal
      places and a non-negative value. Kept as ``Decimal`` to avoid
      float drift at the $10,000.00 / $9,999.99 boundaries.
    - ``date``: a parsed calendar :class:`~datetime.date`.
    - ``transactionType``: the raw transaction type token (e.g. ``CASH_DEPOSIT``).
    - ``isCash``: the resolved cash indicator — ``True`` when the
      explicit flag was ``True`` or the type denotes a cash movement.
    - ``index``: the original position of the transaction in the request, used for
      error reporting and result citations.
    - ``subject``: an optional opaque subject label (no PII required).
    - ``tinLast4``: optional last four digits of a taxpayer identification number
      (full TINs are never stored here).
    """

    amount: Decimal
    date: date
    transactionType: str
    isCash: bool
    index: int
    subject: Optional[str] = None
    tinLast4: Optional[str] = None


def make_transaction(
    *,
    amount: Decimal,
    date: date,
    transactionType: str,
    index: int,
    isCash: Optional[bool] = None,
    subject: Optional[str] = None,
    tinLast4: Optional[str] = None,
) -> Transaction:
    """Construct a normalized :class:`Transaction`, resolving the cash indicator.

    This is the normalization helper used by the validation layer to turn a
    validated request record into the engine's internal :class:`Transaction`. It
    resolves ``isCash`` from the explicit flag or the transaction type
    and carries the original ``index`` and ``tinLast4`` through unchanged.

    The caller is responsible for having already validated the ``amount`` (2 dp,
    non-negative) and parsed the ``date``; this helper performs no I/O and no
    further validation, keeping the module pure.
    """
    return Transaction(
        amount=amount,
        date=date,
        transactionType=transactionType,
        isCash=resolve_is_cash(transactionType, isCash),
        index=index,
        subject=subject,
        tinLast4=tinLast4,
    )

# ---------------------------------------------------------------------------
# Structuring detection
# ---------------------------------------------------------------------------

# The Currency Transaction Report (CTR) reporting threshold under 31 CFR.
# Kept as a Decimal to preserve exact monetary comparisons at the
# $10,000.00 / $9,999.99 boundaries (no float drift).
CTR_THRESHOLD: Decimal = Decimal("10000.00")

# The default Rolling_Window span in days applied when the request does not
# specify a value.
DEFAULT_ROLLING_WINDOW_DAYS: int = 30

# The canonical typology label reported when the structuring pattern is detected.
STRUCTURING_TYPOLOGY: str = "Structuring"

# ---------------------------------------------------------------------------
# Risk and confidence scoring parameters
# ---------------------------------------------------------------------------
# These constants are the deterministic tuning knobs for the pure scoring
# functions below. They are chosen so that:
# - risk stays within the integer range [1, 10];
# - confidence for a satisfied rule stays strictly below 1.0 so that the
#   additive near-threshold boost can raise it while remaining clamped to
#   <= 1.0 (preserving the strict-inequality property).

# Inclusive near-threshold band for contributing cash transactions.
NEAR_THRESHOLD_LOWER: Decimal = Decimal("9000.00")
NEAR_THRESHOLD_UPPER: Decimal = Decimal("9999.99")

# Risk scoring (integer 1..10).
_RISK_BASELINE: int = 3  # baseline severity when structuring is detected
_RISK_MIN: int = 1
_RISK_MAX: int = 10

# Confidence scoring (decimal 0.0..1.0).
# Base confidence for a satisfied rule. Kept well below 1.0 so the near-threshold
# boost has headroom and never pre-saturates (strict inequality).
_CONFIDENCE_BASE: float = 0.6
# Additive boost applied when any contributor is in the near-threshold band.
_CONFIDENCE_NEAR_THRESHOLD_BOOST: float = 0.2
_CONFIDENCE_FLOOR: float = 0.0
_CONFIDENCE_CEIL: float = 1.0


def is_near_threshold(amount: Decimal) -> bool:
    """Return ``True`` when ``amount`` lies in the near-threshold band.

    The near-threshold band is ``$9,000.00``–``$9,999.99`` inclusive.
    Comparisons use :class:`~decimal.Decimal` to stay exact at the band edges.
    """
    return NEAR_THRESHOLD_LOWER <= amount <= NEAR_THRESHOLD_UPPER


def compute_risk_score(
    detected: bool,
    aggregated_amount: Decimal,
    contributor_count: int,
) -> int:
    """Compute the deterministic Risk_Score as an integer in ``[1, 10]``.

    When structuring is not detected the risk is exactly ``1``. When
    detected, the score starts at a baseline and scales with two deterministic
    signals:

    - how far the aggregated contributing amount exceeds the CTR threshold
      (``floor(aggregated / threshold)`` extra points beyond the first multiple);
    - the number of contributing transactions beyond the minimum of two.

    The result is clamped to ``[1, 10]``. The function is pure: equal inputs
    always yield the same score.
    """
    if not detected:
        return _RISK_MIN  # no detection ⇒ risk is exactly 1

    # Scale with aggregated amount relative to the threshold. The first threshold
    # multiple is already implied by detection; each additional full multiple of
    # the threshold adds severity.
    amount_multiple = int(aggregated_amount // CTR_THRESHOLD)
    amount_points = max(0, amount_multiple - 1)

    # Scale with the number of contributors beyond the two required to structure.
    contributor_points = max(0, contributor_count - 2)

    score = _RISK_BASELINE + amount_points + contributor_points
    # Clamp to [1, 10].
    return max(_RISK_MIN, min(_RISK_MAX, score))


def compute_confidence_score(
    detected: bool,
    triggering_transactions: list[Transaction],
) -> float:
    """Compute the deterministic Confidence_Score as a decimal in ``[0.0, 1.0]``.

    When structuring is not detected the confidence is a low ``0.0``. When
    detected, confidence is a base value for a satisfied rule plus an additive
    near-threshold boost applied when any contributing transaction's amount lies
    in the ``$9,000.00``–``$9,999.99`` band. The sum is clamped to
    ``<= 1.0``.

    The base is intentionally below ``1.0`` and leaves room for the boost so that
    an otherwise-identical set that includes a near-threshold contributor yields a
    strictly higher confidence than one that does not — the key property. The
    function is pure and deterministic.
    """
    if not detected:
        return _CONFIDENCE_FLOOR  # sensible low value for the no-detection path

    confidence = _CONFIDENCE_BASE
    if any(is_near_threshold(tx.amount) for tx in triggering_transactions):
        confidence += _CONFIDENCE_NEAR_THRESHOLD_BOOST

    # Clamp to [0.0, 1.0].
    return max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEIL, confidence))


@dataclass
class StructuringVerdict:
    """The result of a structuring-detection evaluation.

    This is the authoritative, deterministic verdict produced by
    :func:`detect_structuring`. It carries the detection outcome, the aggregated
    contributing amount, the transactions that drove the detection, the cash
    transactions set aside as reportable, the window parameters used, and a
    structured rationale.

    Fields:
    - ``detected``: ``True`` when the structuring pattern was satisfied.
    - ``typology``: :data:`STRUCTURING_TYPOLOGY` when detected, otherwise ``None``.
    - ``aggregatedAmount``: the sum of the ``triggeringTransactions`` amounts; the
      value that reached or exceeded the CTR threshold (``Decimal("0.00")`` when
      not detected).
    - ``triggeringTransactions``: the contributing cash transactions in the
      satisfying window. Empty when not detected.
    - ``reportableTransactions``: cash transactions with ``amount >= $10,000.00``,
      recorded separately as reportable rather than as structuring contributors.
    - ``riskScore`` / ``confidenceScore``: deterministic scores. These are field
      slots populated by the scoring logic; they carry neutral
      placeholder defaults here (``1`` and ``0.0``).
    - ``windowDays``: the Rolling_Window value used for the evaluation (specified
      or the default of 30).
    - ``windowSpanDays``: the inclusive date span (in days) of the satisfying
      contributing subset, or ``None`` when not detected.
    - ``ruleRationale``: a structured dict of the facts behind the verdict, used
      to build the human-readable rationale.
    """

    detected: bool
    typology: Optional[str]
    aggregatedAmount: Decimal
    triggeringTransactions: list[Transaction]
    reportableTransactions: list[Transaction]
    windowDays: int
    windowSpanDays: Optional[int]
    ruleRationale: dict
    # Scoring slots — populated by the scoring logic. Neutral placeholder defaults.
    riskScore: int = 1
    confidenceScore: float = 0.0


def detect_structuring(
    transactions: list[Transaction],
    rolling_window_days: Optional[int] = None,
) -> StructuringVerdict:
    """Detect the structuring pattern over a set of normalized transactions.

    Structuring is the pattern in which two or more cash transactions, each
    strictly below the CTR threshold of $10,000.00, fall within a single
    Rolling_Window span and aggregate to a value greater than or equal to the
    threshold.

    Algorithm (mirrors design.md "Structuring detection algorithm", steps 1-7):

    1. Partition by cash indicator. Only cash-type transactions participate;
       non-cash transactions are excluded from the aggregation.
    2. Separate reportable transactions. Any cash transaction with
       ``amount >= $10,000.00`` is reportable, not a contributor, and is recorded
       separately.
    3. Candidate contributors are the remaining cash transactions strictly below
       the threshold.
    4. Sort candidates by date and slide a window: for each start index, extend
       forward while the inclusive span ``(latest - earliest).days`` stays
       ``<= rolling_window_days``, tracking the running sum; the pattern is
       satisfied when any such span's sum reaches ``>= $10,000.00``.
    5. The contributing cash transactions in that satisfying window are the
       ``triggeringTransactions``.
    6. Default the window to 30 days when none is supplied; otherwise use the
       specified value.
    7. If no qualifying subset exists, report that structuring was not detected.

    The function is pure and deterministic: it performs no I/O and depends only on
    its inputs. ``rolling_window_days`` is expected to be ``>= 1`` (values ``< 1``
    are rejected upstream as HTTP 400); ``None`` applies the default.
    """
    window_days = (
        DEFAULT_ROLLING_WINDOW_DAYS
        if rolling_window_days is None
        else rolling_window_days
    )

    # Step 1 & 2 & 3: partition cash vs non-cash, split off reportable cash, and
    # collect sub-threshold cash as candidate contributors.
    reportable: list[Transaction] = []
    candidates: list[Transaction] = []
    for tx in transactions:
        if not tx.isCash:
            continue  # non-cash excluded from the structuring aggregation
        if tx.amount >= CTR_THRESHOLD:
            reportable.append(tx)  # reportable, not a contributor
        else:
            candidates.append(tx)  # strictly < threshold candidate

    # Step 4: sort candidates by date and slide a window looking for a qualifying
    # subset. Preserve original index as a stable tie-breaker so equal-date
    # transactions order deterministically.
    sorted_candidates = sorted(candidates, key=lambda t: (t.date, t.index))

    triggering, span_days = _find_structuring_window(sorted_candidates, window_days)

    if triggering is None:
        # Step 7: no-detection path.
        return StructuringVerdict(
            detected=False,
            typology=None,
            aggregatedAmount=Decimal("0.00"),
            triggeringTransactions=[],
            reportableTransactions=reportable,
            windowDays=window_days,
            windowSpanDays=None,
            ruleRationale={
                "detected": False,
                "typology": None,
                "threshold": str(CTR_THRESHOLD),
                "windowDays": window_days,
                "candidateCount": len(candidates),
                "reportableCount": len(reportable),
                "reason": "No subset of sub-threshold cash transactions aggregated "
                "to the CTR threshold within one rolling-window span.",
            },
            riskScore=compute_risk_score(False, Decimal("0.00"), 0),
            confidenceScore=compute_confidence_score(False, []),
        )

    aggregated = sum((t.amount for t in triggering), Decimal("0.00"))

    # Step 5 & 6: structuring detected; report triggering transactions.
    return StructuringVerdict(
        detected=True,
        typology=STRUCTURING_TYPOLOGY,
        aggregatedAmount=aggregated,
        triggeringTransactions=triggering,
        reportableTransactions=reportable,
        windowDays=window_days,
        windowSpanDays=span_days,
        ruleRationale={
            "detected": True,
            "typology": STRUCTURING_TYPOLOGY,
            "threshold": str(CTR_THRESHOLD),
            "aggregatedAmount": str(aggregated),
            "contributorCount": len(triggering),
            "windowDays": window_days,
            "windowSpanDays": span_days,
            "reportableCount": len(reportable),
        },
        riskScore=compute_risk_score(True, aggregated, len(triggering)),
        confidenceScore=compute_confidence_score(True, triggering),
    )


# ---------------------------------------------------------------------------
# Rule-based rationale text
# ---------------------------------------------------------------------------

# Human-in-the-loop framing shared by both rationale paths. The Rule_Engine
# flags and assesses; it never asserts that a legal violation occurred, so the
# rationale uses review-oriented language and defers the decision to an Analyst
# (mirrored by the Reasoner's Title 21 soft-language framing).
_RATIONALE_REVIEW_SUFFIX: str = (
    "This is an automated assessment that flags a potential pattern for Analyst "
    "review; it does not assert that a violation occurred."
)


def build_rationale(verdict: StructuringVerdict) -> str:
    """Build a non-empty, human-readable rule-based rationale for a verdict.

    For a detected structuring verdict the rationale text explicitly states the
    triggering rule, the aggregated amount, the number of contributing
    transactions, and the Rolling_Window value used. For the
    not-detected path it returns a sensible non-empty message describing that the
    structuring pattern was not satisfied.

    The language is review-oriented and human-in-the-loop: the rationale frames
    the result as a flag for Analyst review and never asserts that a legal
    violation occurred. The function is pure: it derives the text
    solely from ``verdict`` and performs no I/O.
    """
    if not verdict.detected:
        return (
            "The Structuring typology was not detected: no subset of two or more "
            "cash transactions each below the "
            f"${CTR_THRESHOLD:,.2f} CTR reporting threshold aggregated to that "
            f"threshold within a single {verdict.windowDays}-day Rolling_Window "
            f"span. {_RATIONALE_REVIEW_SUFFIX}"
        )

    aggregated = verdict.aggregatedAmount
    contributor_count = len(verdict.triggeringTransactions)
    window_days = verdict.windowDays

    # Pluralize for readable prose without changing the stated facts.
    tx_noun = "transaction" if contributor_count == 1 else "transactions"

    rationale = (
        "Potential Structuring flagged by the aggregation rule: "
        f"{contributor_count} contributing cash {tx_noun}, each below the "
        f"${CTR_THRESHOLD:,.2f} CTR reporting threshold, aggregate to "
        f"${aggregated:,.2f} within a {window_days}-day Rolling_Window"
    )

    if verdict.windowSpanDays is not None:
        span = verdict.windowSpanDays
        span_noun = "day" if span == 1 else "days"
        rationale += f" (contributing {tx_noun} span {span} {span_noun}, inclusive)"

    if verdict.reportableTransactions:
        reportable_count = len(verdict.reportableTransactions)
        reportable_noun = "transaction" if reportable_count == 1 else "transactions"
        rationale += (
            f"; {reportable_count} separate cash {reportable_noun} at or above "
            f"the ${CTR_THRESHOLD:,.2f} threshold recorded as reportable"
        )

    rationale += f". {_RATIONALE_REVIEW_SUFFIX}"
    return rationale


def _find_structuring_window(
    sorted_candidates: list[Transaction],
    window_days: int,
) -> tuple[Optional[list[Transaction]], Optional[int]]:
    """Find the earliest qualifying structuring window in sorted candidates.

    ``sorted_candidates`` must be cash transactions strictly below the threshold,
    ordered by ``(date, index)``. For each start index, the window is extended
    forward while the inclusive date span from the start transaction stays within
    ``window_days``, accumulating the running sum. The first window whose running
    sum reaches ``>= CTR_THRESHOLD`` with two or more contributors satisfies the
    structuring condition.

    Returns ``(triggeringTransactions, windowSpanDays)`` for the satisfying window,
    or ``(None, None)`` when no window qualifies. The span is measured as
    ``(latest - earliest).days`` over the triggering subset, inclusive.

    This is a pure helper with no I/O.
    """
    n = len(sorted_candidates)
    for start in range(n):
        running_sum = Decimal("0.00")
        start_date = sorted_candidates[start].date
        subset: list[Transaction] = []
        for end in range(start, n):
            tx = sorted_candidates[end]
            span = (tx.date - start_date).days
            if span > window_days:
                break  # beyond the window; no further transaction can qualify here
            running_sum += tx.amount
            subset.append(tx)
            if len(subset) >= 2 and running_sum >= CTR_THRESHOLD:
                return list(subset), span
    return None, None
