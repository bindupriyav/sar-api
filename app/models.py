"""Pydantic request/response models for the SAR Classification Agent.

Defines the ``POST /classify`` request and response schemas plus the output
minimization/disclosure helpers.

Interfaces (filled in by later tasks):
    - ``TransactionInput`` / ``ClassifyRequest`` request models
    - ``TransactionRef`` / ``ClassificationResult`` response models
    - output minimization + disclosure assembly (TIN last-4 only, etc.)

This is a scaffolding stub so later tasks have an import target.
"""
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request models (POST /classify request side)
# ---------------------------------------------------------------------------


class TransactionInput(BaseModel):
    """A single transaction supplied by the caller for classification.

    ``amount`` is kept as a :class:`~decimal.Decimal` to preserve exact monetary
    precision (no float drift at the $10,000.00 / $9,999.99 boundaries). The
    amount is validated at the model level to reject negative values and values
    with more than two digits after the decimal point.
    """

    amount: Decimal = Field(..., description="USD, at most 2 decimal places")
    date: str = Field(..., description="Calendar date / timestamp of the transaction")
    transactionType: str = Field(..., description="e.g. CASH_DEPOSIT, WIRE, ACH")
    isCash: Optional[bool] = Field(
        None, description="Explicit cash indicator; inferred from type if omitted"
    )
    subject: Optional[str] = Field(
        None, description="Opaque subject label (no PII required)"
    )
    tin: Optional[str] = Field(
        None, description="If present, only last 4 are ever echoed"
    )

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, value: Decimal) -> Decimal:
        # Reject NaN / infinity, which Decimal would otherwise accept.
        if not value.is_finite():
            raise ValueError("amount must be a finite decimal value")
        # Non-negative.
        if value < 0:
            raise ValueError("amount must be non-negative")
        # At most two decimal places. The exponent of a normalized
        # Decimal is the negative number of fractional digits; e.g. Decimal
        # "100.123" has exponent -3. Using the tuple form avoids float drift and
        # preserves the exact value (e.g. 9999.99 and 10000.00 pass unchanged).
        exponent = value.as_tuple().exponent
        if isinstance(exponent, int) and exponent < -2:
            raise ValueError("amount must have at most two decimal places")
        return value


class ClassifyRequest(BaseModel):
    """The ``POST /classify`` request body.

    Exactly one input source is required downstream: a non-empty
    ``transactions`` set (which takes precedence) or a ``bsaId`` reference.
    Presence/empty/oversize semantics are enforced by the semantic validation
    pass, not here. ``rollingWindowDays`` defaults to 30 in the Rule_Engine; a
    value ``< 1`` is rejected upstream as HTTP 400.
    """

    transactions: Optional[List[TransactionInput]] = None
    bsaId: Optional[str] = None
    rollingWindowDays: Optional[int] = None
    enableModelReasoning: Optional[bool] = False

# ---------------------------------------------------------------------------
# Response models (POST /classify response side)
# ---------------------------------------------------------------------------


class TransactionRef(BaseModel):
    """A minimized reference to a transaction cited in the classification.

    This is the output-side projection of a transaction: it carries only the
    fields that are safe and useful to echo back to the caller. Any subject
    identifier is reduced to the TIN last-4 (``tinLast4``) so that no full TIN
    or raw PII leaves the service. ``amount`` stays a
    :class:`~decimal.Decimal` to preserve exact monetary precision at the
    reporting boundary.
    """

    index: int = Field(..., description="Position of the transaction in the input set")
    amount: Decimal = Field(..., description="USD amount, at most 2 decimal places")
    date: str = Field(..., description="Calendar date / timestamp of the transaction")
    transactionType: str = Field(..., description="e.g. CASH_DEPOSIT, WIRE, ACH")
    tinLast4: Optional[str] = Field(
        None, description="Last 4 of the TIN only; never the full TIN"
    )


class ClassificationResult(BaseModel):
    """The ``POST /classify`` response body.

    Produced by the output minimization/disclosure assembly step. Carries the
    deterministic Rule_Engine outcome (typology, scores, triggering/reportable
    transactions, rationale) plus optional model-reasoning and disclosure
    fields. Every result includes ``modelDisclosure`` stating the output is
    AI-produced and requires Analyst review. Ambiguous input is
    reported via ``ambiguityNotes`` rather than resolved.
    """

    typology: Optional[str] = Field(
        None, description='Detected typology, e.g. "Structuring", or None'
    )
    riskScore: int = Field(..., description="Risk score in the range 1..10")
    confidenceScore: float = Field(..., description="Confidence in the range 0.0..1.0")
    triggeringTransactions: List[TransactionRef] = Field(
        default_factory=list,
        description="Transactions that triggered the classification",
    )
    reportableTransactions: List[TransactionRef] = Field(
        default_factory=list,
        description="Transactions deemed reportable for the SAR",
    )
    rationale: str = Field(..., description="Rule-based rationale (always present)")
    bsaId: Optional[str] = Field(
        None, description="Source citation when the classification referenced stored data"
    )
    modelReasoningAvailable: bool = Field(
        ..., description="True only if a Bedrock rationale was produced"
    )
    modelReasoning: Optional[str] = Field(
        None, description="Natural-language rationale when available"
    )
    modelDisclosure: str = Field(
        ..., description="States the result is AI-produced and requires Analyst review"
    )
    syntheticDataDisclosure: Optional[str] = Field(
        None, description="Populated when the classified data originated from synthetic data"
    )
    requestId: Optional[str] = Field(
        None, description="Echo of the X-Request-Id header; None when absent"
    )
    ambiguityNotes: Optional[List[str]] = Field(
        None, description="Reported ambiguities, not resolved"
    )

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Standing model disclosure. Every Classification_Result carries this statement
# so the output is always framed as an AI-produced flag requiring human review
# before any action is taken. The language is review-oriented and
# never asserts that a violation occurred.
MODEL_DISCLOSURE: str = (
    "This classification was produced by an automated AI agent. It is a flag for "
    "review, not a legal determination, and requires Analyst review before any "
    "action is taken. Any recommended action is advisory only and requires "
    "Analyst approval before execution."
)

# Standing synthetic-data disclosure, surfaced when the classified data
# originates from the synthetic SAR data set (stored records carry
# ``SyntheticData: true``).
SYNTHETIC_DATA_DISCLOSURE: str = (
    "The classified data originates from the synthetic SAR data set and does not "
    "represent real persons, accounts, or transactions."
)


def tin_last4(tin: Optional[str]) -> Optional[str]:
    """Reduce a supplied TIN to its last four digits only.

    Reuses the ``id_num[-4:]`` minimization philosophy from
    ``app.main.minimize_sar_response``: a full taxpayer identification number is
    never emitted. ``None`` or an empty/whitespace-only value maps
    to ``None`` so the output simply omits the field; any longer value is reduced
    to its trailing four characters.
    """
    if not tin:
        return None
    trimmed = tin.strip()
    if not trimmed:
        return None
    return trimmed[-4:]


def _to_transaction_ref(tx) -> "TransactionRef":
    """Project a Rule_Engine ``Transaction`` onto a minimized ``TransactionRef``.

    The date is serialized as an ISO-8601 string (``date.isoformat()``) and the
    TIN is carried through as last-4 only — the full TIN never leaves the service. ``tx`` is the ``app.rule_engine.Transaction`` dataclass
    (duck-typed here to keep this module free of a hard import dependency on the
    engine). The ``tinLast4`` field is assumed already reduced upstream; it is
    re-reduced defensively so a full TIN can never slip through.
    """
    return TransactionRef(
        index=tx.index,
        amount=tx.amount,
        date=tx.date.isoformat(),
        transactionType=tx.transactionType,
        tinLast4=tin_last4(getattr(tx, "tinLast4", None)),
    )


def assemble_classification_result(
    verdict,
    *,
    rationale: Optional[str] = None,
    bsa_id: Optional[str] = None,
    model_reasoning: Optional[str] = None,
    model_reasoning_available: bool = False,
    synthetic_data: bool = False,
    request_id: Optional[str] = None,
    ambiguity_notes: Optional[List[str]] = None,
) -> "ClassificationResult":
    """Assemble a minimized, disclosure-bearing ``ClassificationResult``.

    This is the pure output-assembly step (no I/O, no boto3, no model calls). It
    takes a Rule_Engine ``StructuringVerdict`` plus optional context and produces
    the response model, applying the project's data-handling principles:

    - **TIN minimization:** every cited transaction is projected
      onto a :class:`TransactionRef` with the date as an ISO string and the TIN
      reduced to last-4 only; a full TIN never appears in the output.
    - **Model disclosure:** ``modelDisclosure`` is always a
      non-empty statement that the result is AI-produced, is a flag (not a legal
      determination), and requires Analyst review, and that any recommended
      action requires Analyst approval.
    - **BSAID citation:** ``bsaId`` is set when the classification
      derived from or referenced stored data.
    - **Synthetic-data disclosure:** ``syntheticDataDisclosure`` is set
      when ``synthetic_data`` is ``True`` (stored source carried
      ``SyntheticData: true``).
    - **Ambiguity surfacing:** ``ambiguityNotes`` is passed through
      verbatim so ambiguity is reported, not resolved.
    - **Request correlation:** ``requestId`` echoes the supplied
      ``X-Request-Id`` value (``None`` when absent).

    ``rationale`` defaults to the engine's :func:`build_rationale(verdict)` when
    not explicitly supplied. ``verdict`` is duck-typed as the engine's
    ``StructuringVerdict`` to keep this module decoupled from the engine import.
    """
    # Default the rationale to the engine's rule-based text when not supplied.
    if rationale is None:
        # Local import keeps app.models importable without the engine at import
        # time and avoids a circular dependency.
        from app.rule_engine import build_rationale

        rationale = build_rationale(verdict)

    triggering = [_to_transaction_ref(tx) for tx in verdict.triggeringTransactions]
    reportable = [_to_transaction_ref(tx) for tx in verdict.reportableTransactions]

    # Normalize empty ambiguity lists to None so the field is simply omitted.
    notes = ambiguity_notes if ambiguity_notes else None

    return ClassificationResult(
        typology=verdict.typology,
        riskScore=verdict.riskScore,
        confidenceScore=verdict.confidenceScore,
        triggeringTransactions=triggering,
        reportableTransactions=reportable,
        rationale=rationale,
        bsaId=bsa_id,
        modelReasoningAvailable=model_reasoning_available,
        modelReasoning=model_reasoning,
        modelDisclosure=MODEL_DISCLOSURE,
        syntheticDataDisclosure=SYNTHETIC_DATA_DISCLOSURE if synthetic_data else None,
        requestId=request_id,
        ambiguityNotes=notes,
    )
