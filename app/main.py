"""
SAR Data API — FastAPI service (§4.2 aligned)

Read-only REST API for querying synthetic SAR records from DynamoDB.
Follows the design principles in SAR_Analysis_Agent_Design_Outline §4:
  - No PII in URLs (§4.1)
  - Opaque entity IDs for subject queries
  - PII only in request headers (POST /entities:resolve)
  - Pagination via cursor on list endpoints
  - Common header validation (§4.3)
"""
import base64
import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import boto3
from boto3.dynamodb.conditions import Key

from decimal import Decimal, InvalidOperation

from app.models import (
    ClassifyRequest,
    ClassificationResult,
    assemble_classification_result,
)
from app.validation import normalize_and_validate
from app.rule_engine import detect_structuring, StructuringVerdict
from app.reasoner import generate_rationale

app = FastAPI(title="SAR Data API", version="2.0.0")

# DynamoDB setup
REGION = os.environ.get("AWS_REGION", "us-east-2")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "sar-prototype")
ENTITY_TABLE_NAME = os.environ.get("ENTITY_TABLE", "sar-entities")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

# Entity table may not exist yet — handle gracefully
try:
    entity_table = dynamodb.Table(ENTITY_TABLE_NAME)
except Exception:
    entity_table = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_cursor(cursor: Optional[str]) -> Optional[dict]:
    """Decode a base64-encoded pagination cursor back to DynamoDB ExclusiveStartKey."""
    if not cursor:
        return None
    try:
        return json.loads(base64.b64decode(cursor).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid cursor")


def encode_cursor(last_evaluated_key: Optional[dict]) -> Optional[str]:
    """Encode DynamoDB LastEvaluatedKey as a base64 cursor string."""
    if not last_evaluated_key:
        return None
    return base64.b64encode(json.dumps(last_evaluated_key, default=str).encode("utf-8")).decode("utf-8")


def validate_common_headers(x_case_id: Optional[str], x_request_id: Optional[str]):
    """Validate required common headers per §4.3."""
    # For prototype, we log but don't hard-enforce yet
    # In production, uncomment the raise statements
    # if not x_case_id:
    #     raise HTTPException(status_code=400, detail="X-Case-Id header required")
    # if not x_request_id:
    #     raise HTTPException(status_code=400, detail="X-Request-Id header required")
    pass


def extract_classification_code(sar_item: dict) -> Optional[str]:
    """Extract the primary classification code from a SAR record."""
    activity = sar_item.get("Activity", {})
    suspicious = activity.get("SuspiciousActivity", [])
    if suspicious:
        classifications = suspicious[0].get("SuspiciousActivityClassification", [])
        if classifications:
            return classifications[0].get("SuspiciousActivityTypeCodeDescription")
    return None


def minimize_sar_response(item: dict) -> dict:
    """Return a summary view of a SAR (for list endpoints). No full TIN ever returned."""
    activity = item.get("Activity", {})
    suspicious = activity.get("SuspiciousActivity", [{}])
    classifications = []
    if suspicious:
        for cls in suspicious[0].get("SuspiciousActivityClassification", []):
            classifications.append({
                "code": cls.get("SuspiciousActivityTypeID"),
                "description": cls.get("SuspiciousActivityTypeCodeDescription"),
            })

    # Extract subject info (minimized — name and TIN last-4 only)
    subjects = []
    for party in activity.get("Party", []):
        if party.get("ActivityPartyTypeCode") == "33":
            names = party.get("PartyName", [{}])
            full_name = names[0].get("RawPartyFullName", "") if names else ""
            ids = party.get("PartyIdentification", [{}])
            id_num = ids[0].get("PartyIdentificationNumberText", "") if ids else ""
            subjects.append({
                "displayName": full_name,
                "tinLast4": id_num[-4:] if id_num else "",
            })

    # Extract institution info
    institution = {}
    for party in activity.get("Party", []):
        if party.get("ActivityPartyTypeCode") == "30":
            names = party.get("PartyName", [{}])
            institution = {
                "legalName": names[0].get("RawPartyFullName", "") if names else "",
            }
            break

    return {
        "BSAID": str(activity.get("BSAID", "")),
        "filingDate": activity.get("FilingDateText", ""),
        "filingInstitution": institution,
        "suspiciousActivity": {
            "totalSuspiciousAmount": suspicious[0].get("TotalSuspiciousAmountText") if suspicious else None,
            "classification": classifications,
        },
        "subjects": subjects,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "table": TABLE_NAME, "version": "2.0.0"}


# §4.4.2 — GET /sars/{bsaId}
@app.get("/sars/{bsa_id}")
def get_sar_by_bsaid(
    bsa_id: str,
    x_case_id: Optional[str] = Header(None, alias="X-Case-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """Get a single SAR by BSA ID. bsaId is not PII (§4.1)."""
    validate_common_headers(x_case_id, x_request_id)
    result = table.get_item(Key={"PK": f"SAR#{bsa_id}", "SK": "METADATA"})
    item = result.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="SAR not found")
    return item


# §4.4.3 — GET /institutions/{institutionId}/sars
@app.get("/institutions/{institution_id}/sars")
def get_sars_by_institution(
    institution_id: str,
    since: str = "2000-01-01",
    limit: int = 20,
    cursor: Optional[str] = None,
    x_case_id: Optional[str] = Header(None, alias="X-Case-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """Get SARs filed by an institution (GSI1 query). Paginated."""
    validate_common_headers(x_case_id, x_request_id)

    query_params = {
        "IndexName": "GSI1-InstitutionFilingDate",
        "KeyConditionExpression": Key("GSI1PK").eq(f"INSTITUTION#{institution_id}")
        & Key("GSI1SK").gte(f"FILINGDATE#{since}"),
        "Limit": min(limit, 100),
    }

    start_key = decode_cursor(cursor)
    if start_key:
        query_params["ExclusiveStartKey"] = start_key

    result = table.query(**query_params)

    items = [minimize_sar_response(item) for item in result.get("Items", [])]
    next_cursor = encode_cursor(result.get("LastEvaluatedKey"))

    return {
        "institutionId": institution_id,
        "count": len(items),
        "items": items,
        "nextCursor": next_cursor,
    }


# §4.4.4 — GET /classifications/{code}/sars
@app.get("/classifications/{code}/sars")
def get_sars_by_classification(
    code: str,
    since: str = "2000-01-01",
    limit: int = 20,
    cursor: Optional[str] = None,
    x_case_id: Optional[str] = Header(None, alias="X-Case-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """
    Get SARs by classification/typology code (e.g. 'Structuring', 'Fraud', 'Money Laundering').

    NOTE: In production this would use GSI3. For the prototype, we scan and filter
    since GSI3 doesn't exist on the sar-prototype table yet.
    """
    validate_common_headers(x_case_id, x_request_id)

    # Prototype: scan + filter (no GSI3 yet)
    scan_params = {"Limit": 500}
    start_key = decode_cursor(cursor)
    if start_key:
        scan_params["ExclusiveStartKey"] = start_key

    result = table.scan(**scan_params)
    matches = []

    for item in result.get("Items", []):
        activity = item.get("Activity", {})
        filing_date = activity.get("FilingDateText", "")
        if filing_date < since:
            continue

        classification = extract_classification_code(item)
        if classification and code.lower() in classification.lower():
            matches.append(minimize_sar_response(item))
            if len(matches) >= limit:
                break

    next_cursor = encode_cursor(result.get("LastEvaluatedKey")) if len(matches) >= limit else None

    return {
        "classificationCode": code,
        "count": len(matches),
        "items": matches,
        "nextCursor": next_cursor,
    }


# §4.4.1 — GET /entities/{entityId}/sars
@app.get("/entities/{entity_id}/sars")
def get_sars_by_entity(
    entity_id: str,
    x_case_id: Optional[str] = Header(None, alias="X-Case-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """
    Get all SARs for a resolved entity. entityId is an opaque internal key — not PII (§4.1).

    The entity mapping is stored in a separate DynamoDB table (sar-entities)
    that maps entityId -> subject identifiers used for GSI2 queries.
    """
    validate_common_headers(x_case_id, x_request_id)

    # Look up entity mapping
    if entity_table:
        try:
            entity_result = entity_table.get_item(
                Key={"PK": f"ENTITY#{entity_id}", "SK": "METADATA"}
            )
            entity = entity_result.get("Item")
        except Exception:
            entity = None
    else:
        entity = None

    if not entity:
        # Fallback: for prototype, treat entity_id as a subject identifier
        # and query GSI2 directly (backwards compat during migration)
        result = table.query(
            IndexName="GSI2-Subject",
            KeyConditionExpression=Key("GSI2PK").eq(f"SUBJECT#{entity_id}"),
        )
        items = result.get("Items", [])
        if not items:
            raise HTTPException(status_code=404, detail="Entity not found")

        return {
            "entityId": entity_id,
            "sarCount": len(items),
            "sars": [minimize_sar_response(item) for item in items],
        }

    # Use the mapped subject identifiers to query
    subject_ids = entity.get("subjectIdentifiers", [])
    all_sars = []

    for subject_id in subject_ids:
        result = table.query(
            IndexName="GSI2-Subject",
            KeyConditionExpression=Key("GSI2PK").eq(f"SUBJECT#{subject_id}"),
        )
        all_sars.extend(result.get("Items", []))

    return {
        "entityId": entity_id,
        "sarCount": len(all_sars),
        "sars": [minimize_sar_response(item) for item in all_sars],
    }


# §4.4.5 — POST /entities:resolve (PII in headers only)
@app.post("/entities:resolve")
async def resolve_entity(
    request: Request,
    x_case_id: Optional[str] = Header(None, alias="X-Case-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """
    Entity resolution search. PII comes exclusively from request headers (§4.1, §4.4.5).
    Never in URL, never in request body.

    Required headers (at least one): X-Subject-Name, X-Subject-Tin, X-Subject-Dob
    Optional body: matchThreshold, maxResults, institutionScope, dateRange
    """
    validate_common_headers(x_case_id, x_request_id)

    headers = request.headers
    subject_name = headers.get("x-subject-name", "")
    subject_tin = headers.get("x-subject-tin", "")
    subject_dob = headers.get("x-subject-dob", "")

    if not any([subject_name, subject_tin, subject_dob]):
        raise HTTPException(
            status_code=400,
            detail="At least one of X-Subject-Name, X-Subject-Tin, X-Subject-Dob required",
        )

    # Parse optional body for search refinement
    match_threshold = 0.0
    max_results = 10
    try:
        body = await request.json()
        match_threshold = body.get("matchThreshold", 0.0)
        max_results = body.get("maxResults", 10)
    except Exception:
        pass  # No body is fine

    # Scan and fuzzy match (prototype only — production uses OpenSearch)
    result = table.scan()
    matches = []
    seen = set()

    for item in result.get("Items", []):
        activity = item.get("Activity", {})
        parties = activity.get("Party", [])

        for party in parties:
            if party.get("ActivityPartyTypeCode") != "33":  # Subject only
                continue

            names = party.get("PartyName", [{}])
            full_name = names[0].get("RawPartyFullName", "") if names else ""
            ids = party.get("PartyIdentification", [{}])
            id_num = ids[0].get("PartyIdentificationNumberText", "") if ids else ""
            dob = party.get("BirthDateText", "")
            bsa_id = str(activity.get("BSAID", ""))

            if bsa_id in seen:
                continue

            matched_on = []
            if subject_name and subject_name.lower() in full_name.lower():
                matched_on.append("name")
            if subject_tin and id_num:
                # Support SHA256: prefix or raw last-4 matching
                if subject_tin.startswith("SHA256:"):
                    # In production, compare hashes
                    pass
                elif subject_tin[-4:] == id_num[-4:]:
                    matched_on.append("tin")
            if subject_dob and dob == subject_dob:
                matched_on.append("dob")

            if matched_on:
                confidence = round(len(matched_on) / 3, 2)
                if confidence >= match_threshold:
                    seen.add(bsa_id)
                    matches.append({
                        "entityId": f"ENT-{bsa_id[-4:]}",  # Prototype opaque ID
                        "matchConfidence": confidence,
                        "matchedOn": matched_on,
                        "displayName": full_name,
                        "tinLast4": id_num[-4:] if id_num else "",
                        "sarCount": 1,
                    })

            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    return {"matches": matches}


# §Classification — optional model-reasoning wiring
# ---------------------------------------------------------------------------

# Notice appended to ``ambiguityNotes`` when model reasoning was requested but
# the Bedrock_Reasoner returned no rationale (failure/timeout). The deterministic
# Rule_Engine result is still returned with HTTP 200.
MODEL_UNAVAILABLE_NOTICE: str = (
    "Model reasoning was requested but is currently unavailable; the "
    "deterministic Rule_Engine result is returned."
)


def _build_reasoning_context(verdict: "StructuringVerdict", bsa_id: Optional[str]) -> dict:
    """Build a minimal context dict for the Bedrock_Reasoner.

    Surfaces the aggregated amount, detected typology, and rolling window used
    (plus the BSAID citation when present) without leaking PII or raw inputs.
    """
    context = {
        "aggregateSignal": str(verdict.aggregatedAmount),
        "typology": verdict.typology,
        "windowDays": verdict.windowDays,
    }
    if bsa_id:
        context["bsaId"] = bsa_id
    return context


def _apply_model_reasoning(
    result: "ClassificationResult",
    verdict: "StructuringVerdict",
    enabled: Optional[bool],
    bsa_id: Optional[str] = None,
) -> "ClassificationResult":
    """Opt-in: attach Bedrock_Reasoner rationale to an assembled result.

    When ``enabled`` is falsy the Reasoner is never invoked and the rule-only
    result is returned unchanged (``modelReasoningAvailable=False``, no notice).
    When enabled, the Reasoner is invoked; a non-``None`` rationale is attached
    with ``modelReasoningAvailable=True``, while a ``None`` result sets
    ``modelReasoningAvailable=False`` and appends a model-unavailable notice to
    ``ambiguityNotes`` — still returning the result.
    """
    if not enabled:
        return result

    rationale = generate_rationale(
        verdict, context=_build_reasoning_context(verdict, bsa_id)
    )

    if rationale is not None:
        result.modelReasoning = rationale
        result.modelReasoningAvailable = True
    else:
        result.modelReasoningAvailable = False
        notes = list(result.ambiguityNotes) if result.ambiguityNotes else []
        notes.append(MODEL_UNAVAILABLE_NOTICE)
        result.ambiguityNotes = notes

    return result


# §Classification — BSAID retrieval path helpers
# ---------------------------------------------------------------------------

def _parse_amount_text(text) -> Optional[Decimal]:
    """Parse a stored amount string (e.g. ``"325321.00"``) into a ``Decimal``.

    Returns ``None`` when the value is absent or cannot be parsed. Defensive
    against the aggregate field being missing or malformed in the stored record.
    """
    if text is None:
        return None
    try:
        return Decimal(str(text))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _derive_bsaid_signals(item: dict) -> dict:
    """Derive aggregate/narrative signals from a stored SAR ``Activity``.

    The stored synthetic SAR records do not carry itemized per-transaction line
    items — each ``SuspiciousActivity`` element exposes only an aggregate
    ``TotalSuspiciousAmountText``, a from/to activity date range, a classification
    code, and free-text narrative (design.md "A note on the data model
    constraint"). This helper reads what is available, defensively tolerating
    missing keys, and returns a flat signal dict for rationale assembly.
    """
    activity = item.get("Activity", {}) or {}

    suspicious_list = activity.get("SuspiciousActivity", []) or []
    suspicious = suspicious_list[0] if suspicious_list else {}

    total_amount_text = suspicious.get("TotalSuspiciousAmountText")
    from_date = suspicious.get("SuspiciousActivityFromDateText")
    to_date = suspicious.get("SuspiciousActivityToDateText")

    classification = None
    classifications = suspicious.get("SuspiciousActivityClassification", []) or []
    if classifications:
        classification = classifications[0].get(
            "SuspiciousActivityTypeCodeDescription"
        )

    narratives = []
    for narrative in activity.get("ActivityNarrativeInformation", []) or []:
        text = narrative.get("ActivityNarrativeText")
        if text:
            narratives.append(text)
    narrative_text = " ".join(narratives) if narratives else None

    # SyntheticData may live at the top level of the record or under Activity.
    synthetic_data = bool(item.get("SyntheticData") or activity.get("SyntheticData"))

    return {
        "bsaId": str(activity.get("BSAID")) if activity.get("BSAID") is not None else None,
        "totalAmountText": total_amount_text,
        "totalAmount": _parse_amount_text(total_amount_text),
        "fromDate": from_date,
        "toDate": to_date,
        "classification": classification,
        "narrativeText": narrative_text,
        "syntheticData": synthetic_data,
    }


def _build_bsaid_rationale(bsa_id: str, signals: dict) -> str:
    """Build a non-empty rationale for the BSAID path.

    States plainly that itemized transaction data was unavailable (so no
    structuring verdict could be computed from true per-transaction data) and
    summarizes the aggregate amount, activity date window, and classification
    signal derived from the stored record.
    """
    parts = [
        f"Itemized transaction data was unavailable for the SAR record referenced "
        f"by bsaId {bsa_id}; the stored record carries only aggregate and "
        f"narrative signals, so structuring could not be assessed from "
        f"per-transaction line items."
    ]

    amount_text = signals.get("totalAmountText")
    if amount_text:
        parts.append(f"Reported total suspicious amount: ${amount_text}.")

    from_date = signals.get("fromDate")
    to_date = signals.get("toDate")
    if from_date and to_date:
        parts.append(f"Reported activity window: {from_date} to {to_date}.")
    elif from_date:
        parts.append(f"Reported activity start date: {from_date}.")
    elif to_date:
        parts.append(f"Reported activity end date: {to_date}.")

    classification = signals.get("classification")
    if classification:
        parts.append(f"Stored classification signal: {classification}.")

    parts.append(
        "This is an automated assessment that surfaces the stored aggregate/"
        "narrative signals for Analyst review; it does not assert that a "
        "violation occurred."
    )
    return " ".join(parts)


def _classify_from_bsaid(
    body: ClassifyRequest, x_request_id: Optional[str]
) -> tuple[StructuringVerdict, ClassificationResult]:
    """Resolve a ``bsaId`` from DynamoDB and derive a classification result.

    Returns the ``(verdict, result)`` pair so the caller can optionally apply
    model reasoning using the derived verdict.

    Reuses the existing ``SAR#{bsaId}`` / ``METADATA`` key pattern and the
    module-level ``table`` resource. Because stored records lack itemized line
    items, the result reports that itemized transaction data was unavailable and
    relies on aggregate/narrative signals:

    - Missing ``Item`` → 404 identifying the unresolved BSAID.
    - boto3/client exception → 503 "data source temporarily unavailable".
    """
    bsa_id = body.bsaId

    try:
        result = table.get_item(Key={"PK": f"SAR#{bsa_id}", "SK": "METADATA"})
    except Exception:
        # Any boto3/client-side failure maps to a 503 temporary-unavailability
        # response. The specific exception is intentionally not surfaced to the
        # caller to avoid leaking data-source internals.
        raise HTTPException(
            status_code=503, detail="data source temporarily unavailable"
        )

    item = result.get("Item")
    if not item:
        # Unknown/unresolved BSAID → 404 naming the identifier.
        raise HTTPException(
            status_code=404,
            detail=f"No SAR record resolves to bsaId {bsa_id}.",
        )

    signals = _derive_bsaid_signals(item)

    # Prefer the stored record's own BSAID for the citation, falling back to the
    # requested value when the stored field is absent.
    cited_bsa_id = signals.get("bsaId") or bsa_id

    # Build a not-detected verdict: with no itemized line items we cannot assert
    # structuring from true per-transaction data, so detection is False and the
    # triggering/reportable lists are empty. The aggregate signal is captured in
    # the rationale and ruleRationale rather than fabricated as line items.
    window_days = body.rollingWindowDays or 30
    verdict = StructuringVerdict(
        detected=False,
        typology=None,
        aggregatedAmount=Decimal("0.00"),
        triggeringTransactions=[],
        reportableTransactions=[],
        windowDays=window_days,
        windowSpanDays=None,
        ruleRationale={
            "detected": False,
            "typology": None,
            "source": "bsaId",
            "bsaId": cited_bsa_id,
            "itemizedDataAvailable": False,
            "totalSuspiciousAmountText": signals.get("totalAmountText"),
            "activityFromDate": signals.get("fromDate"),
            "activityToDate": signals.get("toDate"),
            "storedClassification": signals.get("classification"),
            "reason": "Stored SAR record lacks itemized transaction line items; "
            "aggregate/narrative signals derived instead.",
        },
        riskScore=1,
        confidenceScore=0.0,
    )

    ambiguity_notes = [
        "Itemized transaction data was unavailable for the referenced bsaId; "
        "aggregate and narrative signals from the stored record were used instead "
        "of per-transaction line items.",
    ]

    result = assemble_classification_result(
        verdict,
        rationale=_build_bsaid_rationale(cited_bsa_id, signals),
        bsa_id=cited_bsa_id,
        synthetic_data=signals.get("syntheticData", False),
        request_id=x_request_id,
        ambiguity_notes=ambiguity_notes,
    )
    return verdict, result


# §Classification — POST /classify (SAR Classification Agent)
@app.post("/classify", response_model=ClassificationResult)
def classify(
    body: ClassifyRequest,
    x_case_id: Optional[str] = Header(None, alias="X-Case-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ClassificationResult:
    """Classify transactions (or a referenced SAR filing) for suspicious patterns.

    Transaction path: run the semantic validation pass over the submitted
    ``transactions`` set and, when non-empty, delegate structuring detection to
    the deterministic Rule_Engine and assemble a minimized
    ``ClassificationResult``.

    Transaction-set precedence: a non-empty ``transactions[]`` is always
    classified from the submitted transactions — even when a ``bsaId`` is also
    present — with no stored-record substitution. ``normalize_and_validate``
    returns the normalized transactions for this path and an empty list only on
    the BSAID-only path.

    The ``X-Request-Id`` header is echoed into ``requestId`` (``None`` when
    absent) for request correlation. This handler leaves all existing endpoints
    and helpers unchanged.
    """
    # Reuse the shared header-validation convention (§4.3).
    validate_common_headers(x_case_id, x_request_id)

    # Semantic validation + normalization. Returns a non-empty list for the
    # transaction path, or [] for the BSAID-only path (transactions key absent).
    normalized = normalize_and_validate(body)

    if normalized:
        # Transaction-set precedence: classify the submitted transactions
        # regardless of a present bsaId; no stored-record lookup.
        verdict = detect_structuring(
            normalized, rolling_window_days=body.rollingWindowDays
        )
        bsa_id = body.bsaId if body.bsaId else None

        # Opt-in model reasoning: only invoke the Bedrock_Reasoner when the
        # caller enabled it. On success the rationale and
        # modelReasoningAvailable=True are passed into assembly; on None/failure
        # modelReasoningAvailable=False and a model-unavailable notice is
        # surfaced while still returning 200.
        model_reasoning: Optional[str] = None
        model_reasoning_available = False
        ambiguity_notes: Optional[list] = None
        if body.enableModelReasoning:
            model_reasoning = generate_rationale(
                verdict, context=_build_reasoning_context(verdict, bsa_id)
            )
            if model_reasoning is not None:
                model_reasoning_available = True
            else:
                ambiguity_notes = [MODEL_UNAVAILABLE_NOTICE]

        return assemble_classification_result(
            verdict,
            bsa_id=bsa_id,
            request_id=x_request_id,
            model_reasoning=model_reasoning,
            model_reasoning_available=model_reasoning_available,
            ambiguity_notes=ambiguity_notes,
        )

    # BSAID-only path (transactions absent/empty, bsaId present). Resolve the
    # bsaId via table.get_item and derive aggregate/narrative signals from the
    # stored record.
    verdict, result = _classify_from_bsaid(body, x_request_id)
    # Apply the same opt-in reasoning logic to the BSAID path for consistency
    # (the design shows reasoning after the rule engine for both paths). When
    # reasoning is disabled this is a no-op.
    return _apply_model_reasoning(
        result, verdict, body.enableModelReasoning, bsa_id=result.bsaId
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
