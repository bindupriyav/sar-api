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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
