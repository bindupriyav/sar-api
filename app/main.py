"""
SAR Data API — FastAPI service
Read-only REST API for querying synthetic SAR records from DynamoDB.
"""
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import boto3
from boto3.dynamodb.conditions import Key

app = FastAPI(title="SAR Data API", version="1.0.0")

# DynamoDB setup
REGION = os.environ.get("AWS_REGION", "us-east-2")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "sar-prototype")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)


@app.get("/health")
def health():
    return {"status": "healthy", "table": TABLE_NAME}


@app.get("/sars/{bsa_id}")
def get_sar_by_bsaid(bsa_id: str):
    """Get a single SAR by BSA ID (PK query)."""
    result = table.get_item(Key={"PK": f"SAR#{bsa_id}", "SK": "METADATA"})
    item = result.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="SAR not found")
    return item


@app.get("/institutions/{ein}/sars")
def get_sars_by_institution(ein: str, since: str = "2000-01-01"):
    """Get all SARs filed by an institution (GSI1 query)."""
    result = table.query(
        IndexName="GSI1-InstitutionFilingDate",
        KeyConditionExpression=Key("GSI1PK").eq(f"INSTITUTION#{ein}")
        & Key("GSI1SK").gte(f"FILINGDATE#{since}"),
    )
    return {"institutionEIN": ein, "count": len(result["Items"]), "sars": result["Items"]}


@app.get("/subjects/{subject_id}/sars")
def get_sars_by_subject(subject_id: str):
    """Get all SARs naming a subject (GSI2 query)."""
    result = table.query(
        IndexName="GSI2-Subject",
        KeyConditionExpression=Key("GSI2PK").eq(f"SUBJECT#{subject_id}"),
    )
    return {"subjectId": subject_id, "count": len(result["Items"]), "sars": result["Items"]}


@app.post("/entities:resolve")
async def resolve_entity(request: Request):
    """Entity resolution search. PII comes from request headers per §4.1."""
    headers = request.headers
    subject_name = headers.get("x-subject-name", "")
    subject_tin = headers.get("x-subject-tin", "")
    subject_dob = headers.get("x-subject-dob", "")

    if not any([subject_name, subject_tin, subject_dob]):
        raise HTTPException(
            status_code=400,
            detail="At least one of X-Subject-Name, X-Subject-Tin, X-Subject-Dob required",
        )

    # Scan all records and fuzzy match (prototype only)
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
            if subject_tin and id_num and subject_tin[-4:] == id_num[-4:]:
                matched_on.append("tin")
            if subject_dob and dob == subject_dob:
                matched_on.append("dob")

            if matched_on:
                seen.add(bsa_id)
                matches.append({
                    "bsaId": bsa_id,
                    "matchConfidence": round(len(matched_on) / 3, 2),
                    "matchedOn": matched_on,
                    "displayName": full_name,
                    "tinLast4": id_num[-4:] if id_num else "",
                    "filingDate": activity.get("FilingDateText", ""),
                })

    return {"matches": matches}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
