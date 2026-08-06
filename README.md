# FinCEN SAR API Prototype — Synthetic Data Kit

This kit generates **100% synthetic** FinCEN SAR (Suspicious Activity Report)
data as JSON, shaped for storage in DynamoDB, for use in building/testing an
API prototype. No real filings, institutions, or individuals are represented.

Source schema reference: [BSA XML 2.0 XSD](https://www.fincen.gov/system/files/schema/base/BSA_XML_2.0.xsd)
(FinCEN's official schema for BSA report submission/dissemination — SAR is one
of several report types it covers, all rooted at the `Activity` element).

## Files

| File | Purpose |
|---|---|
| `generate_sar_data.py` | Generates N synthetic SAR JSON records |
| `load_to_dynamodb.py` | Loads generated records into DynamoDB (real AWS or DynamoDB Local) |
| `synthetic_sar_data.json` | Example output — 40 generated records |
| `README.md` | This file |

## Why the JSON isn't a 1:1 XML→JSON transliteration

The BSA XML 2.0 schema is FinCEN's **internal system-of-record** schema. Most
of its ~40 complex types carry FinCEN-only processing metadata that a filing
institution never populates and an API consumer would never want: batch load
sequence numbers (`BatchSeqNum`, `MegabatchID`), USPS/geocoding enrichment
fields (`EnhancedCASSStatusText`, `EnhancedGeoCompleteText`, ~60 fields per
address), e-filing housekeeping fields, etc.

For a prototype API, this kit keeps the **element and field names** from the
XSD (so mapping to/from the official schema is direct) but only populates the
subset of `ActivityType` / `PartyType` / `AddressType` / etc. fields that
correspond to actual SAR *form content* — the same fields you'd see on the
FinCEN SAR (Form 111) e-filing form itself:

- `Activity` — the SAR itself (BSAID, filing date, form type, action type)
- `Activity.Party[]` — the three party roles always present on a SAR:
  - `ActivityPartyTypeCode = "30"` — Filing institution
  - `ActivityPartyTypeCode = "35"` — Financial institution where the activity occurred
  - `ActivityPartyTypeCode = "33"` — Subject (person/entity the report is about)
- `Party.PartyName`, `Party.Address`, `Party.PartyIdentification`,
  `Party.PhoneNumber`, `Party.PartyOccupationBusiness`,
  `Party.PartyAccountAssociation.Account` — standard sub-elements per the XSD
- `Activity.SuspiciousActivity[]` → `SuspiciousActivityClassification[]` —
  category/subtype codes (structuring, fraud, money laundering, cyber event, etc.)
- `Activity.ActivityNarrativeInformation[]` — the free-text narrative

Full field-level fidelity to the XSD (all ~400 elements) is straightforward to
add later by extending the `make_*` functions — the generator is intentionally
structured one function per BSA complex type so it's easy to layer in more
fields as the prototype's API surface grows.

## DynamoDB table design

Single-table design, since SAR records are read primarily by BSAID, by filing
institution, or by subject identifier.

**Base table**

| Attribute | Type | Example | Notes |
|---|---|---|---|
| `PK` (partition key) | S | `SAR#37413986524757` | `SAR#<BSAID>` |
| `SK` (sort key) | S | `METADATA` | Reserved for future item types under the same PK (e.g. `ATTACHMENT#<id>`, `AMENDMENT#<id>`) |
| `GSI1PK` | S | `INSTITUTION#38-7265287` | Filing institution's EIN |
| `GSI1SK` | S | `FILINGDATE#2026-05-13` | Enables date-range queries per institution |
| `GSI2PK` | S | `SUBJECT#480-31-2380` | Subject's SSN/EIN |
| `GSI2SK` | S | `SAR#37413986524757` | |
| `RecordType` | S | `SAR` | For future multi-entity-type tables |
| `SchemaVersion` | S | `BSA_XML_2.0` | Tracks which FinCEN schema version the payload maps to |
| `SyntheticData` | BOOL | `true` | Always `true` for generated data — **never remove this flag from real synthetic-only environments** |
| `Activity` | M (map) | *(nested object)* | The BSA XML-aligned SAR payload |

**Global Secondary Indexes**

- `GSI1-InstitutionFilingDate` — "all SARs filed by institution X, most recent first"
- `GSI2-Subject` — "all SARs naming subject Y" (a common investigative query pattern)

**Access patterns supported**

1. Get a single SAR by BSAID → `GetItem(PK=SAR#<bsaid>, SK=METADATA)`
2. List SARs filed by an institution, optionally by date range → `Query` on `GSI1`
3. List SARs naming a given subject (SSN/EIN) → `Query` on `GSI2`
4. (Future) Attach documents/amendments to a SAR under the same `PK` with a different `SK` prefix

## Usage

```bash
pip install faker boto3

# generate 100 synthetic records
python3 generate_sar_data.py 100

# load into DynamoDB Local for prototype API development
docker run -p 8000:8000 amazon/dynamodb-local
python3 load_to_dynamodb.py --table sar-prototype --file synthetic_sar_data.json \
    --endpoint-url http://localhost:8000 --create-table

# load into a real (dev/sandbox) AWS account
python3 load_to_dynamodb.py --table sar-prototype --file synthetic_sar_data.json \
    --region us-east-1 --create-table
```

## Compliance note

This is synthetic test data for software development purposes. Actual SAR
data is highly sensitive (protected from disclosure under 31 U.S.C. § 5318(g)),
subject to strict access, retention, and confidentiality controls under BSA
regulations, and must never be stored or transmitted through non-FinCEN-
authorized systems. Do not point this prototype's storage layer at real SAR
data without the appropriate regulatory, security, and legal review.
