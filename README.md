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

---

# SAR Classification Agent (`POST /classify`)

In addition to the read-only query endpoints above, the service exposes a
**transaction classification** endpoint that flags potential **structuring**
(a.k.a. "smurfing") — multiple sub-$10,000 cash transactions that aggregate to
or above the $10,000 CTR reporting threshold within a rolling window.

The agent is **deterministic first**: a pure rule engine produces the verdict,
scores, and a plain-language rationale. An **optional** Amazon Bedrock reasoner
can add a natural-language explanation; if it is disabled or fails, the service
still returns the deterministic result (graceful fallback).

> This is a flag for **Analyst review**, never a legal determination. Every
> response carries a `modelDisclosure` to that effect.

## Running the API

```bash
pip install -r requirements.txt

# start the service (http://localhost:8000)
python -m uvicorn app.main:app --port 8000
# add --reload for autoreload during development (needs `watchfiles`)
```

Interactive Swagger UI: **http://localhost:8000/docs**

- The **transaction path** (sending `transactions[]`) needs **no AWS**.
- The **BSAID path** (sending `bsaId`) reads DynamoDB and needs AWS creds + the
  `sar-prototype` table in the configured region.
- **Model reasoning** (`enableModelReasoning: true`) calls Amazon Bedrock and
  needs `bedrock:InvokeModel` permission; on any failure it degrades to the
  rule-engine result with `modelReasoningAvailable: false`.

## Request

`POST /classify`

Optional headers: `X-Request-Id` (echoed back as `requestId`), `X-Case-Id`.

| Field | Type | Notes |
|---|---|---|
| `transactions` | array | Each item: `amount` (string/number, non-negative, <=2 dp), `date` (ISO `YYYY-MM-DD`), `transactionType` (e.g. `CASH_DEPOSIT`, `WIRE`), optional `isCash`, `subject`, `tin`. A non-empty set takes precedence over `bsaId`. |
| `bsaId` | string | Classify from a stored SAR record instead of itemized transactions. |
| `rollingWindowDays` | int | Window span; defaults to 30. Values `< 1` are rejected (HTTP 400). |
| `enableModelReasoning` | bool | Opt in to the Bedrock rationale. Defaults to `false`. |

A request must include either a non-empty `transactions` set **or** a `bsaId`.

### Example — structuring detected

```json
{
  "transactions": [
    { "amount": "9999.99", "date": "2024-01-01", "transactionType": "CASH_DEPOSIT" },
    { "amount": "9999.99", "date": "2024-01-02", "transactionType": "CASH_DEPOSIT" }
  ]
}
```

## Response (`ClassificationResult`)

| Field | Type | Notes |
|---|---|---|
| `typology` | string \| null | `"Structuring"` when detected, else `null`. |
| `riskScore` | int | 1–10. `1` when nothing is detected. |
| `confidenceScore` | number | 0.0–1.0. |
| `triggeringTransactions` | array | Contributing transactions (minimized; TIN shown as `tinLast4` only). |
| `reportableTransactions` | array | Cash transactions `>= $10,000` recorded separately. |
| `rationale` | string | Rule-based explanation (always present). |
| `bsaId` | string \| null | Citation when derived from stored data. |
| `modelReasoningAvailable` | bool | `true` only if a Bedrock rationale was produced. |
| `modelReasoning` | string \| null | Natural-language rationale when available. |
| `modelDisclosure` | string | Always present: AI-produced, requires Analyst review/approval. |
| `syntheticDataDisclosure` | string \| null | Present when source data was synthetic. |
| `requestId` | string \| null | Echo of `X-Request-Id`. |
| `ambiguityNotes` | array \| null | Reported ambiguities (not auto-resolved). |

### Validation / error codes

| Condition | Status |
|---|---|
| Neither `transactions` nor `bsaId` provided | 400 |
| Empty `transactions` set | 400 |
| More than 10,000 transactions | 400 |
| Missing required field / negative amount / >2 dp amount | 422 (model) |
| Unparseable `date` | 400 (names field + index) |
| `rollingWindowDays < 1` | 400 |
| Malformed JSON body | 422 |
| `bsaId` not found (BSAID path) | 404 |
| Data source unavailable (BSAID path) | 503 |

## Running the tests

The full suite runs **offline** — DynamoDB and Bedrock are mocked.

```bash
python -m pytest tests/ -q          # all tests
python -m pytest tests/ -v          # verbose
python -m pytest tests/test_classify_api.py -v   # one file
```

Coverage includes: Rule_Engine property tests (Hypothesis), request/response
validation, FastAPI endpoint tests, mocked-DynamoDB BSAID tests, mocked-Bedrock
reasoning tests, a regression test that the existing read endpoints are
unchanged, and human-in-the-loop disclosure assertions.

## Testing with Postman

1. **New request** → method `POST`, URL `http://localhost:8000/classify`.
2. **Headers** tab: add `Content-Type: application/json` and optionally
   `X-Request-Id: req-123`.
3. **Body** tab → **raw** → **JSON**, paste the example request above.
4. **Send**. You should get `200` with `"typology": "Structuring"`.
5. Try the error cases from the table (e.g. empty `transactions`, or
   `"rollingWindowDays": 0`) to see the 400/422 responses.

Tip: you can also import the live OpenAPI schema into Postman from
`http://localhost:8000/openapi.json` to auto-generate all requests.
