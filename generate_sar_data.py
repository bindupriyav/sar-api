#!/usr/bin/env python3
"""
generate_sar_data.py

Generates 100% SYNTHETIC FinCEN SAR (Suspicious Activity Report) records in
JSON format, structured for storage in DynamoDB, for use in a development /
prototype environment ONLY.

The record shape is derived from FinCEN's BSA XML 2.0 schema
(https://www.fincen.gov/system/files/schema/base/BSA_XML_2.0.xsd), specifically
the Activity -> Party -> {PartyName, Address, PartyIdentification,
PartyAccountAssociation} and Activity -> SuspiciousActivity ->
SuspiciousActivityClassification / ActivityNarrativeInformation branches.

The XML schema is deeply nested and includes hundreds of FinCEN-internal
processing fields (batch IDs, enhanced address geocoding, e-filing housekeeping
fields, etc.) that only make sense inside FinCEN's own system of record. For an
API prototype, this generator flattens the schema down to the subset of
elements a filing institution's own system would actually populate/consume,
using the *same element names* as the XSD so the mapping back to the official
schema is direct and traceable.

NONE of the data produced by this script represents a real person, business,
account, or filing. All identifiers (SSN/EIN, account numbers, addresses,
narratives) are randomly generated.
"""

import json
import random
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(None)  # non-deterministic; pass an int here if you want repeatable runs

# ---------------------------------------------------------------------------
# Reference code tables (abbreviated subsets of FinCEN's official SAR code
# lists, used here only to produce plausible-looking coded values)
# ---------------------------------------------------------------------------

SUSPICIOUS_ACTIVITY_CATEGORY_CODES = [
    ("1", "Structuring"),
    ("2", "Terrorist Financing"),
    ("3", "Fraud"),
    ("4", "Money Laundering"),
    ("5", "Identity Theft"),
    ("6", "Cyber Event"),
    ("7", "Elder Financial Exploitation"),
    ("8", "Gaming Activities"),
    ("9", "Mortgage Fraud"),
]

SUSPICIOUS_ACTIVITY_SUBTYPE_CODES = {
    "1": [("1", "Alters or cancels transaction to avoid BSA recordkeeping requirement"),
          ("2", "Suspicious inquiry by customer regarding BSA reporting or recordkeeping requirements"),
          ("3", "Transaction(s) below reporting threshold")],
    "3": [("1", "Account takeover"), ("2", "Business loan fraud"), ("3", "Check fraud"),
          ("4", "Debit card fraud"), ("5", "Wire fraud"), ("6", "Consumer fraud")],
    "4": [("1", "Fundamental review"), ("2", "Suspicious use of multiple accounts"),
          ("3", "Trade based money laundering")],
    "6": [("1", "Botnet activity"), ("2", "Denial of service / DDoS attack"),
          ("3", "Malware"), ("4", "Phishing"), ("5", "Ransomware")],
}

FINANCIAL_INSTITUTION_TYPE_CODES = [
    ("1", "Depository institution"),
    ("2", "Casino/card club"),
    ("3", "Money services business"),
    ("7", "Securities/futures"),
    ("8", "Insurance company"),
    ("9", "Loan or finance company"),
]

PRODUCT_TYPE_CODES = [
    ("1", "Checking"), ("2", "Savings"), ("6", "Credit card"),
    ("11", "Wire transfer"), ("14", "ACH"), ("18", "Prepaid access/card"),
    ("23", "Virtual currency"),
]

PARTY_ROLE_FILER = "35"          # Financial institution where activity occurred
PARTY_ROLE_SUBJECT = "33"        # Subject
PARTY_ROLE_FILING_INSTITUTION = "30"
PARTY_ROLE_CONTACT = "34"        # Contact for assistance

US_STATES = ["CA", "NY", "TX", "FL", "IL", "OH", "PA", "GA", "NC", "MI", "WA", "AZ", "MA", "VA"]

NARRATIVE_TEMPLATES = [
    "Between {start} and {end}, {subject} conducted a series of {count} cash "
    "deposits totaling ${amount} across multiple branches of {institution}, "
    "each structured below the $10,000 reporting threshold. Deposits were "
    "followed within 24-48 hours by outgoing wire transfers to {dest_country}. "
    "The pattern of activity is inconsistent with the stated purpose of the "
    "account and the customer's known business activity.",

    "{institution} identified {count} incoming ACH credits to the account of "
    "{subject} totaling ${amount} between {start} and {end}, originating from "
    "accounts previously associated with reported unemployment insurance fraud. "
    "Funds were rapidly withdrawn via ATM and person-to-person transfer apps "
    "shortly after each credit posted.",

    "On {start}, {subject} attempted to negotiate a counterfeit cashier's check "
    "in the amount of ${amount} drawn on an account at {institution}. The item "
    "was identified as fraudulent by the branch teller and the transaction was "
    "declined. The customer became agitated and left the branch without further "
    "explanation.",

    "{institution} detected {count} login attempts to the online banking profile "
    "of {subject} from IP addresses geolocated to {dest_country} over a "
    "{days}-day period beginning {start}, followed by an unauthorized external "
    "transfer of ${amount}. The transfer was placed on hold pending customer "
    "verification.",

    "Account activity for {subject} at {institution} shows {count} transactions "
    "totaling ${amount} between {start} and {end} involving a virtual currency "
    "exchange with limited or no Know-Your-Customer controls. The transaction "
    "pattern is inconsistent with the customer's profile and stated occupation.",
]


def _rand_date(start_days_ago=365, end_days_ago=1):
    start = date.today() - timedelta(days=start_days_ago)
    end = date.today() - timedelta(days=end_days_ago)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 1)))


def _fmt(d):
    return d.strftime("%Y-%m-%d")


def _synthetic_ein():
    return f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"


def _synthetic_ssn():
    # deliberately uses the reserved 900-series-like pattern is not guaranteed;
    # this is fine for a non-production prototype but should never be treated
    # as PII-safe for anything beyond local dev/testing.
    return f"{random.randint(100,899):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}"


def _synthetic_account_number():
    return str(random.randint(10**9, 10**12 - 1))


def make_address(party_seq):
    return {
        "AddressID": party_seq * 10 + 1,
        "RawStreetAddress1Text": fake.street_address(),
        "RawCityText": fake.city(),
        "RawStateCodeText": random.choice(US_STATES),
        "RawZIPCode": fake.postcode(),
        "RawCountryCodeText": "US",
        "PartyAddressTypeCode": "0",  # 0 = current address (SAR code table value)
    }


def make_phone(party_seq):
    return {
        "PhoneNumberText": fake.numerify("##########"),
        "PhoneNumberTypeCode": random.choice(["1", "2", "3"]),  # Work/Mobile/Fax etc.
    }


def make_party_identification(party_seq, id_type_code, id_text):
    return {
        "PartyIdentificationID": party_seq * 10 + 2,
        "PartyIdentificationTypeCode": id_type_code,   # e.g. 2=EIN, 1=SSN/ITIN, 9=Passport
        "PartyIdentificationNumberText": id_text,
        "IdentificationPresentUnknownIndicator": None,
    }


def make_account(party_seq):
    return {
        "AccountID": party_seq * 10 + 3,
        "AccountNumberText": _synthetic_account_number(),
        "AccountTypeCode": random.choice(["1", "2", "5"]),  # checking/savings/other
    }


def make_filing_institution_party(seq):
    name = fake.company() + random.choice([" Bank", " Bank, N.A.", " Financial", " Credit Union"])
    return {
        "PartyID": seq,
        "ActivityPartyTypeCode": PARTY_ROLE_FILING_INSTITUTION,
        "ActivityPartyTypeCodeDescription": "Filing institution",
        "PrimaryRegulatorTypeCode": random.choice(["1", "2", "3", "7"]),
        "PartyName": [{
            "PartyNameTypeCode": "L",  # Legal name
            "RawPartyFullName": name,
        }],
        "Address": [make_address(seq)],
        "PartyIdentification": [make_party_identification(seq, "2", _synthetic_ein())],
        "PhoneNumber": [make_phone(seq)],
    }


def make_financial_institution_party(seq, institution_name):
    fi_type_code, fi_type_desc = random.choice(FINANCIAL_INSTITUTION_TYPE_CODES)
    return {
        "PartyID": seq,
        "ActivityPartyTypeCode": PARTY_ROLE_FILER,
        "ActivityPartyTypeCodeDescription": "Financial institution where activity occurred",
        "FinancialInstitutionTypeCode": fi_type_code,
        "FinancialInstitutionTypeCodeDescription": fi_type_desc,
        "PartyName": [{
            "PartyNameTypeCode": "L",
            "RawPartyFullName": institution_name,
        }],
        "Address": [make_address(seq)],
        "PartyIdentification": [make_party_identification(seq, "2", _synthetic_ein())],
    }


def make_subject_party(seq):
    is_business = random.random() < 0.3
    if is_business:
        full_name = fake.company()
        id_type, id_num = "2", _synthetic_ein()
        occupation = "N/A - Business"
    else:
        full_name = fake.name()
        id_type, id_num = "1", _synthetic_ssn()
        occupation = fake.job()

    party = {
        "PartyID": seq,
        "ActivityPartyTypeCode": PARTY_ROLE_SUBJECT,
        "ActivityPartyTypeCodeDescription": "Subject",
        "SubjectRoleTypeCode": random.choice(["10", "35", "42"]),  # e.g. accountholder/signer/other
        "PartyName": [{
            "PartyNameTypeCode": "L",
            "RawPartyFullName": full_name,
            "RawEntityIndividualLastName": None if is_business else full_name.split(" ")[-1],
            "RawIndividualFirstName": None if is_business else full_name.split(" ")[0],
        }],
        "Address": [make_address(seq)],
        "PartyIdentification": [make_party_identification(seq, id_type, id_num)],
        "PhoneNumber": [make_phone(seq)],
        "PartyOccupationBusiness": [{
            "OccupationBusinessText": occupation,
            "NAICSCode": fake.numerify("######") if is_business else None,
        }],
        "PartyAccountAssociation": [{
            "PartyAccountAssociationTypeCode": "3",  # e.g. account holder
            "Account": [make_account(seq)],
        }],
    }
    if not is_business:
        party["BirthDateText"] = _fmt(fake.date_of_birth(minimum_age=18, maximum_age=85))
    return party


def make_suspicious_activity(seq, subject_amount):
    cat_code, cat_desc = random.choice(SUSPICIOUS_ACTIVITY_CATEGORY_CODES)
    subtype_pool = SUSPICIOUS_ACTIVITY_SUBTYPE_CODES.get(cat_code, [("99", "Other")])
    subtype_code, subtype_desc = random.choice(subtype_pool)

    activity_start = _rand_date(365, 60)
    activity_end = activity_start + timedelta(days=random.randint(1, 45))

    return {
        "SuspiciousActivityID": seq,
        "AmountUnknownIndicator": None,
        "TotalSuspiciousAmountText": f"{subject_amount:.2f}",
        "CumulativeTotalViolationAmountText": f"{subject_amount:.2f}",
        "SuspiciousActivityFromDateText": _fmt(activity_start),
        "SuspiciousActivityToDateText": _fmt(activity_end),
        "SuspiciousActivityClassification": [{
            "SuspiciousActivityTypeID": cat_code,
            "SuspiciousActivityTypeCodeDescription": cat_desc,
            "SuspiciousActivitySubtypeID": subtype_code,
            "SuspiciousActivitySubtypeCodeDescription": subtype_desc,
        }],
    }


def make_narrative(activity_id, subject_name, institution_name, amount):
    template = random.choice(NARRATIVE_TEMPLATES)
    start = _rand_date(365, 60)
    end = start + timedelta(days=random.randint(1, 30))
    text = template.format(
        start=_fmt(start),
        end=_fmt(end),
        subject=subject_name,
        institution=institution_name,
        amount=f"{amount:,.2f}",
        count=random.randint(3, 22),
        dest_country=random.choice(["Nigeria", "Hong Kong", "Mexico", "Russia", "the United Arab Emirates", "Colombia"]),
        days=random.randint(2, 14),
    )
    return [{
        "ActivityID": activity_id,
        "ActivityNarrativeSequenceNumber": 1,
        "ActivityNarrativeText": text,
    }]


def make_sar_record(index):
    activity_id = 100000000 + index
    bsaid = int(f"3{random.randint(10**12, 10**13 - 1)}")  # 14-digit synthetic BSA ID
    filing_date = _rand_date(90, 1)

    filing_inst_seq = 1
    fi_party_seq = 2
    subject_seq = 3

    filing_institution_party = make_filing_institution_party(filing_inst_seq)
    institution_name = filing_institution_party["PartyName"][0]["RawPartyFullName"]
    fi_party = make_financial_institution_party(fi_party_seq, institution_name)
    subject_party = make_subject_party(subject_seq)
    subject_name = subject_party["PartyName"][0]["RawPartyFullName"]

    total_amount = round(random.uniform(5000, 750000), 2)

    record = {
        # --- DynamoDB key design ---
        "PK": f"SAR#{bsaid}",
        "SK": "METADATA",
        "GSI1PK": f"INSTITUTION#{filing_institution_party['PartyIdentification'][0]['PartyIdentificationNumberText']}",
        "GSI1SK": f"FILINGDATE#{_fmt(filing_date)}",
        "GSI2PK": f"SUBJECT#{subject_party['PartyIdentification'][0]['PartyIdentificationNumberText']}",
        "GSI2SK": f"SAR#{bsaid}",
        "RecordType": "SAR",
        "SchemaVersion": "BSA_XML_2.0",
        "SyntheticData": True,

        # --- BSA XML-aligned payload (Activity element) ---
        "Activity": {
            "ActivityID": activity_id,
            "BSAID": bsaid,
            "SeqNum": 1,
            "ActivityTypeCode": "SAR",
            "ActivityTypeCodeDescription": "Suspicious Activity Report",
            "FilingDateText": _fmt(filing_date),
            "FilingEntryDate": _fmt(filing_date),
            "FormTypeCode": "111",  # FinCEN SAR form 111
            "ActionTypeCode": "A",  # Add
            "InitialReportIndicator": "Y",
            "CorrectsAmendsPriorReportIndicator": None,
            "FilingInstitutionNotetoFinCEN": None,

            "Party": [
                filing_institution_party,
                fi_party,
                subject_party,
            ],

            "SuspiciousActivity": [
                make_suspicious_activity(activity_id, total_amount)
            ],

            "ActivityNarrativeInformation": make_narrative(
                activity_id, subject_name, institution_name, total_amount
            ),
        },

        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "generate_sar_data.py",
            "notice": "Fictitious data generated for software development/testing purposes only. "
                      "No relationship to any real filing, institution, or individual.",
        },
    }
    return record


def main(n=25, out_path="synthetic_sar_data.json"):
    records = [make_sar_record(i) for i in range(1, n + 1)]
    out = Path(out_path)
    out.write_text(json.dumps(records, indent=2, default=str))
    print(f"Wrote {len(records)} synthetic SAR records to {out_path}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    main(n)
