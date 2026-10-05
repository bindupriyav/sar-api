# Feature: sar-classification-agent, Task 10.1: Regression test — existing endpoints unchanged
"""Regression tests asserting the pre-existing SAR Data API endpoints are unchanged.


The SAR Classification Agent work added a new ``POST /classify`` endpoint plus
supporting helpers to ``app/main.py``. requires that the
existing read endpoints keep the same request/response behavior and signatures:

    - ``GET  /health``
    - ``GET  /sars/{bsaId}``
    - ``GET  /institutions/{id}/sars``
    - ``GET  /classifications/{code}/sars``
    - ``GET  /entities/{entityId}/sars``
    - ``POST /entities:resolve``

These tests pin each endpoint's HTTP status code and documented response-key
structure. To keep the suite fully offline (no real AWS), the module-level
boto3 ``table`` / ``entity_table`` resources in ``app.main`` are monkeypatched
per test via ``monkeypatch.setattr(main.table, "<method>", fake)`` so each case
is isolated and deterministic.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — minimal SAR items shaped so ``minimize_sar_response`` works
# ---------------------------------------------------------------------------

def _minimal_sar_item() -> Dict[str, Any]:
    """A minimal stored SAR record that exercises ``minimize_sar_response``.

    Includes an Activity with a SuspiciousActivity list (classification +
    aggregate amount), a subject Party (type 33) with an identification number,
    and an institution Party (type 30) — matching the shapes the minimizer
    reads defensively.
    """
    return {
        "PK": "SAR#31000012345678",
        "SK": "METADATA",
        "Activity": {
            "BSAID": "31000012345678",
            "FilingDateText": "2024-02-01",
            "SuspiciousActivity": [
                {
                    "TotalSuspiciousAmountText": "45000.00",
                    "SuspiciousActivityClassification": [
                        {
                            "SuspiciousActivityTypeID": "9",
                            "SuspiciousActivityTypeCodeDescription": "Structuring",
                        }
                    ],
                }
            ],
            "Party": [
                {
                    "ActivityPartyTypeCode": "33",
                    "PartyName": [{"RawPartyFullName": "Jane Doe"}],
                    "PartyIdentification": [
                        {"PartyIdentificationNumberText": "123456789"}
                    ],
                },
                {
                    "ActivityPartyTypeCode": "30",
                    "PartyName": [{"RawPartyFullName": "First National Bank"}],
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_unchanged() -> None:
    """``GET /health`` → 200 with status/table/version keys (version 2.0.0).

    """
    resp = client.get("/health")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["table"] == main.TABLE_NAME
    assert body["version"] == "2.0.0"


# ---------------------------------------------------------------------------
# GET /sars/{bsa_id}
# ---------------------------------------------------------------------------

def test_get_sar_by_bsaid_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolvable bsaId returns the stored item verbatim with status 200.

    """
    item = _minimal_sar_item()

    def fake_get_item(Key=None, **kwargs):  # noqa: N803 - boto3 uses Key kwarg
        return {"Item": item}

    monkeypatch.setattr(main.table, "get_item", fake_get_item)

    resp = client.get("/sars/31000012345678")

    assert resp.status_code == 200, resp.text
    assert resp.json() == item


def test_get_sar_by_bsaid_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bsaId with no stored Item returns 404 "SAR not found".

    """

    def fake_get_item(Key=None, **kwargs):  # noqa: N803
        return {}  # no "Item" key

    monkeypatch.setattr(main.table, "get_item", fake_get_item)

    resp = client.get("/sars/does-not-exist")

    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "SAR not found"


# ---------------------------------------------------------------------------
# GET /institutions/{id}/sars
# ---------------------------------------------------------------------------

def test_get_sars_by_institution_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Institution list endpoint returns the documented response structure.

    """

    def fake_query(**kwargs):
        return {"Items": [_minimal_sar_item()]}  # no LastEvaluatedKey

    monkeypatch.setattr(main.table, "query", fake_query)

    resp = client.get("/institutions/INST-1/sars")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"institutionId", "count", "items", "nextCursor"}
    assert body["institutionId"] == "INST-1"
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["nextCursor"] is None


# ---------------------------------------------------------------------------
# GET /classifications/{code}/sars
# ---------------------------------------------------------------------------

def test_get_sars_by_classification_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classification list endpoint returns the documented response structure.

    """

    def fake_scan(**kwargs):
        return {"Items": [_minimal_sar_item()]}  # no LastEvaluatedKey

    monkeypatch.setattr(main.table, "scan", fake_scan)

    resp = client.get("/classifications/Structuring/sars")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"classificationCode", "count", "items", "nextCursor"}
    assert body["classificationCode"] == "Structuring"
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["nextCursor"] is None


# ---------------------------------------------------------------------------
# GET /entities/{entityId}/sars
# ---------------------------------------------------------------------------

def test_get_sars_by_entity_fallback_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity endpoint falls back to the GSI2 ``table.query`` path and returns 200.

    Force the entity-table lookup to miss (set ``entity_table`` to ``None``) so
    the handler takes the fallback branch that queries ``table.query`` on GSI2.

    """
    # Simplify: drop the entity table so the handler uses the GSI2 fallback.
    monkeypatch.setattr(main, "entity_table", None)

    def fake_query(**kwargs):
        return {"Items": [_minimal_sar_item()]}

    monkeypatch.setattr(main.table, "query", fake_query)

    resp = client.get("/entities/ENT-1/sars")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"entityId", "sarCount", "sars"}
    assert body["entityId"] == "ENT-1"
    assert body["sarCount"] == 1
    assert len(body["sars"]) == 1


# ---------------------------------------------------------------------------
# POST /entities:resolve
# ---------------------------------------------------------------------------

def test_resolve_entity_with_subject_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a subject header and no matches, resolve returns 200 ``{"matches": []}``.

    """

    def fake_scan(**kwargs):
        return {"Items": []}

    monkeypatch.setattr(main.table, "scan", fake_scan)

    resp = client.post("/entities:resolve", headers={"X-Subject-Name": "Test"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"matches": []}


def test_resolve_entity_requires_subject_header() -> None:
    """With no subject headers, resolve returns 400.

    """
    resp = client.post("/entities:resolve")

    assert resp.status_code == 400, resp.text
