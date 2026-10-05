# Feature: sar-classification-agent, Task 9.3: Mocked-Bedrock integration tests
"""Mocked-Bedrock integration tests for optional model reasoning.


Three complementary layers, none of which touch the network:

A) HANDLER-LEVEL (behavior) — monkeypatch ``app.main.generate_rationale`` to
   control its return value and detect whether it was called, then drive the
   full ``POST /classify`` transaction path through ``TestClient``:
     * ``enableModelReasoning`` True + rationale text → 200,
       ``modelReasoningAvailable`` True, ``modelReasoning`` echoed (Req 5.2).
     * ``enableModelReasoning`` omitted/False → Reasoner NOT called, rule-only
       result with ``modelReasoningAvailable`` False and null ``modelReasoning``
       (Req 5.3).
     * ``enableModelReasoning`` True + ``None`` rationale → 200,
       ``modelReasoningAvailable`` False, and a model-unavailable notice present
       in ``ambiguityNotes`` (Req 5.4).

B) REASONER-LEVEL (invoke_model wiring) — monkeypatch ``boto3.client`` used by
   ``app.reasoner`` so ``client.invoke_model`` is a ``Mock``, then call
   ``generate_rationale`` directly:
     * ``invoke_model`` returns a fake Bedrock response → the parsed
       ``content[0].text`` is returned and ``invoke_model`` was called once
       (Req 5.2).
     * ``invoke_model`` raises → ``generate_rationale`` returns ``None`` so the
       caller degrades gracefully (Req 5.4 fallback).

C) PROMPT-LEVEL — assert the module-level ``REASONER_SYSTEM_PROMPT`` enforces
   Title 21 soft language ("indicia of", "consistent with", "flag for review")
   and forbids asserting a violation / making a legal determination
   (Req 5.5, 7.2).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.reasoner as reasoner
from app.main import app
from app.reasoner import REASONER_SYSTEM_PROMPT, generate_rationale

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_tx(amount: str, date: str, tx_type: str = "CASH_DEPOSIT") -> Dict[str, Any]:
    """A minimal valid transaction payload."""
    return {"amount": amount, "date": date, "transactionType": tx_type}


def _structuring_transactions() -> List[Dict[str, Any]]:
    """Two sub-threshold cash transactions one day apart that aggregate above the
    $10,000.00 CTR threshold within the default 30-day window → Structuring.
    """
    return [
        _valid_tx("9999.99", "2024-01-01"),
        _valid_tx("9999.99", "2024-01-02"),
    ]


# ===========================================================================
# A) HANDLER-LEVEL behavior tests (monkeypatch app.main.generate_rationale)
# ===========================================================================


def test_flag_true_attaches_rationale(monkeypatch):
    """enableModelReasoning True + rationale text → attached (Req 5.2)."""
    calls: List[Any] = []

    def fake_generate_rationale(verdict, context=None):
        calls.append((verdict, context))
        return "LLM rationale"

    monkeypatch.setattr(main, "generate_rationale", fake_generate_rationale)

    resp = client.post(
        "/classify",
        json={
            "transactions": _structuring_transactions(),
            "enableModelReasoning": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["modelReasoningAvailable"] is True
    assert body["modelReasoning"] == "LLM rationale"
    # The Reasoner was actually invoked.
    assert len(calls) == 1


def test_flag_absent_does_not_invoke_reasoner(monkeypatch):
    """enableModelReasoning omitted → Reasoner NOT called, rule-only (Req 5.3)."""

    def exploding_generate_rationale(verdict, context=None):
        raise AssertionError(
            "generate_rationale must not be called when reasoning is disabled"
        )

    monkeypatch.setattr(main, "generate_rationale", exploding_generate_rationale)

    resp = client.post(
        "/classify",
        json={"transactions": _structuring_transactions()},  # no enableModelReasoning
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["modelReasoningAvailable"] is False
    assert body["modelReasoning"] is None


def test_flag_false_does_not_invoke_reasoner(monkeypatch):
    """enableModelReasoning explicitly False → Reasoner NOT called (Req 5.3)."""
    calls: List[Any] = []

    def recording_generate_rationale(verdict, context=None):
        calls.append((verdict, context))
        return "should not be used"

    monkeypatch.setattr(main, "generate_rationale", recording_generate_rationale)

    resp = client.post(
        "/classify",
        json={
            "transactions": _structuring_transactions(),
            "enableModelReasoning": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert calls == []  # never invoked
    assert body["modelReasoningAvailable"] is False
    assert body["modelReasoning"] is None


def test_flag_true_reasoner_none_degrades_with_notice(monkeypatch):
    """enableModelReasoning True + None rationale → 200, unavailable + notice (Req 5.4)."""
    monkeypatch.setattr(main, "generate_rationale", lambda verdict, context=None: None)

    resp = client.post(
        "/classify",
        json={
            "transactions": _structuring_transactions(),
            "enableModelReasoning": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["modelReasoningAvailable"] is False
    assert body["modelReasoning"] is None

    notes = body.get("ambiguityNotes") or []
    # The model-unavailable notice is surfaced among the ambiguity notes.
    assert main.MODEL_UNAVAILABLE_NOTICE in notes


# ===========================================================================
# B) REASONER-LEVEL invoke_model wiring tests (monkeypatch boto3.client)
# ===========================================================================


class _FakeBody:
    """Minimal stand-in for a Bedrock streaming body: supports ``.read()``."""

    def __init__(self, payload: str):
        self._payload = payload

    def read(self) -> str:
        return self._payload


def test_invoke_model_success_returns_parsed_text(monkeypatch):
    """invoke_model returns a fake response → parsed text returned, called once (Req 5.2)."""
    response_payload = json.dumps({"content": [{"text": "soft rationale"}]})
    fake_response = {"body": _FakeBody(response_payload)}

    fake_client = MagicMock()
    fake_client.invoke_model = Mock(return_value=fake_response)

    monkeypatch.setattr(
        reasoner.boto3, "client", lambda *a, **k: fake_client
    )

    verdict = MagicMock()
    verdict.typology = "Structuring"
    verdict.detected = True

    result = generate_rationale(verdict, context={"bsaId": "BSA-1"})

    assert result == "soft rationale"
    fake_client.invoke_model.assert_called_once()


def test_invoke_model_raises_returns_none(monkeypatch):
    """invoke_model raises → generate_rationale returns None (Req 5.4 fallback)."""
    fake_client = MagicMock()
    fake_client.invoke_model = Mock(side_effect=Exception("bedrock down"))

    monkeypatch.setattr(
        reasoner.boto3, "client", lambda *a, **k: fake_client
    )

    verdict = MagicMock()
    verdict.typology = "Structuring"

    result = generate_rationale(verdict, context={})

    assert result is None
    fake_client.invoke_model.assert_called_once()


# ===========================================================================
# C) PROMPT-LEVEL assertions on REASONER_SYSTEM_PROMPT (Req 5.5, 7.2)
# ===========================================================================


def test_prompt_contains_title21_soft_language():
    """Title 21 soft-language tokens are present (case-insensitive) (Req 5.5)."""
    prompt = REASONER_SYSTEM_PROMPT.lower()
    for token in ("indicia of", "consistent with", "flag for review"):
        assert token in prompt, f"expected soft-language token {token!r} in prompt"


def test_prompt_forbids_asserting_violation():
    """Prompt forbids asserting a violation / making a legal determination (Req 7.2)."""
    prompt = REASONER_SYSTEM_PROMPT.lower()
    # The prompt must contain a strong prohibition ("never") against asserting a
    # violation or making a legal determination.
    assert "never" in prompt
    assert ("violation" in prompt) or ("legal determination" in prompt)
