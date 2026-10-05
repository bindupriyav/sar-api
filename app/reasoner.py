"""Optional Bedrock_Reasoner for the SAR Classification Agent.

Produces a natural-language rationale by invoking Amazon Bedrock Claude, reusing
the invocation pattern already established in ``agent.py`` (bedrock-runtime client,
``anthropic_version`` ``bedrock-2023-05-31``, ``temperature`` 0.1, parsing
``content[0].text``). The Reasoner is advisory and opt-in: when it fails or times
out, the deterministic Rule_Engine result is still returned with a notice.

Interfaces (filled in by later tasks):
    - ``generate_rationale(verdict, context) -> str | None``

Deployment note:
    This module invokes Amazon Bedrock. Deploying it requires the ECS task role to
    gain ``bedrock:InvokeModel`` (scoped to the configured inference-profile model
    ARN in ``us-east-2``) IN ADDITION TO the existing DynamoDB read permissions on
    the ``sar-prototype`` table. Without that permission, model reasoning fails and
    the service degrades gracefully to the Rule_Engine result.

This is a scaffolding stub so later tasks have an import target.
"""
import json
import os

import boto3

# ---------------------------------------------------------------------------
# Configuration — mirrors agent.py exactly
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
REGION = os.environ.get("AWS_REGION", "us-east-2")

# ---------------------------------------------------------------------------
# Reasoner system prompt — human-in-the-loop + Title 21 soft language
# ---------------------------------------------------------------------------
#
# This prompt mirrors the relevant rules from the agent.py SYSTEM_PROMPT. The
# soft-language tokens ("indicia of", "consistent with", "flag for review") are
# written literally so a prompt-level test can assert their presence.
REASONER_SYSTEM_PROMPT = """You are a BSA/AML compliance analyst AI agent working for a regulatory agency.
You produce a plain-language rationale explaining a deterministic structuring classification result so that a human Analyst can review and validate it.

RULES (you must follow these strictly):
1. Your output is a flag or assessment that REQUIRES Analyst review. It is never a final decision.
2. NEVER state that a legal violation has occurred and NEVER make a legal determination of guilt.
3. NEVER recommend law-enforcement action — only suggest actions that require Analyst approval.
4. Flag ambiguity rather than resolving it — leave decisions to the human Analyst.
5. Describe the structuring pattern in review-oriented terms: it shows "indicia of" structuring, is "consistent with" structuring, and should "flag for review" by an Analyst.
6. For any controlled-substance (Title 21) indicator, use ONLY "indicia of," "consistent with," "flag for review" language. NEVER assert that a controlled-substance violation occurred.
7. Title 21 observations MUST be framed with this disclaimer: "SAR data alone does not establish a controlled-substance violation. Independent law-enforcement predication is required."
8. Base your rationale ONLY on the facts provided in this request — never from memory.

Keep the rationale concise (a few sentences). Explain the aggregated amount, the number of contributing transactions, the rolling window used, and any near-threshold or narrative signals that warrant Analyst attention."""


def _build_user_message(verdict, context: dict) -> str:
    """Construct the user message from the verdict and context, defensively.

    Summarizes the detected typology, aggregated amount, triggering count, the
    rolling window used, and any narrative/aggregate signals supplied in
    ``context``. Any missing attribute degrades to a neutral placeholder rather
    than raising.
    """
    context = context or {}

    def _attr(obj, name, default=None):
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    typology = _attr(verdict, "typology") or "No typology detected"
    detected = _attr(verdict, "detected", False)
    aggregated = _attr(verdict, "aggregatedAmount", "unknown")
    window_days = _attr(verdict, "windowDays", "unknown")
    window_span = _attr(verdict, "windowSpanDays")

    try:
        triggering = _attr(verdict, "triggeringTransactions", []) or []
        triggering_count = len(triggering)
    except Exception:
        triggering_count = 0

    try:
        reportable = _attr(verdict, "reportableTransactions", []) or []
        reportable_count = len(reportable)
    except Exception:
        reportable_count = 0

    lines = [
        "Explain the following deterministic structuring classification result "
        "for Analyst review. Do not assert any violation.",
        "",
        f"Detected: {detected}",
        f"Typology: {typology}",
        f"Aggregated contributing amount: {aggregated}",
        f"Number of contributing (triggering) transactions: {triggering_count}",
        f"Number of reportable cash transactions (>= $10,000.00): {reportable_count}",
        f"Rolling window used (days): {window_days}",
    ]
    if window_span is not None:
        lines.append(f"Contributing date span (days, inclusive): {window_span}")

    # Any additional narrative/aggregate signals passed by the caller (Title 21
    # narrative framing is handled by the system prompt).
    narrative = context.get("narrative")
    if narrative:
        lines.append("")
        lines.append(f"Narrative signal: {narrative}")

    aggregate_signal = context.get("aggregateSignal") or context.get("aggregate_signal")
    if aggregate_signal:
        lines.append(f"Aggregate signal: {aggregate_signal}")

    bsa_id = context.get("bsaId") or context.get("bsa_id")
    if bsa_id:
        lines.append(f"Source BSAID: {bsa_id}")

    # Fold in any remaining simple context keys without failing on odd types.
    for key, value in context.items():
        if key in {"narrative", "aggregateSignal", "aggregate_signal", "bsaId", "bsa_id"}:
            continue
        try:
            lines.append(f"{key}: {value}")
        except Exception:
            continue

    return "\n".join(lines)


def generate_rationale(verdict, context: dict) -> "str | None":
    """Produce a natural-language rationale via Amazon Bedrock Claude.

    Reuses the ``agent.py`` invocation pattern exactly: a ``bedrock-runtime``
    client, ``anthropic_version`` ``bedrock-2023-05-31``, ``temperature`` 0.1, and
    parsing ``content[0].text``.

    The entire invocation is wrapped so that any exception or timeout returns
    ``None``, letting the caller degrade gracefully to the Rule_Engine result
    with ``modelReasoningAvailable = false`` (the caller handles the notice).

    The Bedrock client is created INSIDE this function so importing this module
    performs no network or boto call.
    """
    try:
        user_message = _build_user_message(verdict, context)

        client = boto3.client("bedrock-runtime", region_name=REGION)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "system": REASONER_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        })

        response = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        return json.loads(response["body"].read())["content"][0]["text"]
    except Exception:
        return None  # graceful fallback → modelReasoningAvailable = false
