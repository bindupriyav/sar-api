#!/usr/bin/env python3
"""
agent.py — SAR Analysis AI Agent (Prototype)

Calls the SAR Data API, sends results to Amazon Bedrock (Claude),
and returns a structured compliance/pattern analysis per §5 and §6
of the SAR Analysis Agent Design Outline.

Usage:
    # Analyze by institution
    python agent.py --institution 59-9893515

    # Analyze by entity
    python agent.py --entity 106-73-4850

    # Analyze by entity name (uses entity resolution)
    python agent.py --name Dixon

Environment variables:
    SAR_API_URL  - Base URL of the SAR FastAPI service (default: http://localhost:8000)
    AWS_REGION   - AWS region for Bedrock (default: us-east-2)
"""

import argparse
import json
import os
import sys

import boto3
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAR_API_URL = os.environ.get("SAR_API_URL", "http://localhost:8000")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Step 1: Call the SAR API to get data
# ---------------------------------------------------------------------------

def get_sars_by_institution(institution_id: str) -> dict:
    """Call GET /institutions/{id}/sars"""
    url = f"{SAR_API_URL}/institutions/{institution_id}/sars?limit=50"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_sars_by_entity(entity_id: str) -> dict:
    """Call GET /entities/{id}/sars"""
    url = f"{SAR_API_URL}/entities/{entity_id}/sars"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def resolve_entity(name: str) -> dict:
    """Call POST /entities:resolve with name in header"""
    url = f"{SAR_API_URL}/entities:resolve"
    headers = {"X-Subject-Name": name}
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Step 2: Build the prompt (per §5.3 and §6)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a BSA/AML compliance analyst AI agent working for a regulatory agency.
You analyze Suspicious Activity Reports (SARs) to identify patterns and non-compliance.

RULES (you must follow these strictly):
1. Every finding MUST cite one or more specific BSAIDs from the data provided.
2. NEVER make claims from memory — only from the SAR data in this session.
3. Separate your analysis into Title 31 and Title 21 sections.
4. For Title 31 (BSA compliance): you may state findings directly.
5. For Title 21 (controlled substances): use ONLY "indicia of," "consistent with," "flag for review" language. NEVER assert a violation occurred.
6. Title 21 section MUST include this disclaimer: "SAR data alone does not establish a controlled-substance violation. Independent law-enforcement predication is required."
7. NEVER state guilt or make legal determinations.
8. NEVER recommend law-enforcement action — only suggest actions that require analyst approval.
9. Flag ambiguity rather than resolving it — leave decisions to the human analyst.

ANALYSIS CHECKLIST:
- Structuring (deposits/transactions just below $10,000 CTR threshold)
- Shell company indicators (business entity, no clear operations, high volume)
- Rapid fund movement (money in and out within 24-48 hours)
- Layering (multiple account/institution hops)
- Funnel accounts (many sources into one account)
- Late filing (filing date > 30 days after suspicious activity end date)
- Missing required information (no TIN, no DOB, no occupation)
- Missing continuation SARs (ongoing activity, only initial filing)
- Narrative/classification mismatch
- Cross-border risk (transfers to high-risk jurisdictions)
- Occupation/income inconsistency (transaction volume vs. stated occupation)

OUTPUT FORMAT:
Return ONLY valid JSON matching this structure:
{
  "overallRiskScore": <number 1-10>,
  "riskTier": "LOW|MEDIUM|HIGH",
  "summary": "<2-3 sentence executive summary>",
  "title31Assessment": {
    "findings": [
      {
        "pattern": "<pattern name>",
        "description": "<what was found>",
        "supportingBSAIDs": ["<bsaid1>", "<bsaid2>"],
        "riskLevel": "LOW|MEDIUM|HIGH"
      }
    ]
  },
  "title21CrossReferenceAssessment": {
    "disclaimer": "SAR data alone does not establish a controlled-substance violation. Independent law-enforcement predication is required.",
    "findings": [
      {
        "indicator": "<description using soft language only>",
        "supportingBSAIDs": ["<bsaid>"],
        "confidence": "LOW|MEDIUM|HIGH"
      }
    ]
  },
  "nonComplianceFlags": [
    {
      "flag": "<flag name>",
      "description": "<what was found>",
      "supportingBSAIDs": ["<bsaid>"],
      "severity": "LOW|MEDIUM|HIGH"
    }
  ],
  "shellCompanyIndicators": [
    {
      "entityName": "<name>",
      "indicators": ["<indicator1>", "<indicator2>"],
      "supportingBSAIDs": ["<bsaid>"],
      "confidence": "LOW|MEDIUM|HIGH"
    }
  ],
  "recommendedActions": [
    "<action 1 — requires analyst approval>",
    "<action 2>"
  ],
  "modelDisclosure": "Analysis produced by AI agent using synthetic data. All findings require analyst review and approval before any action is taken."
}"""


def build_user_prompt(sar_data: dict, query_type: str, query_value: str) -> str:
    """Construct the user message with SAR data."""
    return f"""Analyze the following SAR filings retrieved by {query_type}: {query_value}

Identify all BSA patterns, non-compliance issues, shell company indicators, and Title 21 cross-reference flags.

SAR DATA:
{json.dumps(sar_data, indent=2, default=str)}"""


# ---------------------------------------------------------------------------
# Step 3: Call Bedrock (Claude)
# ---------------------------------------------------------------------------

def call_bedrock(system_prompt: str, user_prompt: str) -> str:
    """Send the prompt to Amazon Bedrock and return the response."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.1,  # Low temperature for consistent, factual analysis
    })

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# ---------------------------------------------------------------------------
# Step 4: Parse and display the output
# ---------------------------------------------------------------------------

def display_analysis(analysis_text: str):
    """Parse and display the structured analysis."""
    # Try to parse as JSON
    try:
        # Sometimes the model wraps JSON in markdown code blocks
        cleaned = analysis_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        analysis = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Raw response (could not parse as JSON):")
        print(analysis_text)
        return

    # Display formatted output
    print("\n" + "=" * 70)
    print("SAR ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nOverall Risk Score: {analysis.get('overallRiskScore', 'N/A')}/10")
    print(f"Risk Tier: {analysis.get('riskTier', 'N/A')}")
    print(f"\nSummary: {analysis.get('summary', 'N/A')}")

    # Title 31
    t31 = analysis.get("title31Assessment", {})
    findings = t31.get("findings", [])
    if findings:
        print(f"\n{'─' * 70}")
        print("TITLE 31 — BSA COMPLIANCE FINDINGS")
        print(f"{'─' * 70}")
        for i, f in enumerate(findings, 1):
            print(f"\n  {i}. {f.get('pattern', 'Unknown')} [{f.get('riskLevel', '')}]")
            print(f"     {f.get('description', '')}")
            print(f"     BSAIDs: {', '.join(f.get('supportingBSAIDs', []))}")

    # Non-compliance
    nc_flags = analysis.get("nonComplianceFlags", [])
    if nc_flags:
        print(f"\n{'─' * 70}")
        print("NON-COMPLIANCE FLAGS")
        print(f"{'─' * 70}")
        for i, f in enumerate(nc_flags, 1):
            print(f"\n  {i}. {f.get('flag', 'Unknown')} [{f.get('severity', '')}]")
            print(f"     {f.get('description', '')}")
            print(f"     BSAIDs: {', '.join(f.get('supportingBSAIDs', []))}")

    # Shell companies
    shell = analysis.get("shellCompanyIndicators", [])
    if shell:
        print(f"\n{'─' * 70}")
        print("SHELL COMPANY INDICATORS")
        print(f"{'─' * 70}")
        for i, s in enumerate(shell, 1):
            print(f"\n  {i}. {s.get('entityName', 'Unknown')} [{s.get('confidence', '')}]")
            print(f"     Indicators: {', '.join(s.get('indicators', []))}")
            print(f"     BSAIDs: {', '.join(s.get('supportingBSAIDs', []))}")

    # Title 21
    t21 = analysis.get("title21CrossReferenceAssessment", {})
    t21_findings = t21.get("findings", [])
    if t21_findings:
        print(f"\n{'─' * 70}")
        print("TITLE 21 — CROSS-REFERENCE INDICATORS (INFORMATIONAL ONLY)")
        print(f"{'─' * 70}")
        print(f"\n  DISCLAIMER: {t21.get('disclaimer', '')}")
        for i, f in enumerate(t21_findings, 1):
            print(f"\n  {i}. [{f.get('confidence', '')}] {f.get('indicator', '')}")
            print(f"     BSAIDs: {', '.join(f.get('supportingBSAIDs', []))}")

    # Recommended actions
    actions = analysis.get("recommendedActions", [])
    if actions:
        print(f"\n{'─' * 70}")
        print("RECOMMENDED ACTIONS (require analyst approval)")
        print(f"{'─' * 70}")
        for i, a in enumerate(actions, 1):
            print(f"  {i}. {a}")

    print(f"\n{'─' * 70}")
    print(f"Model Disclosure: {analysis.get('modelDisclosure', '')}")
    print("=" * 70)

    # Also save the raw JSON output
    output_file = "agent_analysis_output.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull JSON output saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAR Analysis AI Agent")
    parser.add_argument("--institution", help="Institution EIN to analyze")
    parser.add_argument("--entity", help="Entity ID to analyze")
    parser.add_argument("--name", help="Subject name to resolve and analyze")
    parser.add_argument("--api-url", help="Override SAR API URL")
    args = parser.parse_args()

    global SAR_API_URL
    if args.api_url:
        SAR_API_URL = args.api_url

    if not any([args.institution, args.entity, args.name]):
        parser.error("Provide at least one of: --institution, --entity, --name")

    # Step 1: Get SAR data from API
    print("Step 1: Fetching SAR data from API...")

    if args.institution:
        sar_data = get_sars_by_institution(args.institution)
        query_type = "institution"
        query_value = args.institution
        print(f"  Found {sar_data.get('count', 0)} SARs for institution {args.institution}")

    elif args.entity:
        sar_data = get_sars_by_entity(args.entity)
        query_type = "entity"
        query_value = args.entity
        print(f"  Found {sar_data.get('sarCount', 0)} SARs for entity {args.entity}")

    elif args.name:
        print(f"  Resolving entity: {args.name}...")
        matches = resolve_entity(args.name)
        if not matches.get("matches"):
            print(f"  No matches found for '{args.name}'")
            sys.exit(1)
        # Use the first match
        match = matches["matches"][0]
        entity_id = match["entityId"]
        print(f"  Resolved to: {match['displayName']} (entity {entity_id}, confidence {match['matchConfidence']})")
        sar_data = get_sars_by_entity(entity_id.replace("ENT-", ""))
        query_type = "entity (resolved from name)"
        query_value = f"{args.name} → {entity_id}"
        print(f"  Found {sar_data.get('sarCount', 0)} SARs")

    # Step 2: Build prompt
    print("\nStep 2: Constructing analysis prompt (§5.3 + §6 rules)...")
    user_prompt = build_user_prompt(sar_data, query_type, query_value)
    print(f"  Prompt size: {len(user_prompt)} characters")

    # Step 3: Call Bedrock
    print(f"\nStep 3: Calling Bedrock ({MODEL_ID})...")
    print("  This may take 10-20 seconds...")
    analysis_text = call_bedrock(SYSTEM_PROMPT, user_prompt)
    print("  Response received.")

    # Step 4: Display results
    print("\nStep 4: Parsing and displaying analysis...")
    display_analysis(analysis_text)


if __name__ == "__main__":
    main()
