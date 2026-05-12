"""
lead_intel.py — Lead Intelligence "Agent" via direct orchestration.

ARCHITECTURE DECISION:
We do NOT use a CrewAI ReAct loop here. The workflow is deterministic
(account → deals → contacts → activities → synthesize) so we orchestrate
the tool calls in plain Python and use the LLM ONCE for synthesis.

This drops the per-request token cost from ~15K (ReAct, 5 LLM calls each
carrying the growing transcript) to ~2.5K (one focused synthesis call).
We trade autonomous tool-selection (which we don't need here) for cost
efficiency and predictability.

The agent abstraction still exists — same tools, same role/goal framing
in the prompt — we just don't pay for the LLM to plan. We'll use a full
ReAct agent for Deal Coach, where branching reasoning is genuinely needed.
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

# Import the underlying tool functions. We bypass @tool decoration since
# we're calling them directly, not via an agent loop. Using .func gives us
# the raw Python callable from the CrewAI-decorated tools.
from tools.crm_tools import (
    find_account_by_name,
    get_deals_for_account,
    get_contacts_for_account,
    get_recent_activities_for_account,
)

load_dotenv()


# ============================================================
# LLM CLIENT
# ============================================================
# Single LLM call: no agent loop, no ReAct. Just a synthesis prompt.
groq_client = Groq()
MODEL = "llama-3.1-8b-instant"

# ============================================================
# ENTITY EXTRACTION (tiny LLM call to clean up natural-language queries)
# ============================================================
EXTRACT_SYSTEM_PROMPT = """You extract a single company name from a sales rep's \
query. The rep is asking about a customer/prospect company.

Rules:
- Return ONLY the company name, nothing else. No punctuation, no quotes, no \
"Here is" preambles.
- If the query is just a company name already, return it unchanged.
- If there's no identifiable company name, return the literal string: UNKNOWN
- Do not add "Inc", "LLC", etc. that aren't in the original query.
- Strip filler like "Tell me about", "What's going on with", "before my call", etc.

Examples:
Input: Tell me about Acme Corp before my call tomorrow
Output: Acme Corp

Input: What's going on with Pacheco-Smith?
Output: Pacheco-Smith

Input: Davis and Sons
Output: Davis and Sons

Input: Give me the latest on the Microsoft deal
Output: Microsoft

Input: Brief me
Output: UNKNOWN"""


def extract_company_name(query: str) -> str:
    """Use the LLM to pull a clean company name out of a natural-language query."""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,    # zero — we want deterministic extraction
        max_tokens=50,      # company names are short; cap hard
    )
    return response.choices[0].message.content.strip()

# ============================================================
# CONTEXT GATHERING (deterministic, no LLM involved)
# ============================================================
@dataclass
class LeadContext:
    """All the CRM data needed to write a pre-call brief."""
    account: dict | None
    deals: list[dict]
    contacts: list[dict]
    activities: list[dict]
    all_matches: list[dict]  # other companies that matched the search, for transparency


def gather_lead_context(company_query: str) -> LeadContext:
    """
    Pull all CRM context for a company. No LLM involved — just SQL via
    the tool functions. Fast, free, deterministic.

    Returns a LeadContext with the resolved account (or None if not found)
    and all related data, plus any other matches for disambiguation.
    """
    # Step 1: Resolve the company name to an account
    raw_matches = find_account_by_name.func(company_query)
    matches = json.loads(raw_matches)

    if not matches:
        # No match — return an empty context. The LLM prompt will handle this case.
        return LeadContext(account=None, deals=[], contacts=[], activities=[], all_matches=[])

    # Pick the first (best fuzzy match). Note: a smarter version would let
    # the LLM disambiguate; for now we surface all_matches so the brief
    # can note that there are other possibilities.
    account = matches[0]
    account_id = account["id"]

    # Step 2-4: Pull related data in parallel (could be threaded, kept serial for clarity)
    deals = json.loads(get_deals_for_account.func(account_id))
    contacts = json.loads(get_contacts_for_account.func(account_id))
    activities = json.loads(get_recent_activities_for_account.func(account_id, limit=10))

    return LeadContext(
        account=account,
        deals=deals,
        contacts=contacts,
        activities=activities,
        all_matches=matches[1:],  # everyone except the chosen one
    )


# ============================================================
# SYNTHESIS PROMPT (one LLM call)
# ============================================================
SYSTEM_PROMPT = """You are a Lead Intelligence Specialist — an experienced sales \
operations analyst. You write concise, actionable pre-call briefs for B2B sales reps.

Rules:
- Cut through noise. Surface only what matters for the call.
- Never invent data. If a section has no data, write "No data available."
- Be specific. Reference actual deal names, contact titles, recent quotes from \
notes. Vague briefs help no one.
- Keep recommended talking points grounded in what's IN the CRM, not generic \
sales advice.
"""

USER_PROMPT_TEMPLATE = """The sales rep asked: "{query}"

Below is the CRM data for the company they're asking about. Synthesize it \
into a pre-call brief using the exact Markdown structure shown.

## CRM DATA

### Account
{account_json}

### Other matching companies (for disambiguation, mention only if relevant)
{other_matches_json}

### Deals ({n_deals} total)
{deals_json}

### Contacts ({n_contacts} total)
{contacts_json}

### Recent Activities ({n_activities} most recent)
{activities_json}

## OUTPUT FORMAT (use this exact structure)

## Account Snapshot
- One line: name, industry, employees, annual revenue.

## Deal Pipeline
- List of OPEN deals (closed_at is null): name, stage, value, probability.
- One line at the end summarizing closed_won / closed_lost counts for context.

## Key Stakeholders
- List of contacts: name (title).

## Recent Activity Highlights
- 2-4 bullets summarizing the most meaningful recent activities. Quote \
or paraphrase actual notes, don't summarize generically.

## Recommended Talking Points
- 2-3 specific things the rep should bring up, grounded in the activities \
above. Each point should reference a specific deal or note."""


def synthesize_brief(query: str, ctx: LeadContext) -> str:
    """Make the single LLM call that turns CRM data into a pre-call brief."""
    if ctx.account is None:
        return f"## No Match Found\n\nNo account in the CRM matches '{query}'. " \
               "Try a different company name or check for typos."

    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        account_json=json.dumps(ctx.account, indent=2),
        other_matches_json=json.dumps(ctx.all_matches, indent=2) if ctx.all_matches else "None",
        n_deals=len(ctx.deals),
        deals_json=json.dumps(ctx.deals, indent=2),
        n_contacts=len(ctx.contacts),
        contacts_json=json.dumps(ctx.contacts, indent=2),
        n_activities=len(ctx.activities),
        activities_json=json.dumps(ctx.activities, indent=2),
    )

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1500,  # the brief shouldn't exceed this
    )

    return response.choices[0].message.content


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_lead_intel(company_query: str) -> str:
    """
    Full Lead Intel pipeline:
      1. Extract clean company name from natural-language query (LLM #1)
      2. Gather CRM context (no LLM)
      3. Synthesize brief (LLM #2)
    """
    # Step 1: parse the natural-language query into a clean company name
    company_name = extract_company_name(company_query)
    if company_name == "UNKNOWN" or not company_name:
        return ("## No Company Identified\n\n"
                "I couldn't identify a company name in your query. "
                "Try: 'Tell me about Acme Corp' or just 'Acme Corp'.")

    print(f"   📌 Extracted company: {company_name!r}")  # diagnostic; remove later

    # Step 2: gather CRM context (no LLM)
    ctx = gather_lead_context(company_name)

    # Step 3: synthesize the brief (one LLM call)
    return synthesize_brief(company_query, ctx)


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about Davis Group"
    print(f"\n🔍 Lead Intel for: {query!r}\n")
    print("=" * 70)
    brief = run_lead_intel(query)
    print(brief)
    print("=" * 70)