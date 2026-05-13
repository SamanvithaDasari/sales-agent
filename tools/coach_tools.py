"""
coach_tools.py — Tools the Deal Coach agent can call.

Plain Python functions; the Deal Coach loop in agents/deal_coach.py will
expose them to Gemini using its native function-calling. We let the SDK
auto-extract the schema from each function's signature + docstring, so
docstrings matter a lot — they're what the LLM sees.

DESIGN PRINCIPLES (interview hooks):
- Each tool returns a SHORT string (JSON, capped). LLMs read text best
  and short tools save tokens that pile up across ReAct iterations.
- Errors return descriptive strings, never raise. Raising kills the loop.
- Tool descriptions written FOR THE LLM, focused on "when to use this".
- We import the RAG retriever from tools.rag so coach can do semantic search.
"""

import json
import sqlite3
from pathlib import Path

from tools.rag import find_similar_activities

DB_PATH = Path(__file__).parent.parent / "db" / "salesagent.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_json(rows: list[sqlite3.Row], limit: int = 10) -> str:
    """Serialize rows to compact JSON, truncating long lists to save tokens."""
    truncated = list(rows)[:limit]
    return json.dumps([dict(r) for r in truncated], default=str, indent=2)


# ============================================================
# TOOL 1: Get deal details
# ============================================================
def get_deal_details(deal_name: str) -> str:
    """Look up a specific deal by name (fuzzy match) and return its core details.

    Use this FIRST when the user asks about a specific deal. The deal name
    might be partial — 'Streamline Magnetic' will match 'Inc Streamline
    Magnetic Channels'. Returns the deal's id, full name, account, stage,
    value, age in days, and probability. Use the deal_id from this for
    subsequent tool calls.

    Args:
        deal_name: Partial or full deal name to search for.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT d.id, d.name AS deal_name, a.name AS account_name,
                      a.industry, d.stage, d.value, d.probability,
                      d.created_at, d.closed_at,
                      CAST((julianday('now') - julianday(d.created_at)) AS INTEGER) AS age_days
               FROM deals d INNER JOIN accounts a ON d.account_id = a.id
               WHERE d.name LIKE ?
               ORDER BY d.created_at DESC
               LIMIT 5""",
            (f"%{deal_name}%",),
        ).fetchall()
        conn.close()
        if not rows:
            return f"No deal found matching {deal_name!r}."
        return _rows_to_json(rows, limit=5)
    except Exception as e:
        return f"ERROR get_deal_details: {e}"


# ============================================================
# TOOL 2: Get a specific deal's activity history
# ============================================================
def get_deal_activities(deal_id: int) -> str:
    """Get the recent activity history (calls, emails, meetings, notes) for one deal.

    Use this AFTER get_deal_details to understand what's actually been happening
    on a specific deal. Returns up to 10 most recent activities with type and notes.
    Read these carefully — they tell the real story of why a deal might be stuck.

    Args:
        deal_id: The integer ID of the deal, from get_deal_details.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT type, notes, created_at
               FROM activities
               WHERE deal_id = ?
               ORDER BY created_at DESC
               LIMIT 10""",
            (deal_id,),
        ).fetchall()
        conn.close()
        if not rows:
            return f"No activities found for deal {deal_id}."
        return _rows_to_json(rows, limit=10)
    except Exception as e:
        return f"ERROR get_deal_activities: {e}"


# ============================================================
# TOOL 3: Semantic search over all activity notes (RAG)
# ============================================================
def find_similar_deal_activities(query: str, k: int = 5) -> str:
    """Semantic search across ALL deal activity notes for ones matching a description.

    Use this to find patterns across the whole CRM — e.g. 'pricing pushback',
    'champion departed', 'legal redlines stuck'. Uses embedding similarity,
    so 'buyer left the company' will match notes mentioning 'sponsor departed'
    even with no shared words.

    Returns up to k matches with: similarity score, account name, deal stage,
    and the note text. Filter your interpretation by deal_stage to find
    relevant patterns (e.g. compare won vs lost outcomes for similar situations).

    Args:
        query: Natural-language description of the pattern to search for.
        k: How many results to return (default 5, max 10).
    """
    try:
        matches = find_similar_activities(query, k=min(k, 10))
        # Compact JSON: limit fields to what the agent actually needs
        out = [
            {
                "score": round(m.score, 3),
                "account": m.record.account_name,
                "deal_stage": m.record.deal_stage,
                "deal_name": m.record.deal_name,
                "notes": m.record.notes,
            }
            for m in matches
        ]
        return json.dumps(out, indent=2)
    except Exception as e:
        return f"ERROR find_similar_deal_activities: {e}"


# ============================================================
# TOOL 4: Get cohort of deals at the same stage
# ============================================================
def get_deals_at_stage(stage: str, limit: int = 10) -> str:
    """Get other deals currently at a specific pipeline stage, for cohort comparison.

    Use this to answer 'how does this stuck deal compare to others at the same
    stage?'. Returns deal name, account, value, probability, and age in days.

    Valid stages: prospecting, qualification, proposal, negotiation,
    closed_won, closed_lost.

    Args:
        stage: One of the valid stage names above.
        limit: How many deals to return (default 10).
    """
    valid_stages = {"prospecting", "qualification", "proposal",
                    "negotiation", "closed_won", "closed_lost"}
    if stage not in valid_stages:
        return f"Invalid stage {stage!r}. Must be one of: {sorted(valid_stages)}."

    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT d.name AS deal_name, a.name AS account_name,
                      d.value, d.probability,
                      CAST((julianday('now') - julianday(d.created_at)) AS INTEGER) AS age_days
               FROM deals d INNER JOIN accounts a ON d.account_id = a.id
               WHERE d.stage = ?
               ORDER BY d.created_at DESC
               LIMIT ?""",
            (stage, limit),
        ).fetchall()
        conn.close()
        return _rows_to_json(rows, limit=limit)
    except Exception as e:
        return f"ERROR get_deals_at_stage: {e}"


# ============================================================
# TOOL 5: Won vs Lost outcomes for a pattern
# ============================================================
def get_won_lost_outcomes(pattern: str, k: int = 4) -> str:
    """Find activity notes from CLOSED deals (won and lost) matching a pattern.

    Use this to learn 'what worked' and 'what didn't' for similar situations.
    Returns up to k matches from closed_won and k from closed_lost, separately,
    so the agent can compare outcomes.

    Args:
        pattern: Natural-language pattern, e.g. 'discount negotiation' or
                 'champion left'.
        k: How many of EACH outcome (won, lost) to return.
    """
    try:
        matches = find_similar_activities(pattern, k=20)
        won = [m for m in matches if m.record.deal_stage == "closed_won"][:k]
        lost = [m for m in matches if m.record.deal_stage == "closed_lost"][:k]
        out = {
            "won_examples": [
                {"account": m.record.account_name, "notes": m.record.notes,
                 "score": round(m.score, 3)}
                for m in won
            ],
            "lost_examples": [
                {"account": m.record.account_name, "notes": m.record.notes,
                 "score": round(m.score, 3)}
                for m in lost
            ],
        }
        return json.dumps(out, indent=2)
    except Exception as e:
        return f"ERROR get_won_lost_outcomes: {e}"


# Tool registry: name → callable. The Deal Coach loop uses this to dispatch
# function calls emitted by Gemini.
COACH_TOOLS = {
    "get_deal_details": get_deal_details,
    "get_deal_activities": get_deal_activities,
    "find_similar_deal_activities": find_similar_deal_activities,
    "get_deals_at_stage": get_deals_at_stage,
    "get_won_lost_outcomes": get_won_lost_outcomes,
}