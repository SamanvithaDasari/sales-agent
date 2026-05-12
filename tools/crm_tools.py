"""
crm_tools.py — Tools the CRM agents can call.

Tools are plain Python functions wrapped with CrewAI's `@tool` decorator.
The decorator exposes them to the LLM with a name + description schema.

KEY DESIGN DECISIONS:
- Each tool returns a STRING (not raw dicts). LLMs read strings; structured
  output from a tool gets serialized to text anyway, so we control the format.
- Errors return descriptive strings, never raise. Raising crashes the agent loop.
- Tool descriptions are written FOR THE LLM, not for human readers.
  They must specify: when to use, what args look like, what gets returned.
"""

import json
import sqlite3
from pathlib import Path

from crewai.tools import tool

# Resolve DB path relative to project root, regardless of where the script runs from
DB_PATH = Path(__file__).parent.parent / "db" / "salesagent.db"


def _connect() -> sqlite3.Connection:
    """Open a connection with FK enforcement on and row factory set to dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    # row_factory makes rows accessible as dicts: row['name'] instead of row[1]
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_json(rows: list[sqlite3.Row]) -> str:
    """Convert a list of sqlite3.Row to a JSON string. LLMs parse JSON well."""
    return json.dumps([dict(row) for row in rows], default=str, indent=2)


# ============================================================
# TOOL 1: Find account by name (fuzzy match)
# ============================================================
@tool("find_account_by_name")
def find_account_by_name(company_name: str) -> str:
    """
    Find a CRM account (company) by name using fuzzy match. Use this FIRST
    when the user mentions a company by name — you'll need the account_id
    to query for related deals, contacts, and activities.

    Args:
        company_name: The company name to search for, e.g. "Acme Corp" or "Davis".
                      Partial matches work — "Davis" finds "Davis and Sons".

    Returns:
        A JSON list of matching accounts with their id, name, industry, employees,
        and annual_revenue. Returns an empty list "[]" if no match found.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT id, name, industry, employees, annual_revenue
               FROM accounts
               WHERE name LIKE ?
               ORDER BY name
               LIMIT 10""",
            (f"%{company_name}%",),
        ).fetchall()
        conn.close()
        return _rows_to_json(rows)
    except Exception as e:
        return f"ERROR querying accounts: {e}"


# ============================================================
# TOOL 2: Get deals for an account
# ============================================================
@tool("get_deals_for_account")
def get_deals_for_account(account_id: int) -> str:
    """
    Get all deals (sales opportunities) for a given account. Use this after
    find_account_by_name to learn the company's deal history and pipeline.

    Args:
        account_id: The integer ID of the account, from find_account_by_name.

    Returns:
        A JSON list of deals: id, name, stage, value (USD), probability (0-100),
        created_at, closed_at (null if open). Sorted by most recent first.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT id, name, stage, value, probability, created_at, closed_at
               FROM deals
               WHERE account_id = ?
               ORDER BY created_at DESC""",
            (account_id,),
        ).fetchall()
        conn.close()
        return _rows_to_json(rows)
    except Exception as e:
        return f"ERROR querying deals: {e}"


# ============================================================
# TOOL 3: Get contacts for an account
# ============================================================
@tool("get_contacts_for_account")
def get_contacts_for_account(account_id: int) -> str:
    """
    Get all contacts (people) at a given account. Use this to know who the
    key stakeholders are at the company.

    Args:
        account_id: The integer ID of the account.

    Returns:
        A JSON list of contacts: id, name, email, title, phone.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT id, name, email, title, phone
               FROM contacts
               WHERE account_id = ?
               ORDER BY name""",
            (account_id,),
        ).fetchall()
        conn.close()
        return _rows_to_json(rows)
    except Exception as e:
        return f"ERROR querying contacts: {e}"


# ============================================================
# TOOL 4: Get recent activities for an account (via deals)
# ============================================================
@tool("get_recent_activities_for_account")
def get_recent_activities_for_account(account_id: int, limit: int = 20) -> str:
    """
    Get recent activities (calls, emails, meetings, notes) across all deals
    for a given account. This is what reveals the actual story of the
    relationship — what was said, what was promised, what's blocking.

    Args:
        account_id: The integer ID of the account.
        limit: Max number of activities to return (default 20).

    Returns:
        A JSON list of activities: type, notes, created_at, plus the parent
        deal_name and deal_stage for context. Sorted most recent first.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT act.type, act.notes, act.created_at,
                      d.name AS deal_name, d.stage AS deal_stage
               FROM activities act
               INNER JOIN deals d ON act.deal_id = d.id
               WHERE d.account_id = ?
               ORDER BY act.created_at DESC
               LIMIT ?""",
            (account_id, limit),
        ).fetchall()
        conn.close()
        return _rows_to_json(rows)
    except Exception as e:
        return f"ERROR querying activities: {e}"