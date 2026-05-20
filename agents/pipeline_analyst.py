"""
pipeline_analyst.py — NL-to-SQL agent ("Pipeline Analyst").

Translates natural-language questions about the CRM into validated SQL,
runs it read-only against the database, and returns a markdown table plus
a one-paragraph narration.

ARCHITECTURE (interview hooks):
- Two LLM calls per question:
    1. Gemini Flash-Lite: generates structured {sql, explanation} JSON
    2. Groq Llama 3.1 8B: narrates the results in plain English
  Multi-LLM per workload — Gemini for structured generation, Groq for prose.

- Defense in depth against unsafe SQL (the famous SQL injection problem):
    Layer 1: System prompt explicitly says SELECT-only against 4 named tables
    Layer 2: Regex validator rejects forbidden keywords + multi-statements
    Layer 3: Read-only SQLite connection (`mode=ro` URI) — engine refuses writes
    Layer 4: Result-set cap (100 rows) and execution timeout (5s)

- Compact LLM-readable schema description (`SCHEMA_SUMMARY`) with PK/FK,
  enum values, and semantic hints — similar to a "semantic layer" used by
  production NL-to-SQL systems like Snowflake Cortex Analyst.

- Four few-shot examples in the system prompt covering the major query
  shapes: filter, sort+limit, aggregation, join.
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
GEMINI_MODEL = "gemini-2.5-flash-lite"
GROQ_MODEL = "llama-3.1-8b-instant"

# Same env-var trick as the other agents so it works on HF Spaces' /tmp dir.
DB_PATH = (
    Path(os.environ["SALESAGENT_DATA_DIR"]) / "salesagent.db"
    if "SALESAGENT_DATA_DIR" in os.environ
    else Path(__file__).parent.parent / "db" / "salesagent.db"
)

MAX_RESULT_ROWS = 100      # cap result set to prevent giant payloads
SQL_TIMEOUT_SECONDS = 5    # cap execution time

gemini_client = genai.Client()
groq_client = Groq()


# ============================================================
# SAFETY: SQL VALIDATION
# ============================================================
ALLOWED_TABLES = {"accounts", "contacts", "deals", "activities"}

# Keywords that must never appear in a query (as whole words).
# Includes sqlite system tables — querying them leaks schema info.
FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "create", "alter", "truncate",
    "replace", "pragma", "attach", "detach", "reindex", "vacuum",
    "sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_sequence",
}


def _strip_comments(sql: str) -> str:
    """Strip SQL comments so they can't smuggle in forbidden keywords."""
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate generated SQL against safety rules.

    Returns (is_valid, error_message). On failure, error_message describes
    which rule was violated — surfaced to the user, so they can re-phrase.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL"

    s = _strip_comments(sql).strip().lower()

    # Must be a SELECT (or a CTE that starts with WITH)
    if not (s.startswith("select") or s.startswith("with")):
        return False, "Only SELECT queries are allowed"

    # No multiple statements: a single trailing semicolon is OK, more is not.
    stripped = s.rstrip(";").strip()
    if ";" in stripped:
        return False, "Multiple statements not allowed"

    # Forbidden keywords as whole words
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(r"\b" + kw + r"\b", s):
            return False, f"Forbidden keyword: {kw}"

    return True, "OK"


# ============================================================
# SAFETY: READ-ONLY EXECUTION
# ============================================================
def _execute_safe_sql(sql: str) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Execute pre-validated SQL against a read-only connection.

    Returns (rows_as_dicts, column_names). On any error, raises an exception
    that the caller wraps in a user-facing message.
    """
    # URI-form connection string with mode=ro — engine refuses any write.
    # uri=True is mandatory; without it Python treats the whole string as a literal filename.
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        timeout=SQL_TIMEOUT_SECONDS,
    )
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_RESULT_ROWS)
        columns = [d[0] for d in cur.description] if cur.description else []
        return [dict(r) for r in rows], columns
    finally:
        conn.close()


# ============================================================
# SCHEMA SUMMARY (compact, LLM-readable)
# ============================================================
SCHEMA_SUMMARY = """\
Tables:

accounts(id PK, name, industry, employees, annual_revenue, website, created_at)

contacts(id PK, account_id FK→accounts.id, name, email, title, phone)

deals(id PK, account_id FK→accounts.id, name, stage, value, probability, created_at, closed_at)
  - stage ∈ {prospecting, qualification, proposal, negotiation, closed_won, closed_lost}
  - closed_at IS NULL means the deal is still OPEN
  - probability is 0-100
  - value is in USD

activities(id PK, deal_id FK→deals.id, type, notes, created_at)
  - type ∈ {call, email, meeting, note}

Join paths:
  accounts.id = contacts.account_id
  accounts.id = deals.account_id
  deals.id = activities.deal_id
"""

# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================
FEW_SHOTS = [
    {
        "question": "Show me deals worth over $50,000",
        "sql": "SELECT id, name, stage, value FROM deals WHERE value > 50000 ORDER BY value DESC LIMIT 50",
        "explanation": "Filter deals by value over the threshold, sorted highest first."
    },
    {
        "question": "Top 5 open deals in negotiation",
        "sql": "SELECT d.name AS deal, a.name AS account, d.value, d.probability FROM deals d JOIN accounts a ON d.account_id = a.id WHERE d.stage = 'negotiation' AND d.closed_at IS NULL ORDER BY d.value DESC LIMIT 5",
        "explanation": "Join deals to accounts, filter to negotiation stage with closed_at IS NULL (open), top 5 by value."
    },
    {
        "question": "How many deals do we have at each stage?",
        "sql": "SELECT stage, COUNT(*) AS deal_count FROM deals GROUP BY stage ORDER BY deal_count DESC",
        "explanation": "Aggregate count of deals grouped by stage."
    },
    {
        "question": "Which industries have the most accounts?",
        "sql": "SELECT industry, COUNT(*) AS account_count FROM accounts GROUP BY industry ORDER BY account_count DESC LIMIT 10",
        "explanation": "Group accounts by industry, count, order by count descending."
    },
]


# ============================================================
# PROMPTS
# ============================================================
SYSTEM_PROMPT_SQL = f"""You are a SQL analyst for a B2B sales CRM. You translate \
natural-language questions into safe SELECT queries.

{SCHEMA_SUMMARY}

RULES:
- Output ONLY SELECT statements (or WITH ... SELECT for CTEs).
- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or ATTACH.
- NEVER query sqlite_master or any sqlite_* system table.
- Always include a LIMIT (default 50 if unspecified, max 100).
- When a question mentions "open" deals, filter `closed_at IS NULL`.
- When a question mentions "won" / "lost", filter `stage = 'closed_won'` / `'closed_lost'`.
- Use JOINs when the question crosses tables.
- Use COUNT / SUM / AVG when the question asks for totals or aggregates.

OUTPUT FORMAT:
You MUST respond with a JSON object containing exactly two keys:
  - "sql": the SQL query as a single string, no trailing semicolon
  - "explanation": one sentence describing what the query does

Examples:

""" + "\n\n".join(
    f"Question: {ex['question']}\n"
    f"Response: {json.dumps({'sql': ex['sql'], 'explanation': ex['explanation']})}"
    for ex in FEW_SHOTS
) + """

If the question is not answerable from the schema (e.g. asks about something \
not in the CRM), respond with:
  {"sql": "", "explanation": "I can't answer that from the available CRM tables."}
"""


SYSTEM_PROMPT_NARRATE = """You are a sales operations analyst. You receive a \
natural-language question, the SQL that was run, and the result rows. You write \
a single paragraph (2-4 sentences) summarizing the answer in plain English for \
a sales rep — no jargon, no SQL-talk, just the insight.

If the result is empty, say so honestly and suggest what the user might try \
instead. Never invent numbers."""


# ============================================================
# THE AGENT
# ============================================================
@dataclass
class PipelineResult:
    """What the agent returns to callers."""
    question: str
    sql: str
    explanation: str           # the LLM's one-sentence description of the SQL
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    narration: str = ""        # the plain-English summary
    error: str | None = None   # populated if anything failed


def _generate_sql(question: str) -> tuple[str, str]:
    """LLM call #1: ask Gemini for SQL + explanation. Returns (sql, explanation)."""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_SQL,
            temperature=0.0,           # SQL generation should be deterministic
            max_output_tokens=600,
            response_mime_type="application/json",  # force JSON output
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # If the model wrapped its JSON in code fences, strip and retry
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        parsed = json.loads(cleaned)
    return parsed.get("sql", "").strip(), parsed.get("explanation", "").strip()


def _narrate_results(question: str, sql: str, columns: list[str], rows: list[dict]) -> str:
    """LLM call #2: ask Groq to summarize the results in plain English."""
    # Truncate rows for the prompt so we don't waste tokens on giant result sets
    rows_for_prompt = rows[:20]
    user_msg = (
        f"User question: {question}\n\n"
        f"SQL executed: {sql}\n\n"
        f"Columns: {columns}\n"
        f"Result rows (showing up to 20):\n{json.dumps(rows_for_prompt, default=str, indent=2)}\n\n"
        f"Total rows returned: {len(rows)}"
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_NARRATE},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def run_pipeline_analyst(question: str) -> PipelineResult:
    """
    Full NL-to-SQL pipeline:
      1. Gemini generates {sql, explanation}
      2. Validator + read-only executor runs it
      3. Groq narrates the results
    Returns a PipelineResult — never raises.
    """
    result = PipelineResult(question=question, sql="", explanation="")

    # Step 1: SQL generation
    try:
        sql, explanation = _generate_sql(question)
    except Exception as e:
        result.error = f"SQL generation failed: {e}"
        return result
    result.sql = sql
    result.explanation = explanation

    # Edge case: model declined to answer (empty SQL is the agreed signal)
    if not sql:
        result.narration = explanation or "I can't answer that from the available CRM data."
        return result

    # Step 2: validate
    ok, reason = validate_sql(sql)
    if not ok:
        result.error = f"Generated SQL rejected by safety validator: {reason}"
        return result

    # Step 3: execute (read-only)
    try:
        rows, columns = _execute_safe_sql(sql)
    except sqlite3.OperationalError as e:
        result.error = f"Database error: {e}"
        return result
    except Exception as e:
        result.error = f"Execution failed: {e}"
        return result
    result.rows = rows
    result.columns = columns

    # Step 4: narrate
    try:
        result.narration = _narrate_results(question, sql, columns, rows)
    except Exception as e:
        # Narration failure isn't fatal — we still have the data
        result.narration = f"({len(rows)} rows returned. Narration unavailable: {e})"

    return result


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Top 5 open deals in negotiation"
    print(f"\n🔍 Pipeline Analyst: {q!r}\n")
    print("=" * 70)
    r = run_pipeline_analyst(q)

    if r.error:
        print(f"❌ {r.error}")
    else:
        print(f"📝 Narration:\n{r.narration}\n")
        print(f"🔧 SQL: {r.sql}")
        print(f"📊 Rows: {len(r.rows)} returned (showing up to 5):")
        for row in r.rows[:5]:
            print(f"   {row}")