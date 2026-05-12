"""
seed_crm.py — Create the SalesAgent CRM database and fill it with realistic fake data.

Run once at the start of the project, or any time you want a fresh database:
    python db/seed_crm.py

Output: db/salesagent.db (a SQLite file) with:
    - 50 accounts across 8 industries
    - 1-4 contacts per account
    - 200 deals distributed in funnel proportions
    - 2-5 activities per deal with stage-appropriate notes
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

# ============================================================
# CONFIG
# ============================================================
# Setting a fixed seed makes the data deterministic: every run produces
# the SAME fake data. Critical for reproducible demos and tests.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

# Resolve paths relative to this file's location, NOT the working directory.
# This makes the script work whether you run it from project root or db/.
HERE = Path(__file__).parent
DB_PATH = HERE / "salesagent.db"
SCHEMA_PATH = HERE / "schema.sql"

# ============================================================
# DOMAIN CONSTANTS
# ============================================================
INDUSTRIES = [
    "SaaS", "Healthcare", "Finance", "Manufacturing",
    "Retail", "Education", "Real Estate", "Logistics",
]

# Funnel distribution: more deals early in the pipeline, fewer at the end.
# These weights mirror how real sales pipelines look.
# (stage, weight, probability_of_closing)
STAGES = [
    ("prospecting",   30, 10),
    ("qualification", 25, 25),
    ("proposal",      15, 50),
    ("negotiation",   10, 75),
    ("closed_won",    10, 100),
    ("closed_lost",   10, 0),
]

JOB_TITLES = [
    "CEO", "CTO", "CFO", "VP Engineering", "VP Sales",
    "Director of IT", "Head of Procurement", "Product Manager",
    "Operations Manager", "Chief Data Officer",
]

# Stage-appropriate note templates. The agent's RAG retrieval will be
# meaningful only if these notes carry signal about what's happening.
NOTE_TEMPLATES = {
    "prospecting": [
        "Cold outreach via LinkedIn. Brief intro to our platform.",
        "Initial discovery call scheduled. Researching their tech stack.",
        "Sent intro deck. Awaiting response.",
        "Connected with {title} on LinkedIn, exploring fit.",
    ],
    "qualification": [
        "Discovery call completed. Pain point: manual reporting taking 20+ hrs/week.",
        "Confirmed budget approved for Q2. Decision maker is {title}.",
        "Identified 3 stakeholders. Need to map their priorities.",
        "Technical requirements gathered. Integration with Salesforce is critical.",
    ],
    "proposal": [
        "Sent custom proposal. Pricing: ${value:,}. Awaiting feedback.",
        "Demo delivered to broader team. Strong interest from engineering.",
        "Pricing pushback expected — preparing ROI deck.",
        "Proposal under legal review. Procurement reaching out next week.",
    ],
    "negotiation": [
        "Pricing negotiation ongoing. They asked for 15% discount.",
        "Legal redlines received. Working with our team to respond.",
        "Verbal commitment from {title}. Waiting on procurement sign-off.",
        "Competing with another vendor. Differentiating on integration depth.",
        "Stuck on payment terms. They want net-60, we want net-30.",
    ],
    "closed_won": [
        "Contract signed! Kickoff scheduled for next week.",
        "Deal closed. Champion: {title}. Expansion potential in 6 months.",
        "Won on integration capabilities. Implementation starts Monday.",
    ],
    "closed_lost": [
        "Lost to competitor. They went with a cheaper alternative.",
        "Budget got cut in Q4. Will revisit next fiscal year.",
        "Champion left the company. Decision postponed indefinitely.",
        "Lost on technical fit. They needed on-prem deployment.",
    ],
}

ACTIVITY_TYPES = ["call", "email", "meeting", "note"]


# ============================================================
# HELPERS
# ============================================================
def random_past_date(days_back_min: int = 1, days_back_max: int = 365) -> datetime:
    """Return a random datetime between days_back_min and days_back_max ago."""
    days_ago = random.randint(days_back_min, days_back_max)
    return datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))


def pick_stage() -> tuple[str, int]:
    """Pick a stage weighted by funnel proportions. Returns (stage_name, probability)."""
    stage_names = [s[0] for s in STAGES]
    weights = [s[1] for s in STAGES]
    probabilities = {s[0]: s[2] for s in STAGES}
    chosen = random.choices(stage_names, weights=weights, k=1)[0]
    return chosen, probabilities[chosen]


# ============================================================
# SEEDING FUNCTIONS
# ============================================================
def create_schema(conn: sqlite3.Connection) -> None:
    """Run schema.sql to create (or recreate) all tables."""
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    print("✅ Schema created")


def seed_accounts(conn: sqlite3.Connection, n: int = 50) -> list[int]:
    """Insert n accounts. Return the list of account IDs."""
    cur = conn.cursor()
    account_ids = []
    for _ in range(n):
        cur.execute(
            """INSERT INTO accounts (name, industry, employees, annual_revenue, website, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                fake.company(),
                random.choice(INDUSTRIES),
                random.choice([50, 100, 500, 1000, 5000, 10000]),
                random.choice([1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000]),
                fake.url(),
                random_past_date(180, 730),  # accounts existed for 6mo-2yr
            ),
        )
        account_ids.append(cur.lastrowid)
    conn.commit()
    print(f"✅ Inserted {len(account_ids)} accounts")
    return account_ids


def seed_contacts(conn: sqlite3.Connection, account_ids: list[int]) -> int:
    """Insert 1-4 contacts per account. Return total contacts inserted."""
    cur = conn.cursor()
    total = 0
    for account_id in account_ids:
        for _ in range(random.randint(1, 4)):
            name = fake.name()
            # Build email from name + fake domain — looks realistic
            email = f"{name.lower().replace(' ', '.')}@{fake.domain_name()}"
            cur.execute(
                """INSERT INTO contacts (account_id, name, email, title, phone)
                   VALUES (?, ?, ?, ?, ?)""",
                (account_id, name, email, random.choice(JOB_TITLES), fake.phone_number()),
            )
            total += 1
    conn.commit()
    print(f"✅ Inserted {total} contacts")
    return total


def seed_deals(conn: sqlite3.Connection, account_ids: list[int], n: int = 200) -> list[tuple[int, str]]:
    """Insert n deals across funnel stages. Return list of (deal_id, stage) for activity seeding."""
    cur = conn.cursor()
    deals = []
    for _ in range(n):
        account_id = random.choice(account_ids)
        stage, probability = pick_stage()

        # Deal value: log-distributed, $5K to $500K
        value = random.choice([5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000])

        created_at = random_past_date(7, 365)
        # closed_at is set only if the deal is in a terminal stage
        if stage in ("closed_won", "closed_lost"):
            closed_at = created_at + timedelta(days=random.randint(7, 180))
        else:
            closed_at = None

        # Realistic deal name: "{Company} - {Product}"
        deal_name = f"{fake.company_suffix()} {fake.bs().title()}"

        cur.execute(
            """INSERT INTO deals (account_id, name, stage, value, probability, created_at, closed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (account_id, deal_name, stage, value, probability, created_at, closed_at),
        )
        deals.append((cur.lastrowid, stage))
    conn.commit()
    print(f"✅ Inserted {len(deals)} deals")
    return deals


def seed_activities(conn: sqlite3.Connection, deals: list[tuple[int, str]]) -> int:
    """For each deal, insert 2-5 activities with stage-appropriate notes."""
    cur = conn.cursor()
    total = 0
    for deal_id, stage in deals:
        n_activities = random.randint(2, 5)
        templates = NOTE_TEMPLATES[stage]
        for _ in range(n_activities):
            note = random.choice(templates).format(
                title=random.choice(JOB_TITLES),
                value=random.choice([10_000, 50_000, 100_000]),
            )
            cur.execute(
                """INSERT INTO activities (deal_id, type, notes, created_at)
                   VALUES (?, ?, ?, ?)""",
                (deal_id, random.choice(ACTIVITY_TYPES), note, random_past_date(1, 365)),
            )
            total += 1
    conn.commit()
    print(f"✅ Inserted {total} activities")
    return total


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    # Delete the old DB file if it exists — fresh start every run
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"🗑️  Removed old database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    # ⚠️ THE SQLITE FOOTGUN: FKs are off by default. Turn them on.
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        create_schema(conn)
        account_ids = seed_accounts(conn, n=50)
        seed_contacts(conn, account_ids)
        deals = seed_deals(conn, account_ids, n=200)
        seed_activities(conn, deals)

        # Verification: print row counts so we can confirm everything landed
        cur = conn.cursor()
        print("\n📊 Final row counts:")
        for table in ("accounts", "contacts", "deals", "activities"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"   {table:12} {cur.fetchone()[0]:>6}")

        # Spot-check: distribution of deals across stages
        print("\n📈 Deal stage distribution:")
        cur.execute("SELECT stage, COUNT(*) FROM deals GROUP BY stage ORDER BY COUNT(*) DESC")
        for stage, count in cur.fetchall():
            print(f"   {stage:15} {count:>4}")

        print(f"\n✅ Database created at {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()