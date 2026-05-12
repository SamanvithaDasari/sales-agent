-- ============================================================
-- SalesAgent CRM Schema
-- ============================================================
-- Four tables modeling a sales pipeline:
--   accounts (companies) ─┬─→ deals ─→ activities
--                         └─→ contacts
--
-- Conventions:
--   - Every table has an INTEGER PRIMARY KEY (SQLite auto-increments)
--   - Foreign keys enforce referential integrity (requires PRAGMA)
--   - created_at is mandatory; closed_at is nullable (open deals)
-- ============================================================

-- Drop in reverse dependency order so re-running the script is safe
DROP TABLE IF EXISTS activities;
DROP TABLE IF EXISTS deals;
DROP TABLE IF EXISTS contacts;
DROP TABLE IF EXISTS accounts;

-- ------------------------------------------------------------
-- accounts: companies we're selling to
-- ------------------------------------------------------------
CREATE TABLE accounts (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    industry        TEXT,
    employees       INTEGER,
    annual_revenue  INTEGER,
    website         TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- contacts: people at those companies
-- ------------------------------------------------------------
CREATE TABLE contacts (
    id          INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL,
    name        TEXT NOT NULL,
    email       TEXT,
    title       TEXT,
    phone       TEXT,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE RESTRICT
);

-- ------------------------------------------------------------
-- deals: sales opportunities
-- ------------------------------------------------------------
CREATE TABLE deals (
    id            INTEGER PRIMARY KEY,
    account_id    INTEGER NOT NULL,
    name          TEXT NOT NULL,
    stage         TEXT NOT NULL CHECK (stage IN (
                    'prospecting', 'qualification', 'proposal',
                    'negotiation', 'closed_won', 'closed_lost'
                  )),
    value         INTEGER NOT NULL,
    probability   INTEGER CHECK (probability BETWEEN 0 AND 100),
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at     TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE RESTRICT
);

-- ------------------------------------------------------------
-- activities: touchpoints on a deal (calls, emails, meetings, notes)
-- ------------------------------------------------------------
CREATE TABLE activities (
    id          INTEGER PRIMARY KEY,
    deal_id     INTEGER NOT NULL,
    type        TEXT NOT NULL CHECK (type IN ('call', 'email', 'meeting', 'note')),
    notes       TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (deal_id) REFERENCES deals(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- Indexes for the most common query patterns
-- ------------------------------------------------------------
-- "Show me all deals for account X" — runs on every page load of an account
CREATE INDEX idx_deals_account_id   ON deals(account_id);

-- "Show me all activities on deal Y" — runs whenever the agent loads a deal
CREATE INDEX idx_activities_deal_id ON activities(deal_id);

-- "Show me all contacts at account X"
CREATE INDEX idx_contacts_account_id ON contacts(account_id);

-- "Filter deals by stage" — used by the Deal Coach for similar-deal lookups
CREATE INDEX idx_deals_stage        ON deals(stage);