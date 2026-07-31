"""SQL DDL for the SQLite-WAL source-of-truth store.

Eight tables: schema_meta, nodes, edges, evidence, status_history,
posterior_history, claims, events. The nodes table is a denormalized current
cache; authoritative history lives in the *_history tables. The events table
is an audit/replay log written in the same transaction as state mutations.
"""

SCHEMA_VERSION = "8"

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id                  TEXT PRIMARY KEY,
    statement           TEXT NOT NULL,
    status              TEXT NOT NULL,
    evidence_regime     TEXT NOT NULL DEFAULT 'deterministic',
    is_parametric       INTEGER NOT NULL DEFAULT 0,
    param_config        TEXT,
    is_goal             INTEGER NOT NULL DEFAULT 0,
    target_metric       REAL,
    exclusion_group     TEXT,
    -- Depth (rigour / scale / context) of the observation that confirmed this
    -- node. A confirmation obtained at depth d does not license a claim at any
    -- depth greater than d: "it worked in the unit test" is not the same claim
    -- as "it works in production". NULL means never confirmed.
    confirmed_depth     INTEGER,
    alpha               REAL NOT NULL DEFAULT 1.0,
    beta                REAL NOT NULL DEFAULT 1.0,
    evidence_count      INTEGER NOT NULL DEFAULT 0,
    active_claim_id     TEXT,
    claimed_at          TEXT,
    infra_retry_count   INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    first_dispatched_at TEXT,
    first_evidence_at   TEXT,
    verified_at         TEXT,
    invalidated_at      TEXT,
    pruned_at           TEXT,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    src  TEXT NOT NULL,
    dst  TEXT NOT NULL,
    type TEXT NOT NULL,
    PRIMARY KEY (src, dst, type)
);

CREATE TABLE IF NOT EXISTS evidence (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    success       REAL,
    -- The rigour/scale at which this observation was made. Higher means a more
    -- demanding test. Carried so a later composition can tell whether the
    -- confirmations it rests on were deep enough to support it.
    depth         INTEGER NOT NULL DEFAULT 0,
    metrics       TEXT NOT NULL DEFAULT '{}',
    artifacts     TEXT NOT NULL DEFAULT '[]',
    context_hash  TEXT,
    git_branch    TEXT,
    claim_id      TEXT,
    notes         TEXT NOT NULL DEFAULT '',
    delta_success REAL,
    delta_metrics TEXT NOT NULL DEFAULT '{}',
    monotonicity  TEXT NOT NULL DEFAULT 'first',
    recorded_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_node ON evidence(node_id);

CREATE TABLE IF NOT EXISTS status_history (
    node_id    TEXT NOT NULL,
    status     TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to   TEXT,
    reason     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (node_id, valid_from)
);

CREATE TABLE IF NOT EXISTS posterior_history (
    node_id    TEXT NOT NULL,
    alpha      REAL NOT NULL,
    beta       REAL NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to   TEXT,
    reason     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (node_id, valid_from)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id    TEXT PRIMARY KEY,
    node_id     TEXT NOT NULL,
    claimed_at  TEXT NOT NULL,
    lease_ttl_s INTEGER NOT NULL,
    consumed_at TEXT,
    expired     INTEGER NOT NULL DEFAULT 0
);
-- Claims are looked up by node whenever a new lease supersedes an old one.
CREATE INDEX IF NOT EXISTS idx_claims_node ON claims(node_id);
-- The live-claim scan (unconsumed and unexpired) runs on every reclaim pass.
CREATE INDEX IF NOT EXISTS idx_claims_live ON claims(consumed_at, expired);

CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    txn_id     TEXT NOT NULL,
    written_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_txn ON events(txn_id);

-- nogoods: recorded conflict sets.
--
-- When a hypothesis that rests on several assumptions fails, what has actually
-- been learned is "these assumptions cannot ALL hold together" — not that any
-- individual one is wrong. Blaming every parent destroys correct knowledge and
-- forces the agent to rediscover it. Storing the conflict set instead keeps the
-- assumptions intact and lets blame be assigned later, once other evidence
-- exonerates all but one member. (Classic truth-maintenance nogood.)
CREATE TABLE IF NOT EXISTS nogoods (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id      TEXT NOT NULL,   -- the failure that produced the conflict
    member_ids          TEXT NOT NULL,   -- JSON array of assumption node ids
    -- Depth at which the composition failed. Members confirmed at or above this
    -- depth are exonerated: their evidence already covers the context in which
    -- the failure occurred, so they cannot be what the failure revealed.
    conflict_depth      INTEGER NOT NULL DEFAULT 0,
    -- How far the substitution diagnosis has progressed: members are stored
    -- sorted, and everything before this index has been cleared by swapping it
    -- out and watching the composition fail anyway. Persisted because the
    -- diagnosis spans many turns and must survive a context reset; restarting it
    -- would re-run experiments whose answers are already in.
    probe_index         INTEGER NOT NULL DEFAULT 0,
    resolved_culprit_id TEXT,            -- set once narrowing identifies the culprit
    -- Set when the conflict has been shown to be a genuine interaction effect —
    -- every member individually survived a test as demanding as the failure, so
    -- no one of them is at fault and the alternatives they retired have been
    -- reopened. Recorded so that recovery happens once rather than every time
    -- evidence arrives while the conflict is still open.
    reopened_at         TEXT,
    recorded_at         TEXT NOT NULL,
    resolved_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_nogoods_open ON nogoods(resolved_at);
"""
