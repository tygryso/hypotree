"""SQL DDL for the SQLite-WAL source-of-truth store.

Nine tables: schema_meta, nodes, edges, evidence, status_history,
posterior_history, claims, events, nogoods. The nodes table is a denormalized
current cache; authoritative history lives in the *_history tables. The events
table is an audit/replay log written in the same transaction as state mutations.

``MIGRATIONS`` at the foot of this module carries the forward upgrade path; see
its comment for the rules a new entry has to satisfy.
"""

SCHEMA_VERSION = "10"

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
    -- What was actually run to produce this number: a path, a URL, a CI run id.
    -- An audit trail that says "0.85" and one that says "0.85, from run #4412"
    -- are different artifacts, and only the caller knows which one exists.
    source_ref    TEXT,
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
    -- Which members the substitution diagnosis has actually cleared: a JSON
    -- array of node ids, each swapped out of the composition with the failure
    -- persisting anyway. Persisted because the diagnosis spans many turns and
    -- must survive a context reset; restarting it would re-run experiments whose
    -- answers are already in.
    --
    -- A *set* rather than the integer cursor it replaces. A cursor can only say
    -- "the first k were dealt with", which conflates cleared with skipped: a
    -- member the plan had to pass over because no substitute was available was
    -- left behind the cursor and reported as cleared when it had never been
    -- tested, and could never be revisited once a substitute freed up. A set
    -- records exactly what was established and nothing more.
    cleared_ids         TEXT,
    -- Deprecated in 0.4.0, removed in 0.5.0: kept in step with `cleared_ids` so
    -- a reader written against the old shape still sees a truthful count.
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


# Forward migrations, keyed by the version they upgrade *from*. Each entry is
# (target_version, statements) and is applied together with the version stamp in
# a single transaction, so a crash mid-upgrade leaves the database on the old
# version rather than half-way between two.
#
# These exist because hypotree is published. "Delete the DB to reset" is a fair
# answer while nothing is deployed and an unacceptable one once someone's belief
# state is the accumulated record of a month of experiments — the whole product
# claim is that the state survives. A release that silently required starting
# over would refute it.
#
# Rules for adding one:
#   * Forward only. A database written by a newer hypotree is not downgraded;
#     it is refused, because the newer code may have stored things this version
#     cannot represent and dropping them silently is worse than stopping.
#   * Additive where possible. `ALTER TABLE ... ADD COLUMN` with a nullable
#     column is instant and cannot lose data. A migration that rewrites rows
#     needs a far stronger justification than a new field.
#   * Chained, not jumped. 7→9 runs 7→8 then 8→9, so every step is exercised by
#     every longer path instead of accumulating untested direct routes.
#   * Back-fill lazily where the old column is still readable. 9→10 adds
#     `nogoods.cleared_ids` and leaves it NULL; the store derives the set from
#     `probe_index` on read, so an in-flight diagnosis keeps its progress without
#     the migration having to interpret JSON in SQL.
MIGRATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "8": ("9", ("ALTER TABLE evidence ADD COLUMN source_ref TEXT",)),
    "9": ("10", ("ALTER TABLE nogoods ADD COLUMN cleared_ids TEXT",)),
}
