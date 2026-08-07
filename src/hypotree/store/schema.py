"""Schema definition and the ordered migration chain.

The chain is the only description of the schema. `BASE_DDL` is the *original*
shape — not the current one — and every change since is a numbered step applied
in order. A fresh database is therefore built the same way an old one is
upgraded: baseline, then every step. There is no second path, so there is no
second thing to keep in sync.

That ordering is what makes it safe. The previous arrangement had the DDL
describing the *current* shape and running before the migrations, so a table the
file happened to be missing was created already-modern and the migration that
added its column then failed on `duplicate column name`. Every future migration
would have met that.

Two rules follow, and both are load-bearing:

* **The baseline is `CREATE TABLE IF NOT EXISTS` only.** It runs on every open,
  including databases that are already current, so it has to be a no-op for
  anything that exists and a repair for anything that does not.
* **The baseline is frozen.** A new column goes in a new step, never here.
  Editing the baseline rewrites history for every database that already passed
  through it.
"""

from __future__ import annotations

# The schema as it stood at version 8, the oldest release this code upgrades
# from. Frozen: additions go in MIGRATIONS.
BASE_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- schema_state: exactly one row, and the first thing to look at when a database
-- behaves unexpectedly. Which schema version it is on, which release of hypotree
-- last opened it, and when. `CHECK (id = 1)` on the primary key makes "exactly
-- one row" a property of the table rather than a convention someone remembers.
CREATE TABLE IF NOT EXISTS schema_state (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version TEXT NOT NULL,
    app_version    TEXT NOT NULL,
    migrated_at    TEXT NOT NULL
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
    probe_index         INTEGER NOT NULL DEFAULT 0,
    resolved_culprit_id TEXT,            -- set once narrowing identifies the culprit
    -- Set when the conflict has been shown to be a genuine interaction effect —
    -- every member individually survived a test as demanding as the failure, so
    -- no one of them is at fault and the alternatives they retired have been
    -- reopened. Recorded so recovery happens once rather than every time
    -- evidence arrives while the conflict is still open.
    reopened_at         TEXT,
    recorded_at         TEXT NOT NULL,
    resolved_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_nogoods_open ON nogoods(resolved_at);
"""

# The version BASE_DDL produces. Anything older predates the published chain and
# is refused rather than guessed at.
BASELINE_VERSION = "8"

# Ordered steps applied after the baseline, oldest first: (version_it_produces,
# statements). Each runs in one transaction with its own stamp, so a crash leaves
# the database on the previous version rather than between two.
#
# Rules for adding one:
#   * Append, never edit. A shipped step has already run somewhere, and changing
#     it means two databases claiming the same version with different shapes. A
#     step may be edited while its version is *unreleased* — that is the only
#     exception, and it expires the moment the version ships.
#   * Forward only. A database written by a newer hypotree is refused, not
#     downgraded: the newer code may have stored things this version cannot
#     represent, and dropping them silently is worse than stopping.
#   * Additive where possible. `ALTER TABLE ... ADD COLUMN` with a nullable
#     column is instant and cannot lose data.
#   * Back-fill lazily where the old column is still readable. Step 10 adds
#     `nogoods.cleared_ids` and leaves it NULL; the store derives the set from
#     `probe_index` on read, so an in-flight diagnosis keeps its progress without
#     the migration having to interpret JSON in SQL.
MIGRATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "9",
        (
            # What was actually run to produce a number: a path, a URL, a CI run
            # id. "0.85" and "0.85, from run #4412" are different artifacts.
            "ALTER TABLE evidence ADD COLUMN source_ref TEXT",
        ),
    ),
    (
        "10",
        (
            # Which members the substitution diagnosis has cleared, as JSON. A
            # set rather than the integer cursor it replaces: a cursor can only
            # say "the first k were dealt with", conflating cleared with skipped.
            "ALTER TABLE nogoods ADD COLUMN cleared_ids TEXT",
            # When an edge appeared. Without it a timeline replay draws the final
            # topology at every tick.
            "ALTER TABLE edges ADD COLUMN created_at TEXT",
            # Human scheduling instructions, deliberately not beliefs. Writing a
            # click into alpha/beta would make it forever indistinguishable from
            # an experiment and inject unlogged nondeterminism into a seeded
            # sampler. A directive changes what is offered, never what is
            # believed, and it is revocable and attributed.
            """CREATE TABLE IF NOT EXISTS node_directives (
                node_id TEXT PRIMARY KEY,
                mode    TEXT NOT NULL,
                reason  TEXT NOT NULL DEFAULT '',
                actor   TEXT NOT NULL DEFAULT 'human',
                set_at  TEXT NOT NULL
            )""",
            # Whether an exclusion group's candidates are believed to be all of
            # them. The engine had been assuming it: deduction by elimination
            # confirms the last survivor for free, which is sound over a complete
            # list and asserts something false over a partial one. NOT NULL with
            # a default of 1, because that is exactly what every group written
            # before this column meant.
            "ALTER TABLE nodes ADD COLUMN exclusion_closed INTEGER NOT NULL DEFAULT 1",
        ),
    ),
    (
        "11",
        (
            # What the experiment cost, in seconds. Nothing in the engine knew
            # that probes cost different amounts, so Thompson Sampling ranked a
            # three-GPU-day question exactly as it ranked a one-second one — and
            # every metric in every gate is counted in probes, which is only
            # defensible because the eval oracle answers in milliseconds. NULL
            # means unknown, never zero: a workspace that never reports duration
            # must behave exactly as it did before this column existed.
            "ALTER TABLE evidence ADD COLUMN duration_s REAL",
        ),
    ),
    (
        "12",
        (
            # A caller's cost estimate, used only until the node has been timed.
            # The observed model cannot rank the competing answers to one
            # question, because a question is settled once and none of them has
            # any history when the choice is made — and that is the only place
            # ordering by cost saves anything, since the last survivor is deduced
            # for free. NULL keeps the pre-existing behaviour exactly.
            "ALTER TABLE nodes ADD COLUMN estimated_cost REAL",
        ),
    ),
)

# Derived from the chain, so the two cannot drift.
SCHEMA_VERSION = MIGRATIONS[-1][0] if MIGRATIONS else BASELINE_VERSION

__all__ = [
    "BASELINE_VERSION",
    "BASE_DDL",
    "MIGRATIONS",
    "SCHEMA_VERSION",
]
