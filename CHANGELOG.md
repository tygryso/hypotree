# Changelog

All notable changes to hypotree are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - 0.6.0

### Fixed
- **The evaluation harness's hand-written tool schemas are now pinned against the real
  ones.** Because the schemas were unreachable outside the MCP server, the harness wrote
  its own copies of seven of them, and they drifted: arm B's `create_hypotheses` was
  missing `exclusion_closed`, the one field that stops the engine deducing an answer over
  an incomplete list of candidates. A parity test now fails on any field the harness
  advertises that dispatch would silently drop — the `source_ref` class of bug, which has
  shipped once — and on any type disagreement. Deliberate omissions are enumerated with
  the reason each is acceptable, so a restriction nobody chose can no longer pass for one
  that somebody did. Two are worth stating outright: arm B cannot declare an open
  candidate list, and it supplies no `duration_s`, so cost-aware ordering is not under
  test in the current gate.
- **A no-op edge write moved the revision counter and lied to the audit log.** `add_edge`
  paired an `INSERT OR IGNORE` with an unconditional `EdgeAdded` event, so re-adding an
  existing edge wrote a row asserting an addition that never happened and advanced
  `events.seq`. That counter is documented as advancing "exactly when something changed
  and never when nothing did", and two readers take it at its word: the dashboard pushes
  it to every connected browser as the change signal, and keys its whole snapshot cache
  on it — so one phantom bump costs a full graph relayout everywhere. The event is now
  written only when the insert actually inserted.
- **`_handle_infra_error` rewrote every column of a node from a stale snapshot.** The last
  whole-row writer on an engine path, and a lost-update by construction: anything written
  between the read and the save was silently reverted, and an infra error arrives
  interleaved with exactly the posterior and claim writes that would be lost. Replaced
  with a targeted `increment_infra_retries`, mirroring `increment_evidence_count`.
- **`get_next_targets` could mint a lease that was already expired.** The reclaim sweep
  compares inclusively, so `lease_ttl_s=0` returned the node to the frontier on the next
  call while the caller believed it held the work. `renew_claim` has refused non-positive
  TTLs since it was written; the issuing path was guarded only by the tool schema, which
  every direct Python caller bypasses — and the engine is public API now.
- **The dashboard refused IPv6 loopback with a port.** `Host: [::1]:7331` contains three
  colons, so the port split was skipped and the whole string was tested as a hostname.
  It failed closed, so this was never a hole — an IPv6 browser simply could not open the
  dashboard. Parsing is now bracket-aware, and a bracketed foreign host is still refused.

### Changed
- **The CLI can open an explicit belief database with `--db-path` or
  `HYPOTREE_DB_PATH`.** Embedded hosts can keep state in their own isolated
  storage namespace and later launch `hypotree --no-mcp --db-path .../state.db`
  without copying it into hypotree's global workspace resolver.
- **Tool schemas and dispatch moved out of the MCP server into `hypotree.toolkit`.** They
  were always transport-neutral — a name, a dict of arguments, plain data back — but they
  lived behind a module whose first three imports are the MCP SDK. A caller who wanted the
  belief state had to acquire a JSON-RPC stack to reach it, or reimplement the routing.
  - The MCP server is now one projection of the shared specs; an embedding host is the
    other. Neither owns the contract, so neither can drift from it.
  - `import hypotree` no longer imports `mcp`, and there is a test that fails if it starts.
  - Internal moves, unreleased: `mcp_server._dispatch` → `toolkit.dispatch.dispatch`,
    `_evidence_report` → `evidence_report`, `_parse_instant` → `parse_instant`,
    `_hypothesis_item_schema` → `specs.hypothesis_item_schema`. `_tool_definitions()`,
    `dashboard_url()` and `SERVER_INSTRUCTIONS` keep their names and locations.
- **SQLite is tuned for the write path it actually has.** `synchronous=NORMAL` (the
  canonical WAL setting: a crash can lose the last transaction and cannot corrupt the
  file), plus an explicit `busy_timeout`, cache size and in-memory temp store. Every
  mutation writes its audit event in the same transaction, so the default `FULL` was
  buying an fsync per belief change that no caller had asked for.
- **A stochastic node now records whether it settled on evidence or on budget.** The
  convergence gate returned a bare boolean, so a node whose credible interval was still
  wide at the sample ceiling left exactly the trace of one measured to a tight interval —
  and "why did this settle?" was answerable only by recomputing the posterior at the time.
  New `convergence_verdict` returns the reason alongside the decision, and the ceiling
  case is appended to the status-change reason where the audit log and the agent can both
  read it. The boolean `convergence_gate` is unchanged, and so is *whether* any node
  settles; only the record is richer.
- **The git-remote lookup is resolved once per process.** Up to three subprocesses with a
  five-second timeout each sat on the MCP startup path before the first handshake, so a
  slow git or a remote behind a hung mount could look to a client like a server that
  failed to start. Re-pointing a remote mid-session deliberately does not migrate the
  belief state; `reset_identity_cache()` forces re-resolution for anything that needs it.
- **One node read per selection pass instead of several.** `_frontier_nodes` accepts the
  snapshot its caller already holds, and `_confirmed_for_composition` stopped reading the
  table twice inside one call. Deliberately *not* a cache: a snapshot is reused only where
  no write intervenes, because a node list that disagrees with SQLite produces a silently
  wrong selection rather than a slow one — the same reason a long-lived incremental graph
  was rejected.
- **Rewinding the dashboard is one query.** `_posterior_at` asked `posterior_history` per
  node, so every scrubber position paid an N+1 that the snapshot cache cannot amortise
  (the instant is part of its key). New `get_all_posterior_history`, mirroring
  `get_all_status_history`.
- **`HypoTreeGraph.is_frontier_status` is public.** It was the engine's only reach into
  another layer's privates, `# noqa: SLF001` and all.

### Removed
- **`sqlalchemy` is no longer a dependency.** It was declared and imported nowhere — tens
  of megabytes of install weight, extra resolver and attack surface, and a false signal
  that an ORM was in play in a package whose entire product is a hand-written SQLite
  schema. A test now fails if any declared dependency stops being imported.

### Added
- **A Python API: `HypoTreeToolset`, and a package that finally exports its own names.**
  hypotree was described as a library and shipped as a server. `from hypotree import
  HypoTreeEngine` raised — `__init__.py` contained a docstring and a version string and
  nothing else — so every consumer reached through `hypotree.engine` into what was
  nominally internal, including this project's own evaluation harness. The public API is
  now re-exported and covered by `__all__`.
  - `HypoTreeToolset(db_path, …)` is what a Python host holds instead of an MCP client:
    `.tools()` returns OpenAI function-calling schemas, `.call(name, args)` executes and
    returns a JSON string, and it is a context manager. `from_engine()` wraps an engine
    the caller already owns **without** taking over its lifecycle, because two objects
    that both believe they own one SQLite connection close it twice.
  - **`.call` never raises.** A misspelled node id, a missing `success`, an argument that
    is not an object — all come back as `{"error": …}`. Every one of them is recoverable
    by the model that caused it, and propagating them as exceptions turns a correctable
    mistake into a lost session.
  - **Tool selection is composable**: `preset="essential"` exposes the six tools that can
    still run the loop, `read_only=True` exposes the eleven sensors and refuses every
    write, and `include`/`exclude` narrow either. This is not a convenience — most clients
    re-send every schema on every turn, and a host already carrying forty of its own tools
    cannot also carry twenty of ours. A narrowed surface is a real boundary: a hidden tool
    is refused by name, not merely left out of the list, because models routinely call
    tools they were never given.
  - Each spec carries what a host needs and no JSON schema can express: `mutates` (gate
    these) and `essential` (ship these when context is tight). `get_next_targets` is
    classified as a mutation — it reads like a query and it issues leases, so it writes,
    and a host inferring that from the tool's name gets it wrong.

## [0.5.0] - 2026-08-08

### Added
- **`add_edges` — grow a graph forward without destroying a node to do it.** A graph grows two
  ways and only one of them was expressible. *Backward* growth discovers a premise underneath
  something already pinned to the goal and leaves the goal alone. *Forward* growth extends a
  pipeline and **must** re-pin the goal to the new last stage — a goal is met when its
  DEPENDENCY parents are VERIFIED, so one still pinned to stage A reports itself achieved the
  moment stage A verifies while B and C sit untested. The run stops early and calls it a success.
  - Until now the only way to re-pin was recreating the goal with `if_exists="overwrite"`, a
    documented **full replace**: omit `target_metric` and the goal silently loses the bar it is
    measured against. Destroying a node to add an edge to it is not a reasonable price, and the
    failure it invites is silent.
  - **Nothing has to be removed.** DEPENDENCY is AND, and a later stage already depends on the
    earlier one, so a goal wired to both is satisfied exactly when the later one is: the
    condition is tightened, never loosened. That is why there is no edge removal to get wrong.
  - Validated exactly as creation is and **all-or-nothing**: unknown endpoints, a goal used as a
    DEPENDENCY parent (`GoalDependencyError`) and cycles are refused before anything is written.
    An edge that already exists reports `created=False` rather than raising, so re-sending a plan
    is safe. Tool count 18 → **19**.
- **A belief diff over a range: `generate_learning_path(since=…)`.** A belief state's *changes*
  are more interesting than its state. "Three hypotheses were confirmed and one was withdrawn" is
  the sentence a standup, a PR description or a review wants, and reconstructing it by diffing two
  full narratives by eye is what people did instead. Returns `settled_in_window`,
  `withdrawn_in_window` and `probes_in_window` alongside the lifetime counters, which keep
  describing the whole history — "what did this cost in total" does not become a different
  question this week. Combine with `as_of` for a closed window; a window running backwards is
  refused rather than silently returning nothing, which would read as "nothing changed".
  - A **withdrawn** belief is the most interesting thing in a window and the easiest to omit, so
    reversals are counted separately and a window with nothing in it says so in as many words.
  - **On the dashboard**, the scrubber that already picks an instant now picks two: *Diff from
    here* marks a start, the window is shaded on the activity histogram, and the narrative beside
    the graph becomes the diff. Same control, used twice.
  - Exposed as `since` on the MCP tool and `GET /api/learning-path?since=`.
- Recorded `artifacts` are read back in `get_evidence_history` and the dashboard provenance
  panel. Written since the first release and consumed by nothing — an audit trail that cannot
  produce the log it refers to is not an audit trail, and it is the same defect that was fixed
  for `context_hash` a year earlier.
- `status_filter` accepts **`EXHAUSTED`**. It is the status the exclusion inference produces most
  and the one that could not be queried directly; the SDK enforces `inputSchema`, so the filter
  was rejected outright and the `view="settled"` workaround bundles three other statuses.

- **`what_would_change_my_mind` — name the experiment that would overturn the conclusion.** Tool
  count 19 → **20**. For any goal, the beliefs holding it up on the least evidence, and the
  cheapest experiment that would flip each. Not *what do you believe* but *what would it take to
  be wrong*, which is the question a reviewer asks and a status report cannot answer.
  - Ranked by **fragility, not by posterior**, and the two disagree on purpose. A belief confirmed
    by elimination carries a confident posterior and **no observation at all**, so it ranks first
    however sure the engine is — and it is simultaneously the cheapest thing in the graph to
    settle, because one probe touches what no probe has ever touched. Then confirmations
    established shallower than the depth something was built on them at, then single observations.
  - Read-only: no lease, no dispatch, no mutation. Asking what would change your mind must not
    change it, and a test pins that it does not.
  - **An empty list is a finding**, not a failure — nothing is resting on thin evidence — and the
    response says so. A panel that always finds something to say is one nobody reads twice.
  - On the dashboard as a third panel beside the learning path and the frontier, and over the API
    as `GET /api/counterfactual`.
- **A short batch now says a competing answer is being held back.** `get_next_targets` has always
  refused to dispatch two answers to one question — confirming one retires the rest, so handing out
  both spends a probe the inference would have saved. It did that **silently**, and a caller that
  asks for two targets and gets one reads it as an exhausted frontier. Two full evaluation runs say
  what happens next: every redundant probe in both was `claimed=False` — self-initiated — and the
  commonest shape is the caller filling the empty slot with the exact sibling the rule was
  protecting. The response now carries `same_question_withheld` and says so in the `rationale`.
  Silent on a full batch, and silent when the batch was short because the work genuinely ran out.

### Changed
- **`--cost-aware` is now `--experimental-cost-aware`.** The mechanism is measured (77% less cost
  for 1.5% more probes) but only against a scripted caller on a synthetic tariff, which justifies
  the mechanism and not the default. The name now says which of those is true. It is expected to
  become the default in a later minor release once a run with a live model has scored it, and the
  flag disappears at that point; recording `duration_s` and `estimated_cost` is useful either way.

- **Cost-aware selection: rank by value per unit of what a probe actually cost.** Selection
  weighed every candidate purely on how much it would settle, which is right only while all
  probes cost the same. They do not: a three-day fine-tune and a one-second unit test were
  ranked as if interchangeable, so the cheap decisive check waited behind the expensive one.
  - `record_evidence` accepts **`duration_s`**, the wall-clock seconds the experiment took.
    The engine reads back the mean per node and converts the frontier into **median-relative**
    ratios — 4.0 means "four times the typical probe here" — then scores `value / cost`.
    Relative, because absolute seconds do not compare across workspaces, and a median is not
    moved by the one overnight run that would drag a mean.
  - A node nobody has timed inherits its **exclusion-group siblings'** median before falling
    back to the workspace median: the answers to one question are usually tested the same way,
    so the closest available evidence is the sibling's, not the global average.
  - **`estimated_cost` on `create_hypotheses`**, because the observed model was blind exactly
    where the saving is. A question is settled *once*, so at the moment the navigator chooses
    between its competing answers none of them has been timed, and the sibling fallback handed
    every one of them the identical number. And within a question is the only place cost can be
    saved at all: ordering across questions changes nothing, since every question must be settled
    either way, whereas **the last answer standing is deduced rather than probed** — so whichever
    one is never reached is never paid for. Ordering cheap-first puts the expensive answer in that
    free slot. Precedence is observations, then the estimate, then siblings, then the workspace
    median. Safe to consume unmeasured in a way an accuracy prior is not: cost moves only what is
    tried next, never what the belief state asserts, and the first real `duration_s` overrides it.
  - **An expensive premise can be deferred but never starved.** Dividing by cost defers a slow
    node whenever a cheaper one keeps looking marginally better — and the cheap candidates are
    genuinely better value at *every* comparison, so the expensive one loses every comparison it
    is ever in, forever, and a premise that gates the whole graph never runs. The cost weight now
    decays with waiting time: full effect at zero wait, exactly neutral after `COST_PATIENCE_S`
    (one hour). The *exponent* decays rather than the ratio, so a 100x node and a 2x node relax at
    the same rate instead of the expensive one being punished twice for being expensive.
  - **Off by default** (`hypotree --cost-aware`). With it off, every ratio is exactly 1.0 and
    ranking is bit-identical to a build with no cost term at all — the only honest way to add a
    term to an acquisition function that a frozen gate has already scored. Recording
    `duration_s` is always safe; it is stored and displayed either way, and the dashboard now
    shows how long each result took beside it.
  - **And it is now measured.** The phase shipped with a falsifier nobody could run: every
    criterion in the harness counts *probes*, which is defensible only because the oracle answers
    in uniform milliseconds — so a probe is the unit of cost by construction, `θ/c` and `θ` induce
    the same order, and the claimed 20% saving was not merely unmet but unobservable. A falsifier
    that cannot fire is not a falsifier. `eval/cost_gate.py` plays every seed twice against a
    cost-weighted tariff, once with the navigator allowed to see it and once not: **cost falls
    77.3%** against a 20% threshold, **probes rise 1.5%** against a 10% ceiling, 30/30 seeds
    cheaper and 30/30 still solved, capturing **94.2%** of the achievable saving. Both halves come
    from one mechanism, which is what makes it a single claim rather than two coincidences. The
    tariff is assigned independently of which answer is correct — verified at 19% against a 20%
    chance baseline — so the arm is not being credited for finding the answer sooner.
- **A probe on a question the inference had already closed now says so.** `record_evidence`
  returns a **`redundant`** note when the result lands on a node the exclusion inference had
  retired. Run M spent eight probes this way: the agent tested all five members of an axis and
  reported all five in one batch, so every retirement fired after the probes were already
  spent. The saving exists only while the alternatives are unprobed, and nothing in the system
  said so — the existing counters watch dispatches, and these were self-initiated.
  - The note distinguishes the two cases. A **contradiction** (the retired node comes back
    VERIFIED) is two confirmed answers to one question and was worth the spend; anything else
    is waste, and the message names the rule that avoids it.
  - The eval report counts both separately, and the exclusion-yield paragraph now excludes
    redundant probes from its denominator: the blind baseline assumes each question is settled
    once, so a yield below chance with redundant probes present is a recording-discipline
    problem and not an ordering one. Two problems, two fixes, and reporting them as one number
    sends the reader after the wrong one.

### Fixed
- **Artifact paths recorded on Windows did not match the same paths recorded on Linux.**
  `record_evidence` stored `str(Path(p))`, so `/tmp/run.log` became `\tmp\run.log` on Windows
  and the same artifact read back as two different records. A belief state is shared — synced,
  committed, opened from CI — so paths are now normalised to POSIX separators on write.
- **A missing `success` invented a measurement and settled the hypothesis.** `record_evidence`
  defaulted it to 0.5, so `record_evidence(node_id="X")` — an ordinary LLM truncation, and valid
  against the schema — wrote a real evidence row, moved the posterior to Beta(1.5, 1.5) and, in
  the deterministic regime where any reading is conclusive, transitioned the node to
  **EXHAUSTED** and retired its competing answers. The belief state may assert only what was
  observed or soundly inferred, and a default value is the quietest possible way to break that.
  Both fields are now required by the schema *and* by the dispatch, and the refusal names the
  contract and points at `evidence_kind="infra"` for an experiment that could not be run.
- **A schema stamp that is not a number skipped every migration and then re-stamped the file as
  current.** `_is_newer_version` swallowed the `ValueError`, so *both* guards read False: neither
  "newer than this hypotree" nor "predates the chain" fired, `pending` came out empty, and the
  database was left at whatever shape it had — then stamped `10`. That last step made it
  **unrecoverable**: a later open by correct code sees a version needing no work, so the missing
  columns can never be added. Unorderable stamps are now refused up front, and the refusal leaves
  the stamp alone so the file stays diagnosable.
- **The exclusion-group prior was inert on the path the eval exercises.** `live_group_counts`
  counted only INVALIDATED/PRUNED, but the deterministic regime refutes only on an exact 0.0 and
  sends everything else to EXHAUSTED. A five-way question with four candidates measured and
  rejected still reported k=5, so the survivor's prior mean stayed at 0.2 — **below** an
  untouched ungrouped node. The navigator was deprioritising the one candidate nearly certain to
  be the answer. It now takes the engine's own definition of *ruled out* (refuted, or EXHAUSTED
  by its own evidence — the definition the guide states and deduction uses), resolved in one bulk
  query rather than one status history per node. A member set aside *by the inference* still does
  not count: nothing was observed about it, and counting it would let one confirmation deduce the
  rest of its own group from itself.
- **The dashboard's glow ignored the directive the reader had just set.** `p_select` was a pure
  argmax over Beta draws, so a suspended node still glowed and a pin did not collapse the
  frontier — wrong in the one situation the reader has most reason to check it, straight after
  using the dashboard's own buttons, which are the only way those directives are ever set.
- **`?at=` rejected a timestamp carrying its own offset.** `+` is the query-string encoding of a
  space, so `?at=2026-08-07T09:00:00+00:00` arrived with a space where the sign belongs and the
  endpoint answered 400 for a perfectly good instant. The browser client encodes it and was fine;
  anyone hand-writing a URL or pasting a timestamp was not, and the time machine has shipped that
  way since it landed. The refusal also now names the field the caller actually passed.
- **A lease could be issued dead on arrival.** `lease_ttl_s` carried `minimum: 1` on
  `renew_claim` and nothing on the two tools that *issue* leases, while `expire_stale_claims`
  uses `<=` deliberately — so `get_next_targets(lease_ttl_s=0)` handed back a claim that expired
  the instant it was created. The agent runs the experiment and cannot file the result.
- **One dropped HTTP connection turned a 90-episode run into a STOP verdict.** Infra-failed
  episodes are dropped from the paired set on purpose — censoring one would charge an arm for the
  inference server's bad minute — but that leaves fewer than the pre-registered n, which
  `_paired_moat_comparison` reported as a *failed* criterion 1, which forces STOP. A
  data-availability fact was being emitted as a pre-registered instruction to abandon the
  project. New **`INCONCLUSIVE`** decision; the thresholds themselves are untouched.
- **Criterion 3 double-counted every conflict that resolved.** `revision_events` summed conflicts
  *and* culprits, but a resolution is the same conflict later in its life — so the headline rose
  with how *well* the diagnosis worked, up to 2× for a run that narrowed everything.
- **`dispatches never reported` counted leases the agent handed back.** `release_claims` is the
  documented response to work you have decided not to run, and the same events were already
  reported one row below as "leases released" — so one action appeared twice, in contradictory
  terms, one of them labelled "work paid for and lost".
- README: the API is no longer described as read-only two lines above a route table listing a
  write; the architecture box said 9 tables and there are 11.
- **Re-attributing a retired sibling was a silent no-op, and buried the answer forever.** When a
  question has two confirmed answers and the first is withdrawn, the siblings it retired are
  supposed to be re-attributed to the second. That call passed the status the sibling already
  held, which `change_status` correctly refuses as a non-transition — so the marker kept naming a
  node that no longer confirmed anything, and since retraction keys on that marker, **nothing
  could ever reopen the sibling again**. The question ended with zero live answers and its only
  untried candidate EXHAUSTED, reported by nothing: `_eliminated_on_its_own_evidence` returns
  False for it, so not even `dead_question` fired. New `store.reattribute_status` rewrites the
  justification without opening a second interval at the instant the first closes.
- **`_cascade_prune` was the one path out of VERIFIED that bypassed the exclusion sync.** Every
  other route surrenders the authority to keep competing answers retired; the cascade did not, so
  a pruned confirmation left a question with no live answer and the only untried candidate buried
  beneath a node nobody believes.
- **A conflict nobody could swap held its members off the frontier forever.** A member with no
  exclusion group — or whose group has no live alternative left — can never be cleared, so the
  conflict never left `_diagnosing_nogoods`. The caller was handed a bare `empty_frontier` with
  **no rationale at all** while untested hypotheses sat in the store, which is exactly the
  failure the DONE taxonomy exists to prevent.
- **Omitting `claim_id` never consumed the node's own lease.** A *wrong* id was already recovered
  from it; an absent one was not, which is backwards — absent is the documented ordinary path.
  The lease stayed live for its full TTL, holding a settled node off the frontier and reporting
  it as dispatched-and-never-reported.
- **`awaiting_evidence` was checked before every substantive diagnosis**, so one leaked lease
  masked `awaiting_substitution`, `awaiting_composition`, `dead_question` and `blocked_frontier`
  alike — telling the caller to record evidence it had already recorded, when recording could not
  help because nothing was left to record.
- **`PRUNED` counted as a refutation when narrowing a conflict.** It says an ancestor was
  refuted, not that this assumption caused the failure — so a conflict closed on collateral
  damage, released every other member, and then spent the review budget on the wrong question's
  alternatives.
- **`_question_is_dead` counted already-recovered conflicts as live.**
  `_recover_from_interaction` sets `reopened_at` and deliberately never `resolved_at`, so one
  interaction recovery silenced the dead question for that group for the rest of the workspace's
  life.
- **`_eliminate_substitute` called `_deduce_last_member` without `_propagate_dead_question`** —
  the one elimination path missing the pair, and the call site
  `_eliminated_on_its_own_evidence` was specifically extended to cover.

### Eval harness
- **The exclusion-yield baseline described an engine hypotree does not ship.** Yield was scored
  against `(k-1)/2k`, the blind baseline for an **open** group, while every group in the eval is
  **closed** — and a closed group retires one more question for free with no ordering skill at
  all, because the engine confirms the last survivor by elimination. The correct baseline is
  `(k²-k+2)/2k²`, **44%** at k=5, validated against a 400 000-trial simulation. Re-scored, runs
  G (45%), K (43%) and L (44%) all sit **at** the baseline: there was never a regression to
  recover and never a signal to celebrate. The baseline is now summed **per group** rather than
  taken from the mean size, because it is non-linear in k, and `exclusion_closed` is logged on
  `node_created` so the reader can tell the two worlds apart.

### Performance
- **`stale_node_ids` was an N+1** — one evidence query per VERIFIED node, the same shape that was
  most of the 41× in v0.4.0. Now one `latest_context_hash_by_node` query, asserted by counting
  round-trips rather than by timing so it cannot pass by accident on a fast machine.
- **`capture_git_context` spawned two subprocesses per result.** `record_results` calls
  `_record_one` per report, so an eight-result batch spawned **sixteen** processes, each with a
  5 s timeout, to learn one commit hash that cannot have moved between them. Resolved once per
  batch, lazily: a batch whose evidence carries its own context spawns **none**.
- Dispatch measured at **46 ms at 1200 nodes** and flat in the number of open conflicts.


## [0.4.2] - 2026-08-05

### Fixed
- **Align `choose_port` socket flags with `asyncio` on Windows.** On Windows, `SO_REUSEADDR` allows probing sockets to bind over active listeners, reporting occupied ports as free and causing `asyncio.create_server` to crash with `address in use`. `choose_port` now applies `SO_REUSEADDR` only on POSIX (`os.name == 'posix'`), matching `asyncio`'s standard library implementation.

## [0.4.1] - 2026-08-04

### Fixed
- **`pip install hypotree` was broken by the release of `mcp` 2.0.0.** The
  dependency was declared as `mcp>=1.28.1` with no upper bound, so a fresh
  install resolved to 2.0.0, which removed the low-level decorator API
  (`Server.list_tools`, `call_tool`, `read_resource`) that the server binds its
  handlers with. Every new install died on startup with
  `AttributeError: 'Server' object has no attribute 'list_tools'` before the
  first handshake. The requirement is now `mcp>=1.28.1,<2`.

  Existing environments were unaffected — an already-resolved `mcp` 1.x stays
  put — so this only ever hit new installs. Users on 0.4.0 or earlier who see
  the `AttributeError` should upgrade, or pin `mcp<2` themselves.

  Support for `mcp` 2.x is a port to its `add_request_handler` API and will
  land as its own release rather than a silent resolver upgrade.

## [0.4.0] - 2026-08-04

### Performance
- **Selecting a target is 41× faster on a large belief state**: 1200 nodes went from
  **4 494 ms to 110 ms** per dispatch, and the curve from ~3.8× to ~2.3× per doubling of
  nodes. None of this was visible at the 40–60 nodes an evaluation episode produces, which
  is why it survived 662 tests — but every long-lived workspace lands on the wrong side of
  it, and "the belief state survives across sessions" is the product claim.
  - **The dominant cost was an N+1 query.** `get_all_nodes` derives each node's `parent_ids`
    from the edges table and was issuing **one query per node**; the navigator calls it
    several times per dispatch, so handing out two targets cost ~6 000 SQL round-trips.
    Parents are now resolved for every row in a single query.
  - `HypoTreeGraph.add_edge` ran a full `is_directed_acyclic_graph()` check **per edge**, so
    reloading *E* edges cost *O(E·(V+E))*. New `add_edges_bulk` checks the finished set once
    and rolls back if it is cyclic. The per-edge path stays for interactive creation, where
    naming the offending edge is worth the cost.
  - `HypoTreeGraph.parents()` scanned every edge in the graph and the frontier calls it once
    per node. Parents are now indexed by `dst`.
  - `get_next_targets` syncs the in-memory graph **once per batch** instead of once per pick.
    The batch loop takes leases, which change node rows but never the topology.
  - A long-lived incremental graph — invalidating on write, never rebuilding — is
    deliberately **not** done. A cached graph that disagrees with SQLite would produce
    silently wrong selections, and at 110 ms there is no case for taking that risk.

### Added
- **The closed-world assumption is now declared rather than assumed.** Deduction by elimination
  — confirming the last survivor of an exclusion group for free — is sound only if the listed
  candidates really are all of them, and **nothing checked that**. Withhold one axis's winning
  value and the engine refuted the four that remained and confirmed the survivor with *no
  observation of its own*: a free confirmation of a value that is wrong, recorded as a
  confirmation. The run then ended `empty_frontier` with the goal unmet and never said the
  question had been incomplete. This is the mechanism the README credits for the moat, and it
  had shipped unguarded for months.
  - **`exclusion_closed` on `create_hypotheses`** (schema `10`, `nodes.exclusion_closed`,
    `NOT NULL DEFAULT 1` — exactly what every group written before it meant). Defaults to
    `true`. Pass `false` where the next candidate always exists — "which learning rate", "which
    prompt wording", "which threshold" — and the last-one-standing deduction is withheld, along
    with the pruning that rests on it. Confirming a member still retires the others: that is an
    observation about the confirmed value, not a claim about the list.
  - **Openness is per-group and one member declaring it is enough.** It withdraws an inference,
    so the cautious declaration governs; the reverse would let a later call that forgot the flag
    quietly re-enable deduction over a list its author knew was partial.
  - **A deduction can now be withdrawn.** When every question is answered and the assembly still
    falls short, the belief with no measurement behind it is the one to doubt. The engine hands
    it back as `UNTESTED` and the navigator dispatches it, so one probe settles whether the value
    was wrong or the list was incomplete. It never asserts the value false — nothing was ever
    observed about it, and asserting on no evidence is the one thing the belief state may not do.
    The same applies at conviction: a deduction named as a conflict's sole cause is tested rather
    than convicted.
  - **`_is_deduced` reads the whole history, not its last entry.** A deduction that passes
    through conflict review comes back marked "released from review", which describes the last
    thing that happened to it and erases where the confirmation came from — a node never once
    observed looked like one that had stood up to scrutiny.
  - **`dead_question` now says which kind.** Over a closed group it means the list was wrong or
    one of the eliminations was, and what depended on those values is pruned. Over an open one it
    means the answer is very likely one that was never listed — and **nothing is pruned**,
    because an untried candidate could still satisfy it.
  - **Measured.** Across all 30 seeds × 5 axes with the winner withheld, **144/150 (96%)** now
    end `dead_question` naming the incomplete axis, against **0/150** before — every one of which
    was `empty_frontier`. Still 0 solved, which is correct: the winning value was never offered,
    so the goal is genuinely unreachable, and what changed is that the run says *why*. A wrong
    answer reported as wrong is a result; the same answer reported as an empty frontier is not.
    Fully-declared self-play is untouched at 30/30, mean 15.9.
- **The dashboard runs by default.** It was behind `--dashboard` while it was unproven, and the
  result was that the people it was built for never saw it — nobody opts into a feature they
  have not been shown. It binds loopback only, mints a fresh token per start, and **cannot take
  the MCP server down**: a bind failure is reported and the server carries on, because the cost
  of a viewer failing must never be the cost of the tooling failing.
  - `--dashboard-port PORT` picks the first port to try (it still probes upward, so several
    workspaces can be open at once).
  - `--no-dashboard` opens no socket at all.
  - `--no-mcp` replaces `--dashboard-only` and reads better beside its sibling. v0.4.0 is
    unreleased, so the old spelling never shipped and is simply gone.
- **The agent can hand over the dashboard link.** The URL is minted at startup and printed to
  stderr, which the model never sees — so an agent asked "where can I watch this?", which is
  constant, had no way to answer. `get_workspace_info` now returns `dashboard_url`, and the new
  `hypotree://dashboard` resource returns it on its own. Both carry the session token, which is
  the whole credential; `null` means none is running, and the text says how to start one.
- **Backward pruning over a complete question.** The exact dual of deduction by elimination,
  and the half of that symmetry that had never shipped: all-but-one candidate eliminated
  confirms the survivor for free, and *all* of them eliminated says the question has no answer
  among the candidates offered — so nothing that assumes one of them can ever be satisfied.
  Those branches are pruned under their own reason marker and the navigator reports a new DONE
  reason, **`dead_question`**, naming the group that ran out and the goal it was holding up.
  - **Soundness rests entirely on the exclusion group.** "All four approaches I tried failed"
    does not entail "the objective is unreachable" — it entails "I need a fifth idea", and a
    rule that could not tell those apart would let an unimaginative agent declare its own goal
    impossible. What makes this version sound is that the caller *declared* these to be the
    competing answers to one question.
  - Three guards, each closing a way it could fire on a live search: at least two candidates
    (one value is not a question); every member eliminated **on its own evidence**, which is
    strictly stronger than the elimination test deduction uses — `PRUNED` does not count,
    because it is a statement about an ancestor's subtree, and neither does a member with no
    observation of its own; and no open conflict over any member, because conflict diagnosis
    may hand one straight back.
  - It reaches what the refutation cascade deliberately spares. A premise settled as
    `EXHAUSTED` was never refuted, so its dependents stay on the frontier being re-attempted
    long after the premise ran out; that is the case this exists for.
  - `dead_question` is an instruction, not an ending: the fix is to add the candidate that was
    never listed, and the eval harness treats it as a reason to continue.
- **`record_evidence` is batch-native.** Pass `results=[{node_id, success, depth, claim_id}, …]`
  to report every experiment from one turn in one call. `create_hypotheses`, `update_status`
  and `get_next_targets` have all been batch-native for a while; `record_evidence` was the
  last singular high-frequency tool, and it is what pinned a synchronous agent at two turns
  per experiment — one to probe, one to report. The single-result form still works and is the
  `k=1` shorthand.
  - Results are applied **in order**, so a refutation's cascade lands before the next result
    in the same batch is read and the belief state ends where it would have ended had the
    results arrived separately.
  - A refused report does **not** abort the batch. Every result was paid for by an experiment
    that already ran, so discarding k-1 of them to punish one bad `node_id` destroys evidence.
    The refusals come back under `failed`, named and explained.
  - The fused dispatch runs **once**, after the whole batch, and is still a top-up.
- `get_conflicts` now reports `skipped_no_substitute` per member.
- **A read-only web dashboard: `hypotree --dashboard [PORT]` and `hypotree --dashboard-only [PORT]`.**
  A belief state that revises itself is hard to appreciate from a status column. The
  dashboard runs on the MCP server's own event loop and serves the graph, the narrative and
  a timeline over localhost HTTP. `--dashboard-only` starts no MCP server at all — the
  try-before-you-wire path for someone who has not configured a client yet.
  - **The interface.** Vue 3 over server-computed coordinates, with `d3-zoom` for
    hardware-accelerated pan and zoom. Untested nodes glow at their real chance of being
    dispatched next; in-progress nodes pulse; **new nodes fade in** as the agent creates
    them and **pruned branches desaturate rather than disappear**, because the claim being
    demonstrated is that they were considered and cut. Edges into a dead branch turn red.
  - **A timeline scrubber** over the bi-temporal history: drag back to any instant, or press
    play and watch the investigation replay.
  - **Proof mode** overlays what each belief cost — score, depth, commit, `source_ref`.
  - **The learning path is typeset markdown**, ready to paste into a report.
  - **Everything is vendored** (Vue 3, nine d3 micromodules, `marked` — 276 KB). No CDN, no
    npm, no build step: it works on a plane and in an air-gapped network. d3 micromodules
    rather than the full build (63 KB against 273 KB) because the layout is computed
    server-side and nothing else in d3 is wanted.
  - **The observer physically cannot write.** Its store is opened `mode=ro`, so a read path
    that is wrong fails loudly instead of quietly mutating the belief state being watched.
    WAL already permits a second reader, so it never contends for the engine's write lock.
  - **Layout is computed server-side** with `networkx` and shipped as coordinates. No
    JavaScript graph library, so nothing large is downloaded and the layout is deterministic
    and testable in Python rather than eyeballed in a browser.
  - **`p_select` is the real selection probability**, estimated by repeating the Thompson
    draw and counting argmax wins — not a monotone stand-in like the posterior mean.
  - **Time travel is a `WHERE` clause.** `?at=<iso8601>` reconstructs any past instant from
    the SCD2 status and posterior intervals, so a replay shows what was believed then.
  - **SSE carries a revision number, never a payload.** `events.seq` advances inside every
    mutation's transaction, so it is a correct change signal by construction; clients refetch.
    A subscriber that cannot keep up is dropped rather than allowed to apply backpressure.
  - **Everything is cached on the revision**, which is also the SSE payload — without it a
    scrubber would rebuild the whole graph per frame.
  - **Security.** Binds `127.0.0.1` explicitly, mints a session token at startup and requires
    it on every `/api/*` call, validates the `Host` header (the DNS-rebinding defence behind
    CVEs in Jupyter, Ray and TensorBoard), refuses cross-origin requests outright, and serves
    the page from a string so path traversal is unrepresentable rather than filtered. The URL
    is printed to **stderr** — stdout is the JSON-RPC channel.
- **Pin and suspend: human scheduling directives that are not beliefs.** A new
  `node_directives` table records "look at this first" or "leave that alone" with an actor and
  a reason. Directives change what the navigator *offers*, never what is believed — writing a
  click into `alpha`/`beta` would make it forever indistinguishable from an experiment and
  inject unlogged nondeterminism into a seeded sampler. With none set, selection is unchanged.
- `edges.created_at` records when an edge appeared, so a timeline replay no longer draws the
  final topology at every tick. Additive; edges written before it are treated as original.
- **A migration adding a column to a table the DDL had just created failed.** The DDL
  described the *current* schema and ran before the migration chain, so a table the file was
  missing got created already-modern and the step adding its column then hit `duplicate
  column name` — a collision every future migration would have met, not just this one. The
  schema is now defined the way Doctrine and Rails define one: `BASE_DDL` is the **original**
  shape, frozen, `CREATE TABLE IF NOT EXISTS` throughout, and every change since is a
  numbered step applied in order. A fresh database is built exactly as an old one is
  upgraded, so the upgrade path is exercised on every install rather than only on other
  people's machines. `SCHEMA_VERSION` is derived from the last step and cannot drift from it.
- **New `schema_state` table: exactly one row, enforced by `CHECK (id = 1)`.** Carries the
  schema version, the hypotree release that last opened the file, and when. It is the first
  thing to look at when a database behaves unexpectedly or a bug report arrives with one
  attached. The pre-0.4.0 `schema_meta` key is still read as a fallback and kept in step for
  one minor version. A freshly created database logs no migration events — it was created,
  not upgraded.
- **The dashboard did not run in a browser.** `default-src 'self'` blocked Vue's global build
  compiling its in-DOM template with `new Function`, blocked the inline script carrying the
  session token, and blocked `data:` fonts injected by browser extensions. Every directive is
  now named explicitly rather than leaning on the fallback, and the token travels in a
  `<meta>` tag so `script-src` can keep refusing `'unsafe-inline'` — the directive that stops
  an injected event handler from ever running.
- **The learning path rendered untrusted text through `v-html`.** Node statements are written
  by an agent, and `marked` has had no sanitiser since v5, so markup in a statement reached
  the document verbatim. An allowlist sanitiser now runs over the parsed output; the strict
  `script-src` covers what a sanitiser bug would miss, and vice versa.
- **Work on one objective at a time: `goal_id` on `get_next_targets`,
  `generate_learning_path` and `get_goal_status`.** A workspace pursuing several goals
  interleaves them — the frontier offers whatever is most uncertain anywhere, and the
  learning path narrates every goal's dead ends into one story. A `goal_id` restricts all
  three to the case for one objective: that goal, everything it depends on, and the
  competing answers to those questions. Omitting it is the previous behaviour exactly,
  down to selecting the identical node on an identically-seeded engine.
  - Scope is DEPENDENCY-ancestry **plus the exclusion-group siblings of that ancestry**,
    not graph reachability. A competing answer is not an ancestor of the goal, but testing
    it is how the engine learns whether the answer that *is* an ancestor holds — a scope
    without it hands the navigator questions it is forbidden from answering.
  - New DONE reason **`goal_scope_empty`**. Agents create premises before wiring them
    (`awaiting_composition` exists because they do), and unwired nodes are outside every
    goal's scope by construction. Rather than reporting a bare empty frontier — which reads
    as "the search is over" — it names how many untested hypotheses are outside the filter
    and says the fix is usually a missing DEPENDENCY edge.

### Fixed
- **The live stream died every fifteen seconds on Python 3.10.** `asyncio.TimeoutError` only
  became an alias of the builtin `TimeoutError` in 3.11, so the keepalive's `except TimeoutError`
  caught nothing on the project's own minimum version: the first quiet interval killed the SSE
  task, the browser reconnected, and the cycle repeated. It presented as a dashboard that
  flickered between "live" and "reconnecting" for no reason, which reads as a flaky network
  rather than as a bug. Caught as `asyncio.TimeoutError`, which is correct on both.
- **A goal can no longer be given as a DEPENDENCY parent.** This is the one modelling mistake
  strong models make reliably: "the goal decomposes into phases" is how everyone thinks about
  objectives, so `parent_ids=[goal]` on the phase reads as "this phase belongs to that goal". In
  hypotree it means the opposite — the phase cannot be tested until the goal is VERIFIED — and
  both halves then break silently. A goal is never dispatched and `verify_upstream` only promotes
  IN_PROGRESS nodes along REFINEMENT edges, so the goal can never reach VERIFIED and the child is
  blocked forever, while the goal still depends on nothing and so can never be reached either.
  The engine had been handling this **downstream**, reporting the wreckage as `blocked_frontier`;
  it is now refused **upstream** with a `GoalDependencyError` carrying the corrected call. Only
  DEPENDENCY is refused, because only DEPENDENCY gates a child on its parent.
  - A real behaviour change: five existing tests were constructing exactly this state to exercise
    the downstream reporting. They now build the same situations from ordinary nodes —
    `blocked_frontier` from premises wired under an EXHAUSTED combination, which settles without
    cascading — which is the general case and the one still representable.
  - Databases written before this release keep the shape and nothing is rewritten. The engine may
    change what it asserts only from an observation or a sound inference over one, and silently
    flipping an edge someone declared is neither. The dashboard names such a goal's wiring as
    `inverted` rather than merely `unwired`, because "turn your edges around" and "attach some
    work" are different instructions.
  - `parent_ids` is described as *what this rests on* at every point of use, and the guide states
    the direction before it states the edge types.
- **Selecting a goal hid the tree hanging off it.** Agents routinely wire a goal the other way
  round — `goal -> phase0 -> work`, reading the goal as a container rather than as something the
  work supports. The navigator's goal scope is its DEPENDENCY *ancestry*, which is the right set
  to dispatch from and the wrong set to draw: filtering to such a goal showed a single dot beside
  a lecture about wiring while a nine-node tree hung off the node in question. A viewer that
  hides your work to make a point about how you built it is not making the point. The dashboard
  now walks **both** directions — the selection scope plus everything downstream plus the
  exclusion siblings of the result — and reports the unreachability in a dismissible banner over
  the canvas instead of in place of the graph. The navigator's scope is deliberately unchanged:
  it must not start selecting downstream of a goal just because the viewer draws it. The timeline
  follows the drawing for the same reason the graph does — the scrubber moves the picture, so a
  tick it cannot show is a step that does nothing.
- **The left panel flashed on every node click.** Selecting a node cleared `detail` before
  fetching it, which mounted the entire narrative for one frame before replacing it again. The
  panel now branches on *what is selected* rather than on *what has arrived*, so the card
  renders immediately and fills in. A slow response for a node the reader has already moved off
  no longer overwrites the one they are looking at.
- **A directive reloaded the whole view.** Pinning a node rebuilt the narrative and the layout
  to change one badge; it now refreshes the node, the graph and the frontier, which are the only
  things a directive can move.
- **Binary assets were dropped from the manifest.** The table only carried `.js` and `.css`, so
  the logo would have 404'd — a broken header rather than a missing feature. PNG and SVG are
  served with the right content type and ship in the wheel.
- **A dry run wrote to the belief state.** `get_next_targets(dry_run=True)` is documented as a
  peek and issues no claim, but when a conflict's targeted narrowing had run out of advice the
  peek performed the fallback: it marked the conflict recovered, released its members from
  review and reopened every alternative they had retired. A caller asking what it *would* be
  given settled the question it was asking about, and the next real call saw a belief state a
  peek had moved. The sibling fallback on the same path was already guarded, so this was an
  inconsistency rather than a design decision. The peek now reports the instruction it would
  still give and changes nothing.
- **The conflict fallback timestamped itself off the wall clock** rather than the instant the
  selection pass was running at, so the status changes it wrote landed outside the interval
  the rest of that pass recorded — a small tear in a history whose whole value is that it can
  be read back.
- **The dashboard's own buttons were refused as cross-origin.** The write path rejected any
  request carrying an `Origin` header. Browsers attach `Origin` to *same-origin* JSON POSTs
  too, so pin, suspend, back and clear each came back `403 cross-origin requests are refused`
  — the CSRF defence was locking the page out of itself. The header is now validated against
  the bound address instead of being treated as evidence of an attack: a foreign origin, or
  the right host on a different port, is still refused, and an absent header still falls
  through to the token and `Host` checks.
- **The narrative ignored the time-travel scrubber.** Rewinding the graph left the learning
  path describing conclusions the picture had not reached, which is worse than showing
  nothing. `generate_learning_path` takes `as_of` and `/api/learning-path` takes `at`, so the
  briefing stops where the graph stops. Observation tallies are counted at that instant too,
  via the new `HypoTreeStore.count_evidence_by_node(before=...)` — one aggregate that also
  replaces the per-node cached counts on the live path. The rewound render says which line
  still reflects the present, because goal achievement is read off the live graph.
- **The timeline handle sat at the far left while showing the present.** The scrubber ran from
  `-1`, and `-1` meant live — so watching the current state pinned the handle to the start of
  history and playback had nowhere to run. The present is now the right-hand end: the handle
  rides it as new history arrives, `live` returns to it, and `play` pressed at the present
  rewinds to the beginning first.

- **A member the diagnosis could not test was reported as cleared.** Substitution progress was
  an integer cursor, which can only say "the first k were dealt with". A member the plan had
  to pass over — because its question had no live alternative left to swap in — sat *behind*
  the cursor the moment a later member was cleared, and was reported as exonerated by an
  experiment that never ran. It could also never be revisited if an alternative freed up
  later. Progress is now a persisted **set of cleared ids**: cleared and skipped are recorded
  as the opposite claims they are, and the plan reconsiders a skipped member on every call.
  Schema `"9"` → `"10"` adds `nogoods.cleared_ids`; the migration is additive and the set is
  back-filled from the old cursor on read, so a diagnosis in flight across the upgrade keeps
  the members it had genuinely cleared. `probe_index` is kept in step and **deprecated**;
  `HypoTreeStore.advance_nogood_probe` warns and will be removed in 0.5.0.
- **Schema versions were ordered as strings.** `"9" > "10"` and even `"3" > "10"` are true
  lexicographically, so the first two-digit schema version made a database with a corrupt or
  older stamp report itself as "written by a newer hypotree" — sending the user to upgrade
  the package instead of keeping the file. Ordered numerically now; a stamp that is not a
  number is not treated as newer, which routes it to the "back it up and open an issue"
  branch where it belongs.
- **A success could resurrect a pruned branch.** Recording a passing result on a `PRUNED` node
  flipped it to `VERIFIED`, leaving the belief state asserting both that the branch is dead
  and that it holds — a confirmed node sitting on an invalidated parent. A pruned node's
  status is a statement about its ancestry, so the measurement and the posterior are kept and
  the status is left alone. Rare while probe and record alternated in fixed pairs; routine
  once results arrive in batches, because one refutation in a batch can prune the node a
  later result in the same batch belongs to.
- **`source_ref` was advertised and dropped.** 0.3.2 added the field, documented it on the
  `record_evidence` tool schema, and never read it in the dispatch — so every value sent over
  MCP was silently discarded. The single and batch paths now build evidence through one
  helper, so they cannot drift apart again.
- **The shortfall recovery fired on a search that had already won.** When every question is
  answered and the assembled answer falls short, the engine hands back the alternatives those
  answers had retired — the reading being that one of the confirmations must be wrong. It did
  not check whether a *different* assembly had since cleared the bar, and a composition that
  succeeded withdraws exactly that reading: the answers were right, the earlier assembly was
  simply the wrong combination of them. One real episode reopened six settled questions on the
  turn it found its answer, because its goal node had never been wired to the winning
  composition and an empty frontier looks the same from the inside either way. The recovery
  now stands down once any composition is `VERIFIED`.
- **Two different recoveries wrote the same reason marker.** `_recover_from_underperformance`
  reused `INTERACTION_REOPEN_PREFIX`, so a history could not distinguish "a conflict turned
  out to be an interaction effect" from "every answer was in and the assembly still missed" —
  and a run report claimed every conflict had been narrowed to a culprit *and* that six
  alternatives had been reopened because a conflict was an interaction effect, which are
  mutually exclusive endings. New `UNDERPERFORMANCE_REOPEN_PREFIX`. Additive: histories written
  before 0.4.0 keep the old prefix and still read as interaction reopens, which is the only
  label that ever existed for them.

### Changed
- **The dashboard has been redesigned as a light analytical interface.** The previous dark theme
  was hard to read for long and, more to the point, spent colour on chrome — so status, the one
  thing colour should carry, had to compete with it. Now warm paper (`#FAFAFA`), a gridded
  canvas that pans with the graph, white cards on soft shadows instead of heavy borders, and a
  single navy/cyan accent reused for goals, focus and selection so the eye learns one colour
  rather than three. Every status has a pastel badge; `PRUNED` adds a diagonal hatch, because
  grey alone reads as "not started" rather than "cut". No webfont is vendored: `font-src 'self'`
  would mean shipping a ~90 KB binary in the wheel to displace a system stack that is
  Inter-class on every current OS.
  - **Node cards** put the id in its own monospaced bar — set in the prose face it read as part
    of the sentence beneath it — and carry **created** and **settled** timestamps plus the
    engine's own reason string. "Confirmed" and "confirmed last week and untouched since" are
    different situations that a status column cannot tell apart.
  - **The scheduling controls are labelled and explained.** `pin` / `suspend` / `clear` were
    three bare verbs in a row and nobody could tell what any of them would do. They are now
    **Test first** / **Hold back** / **Clear**, under a heading, with the current directive
    stated in words and the standing rule beside them: this changes what the navigator *offers*,
    never what is believed.
  - **The timeline is an activity histogram** rather than a form slider. The shape of a run —
    where the bursts were, where it stalled — is information, and a featureless track threw all
    of it away. The handle travels along the bars.
  - Transitions run at a single `0.3s ease` everywhere, so a status arriving over SSE eases from
    one colour to the next instead of jumping.
  - The goal `<select>`, the panel scrollbar and the zoom controls are all styled; they were
    browser defaults sitting inside a designed page.
- **Wide layers wrap into sub-rows.** Five values on each of five axes puts twenty-five nodes on
  one line, and a drawing 25 wide by 3 deep fits the viewport at a scale where nothing is
  legible. The wrap width is derived from the graph's own size, so the layout is still a pure
  function of the topology — same graph, same picture, which is most of what a viewer is for.
- `/hypotree-next` and the server-level instructions now teach the batch shape.
- **Proof mode is no longer a toggle.** What an experiment cost is the reason the panel is
  worth reading; hiding it behind a button meant the default view was the less useful one.
  Provenance is always shown, and the button is gone.
- **The left panel can be dragged wider**, from its default out to 400 px more, and never
  past half the window — a reading panel that eats the graph defeats putting them side by
  side.
- **Zoom lives on the canvas.** `+`, `−` and `fit` sit at the bottom-right of the graph
  rather than in the header, next to the thing they act on.

### Eval harness
- **The exclusion-yield verdict could not report two of its three outcomes.** It compared
  `yield > baseline + 0.02` against `yield < baseline + 0.02`, which left "at the baseline"
  reachable only on exact float equality and described a run performing *below* chance in the
  same words as one performing at it. Now a symmetric band around the baseline, with a verdict
  for each side of it. Runs J and K are unaffected — both sit above the band — but the
  paragraph is the one the report calls "the only lever on premise cost", so a verdict it
  could not actually produce is worse than no verdict.
- **Group size counted creation events, not members.** The arm-B prompt tells the agent to
  re-create nodes with `if_exists="overwrite"`, and every re-creation logs `created` again —
  so an agent that rewrote its own nodes inflated `k`, which inflates the baseline it is being
  judged against, which flatters its own ordering. Counted by distinct node id now. Latent in
  J and K (both measured exactly 5.0, the true axis size) and corrected before it was not.
- **A second conflict stole the first one's diagnosis.** Attribution lived in a single slot,
  so a conflict opening before the previous one resolved overwrote its mark: the first
  resolution was measured from the wrong starting probe and the second was attributed to
  nothing at all. Keyed by conflict id now, which the runner had to start logging on
  `conflict_recorded`. Logs written before that id existed fall back to oldest-open-first —
  exactly the single-slot behaviour they were read with — so J and K still report the 2.15 and
  2.20 they always did.
- **`dead_question` prunes are counted** as a mechanism of their own in the belief-state table,
  apart from the refutation cascade they are deliberately distinct from.
- `RunLogger.log_lease_event` removed — no caller, and no `lease_event` in any log.
- **Self-play can under-declare a question.** `solve_seed(..., omit_winner_on=<axis>)` leaves
  the winning value off one axis's candidates. Every axis carries its winner by construction,
  so a fully-declared question always confirms and the closed-world assumption is never
  actually put under load; this is the only way this landscape can do it.

### Eval harness — batch recording
- **The closed-world check is a gated pre-flight.** `eval.sh` step 2c withholds one axis's
  winning value on every seed and requires at least 80% of those runs to end `dead_question`
  rather than `empty_frontier`. Gated on the *diagnosis*, not on a solve: the winning value was
  never offered, so 0 solved is the correct answer and the only thing worth checking is whether
  the run says why. Currently 30/30.
- **`solve_seed(..., omit_winner_on=<axis>)`** is how that is constructed. Every axis carries its
  winner by construction, so a fully-declared question always confirms and the closed-world
  assumption is never actually put under load; withholding one is the only way this landscape can
  do it.
- **Withdrawn deductions are counted** in the belief-state table, apart from every other
  mechanism. Each one is the engine catching its own assumption being wrong, and each costs
  exactly one probe to settle.
- **One bad result no longer discards the rest of a batch.** The harness looped the
  single-result call, so a batch containing an unknown node id aborted: the results already
  applied stayed in the belief state while the whole call was reported as an error, and every
  result *after* the bad one was never attempted. Each of those is an experiment that has
  already been paid for — which is exactly what `engine.record_results` per-item isolation
  exists to prevent, and the harness was silently overriding it. Never fired in run H (zero
  record failures), so no result is affected.
- Rejected batch items are logged (`record_rejected`) and reported. Per-item isolation makes
  the containing call succeed, so without this a refusal appeared nowhere: not in the tool
  census, which sees an ok call, and not in the evidence count, which never saw the result.
- **`seed_reader` reports probes to name the culprit.** The report said how many conflicts
  were narrowed and never what they cost, so the member ordering — the thing that decides
  that cost, one probe per position — was invisible. Run G: **1.73**. Run H: **2.13**. A run
  can resolve every conflict and still have got slower, and nothing said so.
- **A win no longer cuts the episode off mid-protocol.** The environment decides the win the
  instant a probe clears the target, but the agent is holding an unreported result at that
  moment. On the seeds where the swap that identifies a culprit *is also* the winning
  combination, the engine resolved the conflict and the scoreboard recorded it as unresolved,
  because the record that proves it was never allowed to land — run G's "11 of 13 conflicts
  resolved" was really 13 of 13. The agent now gets a bounded wind-down (2 turns) to file what
  it holds. Further experiments are refused and the step counter is frozen, so this cannot
  move the headline metric; arms with nothing outstanding wind down in zero turns.
- The harness `record_evidence` accepts and instruments the batch shape, attributing every
  belief-revision side effect to the result that caused it rather than to the last in the
  batch. `seed_reader` reports **results per record call** — 1.00 means the batch shape is not
  being used and turns/step cannot fall.
- **One HTTP error no longer kills a 90-episode sweep.** `_call_openai_api` wrapped `urlopen`
  in nothing at all, so a single 5xx from the inference server propagated out of `run()` and,
  under `eval.sh`'s `set -e`, aborted everything that had not run yet. Run `I` died at seed
  1210 of 30 and took 61 episodes with it. Transient faults (5xx, 429, connection failures,
  read timeouts) are now retried with capped exponential backoff plus jitter — five attempts,
  2 s to 60 s — and every retry is logged so a run with odd numbers can be checked against how
  much trouble the transport was having. A 4xx other than 429 still fails immediately: a
  malformed request does not become well-formed by being sent again.
- **An episode that dies is excluded from scoring, not censored at budget.** Exhausted retries
  raise `LLMUnavailableError`, which becomes a clean `run_end` carrying `infra_failed`. The
  gate drops those episodes from the paired set and the reader names them as transport faults.
  Censoring assumes the agent had its full budget and did not solve the task; an episode killed
  by a dropped connection had neither, and scoring it at budget charges one arm for the
  server's bad minute.
- **`eval.sh` survives a failed episode.** The runner call no longer aborts the sweep, failed
  episodes are collected and reported at the end with the `--resume` command that recovers
  them, and the resume check now requires a terminal record *and* the absence of an infra
  marker — otherwise `--resume` would skip precisely the episodes that need re-running.
- `seed_reader` reports **alternatives reopened after a shortfall** separately from those
  reopened by an interaction effect, and lists infrastructure failures apart from episodes
  with no terminal record: one needs the server looked at, the other needs a re-run.
- **A batch result missing `node_id` was refused with the word `'node_id'`.** That is
  `str(KeyError('node_id'))` — the agent was told a key name, not a contract, and had no way
  to work out what to send instead; one run lost a paid-for probe to it. The refusal now
  names the missing field and states the shape of a result.
- **`exclusions_applied` undercounted re-exclusions.** The counter required a status change,
  which dropped any sibling retired, released when its justification was withdrawn, and
  retired again by a later confirmation. `_apply_exclusion` skips any sibling that is not
  open, so arriving back at EXHAUSTED *requires* having passed through UNTESTED — the guard's
  stated rationale (avoiding double-counted re-attribution) had stopped being reachable.
- **`seed_reader` reports exclusion yield against its blind baseline.** Every question is
  settled exactly once, by a probe or by the exclusion inference retiring it for free, so the
  two are halves of one fixed total and the split is the only lever on premise cost. Probing
  a group of k answers in ignorance costs (k+1)/2 and retires the rest, a yield of (k-1)/2k —
  40% for k=5. Two runs differing by a probe an episode differed *here and nowhere else*, and
  without the baseline there was no way to tell an ordering win from a lucky draw.
- **`eval.sh --skip-arm-a`** runs B and F only. Arm A is the ergonomic baseline and the most
  expensive arm to run — it is the one that duplicates probes — so dropping it roughly halves
  the wall clock while iterating. The flag says loudly that criterion 1a is unscorable and the
  run cannot decide a gate.

## [0.3.2] - 2026-08-01

### Fixed
- **Conflict diagnosis now interrogates the least-corroborated question first.** v0.3.1 replaced
  the naming-dependent alphabetical order with a hash, which removed the bias but carried no
  signal — the culprit landed at position 3.75 of 5, and conflict episodes cost +1.9
  probes. A member is only as trustworthy as the question behind it was searched: beating four
  refuted rivals is a real search, being the first thing tried while the exclusion inference
  quietly retired the rest is not. Ordering by how many siblings were eliminated *on their own
  evidence* moves the culprit to position 2.38 and is still independent of what nodes are named.
  Self-play: 16.2 → **15.9** probes, better than the alphabetical order it replaced (16.0).

### Added
- `list_nodes(view=...)` — named presets (`frontier`, `settled`, `verified`, `revision`, `stale`)
  instead of hand-assembled status filters that silently return an empty table.
- `list_nodes(stale_only=True)` and a `Stale` column: VERIFIED nodes whose newest evidence names
  a commit that is no longer checked out. `context_hash` had been captured on every evidence row.
- `LogicalEvidence.source_ref` — what was actually run to produce a number (path, URL, CI run id).
- `AGENT_GUIDE.md` now documents the 3 MCP prompts and 2 resources.

### Changed
- **Schema upgrades are now migrated**: the belief state is the accumulated
  record of a month of experiments, and surviving across sessions is the entire product claim.
  Opening a database written by an older release now walks a forward migration chain, applying
  each step and its version stamp in one transaction — an interrupted upgrade rolls back to the
  version it started from rather than stranding the file between two. Every migration is written
  to the `events` audit log as `SchemaMigrated`, so "why does this differ from my backup?" is
  answerable from the same place as every other question about the workspace.
- Schema `"8"` → `"9"` adds `evidence.source_ref`; the 8→9 migration is additive
  (`ALTER TABLE ... ADD COLUMN`) and cannot lose data.
- A database written by a **newer** hypotree is refused rather than downgraded — the newer code
  may have stored things this version cannot represent, and dropping them silently is worse than
  stopping. A version with no route forward now says to keep the file and open an issue.

### Known limitation
- A conflict member skipped for want of a usable substitute is left behind the `probe_index`
  cursor and reported as cleared. Distinguishing "cleared" from "skipped" needs a settled-set
  rather than an integer cursor, so it is scoped as a schema change rather than patched.

## [0.3.1] - 2026-08-01

### Fixed
- Windows CI: e2e stdio test timeout increased to 900s for slow GitHub Actions runners
- Conflict diagnosis no longer depends on hypothesis **naming**. Members were stored
  alphabetically and interrogated in that order, so the number of diagnostic swaps was
  decided by the id prefix rather than by the evidence — a workspace could be made to
  converge faster by renaming its hypotheses. Members are now ordered weakest-claim-first
  (`confirmed_depth`, then `evidence_count`), tie-broken on a hash of the id, and frozen
  at conflict creation so `probe_index` keeps counting into a stable list.
- `AGENT_GUIDE.md`: tool table said 17 and listed 17 while the server registers 18 —
  `get_workspace_info` was missing. The identity section documented a 5-layer resolution
  including a `UV_PROJECT_ROOT` layer that no longer exists, a 64-character name limit
  that is now 128, and `~/.local/share` as the only store location; rewritten to the real
  four layers, the `[a-z0-9][a-z0-9._~-]{0,127}` rule, the cross-platform Windows
  reserved-name rejection, and `%LOCALAPPDATA%` / `hypotree --info`.

### Changed
- `HypoTreeStore.add_nogood` now preserves the caller's member order verbatim instead of
  re-sorting it. The order is the diagnosis order and `probe_index` counts into it, so
  re-sorting silently overrode the engine's ranking.

### Eval harness
- `pruned_reexecution` no longer fires when a result was *obtained before* the branch was
  pruned. It could not distinguish spending a fresh probe on a dead branch from filing a
  measurement already paid for, and the batch-probe protocol makes the latter routine —
  a single such event flipped a run's hard gate. Transcript entries now carry a UTC
  timestamp; unknown timings still fail open so fabricated results are caught.
- Status transitions are now diffed on `(status, reason)` rather than status alone.
  Ruling a substitute out retracts its exclusion and re-settles it at the same instant,
  so a status-only diff saw no change and the mechanism reported zero forever.
- Criterion 4 derives its p-value from a seeded 20 000-draw permutation test when the
  smallest expected cell falls below 5, instead of reporting a verdict from an asymptotic
  χ² that does not apply at that sparsity.

## [0.3.0] - 2026-07-31

### Added — The adversarial eval gate passed (GO)
- **18 MCP tools**: `generate_learning_path`, `get_workspace_info` added (was 17)
- 3 MCP prompts as slash commands: `/hypotree-init`, `/hypotree-next`, `/hypotree-status`
- 2 MCP resources: `hypotree://guide`, `hypotree://state`
- Server-level `instructions` delivered at MCP handshake
- `AGENT_GUIDE.md` shipped inside the package, not as a repo-root file
- CLI: `hypotree --version`, `--help`, `--info`
- Cross-platform CI: matrix over ubuntu/windows/macOS × py3.10-3.13
- Cross-platform identity resolution: uses `%LOCALAPPDATA%` on Windows, `$XDG_DATA_HOME` on Linux/macOS
- `eval/runner/config.py`: same XDG_DATA_HOME pattern fixed for cross-platform
- PyPI metadata: authors, license, classifiers, project.urls, py.typed
- `hypotree.yaml.template` updated with cross-platform storage paths

### Gate results (Run D, qwen3.6:27b-q8_0, 30 seeds)
- C1b (B vs F): 7.5 steps median reduction (28.9%), δ=0.89, 30/0/0 wins — **PASS**
- C1a (B vs A): 25.0 steps (57.5%), δ=1.0, 30/0/0 — **PASS**
- C2 (navigator): worst case 55% below greedy, 0/30 lock-ins — **PASS**
- C3 (revision): 0 pruned re-executions, 104 revision events, 14/15 conflicts resolved — **PASS**
- C4 (status utility): χ²=45.2, V=0.185, decision=KEEP — **PASS**

### Engine changes
- Differential ablation: `_eliminate_substitute` now mirrors `_confirm_substitute`
- Fabricated claim_id handling: dropped silently, result kept
- Double-counting fix: `node_created` events no longer added to `bulk_create` counts
- Group adoption denominator: compositions without exclusion_group correctly counted
- Record→dispatch fusion: `count_next_targets` parameter on `record_evidence`

## [0.2.0] - 2026-07-25

### Added
- Thompson Sampling navigator with Beta distributions
- Cascading prune, exclusion-group inference, deduction by elimination
- ATMS-style conflict sets with differential ablation
- SQLite-WAL schema v8 (9 tables, bi-temporal history)
- 16 MCP tools over stdio
- Pre-registered adversarial eval harness (30 seeds × 3 arms)

## [0.1.0] - 2026-07-20

### Added
- Core engine: HypoTreeEngine, DAG model, status lifecycle
- SQLite store with WAL mode
- Basic MCP server (stdio transport)
- Initial test suite

[0.4.0]: https://github.com/tygryso/hypotree/releases/tag/v0.4.0
[0.3.1]: https://github.com/tygryso/hypotree/releases/tag/v0.3.1
[0.3.0]: https://github.com/tygryso/hypotree/releases/tag/v0.3.0