"""Turning a belief state into the shapes a viewer needs, without touching it.

Everything here reads. The store is opened read-only, the layout is computed
from the topology rather than asked of a browser, and the glow on each node is
the actual probability Thompson Sampling picks it next rather than a stand-in
for it.

Three things are deliberately not done the obvious way:

* **`get_next_targets(dry_run=True)` is never called.** It looks like the right
  way to populate a frontier panel and it is not read-only — it calls
  `expire_stale_claims`, which writes. The ranking is computed from a snapshot
  instead.
* **Layout is server-side.** Layer assignment plus a barycentre pass is a dozen
  lines against `networkx`, which is already a dependency, and it removes the
  single largest asset a browser would otherwise have to download. It is also
  deterministic, so it can be tested here rather than eyeballed there.
* **Everything is cached on the revision.** `events.seq` advances exactly when
  the belief state changes, so it is both the change signal pushed to clients
  and the cache key. Without it a viewer holding a scrubber would recompute the
  whole graph per frame, which is the shape of the cost the engine spent a phase
  removing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from hypotree.engine import HypoTreeEngine
from hypotree.models.edge import EdgeType
from hypotree.models.status import posterior_mean
from hypotree.navigator.sampler import effective_posterior, live_group_counts

# Vertical spacing between the sub-rows a wide layer wraps into, as a fraction of
# the gap between layers. Below one, so wrapped siblings still read as one band
# rather than as two separate depths.
_SUB_ROW_GAP = 0.62

# Draws used to estimate each frontier node's chance of being selected next.
# The estimate's standard error is ~sqrt(p(1-p)/N); at 2000 that is under 1.2
# percentage points, which is finer than the eye can read off a glow.
_SELECT_DRAWS = 2000


@dataclass
class Snapshot:
    """One fully-resolved view of the belief state, keyed by the revision it came from."""

    revision: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    scope: list[str] | None
    at: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    # Why a goal filter cannot be satisfied, when it cannot. `"inverted"` means
    # the work hangs *off* the goal instead of supporting it — the mistake a model
    # makes when it reads a goal as a container — and `"unwired"` means there is
    # no work attached at all. The fixes differ, so a single flag would leave the
    # reader to guess which one they are looking at.
    unwired_goal: str | None = None
    goal_wiring: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "at": self.at,
            "scope": self.scope,
            "nodes": self.nodes,
            "edges": self.edges,
            "stats": self.stats,
            "unwired_goal": self.unwired_goal,
            "goal_wiring": self.goal_wiring,
        }


def _layer_positions(
    node_ids: list[str], edges: list[tuple[str, str]]
) -> dict[str, tuple[float, float]]:
    """Assign each node a layer and an order within it, then hand back coordinates.

    A hypothesis DAG reads top-down: premises above, the combinations resting on
    them below. Longest-path layering gives exactly that, and one barycentre
    sweep is enough to stop the edges crossing into spaghetti at the sizes a
    belief state actually reaches. Deliberately not force-directed — a layout
    that moves every time it is opened cannot be recognised across sessions, and
    recognising it is most of what a viewer is for.
    """
    parents: dict[str, list[str]] = {n: [] for n in node_ids}
    children: dict[str, list[str]] = {n: [] for n in node_ids}
    known = set(node_ids)
    for src, dst in edges:
        if src in known and dst in known:
            parents[dst].append(src)
            children[src].append(dst)

    # Longest path from any root. Iterative to a fixed point: a DAG converges in
    # at most depth passes, and the guard stops a cycle (which cannot occur here,
    # but a corrupt file should not hang a viewer) from spinning forever.
    layer = dict.fromkeys(node_ids, 0)
    for _ in range(len(node_ids) + 1):
        changed = False
        for node in node_ids:
            want = max((layer[p] + 1 for p in parents[node]), default=0)
            if want != layer[node]:
                layer[node] = want
                changed = True
        if not changed:
            break

    by_layer: dict[int, list[str]] = {}
    for node in node_ids:
        by_layer.setdefault(layer[node], []).append(node)
    for nodes in by_layer.values():
        nodes.sort()

    order = {n: i for nodes in by_layer.values() for i, n in enumerate(nodes)}
    # One barycentre sweep: put each node near the average position of its
    # parents. Enough to untangle the common shapes, cheap enough to be free.
    for depth in sorted(by_layer)[1:]:
        row = by_layer[depth]
        row.sort(
            key=lambda n: (
                sum(order[p] for p in parents[n]) / len(parents[n]) if parents[n] else 0.0,
                n,
            )
        )
        for i, n in enumerate(row):
            order[n] = i

    positions: dict[str, tuple[float, float]] = {}
    # A layer of competing answers is naturally wide — five values on each of
    # five axes puts twenty-five nodes on one line — and a drawing 25 wide by 3
    # deep fits the viewport at a scale where nothing is legible. Wide layers
    # wrap into sub-rows, which keeps the reading order and the top-down
    # semantics while bringing the aspect ratio back to something a screen is
    # shaped like. The width is derived from the graph's own size so the layout
    # is still a pure function of the topology: same graph, same picture.
    wrap = max(6, math.ceil(math.sqrt(len(node_ids))) * 2)
    y = 0.0
    for depth in sorted(by_layer):
        row = by_layer[depth]
        sub_rows = math.ceil(len(row) / wrap) or 1
        width = math.ceil(len(row) / sub_rows)
        for i, node in enumerate(row):
            col, sub = i % width, i // width
            # Centred on x=0 so a viewport can frame the graph without knowing
            # its width in advance.
            positions[node] = (float(col) - (width - 1) / 2.0, y + sub * _SUB_ROW_GAP)
        y += 1.0 + (sub_rows - 1) * _SUB_ROW_GAP
    return positions


def _selection_probabilities(
    frontier: list[Any],
    all_nodes: list[Any],
    rng: np.random.Generator,
    *,
    eliminated_ids: set[str] | None = None,
    directives: dict[str, Any] | None = None,
) -> dict[str, float]:
    """How likely the navigator is to hand out each frontier node next.

    Thompson Sampling draws one theta per candidate from its posterior and takes
    the argmax, so the chance of being selected *is* the chance of winning that
    draw. Estimating it by doing the draw many times reports the real quantity
    rather than a monotone stand-in like the posterior mean — which matters,
    because the first thing anyone asks about a glowing node is what the
    brightness actually means.

    The directives are applied first, because they are absolute rather than
    probabilistic: a suspended node is never offered, and if anything is pinned
    the navigator offers *only* the pinned set. Leaving them out made the panel
    wrong in the one situation the reader has most reason to check it — straight
    after using the dashboard's own pin or hold-back button, which is the only
    way those directives are ever set.
    """
    if not frontier:
        return {}
    directives = directives or {}

    def _mode(node_id: str) -> str:
        # `get_directives` hands back sqlite3.Row, which indexes but has no .get.
        row = directives.get(node_id)
        return "" if row is None else str(row["mode"] or "")

    offered = [n for n in frontier if _mode(n.id) != "suspend"]
    pinned = [n for n in offered if _mode(n.id) == "pin"]
    if pinned:
        offered = pinned
    if not offered:
        return {n.id: 0.0 for n in frontier}

    counts = live_group_counts(all_nodes, eliminated_ids)
    params = np.array([effective_posterior(n, counts) for n in offered], dtype=float)
    # (draws x candidates); argmax per row is one full Thompson selection.
    draws = rng.beta(params[:, 0], params[:, 1], size=(_SELECT_DRAWS, len(offered)))
    wins = np.bincount(draws.argmax(axis=1), minlength=len(offered))
    probs = {n.id: 0.0 for n in frontier}
    probs.update({n.id: float(w) / _SELECT_DRAWS for n, w in zip(offered, wins, strict=True)})
    return probs


class ReadModel:
    """A read-only window onto one belief state.

    Holds an engine opened read-only rather than reimplementing its reads: the
    scope primitive, the frontier rule and the narrative are all logic that
    already exists and is already tested, and a second opinion about any of them
    is a bug waiting to be found by a user.
    """

    def __init__(self, db_path: Any, rng_seed: int | None = 0) -> None:
        self._engine = HypoTreeEngine(db_path, rng_seed=rng_seed, read_only=True)
        self._rng = np.random.default_rng(rng_seed)
        self._cache: dict[tuple[Any, ...], Snapshot] = {}

    def close(self) -> None:
        self._engine.close()

    @property
    def store(self) -> Any:
        return self._engine._store

    def revision(self) -> int:
        return self.store.latest_event_seq()

    # -- meta ------------------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        """Identity, plus the goal list the viewer's selector is built from."""
        nodes = self.store.get_all_nodes()
        goals = [
            {
                "id": n.id,
                "statement": n.statement,
                "met": self._engine.goal_achieved(n),
                "target_metric": n.target_metric,
            }
            for n in nodes
            if n.is_goal
        ]
        return {
            "revision": self.revision(),
            "db_path": str(self.store._db_path),
            "node_count": len(nodes),
            "goals": goals,
        }

    # -- graph -----------------------------------------------------------------

    def _view_scope(self, goal_id: str | None) -> set[str] | None:
        """What to *draw* for one goal — deliberately wider than what to *select*.

        The navigator's `_goal_scope` is the set of nodes that could satisfy the
        goal: its DEPENDENCY ancestry plus the exclusion siblings that answer
        those questions. That is the right set to dispatch from, and the wrong
        set to draw.

        Agents routinely wire a goal the other way round — `goal -> phase0 ->
        work`, reading it as "the goal contains this work" rather than "the goal
        depends on it". The engine cannot reach such a goal and says so, but the
        nodes are still there and still the user's, and filtering them out left a
        single dot on screen next to a message about wiring while a whole tree
        hung off the node in question. A viewer that hides what you built to make
        a point about how you built it is not making the point.

        So the view walks both ways: the selection scope, plus everything
        downstream of the goal by any edge type, plus the exclusion siblings of
        the result. The unreachability is reported alongside the drawing instead
        of in place of it.
        """
        scope = self._engine._resolve_goal_scope(goal_id)
        if scope is None or goal_id is None:
            return scope
        scope = set(scope) | self._engine._graph.descendants(goal_id)
        groups = {
            node.exclusion_group
            for node in (self.store.get_node(nid) for nid in scope)
            if node is not None and node.exclusion_group
        }
        for group in groups:
            scope.update(n.id for n in self.store.get_nodes_in_exclusion_group(group))
        return scope

    def graph(self, goal_id: str | None = None, at: str | None = None) -> Snapshot:
        """The laid-out graph, live or reconstructed at an instant."""
        key = ("graph", self.revision(), goal_id, at)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        snapshot = self._build_graph(goal_id, at)
        # Only the newest revision is worth keeping: a scrubber walks `at` over
        # one revision, and a mutation invalidates everything anyway.
        self._cache = {key: snapshot}
        return snapshot

    def _build_graph(self, goal_id: str | None, at: str | None) -> Snapshot:
        self._engine._sync_graph_from_store()
        scope = self._view_scope(goal_id)
        nodes = [n for n in self.store.get_all_nodes() if scope is None or n.id in scope]
        historical = _parse_at(at)

        status_at: dict[str, tuple[str, str]] = {}
        posterior_at: dict[str, tuple[float, float]] = {}
        if historical is not None:
            status_at = self._status_at(historical)
            posterior_at = self._posterior_at(historical)
            # A node that did not exist yet has no interval covering the instant.
            nodes = [n for n in nodes if n.id in status_at]

        directives = self.store.get_directives()
        # When the current status was entered. The cards show it beside the
        # creation time, because "confirmed" and "confirmed three days ago and
        # untouched since" are different situations and the status alone cannot
        # tell them apart. Rewound, it is the start of the interval covering that
        # instant — the same question asked of a different present.
        settled_at: dict[str, str] = {}
        for row in self.store.get_all_status_history():
            covering = (
                _covers(row, historical) if historical is not None else row["valid_to"] is None
            )
            if covering:
                settled_at[str(row["node_id"])] = str(row["valid_from"])
        node_ids = [n.id for n in nodes]
        known = set(node_ids)
        edge_rows = [
            r
            for r in self.store.get_edge_rows()
            if r["src"] in known
            and r["dst"] in known
            and (historical is None or _edge_existed(r, historical))
        ]
        positions = _layer_positions(node_ids, [(r["src"], r["dst"]) for r in edge_rows])

        # The glow is only meaningful for the live frontier; reconstructing a
        # historical one would need the claim table's own history, which is not
        # kept. Scrubbing therefore shows what was believed, not what was queued.
        p_select: dict[str, float] = {}
        if historical is None:
            frontier = [n for n in self._engine._frontier_nodes() if n.id in known]
            live_nodes = self.store.get_all_nodes()
            p_select = _selection_probabilities(
                frontier,
                live_nodes,
                self._rng,
                eliminated_ids=self._engine._eliminated_ids(live_nodes),
                directives=self.store.get_directives(),
            )

        out_nodes = []
        for node in nodes:
            status = (
                status_at.get(node.id, (node.status.value, ""))[0]
                if status_at
                else (node.status.value)
            )
            alpha, beta = posterior_at.get(node.id, (node.alpha, node.beta))
            directive = directives.get(node.id)
            x, y = positions.get(node.id, (0.0, 0.0))
            out_nodes.append(
                {
                    "id": node.id,
                    "statement": node.statement,
                    "status": status,
                    "is_goal": node.is_goal,
                    "exclusion_group": node.exclusion_group,
                    "alpha": alpha,
                    "beta": beta,
                    "posterior_mean": posterior_mean(alpha, beta),
                    "evidence_count": node.evidence_count,
                    "p_select": round(p_select.get(node.id, 0.0), 4),
                    "directive": directive["mode"] if directive is not None else None,
                    "created_at": node.created_at.isoformat() if node.created_at else None,
                    "settled_at": settled_at.get(node.id),
                    "x": x,
                    "y": y,
                }
            )

        stats = {"nodes": len(out_nodes), "edges": len(edge_rows)}
        for entry in out_nodes:
            stats[entry["status"]] = int(stats.get(entry["status"], 0)) + 1
        # The engine's own complaint, reported beside the drawing rather than in
        # place of it: a goal wired `goal -> work` instead of `work -> goal` can
        # never be satisfied, and its tree is still worth looking at while that
        # is being fixed.
        unwired = None
        wiring = None
        if goal_id is not None and not self._engine._graph.parents(goal_id, EdgeType.DEPENDENCY):
            unwired = goal_id
            wiring = "inverted" if self._engine._graph.children(goal_id) else "unwired"
        return Snapshot(
            revision=self.revision(),
            nodes=out_nodes,
            edges=[{"src": r["src"], "dst": r["dst"], "type": r["type"]} for r in edge_rows],
            scope=sorted(scope) if scope is not None else None,
            at=at,
            stats=stats,
            unwired_goal=unwired,
            goal_wiring=wiring,
        )

    def _status_at(self, when: datetime) -> dict[str, tuple[str, str]]:
        """Every node's status and reason at an instant, from the SCD2 intervals."""
        out: dict[str, tuple[str, str]] = {}
        for row in self.store.get_all_status_history():
            if _covers(row, when):
                out[str(row["node_id"])] = (str(row["status"]), str(row["reason"] or ""))
        return out

    def _posterior_at(self, when: datetime) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for node in self.store.get_all_nodes():
            for row in self.store.get_posterior_history(node.id):
                if _covers(row, when):
                    out[node.id] = (float(row["alpha"]), float(row["beta"]))
        return out

    # -- panels ----------------------------------------------------------------

    def node_detail(self, node_id: str) -> dict[str, Any] | None:
        """Everything behind one node, including what paid for it."""
        node = self.store.get_node(node_id)
        if node is None:
            return None
        directive = self.store.get_directives().get(node_id)
        history = self.store.get_status_history(node_id)
        return {
            "node": {
                "id": node.id,
                "statement": node.statement,
                "status": node.status.value,
                "is_goal": node.is_goal,
                "exclusion_group": node.exclusion_group,
                "alpha": node.alpha,
                "beta": node.beta,
                "posterior_mean": posterior_mean(node.alpha, node.beta),
                "evidence_count": node.evidence_count,
                "parent_ids": self.store.get_parent_ids(node.id),
                "created_at": node.created_at.isoformat() if node.created_at else None,
                # When the current status was entered — "confirmed" and
                # "confirmed last week and untouched since" are different
                # situations that the status alone cannot tell apart.
                "settled_at": str(history[-1]["valid_from"]) if history else None,
                "reason": str(history[-1]["reason"] or "") if history else "",
            },
            "directive": dict(directive) if directive is not None else None,
            "evidence": [
                {
                    "kind": r["kind"],
                    "success": r["success"],
                    "depth": r["depth"],
                    "context_hash": r["context_hash"],
                    "git_branch": r["git_branch"],
                    "source_ref": r["source_ref"],
                    "duration_s": r["duration_s"],
                    "artifacts": json.loads(r["artifacts"] or "[]"),
                    "notes": r["notes"],
                    "recorded_at": r["recorded_at"],
                }
                for r in self.store.get_evidence_for_node(node_id)
            ],
            "status_history": [
                {
                    "status": r["status"],
                    "reason": r["reason"],
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                }
                for r in self.store.get_status_history(node_id)
            ],
        }

    def frontier(self, goal_id: str | None = None, k: int = 5) -> dict[str, Any]:
        """The top candidates by their real chance of being dispatched next."""
        self._engine._sync_graph_from_store()
        scope = self._engine._resolve_goal_scope(goal_id)
        nodes = self.store.get_all_nodes()
        frontier = [n for n in self._engine._frontier_nodes() if scope is None or n.id in scope]
        directives = self.store.get_directives()
        probs = _selection_probabilities(
            frontier,
            nodes,
            self._rng,
            eliminated_ids=self._engine._eliminated_ids(nodes),
            directives=directives,
        )
        ranked = sorted(frontier, key=lambda n: -probs.get(n.id, 0.0))[: max(k, 0)]
        return {
            "revision": self.revision(),
            "candidates": [
                {
                    "node_id": n.id,
                    "statement": n.statement,
                    "status": n.status.value,
                    "p_select": round(probs.get(n.id, 0.0), 4),
                    "posterior_mean": posterior_mean(n.alpha, n.beta),
                    "exclusion_group": n.exclusion_group,
                    "directive": (directives[n.id]["mode"] if n.id in directives else None),
                }
                for n in ranked
            ],
        }

    def counterfactual(self, goal_id: str | None = None, k: int = 5) -> dict[str, Any]:
        """What it would take for the current conclusion to be wrong.

        The panel a reviewer reads before the graph: a belief state that can
        only show what it thinks is a report, and one that can name the
        experiment that would overturn it is an instrument.
        """
        self._engine._sync_graph_from_store()  # noqa: SLF001
        entries = self._engine.what_would_change_my_mind(goal_id, limit=max(k, 0))
        return {
            "revision": self.revision(),
            "beliefs": [e.model_dump() for e in entries],
        }

    def learning_path(
        self,
        goal_id: str | None = None,
        limit: int = 200,
        at: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """The narrative, straight from the engine so it cannot drift from the tool.

        With ``since`` it is a diff instead: what settled or was withdrawn between
        two points on the same timeline the scrubber already walks.
        """
        return self._engine.generate_learning_path(
            limit=limit, goal_id=goal_id, as_of=_parse_at(at), since=_parse_at(since, "since")
        ).model_dump(mode="json")

    def timeline(self, goal_id: str | None = None) -> dict[str, Any]:
        """Every status change in order — the ticks a scrubber steps through.

        Scoped to what is *drawn*, not to what the navigator would select: the
        scrubber moves the picture, so a tick it cannot show is a step that does
        nothing and a node on screen whose changes are missing is a gap.
        """
        self._engine._sync_graph_from_store()
        scope = self._view_scope(goal_id)
        ticks = []
        for row in self.store.get_all_status_history():
            node_id = str(row["node_id"])
            if scope is not None and node_id not in scope:
                continue
            ticks.append(
                {
                    "t": row["valid_from"],
                    "node_id": node_id,
                    "status": row["status"],
                    "reason": row["reason"] or "",
                }
            )
        ticks.sort(key=lambda e: str(e["t"]))
        return {
            "revision": self.revision(),
            "from": ticks[0]["t"] if ticks else None,
            "to": ticks[-1]["t"] if ticks else None,
            "ticks": ticks,
        }


def _parse_at(at: str | None, field: str = "at") -> datetime | None:
    if not at:
        return None
    raw = at
    # `+` is the query-string encoding of a space, so a URL carrying an offset
    # verbatim — `?at=2026-08-07T09:00:00+00:00` — arrives here with a space
    # where the sign belongs. The browser client encodes it, but a hand-written
    # URL or a copied timestamp does not, and the endpoint answered 400 for a
    # perfectly good instant. A space in that position is unambiguous.
    if len(raw) > 6 and raw[-6] == " " and raw[-3] == ":":
        raw = raw[:-6] + "+" + raw[-5:]
    # Python 3.10's parser rejects the `Z` suffix, which is exactly what a
    # timestamp copied out of a JSON payload carries.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"`{field}` must be an ISO-8601 timestamp, got {at!r}") from exc
    # Stored instants are UTC-aware; a hand-typed naive one would raise on the
    # first comparison rather than mean anything different.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _covers(row: Any, when: datetime) -> bool:
    """True when an SCD2 interval is open over ``when``."""
    start = datetime.fromisoformat(str(row["valid_from"]))
    if start > when:
        return False
    end = row["valid_to"]
    return end is None or datetime.fromisoformat(str(end)) > when


def _edge_existed(row: Any, when: datetime) -> bool:
    """Was this edge there at ``when``?

    Two ways to have no timestamp, and both mean the same thing to a viewer: an
    edge written before the column existed, and one in a database stamped with an
    unreleased schema by an earlier development build. Treated as always having
    been there, so a replay shows the graph slightly too wired rather than
    refusing to draw.
    """
    # `in row` on a sqlite3.Row tests values, not column names, so the column
    # list has to be pulled out first.
    columns = row.keys()
    created = row["created_at"] if "created_at" in columns else None
    if created is None:
        return True
    return datetime.fromisoformat(str(created)) <= when


__all__ = ["ReadModel", "Snapshot", "_layer_positions", "_selection_probabilities"]
