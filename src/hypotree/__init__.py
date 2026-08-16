"""hypotree — persistent, self-revising hypothesis-DAG orchestrator.

Memory that forgets: a belief state that records hypotheses, prunes what the
evidence killed, and deduces what it no longer has to test.

Two ways in, both projections of one tool surface.

**Embedded in a Python host** — no transport, no subprocess, no MCP client::

    from hypotree import HypoTreeToolset

    with HypoTreeToolset("beliefs.db", preset="essential") as ht:
        tools = ht.tools()                      # OpenAI function-calling schemas
        result = ht.call("get_next_targets", {"count": 1})

**As an MCP server** — for any client that already speaks it::

    $ hypotree

The engine is importable on its own when a host wants typed results rather than
JSON strings::

    from hypotree import HypoTreeEngine

    engine = HypoTreeEngine("beliefs.db")
    targets = engine.get_next_targets(count=1)   # list[TargetResponse]

Everything re-exported here is the public API. Reaching through a submodule path
for something not listed in ``__all__`` is reaching for an internal.
"""

from hypotree.engine import (
    ClaimError,
    EvidenceReport,
    GoalDependencyError,
    GoalEvidenceError,
    HypoTreeEngine,
    NodeNotFoundError,
)
from hypotree.models import (
    Edge,
    EdgeType,
    Evidence,
    InfraError,
    LogicalEvidence,
    Node,
    Status,
)
from hypotree.toolkit import (
    TOOL_NAMES,
    TOOL_SPECS,
    HypoTreeToolset,
    ToolSpec,
    openai_tools,
    select_specs,
)

__version__ = "0.6.0"

__all__ = [
    "TOOL_NAMES",
    "TOOL_SPECS",
    "ClaimError",
    "Edge",
    "EdgeType",
    "Evidence",
    "EvidenceReport",
    "GoalDependencyError",
    "GoalEvidenceError",
    "HypoTreeEngine",
    "HypoTreeToolset",
    "InfraError",
    "LogicalEvidence",
    "Node",
    "NodeNotFoundError",
    "Status",
    "ToolSpec",
    "__version__",
    "openai_tools",
    "select_specs",
]
