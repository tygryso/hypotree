"""Read-only observability for a live belief state.

Two pieces, deliberately separable: `readmodel` turns the database into the
shapes a viewer needs (laid-out graph, selection probabilities, a timeline), and
`server` puts them on a localhost socket. Neither can mutate the belief state —
the store underneath is opened `mode=ro`, so a read path that is wrong fails
loudly rather than quietly changing what someone is watching.
"""

from hypotree.dashboard.readmodel import ReadModel, Snapshot
from hypotree.dashboard.server import DashboardServer, choose_port

__all__ = ["DashboardServer", "ReadModel", "Snapshot", "choose_port"]
