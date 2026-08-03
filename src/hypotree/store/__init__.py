"""Store package — SQLite-WAL source of truth, schema, workspace identity."""

from hypotree.store.identity import store_root, workspace_id
from hypotree.store.schema import BASE_DDL, BASELINE_VERSION, SCHEMA_VERSION
from hypotree.store.store import HypoTreeStore, SchemaVersionError

__all__ = [
    "HypoTreeStore",
    "BASELINE_VERSION",
    "BASE_DDL",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    "store_root",
    "workspace_id",
]
