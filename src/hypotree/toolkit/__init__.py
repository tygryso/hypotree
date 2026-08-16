"""The tool surface, described once and reachable from any host.

Three pieces, none of which import a transport:

- ``specs`` — every tool, its schema, whether it mutates and whether it is
  essential. The single source of truth the MCP server and any embedding host
  both project from.
- ``dispatch`` — name plus arguments to engine call, returning plain data.
- ``toolset`` — ``HypoTreeToolset``, the object a Python host holds.
"""

from hypotree.toolkit.dispatch import dashboard_url, dispatch, publish_dashboard_url
from hypotree.toolkit.specs import (
    TOOL_NAMES,
    TOOL_SPECS,
    ToolSpec,
    get_spec,
    hypothesis_item_schema,
    openai_tools,
    select_specs,
)
from hypotree.toolkit.toolset import HypoTreeToolset

__all__ = [
    "TOOL_NAMES",
    "TOOL_SPECS",
    "HypoTreeToolset",
    "ToolSpec",
    "dashboard_url",
    "dispatch",
    "get_spec",
    "hypothesis_item_schema",
    "openai_tools",
    "publish_dashboard_url",
    "select_specs",
]
