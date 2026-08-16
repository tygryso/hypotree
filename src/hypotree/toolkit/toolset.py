"""Embed the belief state in a Python host, without a transport in between.

``HypoTreeToolset`` is the object a host holds: it owns an engine, hands out
tool schemas in the shape the host's model already speaks, and executes a call
by name. Three lines to wire in, against a package that previously could only be
reached by speaking JSON-RPC to a subprocess.

Why in-process rather than MCP, for a Python host:

- **Both sides are already Python and already Pydantic.** An MCP subprocess adds
  a process boundary, a JSON round-trip and an MCP client dependency to move a
  dict between two objects in the same interpreter.
- **It is testable without a network, a subprocess or a model.** The engine runs
  against a temporary SQLite file in milliseconds, so a host can drive the whole
  create → dispatch → record → conclude loop in a unit test. That matters more
  than it sounds: it is the difference between validating an integration for
  free and validating it by spending on inference.
- **MCP remains the answer for everything else.** A non-Python host, a host in a
  different virtualenv, or any client that already speaks MCP should keep using
  the server. This is an addition to the surface, not a replacement: both
  transports project the same specs and route through the same dispatch, so
  neither can drift from the other.

The one rule worth stating: ``call`` returns a string and does not raise. A host
loop that dies on a malformed argument dict has turned a recoverable mistake
into a lost session, and every agent eventually sends a malformed argument dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

from hypotree.engine import HypoTreeEngine
from hypotree.navigator.sampler import DEFAULT_LEASE_TTL_S
from hypotree.toolkit.dispatch import dispatch
from hypotree.toolkit.specs import ToolSpec, get_spec, openai_tools, select_specs


class HypoTreeToolset:
    """A belief state plus its tool surface, ready to embed.

    Construct it with a database path, or wrap an engine you already hold with
    ``from_engine``. Either way the toolset owns the tool schemas and the
    routing; the engine underneath stays reachable as ``.engine`` for a host
    that wants to read the belief state directly rather than through a tool.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        project_path: Path | str | None = None,
        rng_seed: int | None = None,
        lease_ttl_s: int = DEFAULT_LEASE_TTL_S,
        cost_aware: bool = False,
        read_only: bool = False,
        preset: str = "full",
        include: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._engine = HypoTreeEngine(
            db_path,
            rng_seed=rng_seed,
            lease_ttl_s=lease_ttl_s,
            project_path=project_path,
            read_only=read_only,
            cost_aware=cost_aware,
        )
        self._owns_engine = True
        # A read-only store cannot serve a mutating tool, so advertising one
        # would be an invitation to an error the host cannot recover from.
        self._set_surface(
            select_specs(preset=preset, include=include, exclude=exclude, read_only=read_only)
        )

    def _set_surface(self, specs: tuple[ToolSpec, ...]) -> None:
        """Fix the exposed surface, and the membership set `call` checks per call."""
        self._specs = specs
        self._exposed = frozenset(spec.name for spec in specs)

    @classmethod
    def from_engine(
        cls,
        engine: HypoTreeEngine,
        *,
        preset: str = "full",
        include: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        read_only: bool = False,
    ) -> HypoTreeToolset:
        """Wrap an engine the caller already owns.

        The caller keeps responsibility for closing it: two objects that both
        believe they own one SQLite connection is how a host ends up closing a
        store another part of itself is still reading.
        """
        toolset = cls.__new__(cls)
        toolset._engine = engine
        toolset._owns_engine = False
        toolset._set_surface(
            select_specs(preset=preset, include=include, exclude=exclude, read_only=read_only)
        )
        return toolset

    # -- introspection --------------------------------------------------------

    @property
    def engine(self) -> HypoTreeEngine:
        """The underlying engine, for reads a tool call would only wrap."""
        return self._engine

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """The tools this toolset exposes, in a stable order."""
        return self._specs

    @property
    def tool_names(self) -> tuple[str, ...]:
        """The names of the exposed tools, in the same stable order."""
        return tuple(spec.name for spec in self._specs)

    @property
    def mutating_tool_names(self) -> frozenset[str]:
        """Exposed tools that change the belief state.

        What a host with its own gating policy keys off: hyporun puts these
        behind its thinking gate and lets the reads through as free sensors,
        which is the same asymmetry hypotree's own dashboard draws between a
        read model and a directive.
        """
        return frozenset(spec.name for spec in self._specs if spec.mutates)

    def tools(self) -> list[dict[str, Any]]:
        """The exposed tools in OpenAI function-calling form."""
        return [spec.as_openai_tool() for spec in self._specs]

    def is_mutation(self, name: str) -> bool:
        """Whether calling ``name`` would change the belief state."""
        return get_spec(name).mutates

    # -- execution ------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Execute one tool call and return its result as a JSON string.

        **Never raises.** A refusal comes back as ``{"error": ...}`` with the
        tool and arguments echoed, because every one of the failures worth
        designing for here is recoverable by the model that caused it: a
        misspelled node id, a missing ``success``, an argument dict that is not
        an object. Propagating those as exceptions would end a session over a
        mistake the next turn could have corrected — and the engine's own error
        messages are written to be read by a model and acted on.

        A tool the host chose not to expose is refused by name rather than
        executed, so narrowing the surface is an actual boundary and not merely
        a suggestion about which schemas to advertise.
        """
        if arguments is not None and not isinstance(arguments, dict):
            return json.dumps(
                {
                    "error": f"arguments must be a JSON object, got {type(arguments).__name__}",
                    "tool": name,
                }
            )
        args: dict[str, Any] = arguments or {}
        if name not in self._exposed:
            known = ", ".join(self.tool_names)
            hidden = " (it exists but is not exposed by this toolset)" if _known(name) else ""
            return json.dumps(
                {"error": f"unknown tool {name!r}{hidden}; available: {known}", "tool": name}
            )
        try:
            # Copied because dispatch is free to consume what it is handed, and
            # the caller's dict is echoed back in the error path below.
            result = dispatch(self._engine, name, dict(args))
        except Exception as exc:  # noqa: BLE001 — see the docstring: this is the contract.
            return json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "tool": name, "arguments": args},
                default=str,
            )
        return json.dumps(result, default=str)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Close the engine, if this toolset opened it."""
        if self._owns_engine:
            self._engine.close()

    def __enter__(self) -> HypoTreeToolset:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _known(name: str) -> bool:
    """Whether hypotree has a tool by this name at all, exposed here or not."""
    try:
        get_spec(name)
    except KeyError:
        return False
    return True


__all__ = ["HypoTreeToolset", "openai_tools"]
