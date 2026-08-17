"""Adapter for pydantic-ai style agents (async `.run()` with `.run_sync()`)."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

from gradio.agents.protocol import (
    AgentAdapter,
    AgentEvent,
    AgentFinal,
    AgentToolEnd,
    AgentToolStart,
    parse_tool_args,
)
from gradio.exceptions import Error


class PydanticAIAdapter(AgentAdapter):
    """pydantic-ai style agents: async `.run()` with a sync `.run_sync()`
    returning a result object with `.output` (or legacy `.data`) and a
    `.new_messages()` history containing tool-call / tool-return parts."""

    @staticmethod
    def matches(agent: Any) -> bool:
        return not inspect.isroutine(agent) and (
            callable(getattr(agent, "run_sync", None))
            or inspect.iscoroutinefunction(getattr(agent, "run", None))
        )

    def stream(self, agent: Any, message: str, history: list) -> Iterator[AgentEvent]:  # noqa: ARG002
        run_sync = getattr(agent, "run_sync", None)
        if not callable(run_sync):
            raise Error(
                f"{type(agent).__name__}.run is async and no run_sync() is available."
            )
        result = run_sync(message)
        new_messages = getattr(result, "new_messages", None)
        if callable(new_messages):
            for msg in new_messages():
                for part in getattr(msg, "parts", None) or []:
                    kind = getattr(part, "part_kind", "")
                    if kind == "tool-call":
                        yield AgentToolStart(
                            name=getattr(part, "tool_name", None) or "tool",
                            args=parse_tool_args(getattr(part, "args", None)),
                            id=getattr(part, "tool_call_id", None),
                        )
                    elif kind == "tool-return":
                        yield AgentToolEnd(
                            id=getattr(part, "tool_call_id", None),
                            result=getattr(part, "content", None),
                        )
        output = getattr(result, "output", None)
        if output is None:
            output = getattr(result, "data", None)
        if output is not None:
            yield AgentFinal(str(output))
