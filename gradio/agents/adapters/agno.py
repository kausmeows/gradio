"""Adapter for agno Agents — and the last-resort fallback for simple custom
agents with a sync `.run()`."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

from gradio.agents.protocol import (
    AgentAdapter,
    AgentEvent,
    AgentFinal,
    AgentText,
    AgentThought,
    AgentToolEnd,
    AgentToolStart,
)
from gradio.exceptions import Error

# agno's RunEvent enum values (agno/run/agent.py), after stripping the
# Team prefix used by the TeamRunEvent variants (agno/run/team.py).
_TOOL_FINISHED_EVENTS = frozenset({"ToolCallCompleted", "ToolCallError"})
_CONTENT_EVENTS = frozenset({"RunContent"})
_COMPLETED_EVENTS = frozenset({"RunCompleted"})
_ERROR_EVENTS = frozenset({"RunError"})


class AgnoAdapter(AgentAdapter):
    """agno Agents: sync `.run(message, stream=True)` yielding typed events
    (`.event` name, `.content`, `.tool`).

    Deliberately matches any non-callable object with a sync `.run()`, making
    it the last-resort fallback in the registry: unknown frameworks and
    hand-rolled agents that stream plain strings, or return a result with
    `.content`, still work through its degraded paths. Frameworks whose sync
    `.run` yields a different event vocabulary (e.g. smolagents) need their
    own adapter registered ahead of this one, or their events are silently
    ignored."""

    @staticmethod
    def matches(agent: Any) -> bool:
        run = getattr(agent, "run", None)
        return (
            not inspect.isroutine(agent)
            and callable(run)
            and not inspect.iscoroutinefunction(run)
        )

    def stream(self, agent: Any, message: str, history: list) -> Iterator[AgentEvent]:  # noqa: ARG002
        result = None
        for kwargs in ({"stream": True, "stream_events": True}, {"stream": True}, {}):
            try:
                result = agent.run(message, **kwargs)
                break
            except TypeError:
                continue
        if result is None:
            return
        if isinstance(result, str) or not hasattr(result, "__iter__"):
            content = getattr(result, "content", result)
            yield AgentFinal(str(content))
            return
        for event in result:
            if isinstance(event, str):
                yield AgentText(event)
                continue
            name = getattr(event, "event", None) or type(event).__name__
            # agno Teams emit the same vocabulary with a "Team" prefix
            # (TeamRunContent, TeamToolCallStarted, ...): normalize it away.
            name = name.removeprefix("Team")
            tool = getattr(event, "tool", None)
            content = getattr(event, "content", None)
            if tool is not None and name == "ToolCallStarted":
                yield AgentToolStart(
                    name=getattr(tool, "tool_name", None) or "tool",
                    args=getattr(tool, "tool_args", None) or {},
                    id=getattr(tool, "tool_call_id", None) or id(tool),
                )
            elif tool is not None and name in _TOOL_FINISHED_EVENTS:
                result_value = (
                    content if content is not None else getattr(tool, "result", None)
                )
                yield AgentToolEnd(
                    id=getattr(tool, "tool_call_id", None) or id(tool),
                    result=result_value,
                    error=name == "ToolCallError",
                )
            elif name == "ReasoningStep" and content is not None:
                yield AgentThought(str(content))
            elif name in _CONTENT_EVENTS and isinstance(content, str):
                yield AgentText(content)
            elif name in _ERROR_EVENTS:
                raise Error(str(content) if content is not None else name)
            elif name in _COMPLETED_EVENTS and isinstance(content, str):
                yield AgentFinal(content)
