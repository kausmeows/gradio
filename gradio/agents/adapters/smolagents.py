"""Adapter for smolagents (Hugging Face) agents."""

from __future__ import annotations

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
    parse_tool_args,
)


class SmolagentsAdapter(AgentAdapter):
    """smolagents CodeAgent / ToolCallingAgent: sync `.run(task, stream=True)`
    yielding step records — PlanningStep (`.plan`), ActionStep
    (`.model_output`, `.tool_calls`, `.observations`, `.error`),
    FinalAnswerStep (`.output`), and ChatMessageStreamDelta (`.content`)
    when model streaming is enabled.

    Both smolagents and agno expose a sync `.run`, so this adapter
    disambiguates by the agent's module and must be registered before
    AgnoAdapter (the sync-`.run` fallback)."""

    @staticmethod
    def matches(agent: Any) -> bool:
        # No `not callable(agent)` guard: smolagents agents define __call__.
        return (
            callable(getattr(agent, "run", None))
            and type(agent).__module__.split(".")[0] == "smolagents"
        )

    def stream(self, agent: Any, message: str, history: list) -> Iterator[AgentEvent]:  # noqa: ARG002
        try:
            steps = agent.run(message, stream=True)
        except TypeError:
            result = agent.run(message)
            output = getattr(result, "output", result)
            yield AgentFinal(str(output))
            return
        for step in steps:
            step_type = type(step).__name__
            if step_type == "ChatMessageStreamDelta":
                yield AgentText(getattr(step, "content", None) or "")
            elif step_type == "PlanningStep":
                plan = getattr(step, "plan", None)
                if plan:
                    yield AgentThought(str(plan))
            elif step_type == "ActionStep":
                model_output = getattr(step, "model_output", None)
                if model_output:
                    yield AgentThought(str(model_output))
                tool_calls = getattr(step, "tool_calls", None) or []
                for tool_call in tool_calls:
                    yield AgentToolStart(
                        name=getattr(tool_call, "name", None) or "tool",
                        args=parse_tool_args(getattr(tool_call, "arguments", None)),
                        id=getattr(tool_call, "id", None),
                    )
                error = getattr(step, "error", None)
                observations = getattr(step, "observations", None)
                if tool_calls and (observations is not None or error is not None):
                    yield AgentToolEnd(
                        id=getattr(tool_calls[0], "id", None),
                        result=str(error) if error is not None else observations,
                        error=error is not None,
                    )
                elif error is not None:
                    # A failed step without a tool call: surface it as a
                    # panel — smolagents retries, so this must not abort.
                    yield AgentThought(f"⚠️ {error}")
            elif step_type == "FinalAnswerStep":
                output = getattr(step, "output", None)
                if output is None:
                    output = getattr(step, "final_answer", None)
                if output is not None:
                    yield AgentFinal(str(output))
