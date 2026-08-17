"""Tests for gradio.agents: adapter resolution and event mapping.

All framework fakes replicate the *object shapes* the real frameworks
produce (type names, attribute names, module paths), since those are
exactly the contracts the duck-typed adapters dispatch on. No agent
framework is required to run these tests.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import gradio as gr
from gradio.agents import (
    AgentAdapter,
    AgnoAdapter,
    LangChainAdapter,
    PydanticAIAdapter,
    SmolagentsAdapter,
    register_adapter,
    resolve_adapter,
)
from gradio.agents import _ADAPTERS

# --------------------------------------------------------------------------
# agno-shaped fakes
# --------------------------------------------------------------------------


@dataclass
class AgnoTool:
    tool_name: str
    tool_args: dict
    tool_call_id: str
    result: Any = None


@dataclass
class AgnoEvent:
    event: str
    content: Any = None
    tool: Optional[AgnoTool] = None


class FakeAgnoTeam:
    """agno Teams emit the same events with a Team prefix (TeamRunContent,
    TeamToolCallStarted, ...) — the adapter must normalize it away."""

    def run(self, message, stream=False, stream_events=False):
        assert stream
        tool = AgnoTool("get_weather", {"city": "SF"}, "call_1", result="18°C, foggy")
        yield AgnoEvent(event="TeamRunStarted")
        yield AgnoEvent(
            event="TeamReasoningStep", content="Delegating to the weather member."
        )
        yield AgnoEvent(event="TeamToolCallStarted", tool=tool)
        yield AgnoEvent(event="TeamToolCallCompleted", tool=tool, content=tool.result)
        yield AgnoEvent(
            event="TeamRunContent", content="The team says: 18°C and foggy."
        )
        yield AgnoEvent(event="TeamRunCompleted")


class FakeAgnoAgent:
    def run(self, message, stream=False, stream_events=False):
        if not stream:
            return AgnoEvent(event="RunCompleted", content=f"echo: {message}")
        return self._stream(message)

    def _stream(self, message):
        yield AgnoEvent(event="RunStarted")
        yield AgnoEvent(event="ReasoningStep", content="I should check the weather.")
        tool = AgnoTool("get_weather", {"city": "SF"}, "call_1")
        yield AgnoEvent(event="ToolCallStarted", tool=tool)
        tool.result = "18°C, foggy"
        yield AgnoEvent(event="ToolCallCompleted", tool=tool, content=tool.result)
        for chunk in ("The weather ", "is 18°C and foggy."):
            yield AgnoEvent(event="RunContent", content=chunk)
        yield AgnoEvent(event="RunCompleted")


# --------------------------------------------------------------------------
# LangChain / LangGraph-shaped fakes
# --------------------------------------------------------------------------


@dataclass
class FakeAIMessage:
    content: Any = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class FakeToolMessage:
    content: str = ""
    tool_call_id: Optional[str] = None
    type: str = "tool"


class FakeReactGraph:
    """Shaped like a LangGraph compiled graph (stream_mode='updates')."""

    def invoke(self, inputs):
        return {"messages": [FakeAIMessage(content="The weather is 18°C.")]}

    def stream(self, inputs):
        assert isinstance(inputs, dict) and "messages" in inputs
        yield {
            "agent": {
                "messages": [
                    FakeAIMessage(
                        tool_calls=[
                            {"name": "get_weather", "args": {"city": "SF"}, "id": "c1"}
                        ]
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [FakeToolMessage(content="18°C, foggy", tool_call_id="c1")]
            }
        }
        yield {
            "agent": {
                "messages": [FakeAIMessage(content="The weather is 18°C and foggy.")]
            }
        }


class FakeChatModel:
    """Shaped like a LangChain chat model streaming content chunks."""

    def invoke(self, inputs):
        return FakeAIMessage(content="Hello there!")

    def stream(self, inputs):
        for chunk in ("Hello ", "there!"):
            yield FakeAIMessage(content=chunk)


# --------------------------------------------------------------------------
# pydantic-ai-shaped fakes
# --------------------------------------------------------------------------


@dataclass
class FakePart:
    part_kind: str
    tool_name: Optional[str] = None
    args: Any = None
    tool_call_id: Optional[str] = None
    content: Any = None


@dataclass
class FakeModelMessage:
    parts: list = field(default_factory=list)


class FakeRunResult:
    output = "It's 18°C in SF."

    def new_messages(self):
        return [
            FakeModelMessage(
                parts=[
                    FakePart(
                        part_kind="tool-call",
                        tool_name="get_weather",
                        args='{"city": "SF"}',
                        tool_call_id="c1",
                    )
                ]
            ),
            FakeModelMessage(
                parts=[
                    FakePart(part_kind="tool-return", tool_call_id="c1", content="18°C")
                ]
            ),
        ]


class FakePydanticAgent:
    async def run(self, message):
        raise AssertionError("sync path should use run_sync")

    def run_sync(self, message):
        return FakeRunResult()


# --------------------------------------------------------------------------
# smolagents-shaped fakes. The adapter dispatches on step type names and the
# agent class's root module, so the fakes carry the real names and claim the
# smolagents module.
# --------------------------------------------------------------------------


@dataclass
class SmolToolCall:
    name: str
    arguments: Any
    id: Optional[str] = None


@dataclass
class PlanningStep:
    plan: str = ""


@dataclass
class ActionStep:
    model_output: Optional[str] = None
    tool_calls: Optional[list] = None
    observations: Optional[str] = None
    error: Any = None


@dataclass
class FinalAnswerStep:
    output: Any = None


class FakeSmolAgent:
    def run(self, task, stream=False, **kwargs):
        assert stream
        yield PlanningStep(plan="I will look up the weather, then answer.")
        yield ActionStep(
            model_output="I should call the weather tool for SF.",
            tool_calls=[SmolToolCall("get_weather", {"city": "SF"}, "c1")],
            observations="18°C, foggy",
        )
        yield FinalAnswerStep(output="It's 18°C and foggy in SF.")


FakeSmolAgent.__module__ = "smolagents.agents"


class CallableFakeSmolAgent(FakeSmolAgent):
    """Real smolagents agents define __call__ — detection must not bail on
    callable instances (regression test for MultiStepAgent.__call__)."""

    def __call__(self, task):
        raise AssertionError("agent must be driven by the adapter, not called")


CallableFakeSmolAgent.__module__ = "smolagents.agents"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def updates_for(agent):
    demo = gr.ChatInterface(agent)
    return list(demo.fn("what's the weather in SF?", []))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


class TestResolution:
    def test_plain_callables_are_not_agents(self):
        assert resolve_adapter(lambda m, h: m) is None
        assert resolve_adapter(updates_for) is None
        assert resolve_adapter(str) is None
        assert resolve_adapter("hello") is None

    def test_adapter_priority(self):
        assert isinstance(resolve_adapter(FakeAgnoAgent()), AgnoAdapter)
        assert isinstance(resolve_adapter(FakeSmolAgent()), SmolagentsAdapter)
        assert isinstance(resolve_adapter(FakePydanticAgent()), PydanticAIAdapter)
        assert isinstance(resolve_adapter(FakeReactGraph()), LangChainAdapter)

    def test_callable_agent_instance_is_detected(self):
        assert isinstance(resolve_adapter(CallableFakeSmolAgent()), SmolagentsAdapter)

    def test_register_adapter_takes_precedence(self):
        class Sentinel(AgentAdapter):
            @staticmethod
            def matches(agent):
                return isinstance(agent, FakeAgnoAgent)

            def stream(self, agent, message, history):
                yield from ()

        register_adapter(Sentinel)
        try:
            assert isinstance(resolve_adapter(FakeAgnoAgent()), Sentinel)
        finally:
            _ADAPTERS.remove(Sentinel)

    def test_broken_matches_does_not_break_detection(self):
        class Broken(AgentAdapter):
            @staticmethod
            def matches(agent):
                raise RuntimeError("boom")

            def stream(self, agent, message, history):
                yield from ()

        register_adapter(Broken)
        try:
            assert isinstance(resolve_adapter(FakeAgnoAgent()), AgnoAdapter)
        finally:
            _ADAPTERS.remove(Broken)


class TestAgnoAdapter:
    def test_full_stream_mapping(self):
        final = updates_for(FakeAgnoAgent())[-1]
        thought, tool, text = final
        assert thought.metadata["title"] == "🧠 Thinking"
        assert "check the weather" in str(thought.content)
        assert tool.metadata["title"] == "🛠️ get_weather"
        assert tool.metadata["status"] == "done"
        assert "foggy" in str(tool.content)
        assert text.content == "The weather is 18°C and foggy."

    def test_tool_panel_pending_before_completion(self):
        # The renderer mutates panel objects in place across yields (each SSE
        # frame is serialized at emit time), so status must be sampled during
        # iteration, not after collecting all updates.
        demo = gr.ChatInterface(FakeAgnoAgent())
        seen_states = []
        for update in demo.fn("what's the weather?", []):
            seen_states.extend(
                msg.metadata.get("status")
                for msg in update
                if msg.metadata.get("title") == "🛠️ get_weather"
            )
        assert "pending" in seen_states
        assert "done" in seen_states


class TestAgnoTeamEvents:
    def test_team_prefixed_events_are_mapped(self):
        final = updates_for(FakeAgnoTeam())[-1]
        thought, tool, text = final
        assert thought.metadata["title"] == "🧠 Thinking"
        assert "Delegating" in str(thought.content)
        assert tool.metadata["title"] == "🛠️ get_weather"
        assert tool.metadata["status"] == "done"
        assert text.content == "The team says: 18°C and foggy."


class TestLangChainAdapter:
    def test_langgraph_updates_mapping(self):
        final = updates_for(FakeReactGraph())[-1]
        tool, text = final
        assert tool.metadata["title"] == "🛠️ get_weather"
        assert tool.metadata["status"] == "done"
        assert "foggy" in str(tool.content)
        assert text.content == "The weather is 18°C and foggy."

    def test_chat_model_deltas_accumulate(self):
        final = updates_for(FakeChatModel())[-1]
        assert final[-1].content == "Hello there!"


class TestPydanticAIAdapter:
    def test_history_parts_and_output(self):
        final = updates_for(FakePydanticAgent())[-1]
        tool, text = final
        assert tool.metadata["title"] == "🛠️ get_weather"
        assert tool.metadata["status"] == "done"
        assert text.content == "It's 18°C in SF."


class TestSmolagentsAdapter:
    def test_step_mapping(self):
        final = updates_for(FakeSmolAgent())[-1]
        plan, reasoning, tool, text = final
        assert plan.metadata["title"] == "🧠 Thinking"
        assert "look up" in str(plan.content)
        assert "weather tool" in str(reasoning.content)
        assert tool.metadata["title"] == "🛠️ get_weather"
        assert tool.metadata["status"] == "done"
        assert "foggy" in str(tool.content)
        assert text.content == "It's 18°C and foggy in SF."

    def test_callable_agent_runs_via_adapter(self):
        final = updates_for(CallableFakeSmolAgent())[-1]
        assert final[-1].content == "It's 18°C and foggy in SF."
