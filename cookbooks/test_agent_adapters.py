"""
Adapter tests with framework-shaped fakes — no API keys or framework
installs needed. Each fake replicates the object shapes the real framework
produces, so these verify the adapters' duck-typing contracts.

Run:  source .venv/bin/activate && python cookbooks/test_agent_adapters.py
"""

from dataclasses import dataclass, field
from typing import Any

from gradio.agents import (
    GenericRunAdapter,
    LangChainAdapter,
    PydanticAIAdapter,
    resolve_adapter,
)
from gradio.chat_interface import ChatInterface

# --- Agno-shaped fake (also covers smolagents-style .run streaming) -------

from mock_agent_chat import MockAgent  # noqa: E402


# --- LangGraph-shaped fakes ------------------------------------------------


@dataclass
class FakeAIMessage:
    content: Any = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class FakeToolMessage:
    content: str = ""
    tool_call_id: str | None = None
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
                            {
                                "name": "get_weather",
                                "args": {"city": "SF"},
                                "id": "call_1",
                            }
                        ]
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [FakeToolMessage(content="18°C, foggy", tool_call_id="call_1")]
            }
        }
        yield {"agent": {"messages": [FakeAIMessage(content="The weather is 18°C and foggy.")]}}


class FakeChatModel:
    """Shaped like a LangChain chat model streaming content chunks."""

    def invoke(self, inputs):
        return FakeAIMessage(content="Hello there!")

    def stream(self, inputs):
        for chunk in ("Hello ", "there!"):
            yield FakeAIMessage(content=chunk)


# --- pydantic-ai-shaped fakes ----------------------------------------------


@dataclass
class FakePart:
    part_kind: str
    tool_name: str | None = None
    args: Any = None
    tool_call_id: str | None = None
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
    async def run(self, message):  # noqa: ARG002
        raise AssertionError("sync path should use run_sync")

    def run_sync(self, message):  # noqa: ARG002
        return FakeRunResult()


# --- assertions -------------------------------------------------------------


def updates_for(agent):
    demo = ChatInterface(agent)
    return list(demo.fn("what's the weather in SF?", []))


def check(label, condition):
    assert condition, label
    print(f"  ok: {label}")


print("GenericRunAdapter (agno-shaped):")
check("resolves", isinstance(resolve_adapter(MockAgent(delay=0)), GenericRunAdapter))
updates = updates_for(MockAgent(delay=0))
final = updates[-1]
check("thinking panel", final[0].metadata.get("title") == "🧠 Thinking")
check("tool panel done", final[1].metadata.get("status") == "done")
check("tool result", "foggy" in str(final[1].content))
check("streamed text", final[-1].content.endswith("jacket."))

print("LangChainAdapter (LangGraph-shaped):")
check("resolves", isinstance(resolve_adapter(FakeReactGraph()), LangChainAdapter))
updates = updates_for(FakeReactGraph())
final = updates[-1]
check("tool panel", final[0].metadata.get("title") == "🛠️ get_weather")
check("tool closed with result", final[0].metadata.get("status") == "done" and "foggy" in str(final[0].content))
check("final text", final[-1].content == "The weather is 18°C and foggy.")

print("LangChainAdapter (chat-model-shaped):")
check("resolves", isinstance(resolve_adapter(FakeChatModel()), LangChainAdapter))
updates = updates_for(FakeChatModel())
check("deltas accumulate", updates[-1][-1].content == "Hello there!")

print("PydanticAIAdapter (pydantic-ai-shaped):")
check("resolves", isinstance(resolve_adapter(FakePydanticAgent()), PydanticAIAdapter))
updates = updates_for(FakePydanticAgent())
final = updates[-1]
check("tool panel from history", final[0].metadata.get("title") == "🛠️ get_weather")
check("json args parsed", "SF" in str(final[0].content) or "18°C" in str(final[0].content))
check("tool closed", final[0].metadata.get("status") == "done")
check("final output", final[-1].content == "It's 18°C in SF.")

print("Negative cases:")
check("plain function not detected", resolve_adapter(lambda m, h: m) is None)
check("string not detected", resolve_adapter("hello") is None)

print("\nALL ADAPTER TESTS PASSED")
