"""
Cookbook 1: gr.ChatInterface(agent) with a mock agent — no dependencies.

Demonstrates the new agent protocol: any non-callable object with a
`.run(message, stream=True)` method that yields events can be passed
directly to gr.ChatInterface. This mock emits the same event shapes as
agno (RunContentEvent, ToolCallStartedEvent, ...) so you can see the
full UI mapping — tool panels with pending spinners, thinking panels,
and streamed text — without an API key.

Run:  source .venv/bin/activate && python cookbooks/mock_agent_chat.py
"""

import time
from dataclasses import dataclass, field
from typing import Any

import gradio as gr


@dataclass
class Tool:
    tool_name: str
    tool_args: dict
    tool_call_id: str
    result: Any = None


@dataclass
class Event:
    event: str
    content: Any = None
    tool: Tool | None = None


@dataclass
class MockAgent:
    """Mimics agno's Agent.run(message, stream=True, stream_events=True)."""

    delay: float = 0.15
    calls: list = field(default_factory=list)

    def run(self, message: str, stream: bool = False, stream_events: bool = False):
        self.calls.append(message)
        if not stream:
            return Event(event="RunCompleted", content=f"You said: {message}")
        return self._stream(message)

    def _stream(self, message: str):
        yield Event(event="RunStarted")
        yield Event(
            event="ReasoningStep",
            content=f"The user asked: {message!r}. I should check the weather first.",
        )
        time.sleep(self.delay)

        tool = Tool(
            tool_name="get_weather",
            tool_args={"city": "San Francisco", "unit": "celsius"},
            tool_call_id="call_1",
        )
        yield Event(event="ToolCallStarted", tool=tool)
        time.sleep(self.delay * 4)  # watch the pending spinner in the UI
        tool.result = "18°C, foggy (of course)"
        yield Event(event="ToolCallCompleted", tool=tool, content=tool.result)

        for chunk in (
            "The weather in San Francisco ",
            "is currently 18°C and foggy. ",
            "Classic! Bring a light jacket.",
        ):
            time.sleep(self.delay)
            yield Event(event="RunContent", content=chunk)
        yield Event(event="RunCompleted")


demo = gr.ChatInterface(
    MockAgent(),
    title="Agent protocol demo (mock)",
    description="A mock agent streaming tool calls, reasoning, and text through the new `gr.ChatInterface(agent)` protocol.",
)

if __name__ == "__main__":
    demo.launch()
