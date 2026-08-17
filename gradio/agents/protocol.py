"""
The agent protocol: the normalized event vocabulary, the AgentAdapter base
class, and the shared renderer that folds events into Chatbot values.

This is the stable contract of the package — adapters (in `adapters/`) are
the parts expected to churn as frameworks evolve.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from gradio.components.chatbot import ChatMessage


@dataclass
class AgentText:
    """A streamed delta of the assistant's text response."""

    text: str


@dataclass
class AgentThought:
    """An intermediate reasoning step, rendered as a collapsible panel."""

    text: str


@dataclass
class AgentToolStart:
    """A tool call has started; renders a panel with a pending spinner."""

    name: str
    args: dict | None = None
    id: Any = None


@dataclass
class AgentToolEnd:
    """A tool call finished; closes the matching panel with its result."""

    id: Any = None
    result: Any = None
    error: bool = False


@dataclass
class AgentFinal:
    """The complete final response (replaces any streamed text)."""

    text: str


AgentEvent = AgentText | AgentThought | AgentToolStart | AgentToolEnd | AgentFinal


class AgentAdapter(ABC):
    """Drives one family of agent objects, translating their native streaming
    events into the normalized AgentEvent vocabulary."""

    @staticmethod
    @abstractmethod
    def matches(agent: Any) -> bool:
        """Return True if this adapter can drive the given object."""

    @abstractmethod
    def stream(self, agent: Any, message: str, history: list) -> Iterator[AgentEvent]:
        """Run the agent on `message` and yield normalized events."""


def parse_tool_args(args: Any) -> dict:
    """Best-effort coercion of a framework's tool-call arguments (dict, JSON
    string, or anything else) into a dict for display."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"args": parsed}
        except (ValueError, TypeError):
            return {"args": args}
    return {}


def events_to_updates(events: Iterable[AgentEvent]) -> Iterator[list[ChatMessage]]:
    """Shared renderer: folds a stream of normalized AgentEvents into
    successive Chatbot values (tool/thought panels followed by the running
    text response)."""
    panels: list[ChatMessage] = []
    pending_tools: dict[Any, ChatMessage] = {}
    text = ""
    for event in events:
        if isinstance(event, AgentToolStart):
            tool_msg = ChatMessage(
                role="assistant",
                content="```json\n"
                + json.dumps(event.args or {}, indent=2, default=str)
                + "\n```",
                metadata={"title": f"🛠️ {event.name}", "status": "pending"},
            )
            panels.append(tool_msg)
            pending_tools[event.id if event.id is not None else id(event)] = tool_msg
        elif isinstance(event, AgentToolEnd):
            tool_msg = pending_tools.pop(
                event.id,
                next(reversed(pending_tools.values()), None)
                if event.id is None
                else None,
            )
            if tool_msg is None and event.id is None and panels:
                tool_msg = panels[-1]
            if tool_msg is not None:
                if event.result is not None:
                    tool_msg.content = str(event.result)
                tool_msg.metadata["status"] = "done"
                if event.error:
                    tool_msg.metadata["title"] += " (error)"
        elif isinstance(event, AgentThought):
            panels.append(
                ChatMessage(
                    role="assistant",
                    content=event.text,
                    metadata={"title": "🧠 Thinking", "status": "done"},
                )
            )
        elif isinstance(event, AgentText):
            text += event.text
        elif isinstance(event, AgentFinal):
            if event.text:
                text = event.text
        yield panels + ([ChatMessage(role="assistant", content=text)] if text else [])


def agent_to_chat_fn(agent: Any, adapter: AgentAdapter) -> Callable:
    """Wrap an agent object and its adapter into a ChatInterface-compatible
    generator function."""

    def chat(message: str, history: list):
        yield from events_to_updates(adapter.stream(agent, message, history))

    return chat
