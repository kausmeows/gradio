"""
Framework-agnostic agent support for gr.ChatInterface.

This module defines a small protocol that lets ChatInterface accept agent
objects from any framework:

1. A normalized event vocabulary (AgentText, AgentThought, AgentToolStart,
   AgentToolEnd, AgentFinal) that a shared renderer maps onto the Chatbot UI
   (tool panels with pending spinners, "thinking" panels, streamed text).
2. An `AgentAdapter` base class: `matches(agent)` decides whether the adapter
   can drive a given object, and `stream(agent, message, history)` yields
   normalized events.
3. A registry of built-in adapters, checked in order. Third-party frameworks
   can plug in via `register_adapter` without any changes to Gradio.

No agent framework is imported here — all detection is duck-typed, so this
module adds zero dependencies.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from gradio.components.chatbot import ChatMessage
from gradio.exceptions import Error


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


def _parse_args(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"args": parsed}
        except (ValueError, TypeError):
            return {"args": args}
    return {}


class GenericRunAdapter(AgentAdapter):
    """Agents following the `.run(message, stream=True)` convention that yield
    Agno-shaped events (`.event` name, `.content`, `.tool`) or plain strings.
    Covers agno and smolagents-style APIs, and non-streaming `.run()` results
    via their `.content`."""

    @staticmethod
    def matches(agent: Any) -> bool:
        run = getattr(agent, "run", None)
        return (
            not callable(agent)
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
            tool = getattr(event, "tool", None)
            content = getattr(event, "content", None)
            if tool is not None and "Started" in name:
                yield AgentToolStart(
                    name=getattr(tool, "tool_name", None) or "tool",
                    args=getattr(tool, "tool_args", None) or {},
                    id=getattr(tool, "tool_call_id", None) or id(tool),
                )
            elif tool is not None and ("Completed" in name or "Error" in name):
                result_value = (
                    content if content is not None else getattr(tool, "result", None)
                )
                yield AgentToolEnd(
                    id=getattr(tool, "tool_call_id", None) or id(tool),
                    result=result_value,
                    error="Error" in name,
                )
            elif name == "ReasoningStep" and content is not None:
                yield AgentThought(str(content))
            elif "Content" in name and isinstance(content, str):
                yield AgentText(content)
            elif "Error" in name:
                raise Error(str(content) if content is not None else name)
            elif "Completed" in name and isinstance(content, str):
                yield AgentFinal(content)


class PydanticAIAdapter(AgentAdapter):
    """pydantic-ai style agents: async `.run()` with a sync `.run_sync()`
    returning a result object with `.output` (or legacy `.data`) and a
    `.new_messages()` history containing tool-call / tool-return parts."""

    @staticmethod
    def matches(agent: Any) -> bool:
        return not callable(agent) and (
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
                            args=_parse_args(getattr(part, "args", None)),
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


class LangChainAdapter(AgentAdapter):
    """LangChain Runnables and LangGraph compiled graphs: objects with
    `.stream()`/`.invoke()`. Handles chat models streaming message chunks
    (`.content` as str or content-block list) and agent graphs streaming
    update dicts containing AIMessage/ToolMessage lists."""

    @staticmethod
    def matches(agent: Any) -> bool:
        return (
            callable(getattr(agent, "stream", None))
            and callable(getattr(agent, "invoke", None))
            and not callable(getattr(agent, "run", None))
        )

    def stream(self, agent: Any, message: str, history: list) -> Iterator[AgentEvent]:  # noqa: ARG002
        try:
            chunks = agent.stream({"messages": [{"role": "user", "content": message}]})
        except Exception:  # noqa: BLE001 — input-shape probe: graphs take dicts, models take strings
            chunks = agent.stream(message)
        for chunk in chunks:
            yield from self._normalize_chunk(chunk)

    def _normalize_chunk(self, chunk: Any) -> Iterator[AgentEvent]:
        if isinstance(chunk, dict):
            messages: list = []
            if isinstance(chunk.get("messages"), list):
                messages.extend(chunk["messages"])
            for value in chunk.values():
                if isinstance(value, dict) and isinstance(value.get("messages"), list):
                    messages.extend(value["messages"])
            for msg in messages:
                yield from self._normalize_message(msg, delta=False)
        elif isinstance(chunk, tuple) and chunk and hasattr(chunk[0], "content"):
            yield from self._normalize_message(chunk[0], delta=True)
        elif hasattr(chunk, "content"):
            yield from self._normalize_message(chunk, delta=True)

    def _normalize_message(self, msg: Any, delta: bool) -> Iterator[AgentEvent]:
        for tool_call in getattr(msg, "tool_calls", None) or []:
            get = (
                tool_call.get
                if isinstance(tool_call, dict)
                else lambda k, tc=tool_call: getattr(tc, k, None)
            )
            yield AgentToolStart(
                name=get("name") or "tool",
                args=_parse_args(get("args")),
                id=get("id"),
            )
        content = getattr(msg, "content", None)
        is_tool_result = (
            type(msg).__name__ == "ToolMessage" or getattr(msg, "type", "") == "tool"
        )
        if is_tool_result:
            yield AgentToolEnd(
                id=getattr(msg, "tool_call_id", None),
                result=content,
            )
        elif isinstance(content, str) and content:
            yield AgentText(content) if delta else AgentFinal(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    yield AgentText(block.get("text") or "")


_ADAPTERS: list[type[AgentAdapter]] = [
    PydanticAIAdapter,
    GenericRunAdapter,
    LangChainAdapter,
]


def register_adapter(adapter: type[AgentAdapter]) -> None:
    """Register a custom AgentAdapter. Custom adapters take precedence over
    the built-in ones."""
    _ADAPTERS.insert(0, adapter)


def resolve_adapter(agent: Any) -> AgentAdapter | None:
    """Return an instance of the first adapter that can drive `agent`, or
    None if the object doesn't look like an agent."""
    if callable(agent):
        return None
    for adapter_cls in _ADAPTERS:
        try:
            if adapter_cls.matches(agent):
                return adapter_cls()
        except Exception:  # noqa: BLE001, S112 — a broken matches() must not break detection
            continue
    return None


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
