"""Adapter for LangChain Runnables and LangGraph compiled graphs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from gradio.agents.protocol import (
    AgentAdapter,
    AgentEvent,
    AgentFinal,
    AgentText,
    AgentToolEnd,
    AgentToolStart,
    parse_tool_args,
)


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
                args=parse_tool_args(get("args")),
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
