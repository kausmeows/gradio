"""
Framework-agnostic agent support for gr.ChatInterface.

This package lets ChatInterface accept agent objects from any framework:

1. `protocol.py` — the stable contract: a normalized event vocabulary
   (AgentText, AgentThought, AgentToolStart, AgentToolEnd, AgentFinal), the
   `AgentAdapter` base class, and the shared renderer that maps events onto
   the Chatbot UI (tool panels with pending spinners, "thinking" panels,
   streamed text).
2. `adapters/` — one module per framework family, all duck-typed with zero
   framework imports.
3. The registry below, checked in order. Third-party frameworks can plug in
   via `register_adapter` without any changes to Gradio.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from gradio.agents.adapters import (
    AgnoAdapter,
    LangChainAdapter,
    PydanticAIAdapter,
    SmolagentsAdapter,
)
from gradio.agents.protocol import (
    AgentAdapter,
    AgentEvent,
    AgentFinal,
    AgentText,
    AgentThought,
    AgentToolEnd,
    AgentToolStart,
    agent_to_chat_fn,
    events_to_updates,
)

__all__ = [
    "AgentAdapter",
    "AgentEvent",
    "AgentFinal",
    "AgentText",
    "AgentThought",
    "AgentToolEnd",
    "AgentToolStart",
    "AgnoAdapter",
    "LangChainAdapter",
    "PydanticAIAdapter",
    "SmolagentsAdapter",
    "agent_to_chat_fn",
    "events_to_updates",
    "register_adapter",
    "resolve_adapter",
]

# Checked in order. AgnoAdapter doubles as the sync-`.run` fallback, so every
# adapter that also matches on `.run` must precede it: PydanticAIAdapter
# (async `.run` would poison the sync path) and SmolagentsAdapter (its step
# records don't fit agno's event vocabulary).
_ADAPTERS: list[type[AgentAdapter]] = [
    PydanticAIAdapter,
    SmolagentsAdapter,
    AgnoAdapter,
    LangChainAdapter,
]


def register_adapter(adapter: type[AgentAdapter]) -> None:
    """Register a custom AgentAdapter. Custom adapters take precedence over
    the built-in ones."""
    _ADAPTERS.insert(0, adapter)


def resolve_adapter(agent: Any) -> AgentAdapter | None:
    """Return an instance of the first adapter that can drive `agent`, or
    None if the object doesn't look like an agent.

    Only plain routines (functions, lambdas, methods, partials) and classes
    skip detection — a callable *instance* may still be an agent (e.g.
    smolagents agents define both `__call__` and `.run`). An instance no
    adapter matches falls back to being treated as an ordinary chat fn."""
    if (
        inspect.isroutine(agent)
        or inspect.isclass(agent)
        or isinstance(agent, functools.partial)
    ):
        return None
    for adapter_cls in _ADAPTERS:
        try:
            if adapter_cls.matches(agent):
                return adapter_cls()
        except Exception:  # noqa: BLE001, S112 — a broken matches() must not break detection
            continue
    return None
