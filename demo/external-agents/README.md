# External agents in `gr.ChatInterface`

Demos for the (prototype) agent protocol in `gradio/agents/`: pass an agent
object from any supported framework straight to `gr.ChatInterface(agent)` —
no glue function. Tool calls render as collapsible panels (pending spinner →
result), reasoning steps as "Thinking" panels, and content deltas stream into
the response bubble.

```python
agent = Agent(model=..., tools=[...])       # any supported framework
gr.ChatInterface(agent).launch()            # the whole app
```

## Demos

| File | Framework | Needs |
|---|---|---|
| `mock_agent_chat.py` | none (fake agent emitting agno-shaped events) | nothing — start here |
| `agno_agent_chat.py` | [agno](https://github.com/agno-agi/agno) | `pip install agno`, `OPENAI_API_KEY` |
| `agno_team_chat.py` | agno Team (multi-agent delegation) | `pip install agno`, `OPENAI_API_KEY` |
| `smolagents_chat.py` | [smolagents](https://github.com/huggingface/smolagents) | `pip install smolagents`, `HF_TOKEN` |
| `langchain_agent_chat.py` | LangChain / LangGraph | `pip install langchain langgraph langchain-openai`, `OPENAI_API_KEY` |
| `pydantic_ai_chat.py` | [pydantic-ai](https://github.com/pydantic/pydantic-ai) | `pip install pydantic-ai`, `OPENAI_API_KEY` |

Run from the repo root with the dev venv active, e.g.:

```bash
python demo/external-agents/mock_agent_chat.py
```

> Run these with plain `python`, not `gradio` reload mode: reload mode
> re-executes the file from a string, which breaks libraries that call
> `inspect.getsource()` at decoration time (e.g. smolagents' `@tool`).

## How it works

- `gradio/agents/protocol.py` — normalized event vocabulary
  (`AgentText`, `AgentThought`, `AgentToolStart`, `AgentToolEnd`,
  `AgentFinal`), the `AgentAdapter` base class, and the shared renderer
  that folds events into Chatbot values.
- `gradio/agents/adapters/` — one duck-typed adapter per framework family,
  zero framework imports. `AgnoAdapter` doubles as the fallback for any
  custom object with a sync `.run()` that streams plain strings or returns
  a result with `.content`.
- Other frameworks plug in via
  `gradio.agents.register_adapter(MyAdapter)` — custom adapters take
  precedence over built-ins.

Tests live in `test/test_agents.py` (pytest, framework-shaped fakes, no
framework installs needed).
