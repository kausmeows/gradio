"""
Cookbook 4: gr.ChatInterface(agent) with a pydantic-ai Agent.

The PydanticAIAdapter runs the agent via `.run_sync()` and reconstructs
tool-call panels from the run's message history parts, then shows the
final `.output`.

Setup (from the gradio repo root):
    source .venv/bin/activate
    pip install pydantic-ai
    export OPENAI_API_KEY=sk-...
    python demo/external-agents/pydantic_ai_chat.py
"""

import random

import gradio as gr
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a concise, friendly weather assistant.",
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{random.randint(5, 30)}°C and {random.choice(['sunny', 'foggy', 'rainy'])} in {city}"


demo = gr.ChatInterface(
    agent,
    title="pydantic-ai × Gradio",
    description="A pydantic-ai Agent passed directly to gr.ChatInterface.",
)

if __name__ == "__main__":
    demo.launch()
