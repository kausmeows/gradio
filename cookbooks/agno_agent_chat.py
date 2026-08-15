"""
Cookbook 2: gr.ChatInterface(agent) with a real agno Agent.

The whole integration is now one line: pass the Agent straight in.
Tool calls render as collapsible panels (via ChatMessage.metadata),
and content deltas stream into the response bubble.

Setup (from the gradio repo root):
    source .venv/bin/activate
    pip install -e /Users/kaustubh/Desktop/Agno/agno/libs/agno
    export OPENAI_API_KEY=sk-...
    python cookbooks/agno_agent_chat.py
"""

import random

import gradio as gr
from agno.agent import Agent
from agno.models.openai import OpenAIChat


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{random.randint(5, 30)}°C and {random.choice(['sunny', 'foggy', 'rainy'])} in {city}"


agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_weather],
    instructions="You are a concise, friendly weather assistant.",
)

demo = gr.ChatInterface(
    agent,
    title="Agno × Gradio",
    description="An agno Agent passed directly to gr.ChatInterface — no glue code.",
)

if __name__ == "__main__":
    demo.launch()
