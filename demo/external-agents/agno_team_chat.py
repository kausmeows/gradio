"""
Cookbook 6: gr.ChatInterface(team) with an agno Team.

Teams work exactly like single agents — pass the Team straight in. Agno
Teams emit the same event vocabulary with a `Team` prefix (TeamRunContent,
TeamToolCallStarted, ...), which the AgnoAdapter normalizes away, so
delegation tool calls render as panels and the coordinator's answer
streams into the chat bubble. With `stream_events=True` (which the adapter
requests automatically), member agents' own tool calls and content events
interleave into the same stream and render too.

Setup (from the gradio repo root):
    source .venv/bin/activate
    pip install agno            # or: pip install -e /Users/kaustubh/Desktop/Agno/agno/libs/agno
    export OPENAI_API_KEY=sk-...
    python demo/external-agents/agno_team_chat.py
"""

import random

import gradio as gr
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{random.randint(5, 30)}°C and {random.choice(['sunny', 'foggy', 'rainy'])} in {city}"


def get_attractions(city: str) -> str:
    """Get the top attractions for a city."""
    return f"Top sights in {city}: the old town, the riverside market, and the science museum."


weather_agent = Agent(
    name="Weather Agent",
    role="Answers questions about the current weather.",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_weather],
)

sights_agent = Agent(
    name="Sightseeing Agent",
    role="Recommends attractions and things to do in a city.",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_attractions],
)

team = Team(
    name="Trip Planning Team",
    members=[weather_agent, sights_agent],
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Plan short city visits: delegate weather questions and sightseeing questions to the right member, then combine their answers.",
)

demo = gr.ChatInterface(
    team,
    title="agno Team × Gradio",
    description="An agno Team passed directly to gr.ChatInterface — delegation and member tool calls render as panels. Try: “I'm visiting Tokyo tomorrow — what should I do, and what's the weather?”",
)

if __name__ == "__main__":
    demo.launch()
