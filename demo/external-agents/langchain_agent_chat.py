"""
Cookbook 3: gr.ChatInterface(agent) with a LangGraph react agent.

The LangChainAdapter drives any Runnable (`.stream()`/`.invoke()`):
chat models stream text deltas; agent graphs stream update dicts whose
AIMessage tool calls and ToolMessages become collapsible panels.

Setup (from the gradio repo root):
    source .venv/bin/activate
    pip install langchain langgraph langchain-openai
    export OPENAI_API_KEY=sk-...
    python demo/external-agents/langchain_agent_chat.py
"""

import random

import gradio as gr
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{random.randint(5, 30)}°C and {random.choice(['sunny', 'foggy', 'rainy'])} in {city}"


agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[get_weather])

demo = gr.ChatInterface(
    agent,
    title="LangGraph × Gradio",
    description="A LangGraph react agent passed directly to gr.ChatInterface.",
)

if __name__ == "__main__":
    demo.launch()
