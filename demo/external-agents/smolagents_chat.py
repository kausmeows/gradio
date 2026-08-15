"""
Cookbook 5: gr.ChatInterface(agent) with a smolagents agent.

The SmolagentsAdapter streams the agent's step records: each ActionStep's
reasoning becomes a "Thinking" panel, its tool calls become tool panels
closed with the step's observations, and the FinalAnswerStep becomes the
response text.

Setup (from the gradio repo root):
    source .venv/bin/activate
    pip install smolagents
    export HF_TOKEN=hf_...        # for InferenceClientModel
    python demo/external-agents/smolagents_chat.py

Note: run with `python`, not `gradio` reload mode. Reload mode re-executes
this file from a string, and smolagents' @tool needs inspect.getsource() on
the decorated function, which raises "OSError: could not get source code"
in that context.
"""

import os
import random

import gradio as gr
from smolagents import InferenceClientModel, ToolCallingAgent, tool

# InferenceClientModel's default model may not be served by any provider
# enabled on your HF account ("model_not_supported"). Pick one you have a
# provider for — see https://huggingface.co/settings/inference-providers —
# or override without editing: SMOLAGENTS_MODEL=... python demo/external-agents/smolagents_chat.py
MODEL_ID = os.environ.get("SMOLAGENTS_MODEL", "Qwen/Qwen2.5-72B-Instruct")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get the weather for.
    """
    return f"{random.randint(5, 30)}°C and {random.choice(['sunny', 'foggy', 'rainy'])} in {city}"


agent = ToolCallingAgent(tools=[get_weather], model=InferenceClientModel(model_id=MODEL_ID))

demo = gr.ChatInterface(
    agent,
    title="smolagents × Gradio",
    description="A smolagents ToolCallingAgent passed directly to gr.ChatInterface.",
)

if __name__ == "__main__":
    demo.launch()
