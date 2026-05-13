"""Verify Gemini's native tool-calling works before building the full Deal Coach."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()


# A tiny fake "tool" — Gemini doesn't actually call this; it tells us it WANTS to call it,
# and we'd run the real function in our own loop. This proves the tool-calling round-trip.
def get_weather(location: str) -> str:
    """Get the current weather in a location.

    Args:
        location: City name, e.g. 'San Francisco' or 'Tokyo'.
    """
    return f"Weather in {location}: sunny, 72°F"  # not actually called by Gemini


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather like in Tokyo right now?",
    config=types.GenerateContentConfig(
        tools=[get_weather],            # pass the function directly; SDK auto-extracts schema
        temperature=0.0,
        max_output_tokens=1000,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    ),
)

print("✅ Gemini tool-calling response received\n")

# When tools are configured, Gemini may emit "function calls" instead of plain text.
# We inspect the response parts to see which.
for part in response.candidates[0].content.parts:
    if part.function_call:
        print(f"🔧 Tool call requested: {part.function_call.name}")
        print(f"   Arguments: {dict(part.function_call.args)}")
    elif part.text:
        print(f"💬 Text: {part.text}")

print(f"\nFinish reason: {response.candidates[0].finish_reason}")