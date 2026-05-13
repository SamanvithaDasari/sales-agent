"""Quick sanity check: confirm the Gemini API key works (new google-genai SDK)."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# The SDK auto-reads GOOGLE_API_KEY (or GEMINI_API_KEY) from the environment
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="In one sentence, what does a sales rep do?",
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.3,
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable thinking
    ),
)

print("✅ Gemini connection works!")
print()
print("Model said:", response.text)
print()
print(f"Finish reason: {response.candidates[0].finish_reason}")
print(f"Full usage: {response.usage_metadata}")