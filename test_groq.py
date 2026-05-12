"""Quick sanity check: confirm the Groq API key works."""

import os
from dotenv import load_dotenv
from groq import Groq

# Load GROQ_API_KEY from .env into os.environ
load_dotenv()

# The Groq SDK auto-reads os.environ["GROQ_API_KEY"]
client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful sales assistant."},
        {"role": "user", "content": "In one sentence, what does a sales rep do?"},
    ],
    max_tokens=100,
)

print("✅ Groq connection works!")
print()
print("Model said:", response.choices[0].message.content)
print()
print(f"Tokens used: {response.usage.total_tokens} "
      f"(prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")