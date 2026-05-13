"""
deal_coach.py — Deal Coach agent. Hand-written ReAct loop using Gemini's
native function calling.

WHY HAND-WRITTEN (not CrewAI):
- CrewAI hides the loop. Writing it ourselves is ~80 lines and we control
  every iteration — visible thought/action/observation, explicit iteration
  caps, observation logging, fail-soft error handling.
- Production agent codebases (LangChain agent executors, OpenAI Assistants
  runner, etc.) are this same pattern with more polish. Knowing it cold is
  a real interview signal.

WHY GEMINI (not Groq):
- Per-iteration tokens grow with accumulated history; Groq's 6K TPM on the
  free tier was too tight. Gemini 2.5 Flash gives ~250K TPM, plenty of room.
- Gemini's native tool-calling is more reliable than prompting Llama 3.1 8B
  into emitting JSON tool calls, especially across 4-6 iterations.

LOOP ARCHITECTURE:
    user query → seed messages
    loop up to max_iter:
        gemini.generate_content(messages, tools=...)
        if response has function_call:
            execute tool, append model's call + our response to messages
        elif response has text:
            return text   # final answer
    raise/return: agent exceeded max iterations
"""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools.coach_tools import COACH_TOOLS

load_dotenv()



# ============================================================
# CONFIG
# ============================================================
MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 8           # hard cap on the loop
THINKING_BUDGET = 1024       # let Gemini reason internally; we still see actions

client = genai.Client()


# ============================================================
# AGENT PERSONA (the "system prompt" — Gemini calls it system_instruction)
# ============================================================
SYSTEM_INSTRUCTION = """You are a Deal Coach for B2B sales reps. You analyze \
deals in a CRM and recommend concrete next actions.

Your reasoning style:
- Always start by looking up the specific deal (get_deal_details).
- Then look at its recent activity (get_deal_activities) to understand what \
has actually been happening.
- Then DECIDE what kind of analysis is needed:
  * If the deal looks stuck on pricing — search similar pricing situations.
  * If a champion has left — search for similar 'champion departed' patterns.
  * If progress has just been slow — compare to other deals at the same stage.
- Use find_similar_deal_activities and get_won_lost_outcomes to learn from \
similar past situations.
- Stop calling tools once you have enough to recommend specific actions. \
Don't keep retrieving for the sake of it.

Your final answer must be Markdown in this exact structure:

## Deal Snapshot
Brief: name, account, stage, value, age in days, probability.

## What's Happening
2-3 bullets summarizing the activity history. Quote real notes.

## Pattern Analysis
What did similar past deals look like? What did won ones do that lost ones \
didn't? Be specific.

## Recommended Actions
2-4 concrete next steps the rep should take this week, each grounded in a \
specific finding above. No generic sales advice.

Be honest. If the data doesn't support a confident recommendation, say so."""


# ============================================================
# TOOL EXPOSURE
# ============================================================
# Gemini extracts the schema from each function's signature + docstring.
# We pass the actual function objects; the SDK auto-converts to its
# FunctionDeclaration format.
TOOL_FUNCTIONS = list(COACH_TOOLS.values())


# ============================================================
# THE LOOP
# ============================================================
@dataclass
class Trace:
    """One iteration of the loop, for logging."""
    iteration: int
    tool_called: str | None
    tool_args: dict[str, Any] | None
    tool_result_preview: str | None
    final_text: str | None


# ============================================================
# RETRY HELPER for transient 503 / rate-limit failures
# ============================================================
import time
from google.genai import errors as genai_errors

def _generate_with_retry(messages, max_attempts: int = 4) -> Any:
    """Call Gemini with exponential backoff on 503/429."""
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=TOOL_FUNCTIONS,
                    temperature=0.2,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                raise
            print(f"   ⚠️  Gemini {e.code} on attempt {attempt}; retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2  # exponential: 2s, 4s, 8s
        except genai_errors.ClientError as e:
            # 429 (rate limit) gets the same backoff treatment
            if e.code == 429 and attempt < max_attempts:
                print(f"   ⚠️  Rate limited on attempt {attempt}; retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise

def run_deal_coach(user_query: str, verbose: bool = True) -> tuple[str, list[Trace]]:
    """
    Run the Deal Coach ReAct loop for a user query.

    Returns (final_answer, trace) where trace is the per-iteration log
    (useful for debugging, demos, and showing the agent's reasoning).
    """
    # The conversation history we'll keep extending. Gemini's API is stateless;
    # we send the full transcript on each iteration.
    messages: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    ]

    trace: list[Trace] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Send the current transcript + tool definitions to Gemini
        response = _generate_with_retry(messages)


        candidate = response.candidates[0]

        # Did the model ask to call a tool?
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if function_calls:
            # For simplicity we handle one call per iteration. Gemini occasionally
            # emits parallel calls; we take the first and let the loop continue.
            fc = function_calls[0]
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            if verbose:
                print(f"\n[Iter {iteration}] 🔧 Tool: {tool_name}({tool_args})")

            # Execute the tool
            if tool_name not in COACH_TOOLS:
                tool_result = f"ERROR: unknown tool {tool_name!r}"
            else:
                try:
                    tool_result = COACH_TOOLS[tool_name](**tool_args)
                except Exception as e:
                    tool_result = f"ERROR running {tool_name}: {e}"

            if verbose:
                preview = tool_result[:300].replace("\n", " ")
                print(f"[Iter {iteration}] 📊 Result: {preview}{'...' if len(tool_result) > 300 else ''}")

            # Append BOTH the model's call AND our response to history.
            # This is the ReAct pattern: every Action gets its Observation.
            messages.append(candidate.content)  # model's function_call
            messages.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": tool_result},
                        )
                    ],
                )
            )

            trace.append(Trace(
                iteration=iteration,
                tool_called=tool_name,
                tool_args=tool_args,
                tool_result_preview=tool_result[:200],
                final_text=None,
            ))
            continue  # next iteration

        # No function call — the model is done. Return its text.
        final_text = response.text or "(no answer produced)"
        if verbose:
            print(f"\n[Iter {iteration}] ✅ Final answer received ({len(final_text)} chars)")
        trace.append(Trace(
            iteration=iteration,
            tool_called=None,
            tool_args=None,
            tool_result_preview=None,
            final_text=final_text,
        ))
        return final_text, trace

    # Fell off the end of the loop
    msg = f"⚠️  Agent did not finish within {MAX_ITERATIONS} iterations."
    return msg, trace


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "Why is the Inc Streamline Magnetic Channels deal stuck? "
        "What should I do this week?"
    )
    print(f"\n🤔 Deal Coach: {query!r}\n")
    print("=" * 70)
    answer, _ = run_deal_coach(query, verbose=True)
    print("\n" + "=" * 70)
    print("📋 FINAL ANSWER\n")
    print(answer)