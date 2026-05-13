"""
app.py — Streamlit chat UI for the SalesAgent.

ARCHITECTURE NOTES:
- The UI is a thin client over the FastAPI backend at http://localhost:8000.
  Separation matters: in production you could deploy the API independently
  (e.g. on a beefy box) and the UI as a static-ish frontend.
- Streamlit re-runs the entire script on every interaction. We use
  st.session_state to persist conversation history across re-runs.
- We never call agent code directly here — only through HTTP. That keeps
  the UI deployable separately and tests the API surface like a real client.

RUN:
    streamlit run app.py
"""

import json
from datetime import datetime

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
API_BASE = "http://localhost:8000"
TIMEOUT_LEAD_INTEL = 30      # seconds — Lead Intel is fast (~5s typical)
TIMEOUT_DEAL_COACH = 180     # seconds — Deal Coach is a ReAct loop


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="SalesAgent",
    page_icon="💼",
    layout="wide",
)

st.title("💼 SalesAgent — AI Copilot for B2B Sales Reps")
st.caption(
    "A multi-agent system: Lead Intel for pre-call briefs, Deal Coach for stuck-deal analysis. "
    "Powered by Groq, Gemini, FAISS, and CrewAI patterns."
)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Agent")
    agent_choice = st.radio(
        "Which agent do you want to talk to?",
        ["Lead Intel", "Deal Coach"],
        captions=[
            "Pre-call briefs for a company (~5s)",
            "Why is this deal stuck? Recommend next steps (~30-60s)",
        ],
    )

    st.divider()
    st.subheader("Try a sample query")
    if agent_choice == "Lead Intel":
        samples = [
            "Tell me about Davis and Sons",
            "What's going on with Pacheco-Smith?",
            "Brief me on Cohen Inc before my call",
        ]
    else:
        samples = [
            "Why is the Inc Streamline Magnetic Channels deal stuck?",
            "How should I unstick the and Sons Enable Sticky Eyeballs deal?",
            "What's blocking PLC Generate Vertical Architectures?",
        ]
    for s in samples:
        if st.button(s, key=f"sample_{s}", use_container_width=True):
            st.session_state.pending_query = s

    st.divider()
    st.subheader("API health")
    try:
        r = requests.get(f"{API_BASE}/", timeout=2)
        if r.status_code == 200:
            st.success(f"✅ API up — {r.json().get('version', '?')}")
        else:
            st.error(f"API returned {r.status_code}")
    except Exception as e:
        st.error(f"❌ API down: {e.__class__.__name__}")
        st.caption("Make sure `uvicorn api.main:app --port 8000` is running.")

    st.divider()
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# RENDER CONVERSATION HISTORY
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.caption(f"**{msg['agent']}** · {msg['meta'].get('elapsed_seconds', '?')}s · "
                       f"{msg['meta'].get('timestamp', '')}")
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander(f"🧠 Reasoning trace — {len(msg['trace'])} iterations"):
                for step in msg["trace"]:
                    if step.get("tool_called"):
                        st.markdown(
                            f"**Iter {step['iteration']}** · "
                            f"`{step['tool_called']}({json.dumps(step.get('tool_args', {}))})`"
                        )
                    else:
                        st.markdown(f"**Iter {step['iteration']}** · _final answer_")


# ============================================================
# INPUT
# ============================================================
prompt = st.chat_input(f"Ask {agent_choice} something...")

if "pending_query" in st.session_state:
    prompt = st.session_state.pending_query
    del st.session_state.pending_query


# ============================================================
# HANDLE A USER MESSAGE
# ============================================================
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "agent": agent_choice,
        "content": prompt,
        "meta": {},
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    if agent_choice == "Lead Intel":
        endpoint = f"{API_BASE}/lead-intel"
        timeout = TIMEOUT_LEAD_INTEL
        spinner_msg = "📋 Pulling the brief..."
    else:
        endpoint = f"{API_BASE}/deal-coach"
        timeout = TIMEOUT_DEAL_COACH
        spinner_msg = "🧠 Running the ReAct loop — this takes ~30-60s..."

    with st.chat_message("assistant"):
        with st.spinner(spinner_msg):
            try:
                resp = requests.post(
                    endpoint,
                    json={"query": prompt},
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    error_detail = resp.json().get("detail", resp.text)
                    st.error(f"API returned {resp.status_code}: {error_detail}")
                    st.stop()
                data = resp.json()
            except requests.Timeout:
                st.error(f"Request timed out after {timeout}s.")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if agent_choice == "Lead Intel":
            answer = data["brief"]
            trace = None
        else:
            answer = data["answer"]
            trace = data.get("trace", [])

        meta = {
            "elapsed_seconds": data.get("elapsed_seconds", "?"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.caption(f"**{agent_choice}** · {meta['elapsed_seconds']}s · {meta['timestamp']}")
        st.markdown(answer)
        if trace:
            with st.expander(f"🧠 Reasoning trace — {len(trace)} iterations"):
                for step in trace:
                    if step.get("tool_called"):
                        st.markdown(
                            f"**Iter {step['iteration']}** · "
                            f"`{step['tool_called']}({json.dumps(step.get('tool_args', {}))})`"
                        )
                    else:
                        st.markdown(f"**Iter {step['iteration']}** · _final answer_")

        st.session_state.messages.append({
            "role": "assistant",
            "agent": agent_choice,
            "content": answer,
            "trace": trace,
            "meta": meta,
        })
