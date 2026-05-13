"""
main.py — FastAPI service exposing the SalesAgent agents over HTTP.

ENDPOINTS:
  GET  /                       — health check + version
  POST /lead-intel             — Lead Intelligence brief for a company
  POST /deal-coach             — Deal Coach analysis for a stuck deal
  GET  /docs                   — interactive API docs (FastAPI auto-generates)

ARCHITECTURE NOTES (interview hooks):
- Pydantic models on requests/responses give us auto-validation + auto-docs.
- The agent functions are sync (they block on LLM calls). We define endpoints
  as `def` not `async def`, so FastAPI runs them in a worker thread pool
  instead of blocking the event loop. This is the correct pattern for
  sync I/O inside an ASGI server.
- CORS is wide-open in dev (`allow_origins=["*"]`) so the Streamlit UI we
  build tomorrow can hit it. We'd lock this down in production.
"""

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.lead_intel import run_lead_intel
from agents.deal_coach import run_deal_coach


# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="SalesAgent API",
    description="AI copilot for B2B sales reps: Lead Intel briefs and Deal Coach analysis.",
    version="0.1.0",
)

# CORS: in dev, accept requests from anywhere (the Streamlit UI's browser).
# In production we'd restrict to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS (Pydantic models for request/response validation)
# ============================================================
class LeadIntelRequest(BaseModel):
    query: str = Field(
        ...,  # ... means required
        min_length=2,
        max_length=500,
        description="Natural-language query about a company, e.g. 'Tell me about Pacheco-Smith'",
        examples=["Tell me about Davis and Sons"],
    )


class LeadIntelResponse(BaseModel):
    brief: str = Field(description="Markdown-formatted pre-call brief")
    elapsed_seconds: float = Field(description="Wall-clock time to generate the brief")


class DealCoachRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Natural-language question about a specific deal",
        examples=["Why is the Inc Streamline Magnetic Channels deal stuck?"],
    )


class DealCoachTraceStep(BaseModel):
    iteration: int
    tool_called: str | None
    tool_args: dict[str, Any] | None


class DealCoachResponse(BaseModel):
    answer: str = Field(description="Markdown-formatted coaching analysis")
    trace: list[DealCoachTraceStep] = Field(description="Per-iteration tool calls for transparency")
    iterations: int
    elapsed_seconds: float


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def health_check() -> dict[str, str]:
    """Health check + service metadata. Useful for uptime monitoring."""
    return {
        "service": "SalesAgent API",
        "version": "0.1.0",
        "status": "ok",
    }


@app.post("/lead-intel", response_model=LeadIntelResponse)
def lead_intel(req: LeadIntelRequest) -> LeadIntelResponse:
    """
    Generate a pre-call brief for a company.

    Pipeline: extraction LLM → CRM tools (direct) → synthesis LLM.
    Typical latency: ~5 seconds.
    """
    start = time.perf_counter()
    try:
        brief = run_lead_intel(req.query)
    except Exception as e:
        # Convert any agent-level failure into a 500 with a clean message.
        raise HTTPException(status_code=500, detail=f"Lead Intel failed: {e}")
    elapsed = time.perf_counter() - start
    return LeadIntelResponse(brief=brief, elapsed_seconds=round(elapsed, 2))


@app.post("/deal-coach", response_model=DealCoachResponse)
def deal_coach(req: DealCoachRequest) -> DealCoachResponse:
    """
    Run the Deal Coach ReAct agent for a deal question.

    Pipeline: Gemini-driven ReAct loop with 5 tools (CRM + RAG).
    Typical latency: ~30-60 seconds.
    """
    start = time.perf_counter()
    try:
        answer, trace = run_deal_coach(req.query, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deal Coach failed: {e}")
    elapsed = time.perf_counter() - start

    return DealCoachResponse(
        answer=answer,
        trace=[
            DealCoachTraceStep(
                iteration=t.iteration,
                tool_called=t.tool_called,
                tool_args=t.tool_args,
            )
            for t in trace
        ],
        iterations=len(trace),
        elapsed_seconds=round(elapsed, 2),
    )