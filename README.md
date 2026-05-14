---
title: SalesAgent
emoji: 💼
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SalesAgent

An AI copilot for B2B sales reps. Multi-agent system with two specialized agents:

- **Lead Intel** — pre-call briefs synthesizing CRM context for any account
- **Deal Coach** — ReAct-powered analysis of why a deal is stuck and what to do

## Architecture

- **Lead Intel:** entity extraction (Groq Llama) → direct SQL orchestration → synthesis (Groq Llama)
- **Deal Coach:** ReAct loop over Gemini 2.5 Flash-Lite with 5 tools (CRM + RAG)
- **RAG:** sentence-transformers MiniLM-L6 embeddings + FAISS index over CRM activity notes
- **Backend:** FastAPI on internal port 8000
- **Frontend:** Streamlit on port 7860 (what you see)

## Tech stack

Python · FastAPI · Streamlit · SQLite · FAISS · sentence-transformers · Groq · Google Gemini · Docker

## First request after sleep

This Space sleeps after 48 hours of inactivity. The first request after wake takes ~60 seconds while the embedding model loads. Subsequent requests are fast.

## Repo

[github.com/samanvithaDasari/sales-agent](https://github.com/samanvithaDasari/sales-agent)
