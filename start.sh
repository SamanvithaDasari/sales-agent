#!/bin/bash
set -e

echo "🌱 Seeding CRM database..."
python db/seed_crm.py

echo "🧮 Building RAG index..."
python -m tools.rag --rebuild

echo "🚀 Starting FastAPI on internal port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Give uvicorn a few seconds to bind before Streamlit tries to call it
sleep 5

echo "🎨 Starting Streamlit on port 7860..."
streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
