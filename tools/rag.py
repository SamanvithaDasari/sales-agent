import os
"""
rag.py — RAG index over CRM activity notes.

Index time (run once or whenever the CRM changes):
    python -m tools.rag --rebuild

Query time:
    from tools.rag import find_similar_activities
    matches = find_similar_activities("deal stuck on pricing", k=5)

ARCHITECTURE NOTES (interview hooks):
- Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80MB, CPU-fast)
- Index type: FAISS IndexFlatIP — exact (not approximate) inner-product search.
  For 660 vectors this is the right choice; approximate indexes (IVF, HNSW)
  only earn their cost above ~100K vectors. Below that, exact search is faster.
- Normalization: we L2-normalize all vectors before indexing, so inner product
  == cosine similarity. This is the standard trick to use FAISS's fast IP search
  for what is really cosine search.
- Persistence: index + metadata are jsoned to disk. Re-embedding 660 notes
  takes ~10 seconds; loading from disk takes <1 second.
"""

import json
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
DB_PATH = Path(os.environ["SALESAGENT_DATA_DIR"]) / "salesagent.db" if "SALESAGENT_DATA_DIR" in os.environ else PROJECT_ROOT / "db" / "salesagent.db"
INDEX_DIR = (Path(os.environ["SALESAGENT_DATA_DIR"]) / "rag_index") if "SALESAGENT_DATA_DIR" in os.environ else (PROJECT_ROOT / "db" / "rag_index")
INDEX_PATH = INDEX_DIR / "activities.faiss"
META_PATH = INDEX_DIR / "activities.meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # MiniLM-L6's output dimension


# ============================================================
# DATA SHAPES
# ============================================================
@dataclass
class ActivityRecord:
    """One activity note with its surrounding context."""
    activity_id: int
    deal_id: int
    deal_name: str
    deal_stage: str
    account_id: int
    account_name: str
    type: str          # call / email / meeting / note
    notes: str
    created_at: str


@dataclass
class SimilarityMatch:
    """A retrieved activity plus its similarity score."""
    record: ActivityRecord
    score: float       # cosine similarity, 0–1


# ============================================================
# MODEL LOADING (lazy, singleton-style)
# ============================================================
_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it. Avoids re-loading on each call."""
    global _model
    if _model is None:
        print(f"Loading embedding model {MODEL_NAME!r}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts and L2-normalize the vectors so that
    FAISS inner-product search == cosine similarity.
    """
    model = get_model()
    # convert_to_numpy: return numpy array; show_progress_bar=False: quiet
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-normalize so dot product == cosine sim. FAISS has a helper for this.
    faiss.normalize_L2(vectors)
    return vectors


# ============================================================
# INDEX BUILD
# ============================================================
def fetch_all_activities() -> list[ActivityRecord]:
    """Pull every activity with its parent deal and account joined in."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT act.id          AS activity_id,
                  act.deal_id     AS deal_id,
                  d.name          AS deal_name,
                  d.stage         AS deal_stage,
                  d.account_id    AS account_id,
                  a.name          AS account_name,
                  act.type        AS type,
                  act.notes       AS notes,
                  act.created_at  AS created_at
           FROM activities act
           INNER JOIN deals    d ON act.deal_id = d.id
           INNER JOIN accounts a ON d.account_id = a.id
           ORDER BY act.id"""
    ).fetchall()
    conn.close()
    return [ActivityRecord(**dict(r)) for r in rows]


def build_index() -> None:
    """Embed every activity note and persist the FAISS index + metadata to disk."""
    print("📥 Fetching activities from CRM...")
    records = fetch_all_activities()
    print(f"   {len(records)} activities to index")

    # Build an enriched "text to embed" string per record. We don't just embed
    # the raw notes — we include the deal stage so a query like "stuck in
    # negotiation" can match notes whose stage is 'negotiation' even if the
    # word doesn't appear in the note itself.
    texts_to_embed = [
        f"[{r.deal_stage}] [{r.type}] {r.notes}"
        for r in records
    ]

    print("🧮 Embedding (~10 sec for 660 notes on CPU)...")
    vectors = embed_texts(texts_to_embed)
    print(f"   Vector matrix shape: {vectors.shape}")

    # FAISS IndexFlatIP: exact inner-product search.
    # For our scale (~hundreds of vectors), exact beats approximate on both
    # speed and accuracy. We'd switch to IVFFlat or HNSW above ~100K vectors.
    print("🏗️  Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    print(f"   Index contains {index.ntotal} vectors")

    # Persist both the index and the metadata side-by-side
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w") as f:
        json.dump([asdict(r) for r in records], f)

    print(f"✅ Index saved to {INDEX_PATH}")
    print(f"✅ Metadata saved to {META_PATH}")


# ============================================================
# QUERY TIME
# ============================================================
_index_cache: faiss.Index | None = None
_meta_cache: list[ActivityRecord] | None = None


def _load_index() -> tuple[faiss.Index, list[ActivityRecord]]:
    """Load the FAISS index and metadata once, cache for subsequent calls."""
    global _index_cache, _meta_cache
    if _index_cache is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"No index at {INDEX_PATH}. Run: python -m tools.rag --rebuild"
            )
        _index_cache = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r") as f:
            raw = json.load(f)
        _meta_cache = [ActivityRecord(**r) for r in raw]
    return _index_cache, _meta_cache


def find_similar_activities(query: str, k: int = 5) -> list[SimilarityMatch]:
    """
    Return the top-k activity notes most semantically similar to `query`.

    Args:
        query: a natural-language query, e.g. "deals stuck on pricing".
        k: how many results to return.

    Returns:
        List of SimilarityMatch, ordered by similarity (highest first).
    """
    index, meta = _load_index()
    query_vec = embed_texts([query])  # shape (1, 384), already normalized
    # FAISS search returns (scores, indices). With IP + normalized vectors,
    # scores ARE cosine similarities, range -1 to 1.
    scores, indices = index.search(query_vec, k)

    matches = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS uses -1 for "no result" (rare with k <= ntotal)
        matches.append(SimilarityMatch(record=meta[idx], score=float(score)))
    return matches


# ============================================================
# CLI
# ============================================================
def main() -> None:
    if "--rebuild" in sys.argv:
        build_index()
        return

    # Default: run a few demo queries against the existing index
    print("Demo queries against the RAG index:\n")
    demo_queries = [
        "deal stuck because the buyer's champion left",
        "pricing pushback, customer wants discount",
        "legal review and contract redlines",
        "lost to a cheaper competitor",
    ]
    for query in demo_queries:
        print(f"🔍 Query: {query!r}")
        matches = find_similar_activities(query, k=3)
        for i, m in enumerate(matches, 1):
            print(f"  {i}. [{m.score:.3f}] [{m.record.deal_stage}] "
                  f"{m.record.account_name} — {m.record.notes[:80]}")
        print()


if __name__ == "__main__":
    main()