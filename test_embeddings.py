"""Smoke test: load the embedding model and look at what it produces."""

import numpy as np
from sentence_transformers import SentenceTransformer

# First time this runs, it downloads ~80MB of model weights. Cached after.
print("Loading model (first run downloads ~80MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"✅ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Three sentences: two about stalled deals, one about something unrelated
sentences = [
    "Champion left the company, deal frozen indefinitely",
    "Decision postponed, our sponsor departed for a new role",
    "Discovery call scheduled, researching their tech stack",
]

print(f"\nEmbedding {len(sentences)} sentences...")
vectors = model.encode(sentences)
print(f"Got vectors with shape: {vectors.shape}")  # (3, 384)

# Compute pairwise cosine similarity manually so you SEE what's happening
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("\nPairwise cosine similarity (1.0 = identical, 0 = orthogonal):")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(vectors[i], vectors[j])
        print(f"  [{i}] vs [{j}]: {sim:.3f}")
        print(f"      {sentences[i][:50]!r}")
        print(f"      {sentences[j][:50]!r}")