# semantic_search_mcp_server.py

import json
from typing import List, Dict

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from fastmcp import FastMCP

# -------------------------------------------------------------------
# MCP server instance
# -------------------------------------------------------------------
mcp = FastMCP("Semantic-Search-MD-SOAR")

# -------------------------------------------------------------------
# Config – you can also move these to env vars if you like
# -------------------------------------------------------------------
VECTOR_CSV = "vectors.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------------------------------------------
# Load model & vectors once at startup
# -------------------------------------------------------------------
print("Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME)

print(f"Loading vectors from {VECTOR_CSV} ...")
df = pd.read_csv(VECTOR_CSV)
df["vector"] = df["vector"].apply(lambda x: json.loads(x))  # JSON string → list

# -------------------------------------------------------------------
# Helper functions (adapted from your script)
# -------------------------------------------------------------------
def _find_closest_matches(query: str, top_n: int = 5) -> List[Dict]:
    query_vector = model.encode(query).tolist()

    similarities: List[Dict] = []
    for _, row in df.iterrows():
        similarity = 1 - cosine(query_vector, row["vector"])
        similarities.append(
            {
                "filename": row["filename"],
                "score": float(similarity),
                "method": "cosine",
            }
        )

    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_n]


def _hybrid_search(query: str, top_n: int = 5) -> List[Dict]:
    query_vector = model.encode(query).tolist()
    keywords = set(query.lower().split())

    results: List[Dict] = []
    for _, row in df.iterrows():
        semantic_score = float(np.dot(query_vector, row["vector"]))
        filename_words = set(row["filename"].lower().split("_"))
        keyword_match_score = len(keywords & filename_words) * 0.1

        results.append(
            {
                "filename": row["filename"],
                "score": semantic_score + keyword_match_score,
                "semantic_score": semantic_score,
                "keyword_boost": keyword_match_score,
                "method": "hybrid",
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# -------------------------------------------------------------------
# MCP tool
# -------------------------------------------------------------------
@mcp.tool
def semantic_search_mdsoar(query: str, top_n: int = 5) -> List[Dict]:
    """
    Run a semantic search over the MD-SOAR vector index.

    For queries > 5 words: cosine similarity.
    For shorter queries: hybrid (dot-product + filename keyword boost).

    Returns a list of {filename, score, ...}.
    """
    word_count = len(query.split())
    if word_count > 5:
        return _find_closest_matches(query, top_n=top_n)
    else:
        return _hybrid_search(query, top_n=top_n)


# -------------------------------------------------------------------
# HTTP entrypoint for local / K8s usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Run MCP server over HTTP so it is reachable at http://localhost:8000/mcp
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
    )
