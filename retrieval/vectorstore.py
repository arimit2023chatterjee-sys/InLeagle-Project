# For Local vectorstore retrieval using Qdrant and HuggingFaceEmbeddings

# QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "InLegalDocs")
# MODEL_NAME      = os.getenv("EMBEDDING_MODEL", "bhavyagiri/InLegal-Sbert")
# TOP_K           = 5 

# def get_embeddings() -> HuggingFaceEmbeddings:
#     global _embeddings
#     if _embeddings is None:
#         log.info(f"Loading embedding model: {MODEL_NAME}")
#         _embeddings = HuggingFaceEmbeddings(
#             model_name=MODEL_NAME,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )
#     return _embeddings

# def get_client() -> QdrantClient:
#     global _client
#     if _client is None:
#         _client = QdrantClient(url=QDRANT_URL)
#         log.info(f"Qdrant connected: {QDRANT_URL}")
#     return _client


""" No LangChain wrapper here as directly Qdrant client gives us
full control over filtering and scoring.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()
log = logging.getLogger("retrieval")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

QDRANT_URL      = os.getenv("QDRANT_CLOUD_CLUSTER_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_CLOUD_API_KEY", "")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "")
MODEL_NAME      = os.getenv("EMBEDDING_MODEL", "")
TOP_K           = 5   # number of chunks to retrieve per query


# ─────────────────────────────────────────────────────────────
# SINGLETON LOADER
# Load model and client once — reused across all queries
# ─────────────────────────────────────────────────────────────

_embeddings: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def get_embeddings() -> SentenceTransformer:
    global _embeddings
    if _embeddings is None:
        log.info(f"Loading embedding model: {MODEL_NAME}")
        _embeddings = SentenceTransformer(
            MODEL_NAME,
            cache_folder=os.getenv("HF_HOME", "/app/hf_cache")
        )
    return _embeddings


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=30)
        log.info(f"Qdrant connected: {QDRANT_URL}")
    return _client

# ─────────────────────────────────────────────────────────────
# CORE SEARCH
# ─────────────────────────────────────────────────────────────

def search(
    query:       str,
    top_k:       int            = TOP_K,
    source_type: Optional[str]  = None,   # "act" | "rbi_circular" | "case"
    doc_name:    Optional[str]  = None,   # e.g. "SARFAESI Act 2002"
) -> list[dict]:
    """
    Embed query → search Qdrant → return top-K chunks.

    Each returned dict has:
        text          : the chunk text
        document_name : which document it came from
        source_type   : act / rbi_circular / case
        year          : year of the document
        score         : cosine similarity score (0–1)
        + any other metadata stored at ingest time
    """
    embeddings = get_embeddings()
    client     = get_client()

    # Embed the query
    query_vector = embeddings.encode(query, normalize_embeddings=True).tolist()

    # Build optional metadata filter
    qdrant_filter = None
    conditions    = []

    if source_type:
        conditions.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )
    if doc_name:
        conditions.append(
            FieldCondition(key="document_name", match=MatchValue(value=doc_name))
        )
    if conditions:
        qdrant_filter = Filter(must=conditions)

    # Search
    hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=qdrant_filter,
    limit=top_k,
    with_payload=True,
    score_threshold=0.3,
    ).points

    # Normalise into clean dicts
    results = []
    for hit in hits:
        payload  = hit.payload or {}
        metadata = payload.get("metadata", payload)  
        results.append({
            "text":          payload.get("page_content") or metadata.get("text", ""),
            "document_name": metadata.get("document_name", "Unknown"),
            "source_type":   metadata.get("source_type", "unknown"),
            "year":          metadata.get("year", ""),
            "court":         metadata.get("court", ""),
            "score":         round(hit.score, 4),
        })

    log.info(f"Query: '{query[:60]}...' → {len(results)} chunks retrieved")
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    test_queries = [
        "bank sent notice to take my property what can I do",
        "cheque bounce case filed against me what happens",
        "loan app calling my family members is that legal",
        "bank froze my account without explanation",
        "I am guarantor company went bankrupt am I liable",
    ]
    print("\n" + "=" * 65)
    print("  RETRIEVAL TEST")
    print("=" * 65)
    for q in test_queries:
        print(f"\n🔍 {q}")
        print("─" * 65)
        results = search(q, top_k=2)
        for i, r in enumerate(results):
            print(f"  #{i+1}  [{r['score']}]  {r['document_name']}")
            print(f"       {r['text'][:160]}...")