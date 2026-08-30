"""
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
http://localhost:8000
"""

import sys
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent))
from pipeline.rag_pipeline import ask
from retrieval.vectorstore import get_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("app")

app = FastAPI(
    title       = "InLeagle - Indian Banking Legal AI",
    description = "RAG assistant for Indian banking law",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────

class Message(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)

class QueryRequest(BaseModel):
    query:   str = Field(..., min_length=5, max_length=1000)
    mode:    str = Field(default="auto", pattern="^(auto|layman|legal)$")
    top_k:   int = Field(default=5, ge=1, le=10)
    history: list[Message] = Field(default=[])

class SourceChunk(BaseModel):
    document_name: str
    source_type:   str
    year:          str | int
    score:         float

class QueryResponse(BaseModel):
    query:     str
    answer:    str
    mode:      str
    citations: list[str]
    sources:   list[SourceChunk]
    error:     str | None

# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        client = get_client()
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        log.error(f"Qdrant health check failed: {e}")
        qdrant_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "service": "Indian Banking Legal AI",
        "qdrant": qdrant_status
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    log.info(f"Query [{request.mode}]: {request.query[:80]}")

    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    result = ask(
        query=request.query.strip(),
        mode=request.mode,
        top_k=request.top_k,
        history=history_dicts
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    sources, seen = [], set()
    for chunk in result.get("chunks", []):
        name = chunk.get("document_name", "")
        if name and name not in seen:
            seen.add(name)
            sources.append(SourceChunk(
                document_name=name,
                source_type=chunk.get("source_type", ""),
                year=chunk.get("year", ""),
                score=chunk.get("score", 0.0),
            ))
    return QueryResponse(
        query=result["query"], answer=result["answer"],
        mode=result["mode"],   citations=result["citations"],
        sources=sources,       error=result.get("error"),
    )

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>index.html not found in Rag/ folder</h2>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))