"""
FastAPI REST + SSE API layer for the RAG-LLM Financial Insights engine.
Wraps SimpleRAG, CharlieAnalyzer, and health checks as HTTP endpoints.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config import settings
from healthcheck import get_health_status
from local_llm import LLMConnectionError
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.max_query_length)
    top_k: int = Field(default=settings.top_k, ge=1, le=settings.max_top_k)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    document_count: int


class AnalyzeRequest(BaseModel):
    financial_data: Dict[str, Any]


class DocumentInfo(BaseModel):
    source: str
    type: str
    content_preview: str


# ---------------------------------------------------------------------------
# Application lifespan – lazy-init RAG on first real request
# ---------------------------------------------------------------------------

_rag_instance = None


def _get_rag():
    """Lazy-initialise the RAG singleton (avoids slow startup when importing)."""
    global _rag_instance
    if _rag_instance is None:
        import os
        from app_local import SimpleRAG
        _rag_instance = SimpleRAG(
            docs_folder="./documents",
            llm_model=os.getenv("OLLAMA_MODEL", settings.llm_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", settings.embedding_model),
        )
        logger.info("RAG engine initialised (%d chunks)", len(_rag_instance.documents))
    return _rag_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI starting up")
    yield
    logger.info("FastAPI shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial Report Insights API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Service health check."""
    status = get_health_status()
    code = 200 if status["healthy"] else 503
    if code == 503:
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Synchronous RAG question answering."""
    rag = _get_rag()

    # Check circuit breaker early
    if hasattr(rag.llm, "circuit_state") and rag.llm.circuit_state == "OPEN":
        raise HTTPException(
            status_code=503,
            detail="LLM circuit breaker is OPEN – service temporarily unavailable.",
        )

    try:
        relevant_docs = rag.retrieve(req.text, top_k=req.top_k)
        answer = await asyncio.to_thread(rag.answer, req.text, relevant_docs)
        sources = list({doc.get("source", "unknown") for doc in relevant_docs})
        return QueryResponse(
            answer=answer,
            sources=sources,
            document_count=len(relevant_docs),
        )
    except LLMConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query-stream")
async def query_stream(req: QueryRequest):
    """SSE streaming RAG question answering."""
    rag = _get_rag()

    if hasattr(rag.llm, "circuit_state") and rag.llm.circuit_state == "OPEN":
        raise HTTPException(
            status_code=503,
            detail="LLM circuit breaker is OPEN – service temporarily unavailable.",
        )

    relevant_docs = rag.retrieve(req.text, top_k=req.top_k)

    async def event_generator():
        try:
            for chunk in rag.answer_stream(req.text, relevant_docs):
                yield {"data": chunk}
            yield {"event": "done", "data": ""}
        except LLMConnectionError as exc:
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run CharlieAnalyzer on provided financial data."""
    rag = _get_rag()

    if not rag.charlie_analyzer:
        raise HTTPException(status_code=501, detail="Financial analyzer not available.")

    try:
        from financial_analyzer import FinancialData
        # Build FinancialData from the dict, ignoring unknown keys
        known_fields = {f.name for f in FinancialData.__dataclass_fields__.values()}
        filtered = {k: v for k, v in req.financial_data.items() if k in known_fields}
        data = FinancialData(**filtered)

        report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)
        return {
            "executive_summary": report.executive_summary,
            "sections": report.sections,
            "generated_at": report.generated_at,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List indexed document chunks."""
    rag = _get_rag()
    results = []
    seen = set()
    for doc in rag.documents:
        source = doc.get("source", "unknown")
        if source in seen:
            continue
        seen.add(source)
        results.append(DocumentInfo(
            source=source,
            type=doc.get("type", "unknown"),
            content_preview=doc.get("content", "")[:200],
        ))
    return results
