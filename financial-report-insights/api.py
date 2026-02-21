"""
FastAPI REST + SSE API layer for the RAG-LLM Financial Insights engine.
Wraps SimpleRAG, CharlieAnalyzer, and health checks as HTTP endpoints.
"""

import asyncio
import io
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (per-IP, sliding window)
# ---------------------------------------------------------------------------

_RATE_WINDOW = 60  # seconds
_RATE_LIMIT = 60  # max requests per window
_rate_log: Dict[str, List[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP request rate limit. Skips /health for monitoring."""
    if request.url.path == "/health":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    # Prune expired entries
    timestamps = _rate_log[client_ip]
    cutoff = now - _RATE_WINDOW
    _rate_log[client_ip] = [t for t in timestamps if t > cutoff]
    if len(_rate_log[client_ip]) >= _RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )
    _rate_log[client_ip].append(now)
    return await call_next(request)


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
        data = _parse_financial_data(req.financial_data)

        report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)
        return {
            "executive_summary": report.executive_summary,
            "sections": report.sections,
            "generated_at": report.generated_at,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Graph-specific endpoints (Phase 3)
# ---------------------------------------------------------------------------


class PeriodContext(BaseModel):
    period_label: str
    ratios: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]


class RatioEntry(BaseModel):
    name: str
    value: Optional[float]
    category: str


def _require_graph_store():
    """Return the graph store or raise 501."""
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)
    if store is None:
        raise HTTPException(status_code=501, detail="Graph store not configured (NEO4J_URI not set).")
    return store


@app.get("/graph/context/{period_label}", response_model=PeriodContext)
async def graph_context(period_label: str):
    """Return all ratios and scores for a fiscal period."""
    store = _require_graph_store()
    ratios = await asyncio.to_thread(store.ratios_by_period_label, period_label)
    scores = await asyncio.to_thread(store.scores_by_period_label, period_label)
    return PeriodContext(period_label=period_label, ratios=ratios, scores=scores)


@app.get("/graph/ratios/{period_label}", response_model=List[RatioEntry])
async def graph_ratios(period_label: str, category: Optional[str] = None):
    """Return ratios for a fiscal period with optional category filter."""
    store = _require_graph_store()
    ratios = await asyncio.to_thread(store.ratios_by_period_label, period_label)
    if category:
        ratios = [r for r in ratios if r.get("category", "").lower() == category.lower()]
    return [RatioEntry(name=r["name"], value=r.get("value"), category=r.get("category", "")) for r in ratios]


# ---------------------------------------------------------------------------
# Multi-document comparison (Phase 4)
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    period_labels: List[str] = Field(..., min_length=2, max_length=10)


class PeriodDelta(BaseModel):
    ratio_name: str
    periods: Dict[str, Optional[float]]
    delta: Optional[float] = None


class CompareResponse(BaseModel):
    periods_compared: List[str]
    improvements: List[str]
    deteriorations: List[str]
    deltas: List[PeriodDelta]
    graph_trend_data: Optional[List[Dict[str, Any]]] = None
    summary: str


@app.post("/compare", response_model=CompareResponse)
async def compare_periods(req: CompareRequest):
    """Compare financial metrics across multiple fiscal periods."""
    rag = _get_rag()
    store = getattr(rag, "_graph_store", None)

    graph_trend_data = None
    deltas: List[PeriodDelta] = []
    improvements: List[str] = []
    deteriorations: List[str] = []

    # Graph path: query cross-period trends
    if store is not None:
        try:
            raw_trends = await asyncio.to_thread(
                store.cross_period_ratio_trend, req.period_labels
            )
            if raw_trends:
                graph_trend_data = raw_trends
                # Build deltas from graph data
                ratio_periods: Dict[str, Dict[str, Optional[float]]] = {}
                for row in raw_trends:
                    rname = row["ratio_name"]
                    ratio_periods.setdefault(rname, {})
                    ratio_periods[rname][row["period"]] = row.get("value")

                for rname, periods in ratio_periods.items():
                    values = [periods.get(p) for p in req.period_labels]
                    first_val = next((v for v in values if v is not None), None)
                    last_val = next((v for v in reversed(values) if v is not None), None)
                    delta = (last_val - first_val) if first_val is not None and last_val is not None else None
                    deltas.append(PeriodDelta(ratio_name=rname, periods=periods, delta=delta))
                    if delta is not None:
                        if delta > 0:
                            improvements.append(f"{rname}: +{delta:.4f}")
                        elif delta < 0:
                            deteriorations.append(f"{rname}: {delta:.4f}")
        except Exception as exc:
            logger.debug("Graph comparison failed, using in-memory: %s", exc)

    # In-memory fallback: use cached FinancialData
    if not deltas:
        period_data = getattr(rag, "_period_financial_data", {})
        # Ensure financial analysis context is computed
        if not period_data:
            await asyncio.to_thread(rag._get_financial_analysis_context)
            period_data = getattr(rag, "_period_financial_data", {})

        if period_data and rag.charlie_analyzer:
            try:
                from ratio_framework import run_all_ratios
                for label in req.period_labels:
                    fd = period_data.get(label)
                    if fd:
                        results = run_all_ratios(fd)
                        for key, result in results.items():
                            if result.value is not None:
                                found = False
                                for d in deltas:
                                    if d.ratio_name == result.name:
                                        d.periods[label] = result.value
                                        found = True
                                        break
                                if not found:
                                    deltas.append(PeriodDelta(
                                        ratio_name=result.name,
                                        periods={label: result.value},
                                    ))

                # Compute deltas between first and last period
                for d in deltas:
                    values = [d.periods.get(p) for p in req.period_labels]
                    first_val = next((v for v in values if v is not None), None)
                    last_val = next((v for v in reversed(values) if v is not None), None)
                    if first_val is not None and last_val is not None:
                        d.delta = last_val - first_val
                        if d.delta > 0:
                            improvements.append(f"{d.ratio_name}: +{d.delta:.4f}")
                        elif d.delta < 0:
                            deteriorations.append(f"{d.ratio_name}: {d.delta:.4f}")
            except Exception as exc:
                logger.debug("In-memory comparison failed: %s", exc)

    n_improvements = len(improvements)
    n_deteriorations = len(deteriorations)
    summary = (
        f"Compared {len(req.period_labels)} periods: "
        f"{n_improvements} improvements, {n_deteriorations} deteriorations."
    )

    return CompareResponse(
        periods_compared=req.period_labels,
        improvements=improvements,
        deteriorations=deteriorations,
        deltas=deltas,
        graph_trend_data=graph_trend_data,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    financial_data: Dict[str, Any]
    company_name: str = ""


@app.post("/export/xlsx")
async def export_xlsx(req: ExportRequest):
    """Export financial analysis as Excel workbook."""
    from export_xlsx import FinancialExcelExporter

    data = _parse_financial_data(req.financial_data)

    rag = _get_rag()
    if not rag.charlie_analyzer:
        raise HTTPException(status_code=501, detail="Financial analyzer not available.")

    analysis = await asyncio.to_thread(rag.charlie_analyzer.analyze, data)
    report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)

    exporter = FinancialExcelExporter()
    xlsx_bytes = exporter.export_full_report(data, analysis, report=report)

    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="financial_report.xlsx"'},
    )


@app.post("/export/pdf")
async def export_pdf(req: ExportRequest):
    """Export financial analysis as PDF report."""
    from export_pdf import FinancialPDFExporter

    data = _parse_financial_data(req.financial_data)

    rag = _get_rag()
    if not rag.charlie_analyzer:
        raise HTTPException(status_code=501, detail="Financial analyzer not available.")

    analysis = await asyncio.to_thread(rag.charlie_analyzer.analyze, data)
    report = await asyncio.to_thread(rag.charlie_analyzer.generate_report, data)

    exporter = FinancialPDFExporter()
    pdf_bytes = exporter.export_full_report(data, analysis, report=report)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="financial_report.pdf"'},
    )


# ---------------------------------------------------------------------------
# Portfolio endpoints (Phase 7)
# ---------------------------------------------------------------------------


class PortfolioRequest(BaseModel):
    companies: Dict[str, Dict[str, Any]] = Field(
        ..., min_length=1,
        description="Map of company_name -> financial_data dict (at least 1)",
    )


class PortfolioResponse(BaseModel):
    num_companies: int
    avg_health_score: float
    diversification_score: int
    diversification_grade: str
    risk_level: str
    risk_flags: List[str]
    strongest: str
    weakest: str
    summary: str


class CorrelationResponse(BaseModel):
    company_names: List[str]
    ratio_names: List[str]
    matrix: List[List[float]]
    avg_correlation: float
    interpretation: str


def _parse_financial_data(raw: Dict[str, Any]) -> "FinancialData":
    """Parse a raw dict into FinancialData, filtering unknown fields."""
    from financial_analyzer import FinancialData

    if not hasattr(_parse_financial_data, "_fields"):
        _parse_financial_data._fields = frozenset(FinancialData.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in _parse_financial_data._fields}
    return FinancialData(**filtered)


@app.post("/portfolio/analyze", response_model=PortfolioResponse)
async def portfolio_analyze(req: PortfolioRequest):
    """Run full portfolio analysis across multiple companies."""
    from portfolio_analyzer import PortfolioAnalyzer

    try:
        companies = {name: _parse_financial_data(raw) for name, raw in req.companies.items()}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid financial data: {exc}") from exc

    pa = PortfolioAnalyzer()
    report = await asyncio.to_thread(pa.full_portfolio_analysis, companies)

    return PortfolioResponse(
        num_companies=report.num_companies,
        avg_health_score=report.risk_summary.avg_health_score,
        diversification_score=report.diversification.overall_score,
        diversification_grade=report.diversification.grade,
        risk_level=report.risk_summary.overall_risk_level,
        risk_flags=report.risk_summary.risk_flags,
        strongest=report.risk_summary.strongest_company,
        weakest=report.risk_summary.weakest_company,
        summary=report.summary,
    )


@app.post("/portfolio/correlation", response_model=CorrelationResponse)
async def portfolio_correlation(req: PortfolioRequest):
    """Compute correlation matrix across portfolio companies."""
    from portfolio_analyzer import PortfolioAnalyzer

    try:
        companies = {name: _parse_financial_data(raw) for name, raw in req.companies.items()}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid financial data: {exc}") from exc

    pa = PortfolioAnalyzer()
    corr = await asyncio.to_thread(pa.correlation_matrix, companies)

    return CorrelationResponse(
        company_names=corr.company_names,
        ratio_names=corr.ratio_names,
        matrix=corr.matrix,
        avg_correlation=corr.avg_correlation,
        interpretation=corr.interpretation,
    )


# ---------------------------------------------------------------------------
# Compliance endpoints (Phase 7)
# ---------------------------------------------------------------------------


class ComplianceResponse(BaseModel):
    sox_risk: str
    sox_score: int
    sec_score: int
    sec_grade: str
    regulatory_pct: float
    regulatory_pass: int
    regulatory_fail: int
    audit_risk: str
    audit_score: int
    audit_grade: str
    going_concern: bool
    summary: str


@app.post("/compliance/analyze", response_model=ComplianceResponse)
async def compliance_analyze(req: AnalyzeRequest):
    """Run full compliance analysis on financial data."""
    from compliance_scorer import ComplianceScorer

    try:
        data = _parse_financial_data(req.financial_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid financial data: {exc}") from exc

    cs = ComplianceScorer()
    report = await asyncio.to_thread(cs.full_compliance_report, data)

    return ComplianceResponse(
        sox_risk=report.sox.overall_risk,
        sox_score=report.sox.risk_score,
        sec_score=report.sec.disclosure_score,
        sec_grade=report.sec.grade,
        regulatory_pct=report.regulatory.compliance_pct,
        regulatory_pass=report.regulatory.pass_count,
        regulatory_fail=report.regulatory.fail_count,
        audit_risk=report.audit_risk.risk_level,
        audit_score=report.audit_risk.score,
        audit_grade=report.audit_risk.grade,
        going_concern=report.audit_risk.going_concern_risk,
        summary=report.summary,
    )


@app.post("/compliance/sox")
async def compliance_sox(req: AnalyzeRequest):
    """Run SOX compliance check only."""
    from compliance_scorer import ComplianceScorer

    try:
        data = _parse_financial_data(req.financial_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid financial data: {exc}") from exc

    cs = ComplianceScorer()
    sox = await asyncio.to_thread(cs.sox_compliance, data)
    return {
        "overall_risk": sox.overall_risk,
        "risk_score": sox.risk_score,
        "flags": sox.flags,
        "material_weakness_indicators": sox.material_weakness_indicators,
        "significant_deficiency_indicators": sox.significant_deficiency_indicators,
        "checks_performed": sox.checks_performed,
        "checks_passed": sox.checks_passed,
    }


@app.post("/compliance/regulatory")
async def compliance_regulatory(req: AnalyzeRequest):
    """Run regulatory threshold check only."""
    from compliance_scorer import ComplianceScorer

    try:
        data = _parse_financial_data(req.financial_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid financial data: {exc}") from exc

    cs = ComplianceScorer()
    reg = await asyncio.to_thread(cs.regulatory_ratios, data)
    return {
        "pass_count": reg.pass_count,
        "fail_count": reg.fail_count,
        "compliance_pct": reg.compliance_pct,
        "critical_failures": reg.critical_failures,
        "thresholds": [
            {
                "rule_name": t.rule_name,
                "framework": t.framework,
                "metric_name": t.metric_name,
                "current_value": t.current_value,
                "threshold_value": t.threshold_value,
                "passes": t.passes,
                "severity": t.severity,
            }
            for t in reg.thresholds_checked
        ],
    }


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
