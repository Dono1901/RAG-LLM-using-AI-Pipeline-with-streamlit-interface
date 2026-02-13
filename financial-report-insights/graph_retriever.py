"""
Graph-augmented financial retrieval.

Extends standard RAG retrieval with graph traversal capabilities:
- Vector search for relevant text chunks (via Neo4j HNSW or numpy)
- Graph traversal for connected financial metrics, ratios, and scores
- Structured context injection alongside text chunks
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def graph_enhanced_search(
    store,
    query_embedding: List[float],
    top_k: int = 5,
    model_name: str = "mxbai-embed-large",
) -> Dict[str, Any]:
    """Perform vector search with graph-traversal enrichment.

    Returns both text chunks and structured financial data connected
    to those chunks via the graph.

    Args:
        store: A Neo4jStore instance.
        query_embedding: Embedded query vector.
        top_k: Number of chunks to retrieve.
        model_name: Embedding model name (for index lookup).

    Returns:
        Dict with 'chunks' (text results) and 'financial_context' (structured data).
    """
    results = store.graph_search(query_embedding, top_k, model_name)

    chunks = []
    financial_context = []

    for r in results:
        chunks.append({
            "source": r.get("source", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        })

        # Collect connected financial data
        ratios = r.get("ratios", [])
        scores = r.get("scores", [])
        if ratios or scores:
            financial_context.append({
                "document": r.get("document", ""),
                "period": r.get("period", ""),
                "ratios": [rt for rt in ratios if rt.get("name")],
                "scores": [sc for sc in scores if sc.get("model")],
            })

    return {
        "chunks": chunks,
        "financial_context": financial_context,
    }


def format_graph_context(financial_context: List[Dict[str, Any]]) -> str:
    """Format graph-retrieved financial context as text for LLM prompts.

    Args:
        financial_context: List of dicts from graph_enhanced_search.

    Returns:
        Formatted string ready for prompt injection.
    """
    if not financial_context:
        return ""

    parts = []
    for ctx in financial_context:
        doc = ctx.get("document", "Unknown")
        period = ctx.get("period", "Unknown")
        header = f"[{doc} / {period}]"

        ratio_lines = []
        for r in ctx.get("ratios", []):
            name = r.get("name", "")
            value = r.get("value", "N/A")
            category = r.get("category", "")
            ratio_lines.append(f"  - {name}: {value} ({category})")

        score_lines = []
        for s in ctx.get("scores", []):
            model = s.get("model", "")
            value = s.get("value", "N/A")
            grade = s.get("grade", "")
            score_lines.append(f"  - {model}: {value} (Grade: {grade})")

        section = header
        if ratio_lines:
            section += "\n  Ratios:\n" + "\n".join(ratio_lines)
        if score_lines:
            section += "\n  Scores:\n" + "\n".join(score_lines)

        parts.append(section)

    return "\n\n".join(parts)


def persist_analysis_to_graph(
    store,
    doc_id: str,
    period_label: str,
    report,
) -> None:
    """Persist CharlieAnalyzer report results as graph nodes.

    Extracts ratios and scores from a FinancialReport and stores them
    in Neo4j for later graph-based retrieval.

    Args:
        store: A Neo4jStore instance.
        doc_id: Source document identifier.
        period_label: Fiscal period label (e.g., "FY2024").
        report: FinancialReport instance from CharlieAnalyzer.
    """
    if not store:
        return

    ratios = {}
    scores = {}

    # Extract ratio data from report sections
    ratio_section = report.sections.get("ratio_analysis", "")
    if ratio_section:
        for line in ratio_section.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("="):
                parts = line.split(":", 1)
                name = parts[0].strip().lower().replace(" ", "_")
                value_str = parts[1].strip()
                try:
                    # Try to parse numeric value
                    value = float(value_str.replace("%", "").replace("x", "").split()[0])
                    ratios[name] = {"value": value, "category": "computed"}
                except (ValueError, IndexError):
                    pass

    # Extract scoring data from report sections
    score_section = report.sections.get("scoring_models", "")
    if score_section:
        for line in score_section.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("="):
                parts = line.split(":", 1)
                model = parts[0].strip().lower().replace(" ", "_")
                rest = parts[1].strip()
                try:
                    value = float(rest.split()[0])
                    grade = ""
                    if "(" in rest and ")" in rest:
                        grade = rest[rest.index("(") + 1:rest.index(")")]
                    scores[model] = {"value": value, "grade": grade, "interpretation": rest}
                except (ValueError, IndexError):
                    pass

    if ratios or scores:
        store.store_financial_data(
            doc_id=doc_id,
            period_label=period_label,
            ratios=ratios,
            scores=scores,
        )
        logger.info("Persisted %d ratios and %d scores for %s/%s",
                     len(ratios), len(scores), doc_id, period_label)
