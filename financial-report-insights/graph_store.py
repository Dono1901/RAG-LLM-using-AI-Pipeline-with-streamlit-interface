"""
Neo4j graph+vector storage layer.

Optional backend that activates when NEO4J_URI is set.
Falls back gracefully so the system works identically without Neo4j.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _neo4j_configured() -> bool:
    """Return True if NEO4J_URI is set in the environment."""
    return bool(os.environ.get("NEO4J_URI", "").strip())


class Neo4jStore:
    """Persistent graph+vector store backed by Neo4j.

    Usage::

        store = Neo4jStore.connect()  # returns None if NEO4J_URI not set
        if store:
            store.ensure_schema(embedding_dim=1024, model_name="mxbai-embed-large")
            store.store_chunks(chunks, embeddings, doc_id="report.pdf")
            results = store.vector_search(query_embedding, top_k=5)
            store.close()
    """

    def __init__(self, driver):
        self._driver = driver

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls) -> Optional["Neo4jStore"]:
        """Create a Neo4jStore if NEO4J_URI is configured, else return None."""
        if not _neo4j_configured():
            return None

        try:
            import neo4j
            uri = os.environ["NEO4J_URI"]
            username = os.environ.get("NEO4J_USERNAME", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "password")
            driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
            driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", uri)
            return cls(driver)
        except Exception as exc:
            logger.warning("Neo4j connection failed (%s) – falling back to in-memory", exc)
            return None

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def ensure_schema(self, embedding_dim: int, model_name: str) -> None:
        """Create constraints and vector index if they don't exist."""
        from graph_schema import CONSTRAINTS, vector_index_statement

        with self._driver.session() as session:
            for stmt in CONSTRAINTS:
                session.run(stmt)
            session.run(vector_index_statement(embedding_dim, model_name))
        logger.info("Neo4j schema ensured (dim=%d, model=%s)", embedding_dim, model_name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_id: str,
    ) -> int:
        """Persist document chunks with embedding vectors.

        Args:
            chunks: List of chunk dicts with 'content', 'source', and optional metadata.
            embeddings: Parallel list of embedding vectors.
            doc_id: Document identifier (typically filename).

        Returns:
            Number of chunks stored.
        """
        from graph_schema import MERGE_DOCUMENT, MERGE_CHUNK

        stored = 0
        try:
            with self._driver.session() as session:
                # Ensure document node exists
                file_type = chunks[0].get("type", "unknown") if chunks else "unknown"
                session.run(MERGE_DOCUMENT, doc_id=doc_id, filename=doc_id, file_type=file_type)

                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = hashlib.sha256(
                        f"{doc_id}:{i}:{chunk.get('content', '')[:100]}".encode()
                    ).hexdigest()
                    session.run(
                        MERGE_CHUNK,
                        chunk_id=chunk_id,
                        content=chunk.get("content", ""),
                        embedding=embedding,
                        source=chunk.get("source", doc_id),
                        chunk_index=i,
                        doc_id=doc_id,
                    )
                    stored += 1

            logger.info("Stored %d chunks for %s in Neo4j", stored, doc_id)
        except Exception as exc:
            logger.warning("Neo4j store_chunks failed: %s", exc)

        return stored

    def store_financial_data(
        self,
        doc_id: str,
        period_label: str,
        ratios: Optional[Dict[str, Dict[str, Any]]] = None,
        scores: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Persist computed financial ratios and scores as graph nodes.

        Args:
            doc_id: Document the data comes from.
            period_label: Fiscal period label (e.g. "FY2024").
            ratios: Dict of ratio_name -> {value, category}.
            scores: Dict of model_name -> {value, grade, interpretation}.
        """
        from graph_schema import MERGE_FISCAL_PERIOD, MERGE_RATIO, MERGE_SCORE

        period_id = hashlib.sha256(f"{doc_id}:{period_label}".encode()).hexdigest()

        try:
            with self._driver.session() as session:
                session.run(MERGE_FISCAL_PERIOD, period_id=period_id, label=period_label, doc_id=doc_id)

                for name, data in (ratios or {}).items():
                    ratio_id = hashlib.sha256(f"{period_id}:{name}".encode()).hexdigest()
                    session.run(
                        MERGE_RATIO,
                        ratio_id=ratio_id,
                        name=name,
                        value=data.get("value"),
                        category=data.get("category", ""),
                        period_id=period_id,
                    )

                for model, data in (scores or {}).items():
                    score_id = hashlib.sha256(f"{period_id}:{model}".encode()).hexdigest()
                    session.run(
                        MERGE_SCORE,
                        score_id=score_id,
                        model=model,
                        value=data.get("value"),
                        grade=data.get("grade", ""),
                        interpretation=data.get("interpretation", ""),
                        period_id=period_id,
                    )

            logger.info("Stored financial data for %s / %s", doc_id, period_label)
        except Exception as exc:
            logger.warning("Neo4j store_financial_data failed: %s", exc)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        model_name: str = "mxbai-embed-large",
    ) -> List[Dict[str, Any]]:
        """Search chunks by embedding vector using Neo4j's HNSW index.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            model_name: Embedding model (determines index name).

        Returns:
            List of dicts with chunk_id, content, source, score.
        """
        from graph_schema import VECTOR_SEARCH

        safe_model = model_name.replace("-", "_").replace("/", "_")
        index_name = f"chunk_embedding_{safe_model}"

        try:
            with self._driver.session() as session:
                result = session.run(
                    VECTOR_SEARCH,
                    index_name=index_name,
                    top_k=top_k,
                    query_embedding=query_embedding,
                )
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("Neo4j vector_search failed: %s", exc)
            return []

    def graph_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        model_name: str = "mxbai-embed-large",
    ) -> List[Dict[str, Any]]:
        """Vector search + graph traversal for enriched financial context.

        Returns vector search results augmented with connected ratios and scores.
        """
        from graph_schema import GRAPH_CONTEXT_FOR_CHUNK

        vector_results = self.vector_search(query_embedding, top_k, model_name)

        enriched = []
        try:
            with self._driver.session() as session:
                for vr in vector_results:
                    context = session.run(
                        GRAPH_CONTEXT_FOR_CHUNK, chunk_id=vr["chunk_id"]
                    )
                    record = context.single()
                    entry = {**vr}
                    if record:
                        entry["document"] = record["document"]
                        entry["period"] = record["period"]
                        entry["ratios"] = record["ratios"]
                        entry["scores"] = record["scores"]
                    enriched.append(entry)
        except Exception as exc:
            logger.warning("Neo4j graph_search traversal failed: %s", exc)
            return vector_results  # degrade to vector-only

        return enriched

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Neo4j driver."""
        try:
            self._driver.close()
            logger.info("Neo4j connection closed")
        except Exception:
            pass
