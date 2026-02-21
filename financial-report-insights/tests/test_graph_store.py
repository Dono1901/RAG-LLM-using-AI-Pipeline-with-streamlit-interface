"""Tests for the Neo4j graph store layer (graph_store.py + graph_schema.py)."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# graph_schema tests
# ---------------------------------------------------------------------------


class TestGraphSchema:
    def test_vector_index_statement_encodes_model_and_dim(self):
        from graph_schema import vector_index_statement
        stmt = vector_index_statement(1024, "mxbai-embed-large")
        assert "chunk_embedding_mxbai_embed_large" in stmt
        assert "1024" in stmt
        assert "cosine" in stmt

    def test_vector_index_statement_sanitises_slashes(self):
        from graph_schema import vector_index_statement
        stmt = vector_index_statement(384, "org/model-name")
        assert "org_model_name" in stmt
        assert "/" not in stmt.split("IF NOT EXISTS")[0]

    def test_vector_index_statement_rejects_injection_chars(self):
        from graph_schema import vector_index_statement
        malicious = "model`; DROP INDEX foo; //"
        stmt = vector_index_statement(512, malicious)
        # Cypher control chars stripped from index name portion
        idx_part = stmt.split("IF NOT EXISTS")[0]
        assert "`" not in idx_part
        assert ";" not in stmt.split("OPTIONS")[0]
        assert "//" not in idx_part
        # Dimension still correct
        assert "512" in stmt

    def test_constraints_are_idempotent(self):
        from graph_schema import CONSTRAINTS
        for c in CONSTRAINTS:
            assert "IF NOT EXISTS" in c

    def test_cypher_templates_exist(self):
        from graph_schema import (
            MERGE_DOCUMENT, MERGE_CHUNK, MERGE_FISCAL_PERIOD,
            MERGE_LINE_ITEM, MERGE_RATIO, MERGE_SCORE,
            MERGE_CHUNKS_BATCH, MERGE_RATIOS_BATCH, MERGE_SCORES_BATCH,
            VECTOR_SEARCH, GRAPH_CONTEXT_FOR_CHUNK, GRAPH_CONTEXT_FOR_CHUNKS_BATCH,
            RATIOS_BY_PERIOD, SCORES_BY_PERIOD,
        )
        # All should be non-empty strings
        for tmpl in [
            MERGE_DOCUMENT, MERGE_CHUNK, MERGE_FISCAL_PERIOD,
            MERGE_LINE_ITEM, MERGE_RATIO, MERGE_SCORE,
            MERGE_CHUNKS_BATCH, MERGE_RATIOS_BATCH, MERGE_SCORES_BATCH,
            VECTOR_SEARCH, GRAPH_CONTEXT_FOR_CHUNK, GRAPH_CONTEXT_FOR_CHUNKS_BATCH,
            RATIOS_BY_PERIOD, SCORES_BY_PERIOD,
        ]:
            assert isinstance(tmpl, str) and len(tmpl) > 10


# ---------------------------------------------------------------------------
# Neo4jStore unit tests (mocked driver)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture()
def store(mock_driver):
    from graph_store import Neo4jStore
    driver, _ = mock_driver
    return Neo4jStore(driver)


class TestNeo4jStoreConnect:
    def test_returns_none_when_uri_not_set(self):
        from graph_store import Neo4jStore
        with patch.dict("os.environ", {}, clear=True):
            assert Neo4jStore.connect() is None

    def test_returns_none_when_uri_empty(self):
        from graph_store import Neo4jStore
        with patch.dict("os.environ", {"NEO4J_URI": "  "}):
            assert Neo4jStore.connect() is None

    def test_returns_store_on_success(self):
        from graph_store import Neo4jStore
        mock_driver = MagicMock()
        with patch.dict("os.environ", {"NEO4J_URI": "bolt://localhost:7687"}):
            with patch("neo4j.GraphDatabase") as mock_gdb:
                mock_gdb.driver.return_value = mock_driver
                store = Neo4jStore.connect()
                assert store is not None
                mock_driver.verify_connectivity.assert_called_once()

    def test_returns_none_on_connection_failure(self):
        from graph_store import Neo4jStore
        with patch.dict("os.environ", {"NEO4J_URI": "bolt://badhost:9999"}):
            with patch("neo4j.GraphDatabase") as mock_gdb:
                mock_gdb.driver.side_effect = Exception("Connection refused")
                assert Neo4jStore.connect() is None


class TestNeo4jStoreEnsureSchema:
    def test_runs_constraints_and_vector_index(self, store, mock_driver):
        _, session = mock_driver
        store.ensure_schema(1024, "mxbai-embed-large")
        # Should have run constraints + vector index
        from graph_schema import CONSTRAINTS
        assert session.run.call_count >= len(CONSTRAINTS) + 1


class TestNeo4jStoreChunks:
    def test_store_chunks_creates_nodes(self, store, mock_driver):
        _, session = mock_driver
        chunks = [
            {"content": "Revenue was $1M", "source": "report.pdf", "type": "pdf"},
            {"content": "Assets are $5M", "source": "report.pdf", "type": "pdf"},
        ]
        embeddings = [[0.1] * 10, [0.2] * 10]
        result = store.store_chunks(chunks, embeddings, "report.pdf")
        assert result == 2
        # MERGE_DOCUMENT + 1x MERGE_CHUNKS_BATCH (batched UNWIND)
        assert session.run.call_count == 2

    def test_store_chunks_handles_failure(self, store, mock_driver):
        _, session = mock_driver
        session.run.side_effect = Exception("Neo4j down")
        result = store.store_chunks([{"content": "x"}], [[0.1]], "f.pdf")
        assert result == 0  # graceful failure


class TestNeo4jStoreFinancialData:
    def test_store_ratios_and_scores(self, store, mock_driver):
        _, session = mock_driver
        store.store_financial_data(
            doc_id="report.pdf",
            period_label="FY2024",
            ratios={
                "current_ratio": {"value": 2.1, "category": "liquidity"},
                "debt_to_equity": {"value": 0.5, "category": "leverage"},
            },
            scores={
                "altman_z": {"value": 3.2, "grade": "Safe", "interpretation": "Low risk"},
            },
        )
        # MERGE_FISCAL_PERIOD + 1 MERGE_RATIOS_BATCH + 1 MERGE_SCORES_BATCH
        assert session.run.call_count == 3


class TestNeo4jStoreVectorSearch:
    def test_vector_search_returns_results(self, store, mock_driver):
        _, session = mock_driver
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"chunk_id": "abc", "content": "Revenue data", "source": "r.pdf", "score": 0.95},
        ]))
        session.run.return_value = mock_result

        results = store.vector_search([0.1] * 10, top_k=3, model_name="mxbai-embed-large")
        assert len(results) == 1
        assert results[0]["content"] == "Revenue data"

    def test_vector_search_returns_empty_on_failure(self, store, mock_driver):
        _, session = mock_driver
        session.run.side_effect = Exception("index not found")
        results = store.vector_search([0.1] * 10)
        assert results == []


class TestNeo4jStoreGraphSearch:
    def test_graph_search_enriches_results(self, store, mock_driver):
        _, session = mock_driver

        # First call is vector search, second is batched graph context
        vector_result = MagicMock()
        vector_result.__iter__ = MagicMock(return_value=iter([
            {"chunk_id": "abc", "content": "Revenue data", "source": "r.pdf", "score": 0.95},
        ]))
        graph_batch_result = MagicMock()
        graph_batch_result.__iter__ = MagicMock(return_value=iter([
            {
                "chunk_id": "abc",
                "document": "r.pdf",
                "period": "FY2024",
                "ratios": [{"name": "current_ratio", "value": 2.0, "category": "liquidity"}],
                "scores": [{"model": "altman_z", "value": 3.0, "grade": "Safe"}],
            },
        ]))

        session.run.side_effect = [vector_result, graph_batch_result]
        results = store.graph_search([0.1] * 10, top_k=3)
        assert len(results) == 1
        assert results[0]["document"] == "r.pdf"
        assert results[0]["ratios"][0]["name"] == "current_ratio"


class TestNeo4jStoreClose:
    def test_close_closes_driver(self, store, mock_driver):
        driver, _ = mock_driver
        store.close()
        driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration with SimpleRAG (NEO4J_URI unset = in-memory only)
# ---------------------------------------------------------------------------


class TestSimpleRAGGraphIntegration:
    """Verify SimpleRAG still works without Neo4j (backward compat)."""

    def test_graph_store_is_none_without_neo4j(self):
        """When NEO4J_URI is not set, _graph_store should be None."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure NEO4J_URI is not set
            import os
            os.environ.pop("NEO4J_URI", None)

            from graph_store import Neo4jStore
            store = Neo4jStore.connect()
            assert store is None


# ---------------------------------------------------------------------------
# Phase 1: Graph-enhanced retrieval wiring tests
# ---------------------------------------------------------------------------


class TestNeo4jStoreLineItems:
    """Phase 2: FinancialStatement + LineItem node creation."""

    def test_store_line_items_creates_nodes(self, store, mock_driver):
        _, session = mock_driver
        from financial_analyzer import FinancialData
        fd = FinancialData(revenue=1_000_000, net_income=200_000, total_assets=5_000_000)
        period_id = "test_period_hash"
        count = store.store_line_items(fd, period_id)
        assert count >= 3  # revenue, net_income, total_assets at minimum
        assert session.run.call_count >= 2  # at least 1 statement + 1 batch

    def test_store_line_items_skips_none_values(self, store, mock_driver):
        _, session = mock_driver
        from financial_analyzer import FinancialData
        fd = FinancialData(revenue=100)  # Only revenue set, rest None
        count = store.store_line_items(fd, "p1")
        assert count == 1  # Only revenue stored

    def test_store_line_items_handles_failure(self, store, mock_driver):
        _, session = mock_driver
        session.run.side_effect = Exception("Neo4j down")
        from financial_analyzer import FinancialData
        fd = FinancialData(revenue=100, total_assets=500)
        count = store.store_line_items(fd, "p1")
        assert count == 0


class TestNeo4jStoreDerivedFromEdges:
    """Phase 2: DERIVED_FROM edge creation from RATIO_CATALOG."""

    def test_store_derived_from_edges_links_ratios(self, store, mock_driver):
        _, session = mock_driver
        # Use a known period_id that matches the hashing scheme
        import hashlib
        period_id = "test_period_id"
        count = store.store_derived_from_edges(period_id)
        # Should create edges for all catalog ratios that have matching fields
        assert count > 0
        session.run.assert_called_once()

    def test_store_derived_from_edges_handles_failure(self, store, mock_driver):
        _, session = mock_driver
        session.run.side_effect = Exception("Neo4j down")
        count = store.store_derived_from_edges("p1")
        assert count == 0


class TestNeo4jStorePeriodLabelReads:
    """Phase 2/3: Period-label-based read methods."""

    def test_ratios_by_period_label(self, store, mock_driver):
        _, session = mock_driver
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"name": "current_ratio", "value": 2.1, "category": "liquidity"},
        ]))
        session.run.return_value = mock_result
        results = store.ratios_by_period_label("FY2024")
        assert len(results) == 1
        assert results[0]["name"] == "current_ratio"

    def test_scores_by_period_label(self, store, mock_driver):
        _, session = mock_driver
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"model": "altman_z", "value": 3.2, "grade": "Safe", "interpretation": "Low risk"},
        ]))
        session.run.return_value = mock_result
        results = store.scores_by_period_label("FY2024")
        assert len(results) == 1
        assert results[0]["model"] == "altman_z"


class TestSemanticSearchCallsGraphSearch:
    """Verify _semantic_search uses graph_search when store is present."""

    def test_semantic_search_calls_graph_search_when_store_present(self):
        from app_local import SimpleRAG
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 10
        mock_store = MagicMock()
        mock_store.graph_search.return_value = [
            {
                "chunk_id": "c1",
                "content": "Revenue $1M",
                "source": "report.pdf",
                "score": 0.9,
                "document": "report.pdf",
                "period": "FY2024",
                "ratios": [{"name": "current_ratio", "value": 2.0, "category": "liquidity"}],
                "scores": [],
            }
        ]

        rag = SimpleRAG.__new__(SimpleRAG)
        rag.documents = [{"source": "report.pdf", "content": "Revenue $1M", "type": "pdf"}]
        rag.embeddings = [[0.1] * 10]
        rag._doc_matrix = None
        rag._doc_norms = None
        rag.embedder = mock_embedder
        rag._graph_store = mock_store
        rag._embedding_model_name = "test-model"

        results = rag._semantic_search("What is the revenue?", top_k=3)
        mock_store.graph_search.assert_called_once()
        assert len(results) == 1
        assert results[0]["_graph_context"]["period"] == "FY2024"

    def test_graph_context_in_financial_prompt(self):
        from app_local import SimpleRAG
        rag = SimpleRAG.__new__(SimpleRAG)
        rag._financial_analysis_cache = ""
        rag._charlie_analyzer = MagicMock()

        relevant_docs = [
            {
                "source": "r.pdf",
                "content": "data",
                "type": "unknown",
                "_graph_context": {
                    "document": "r.pdf",
                    "period": "FY2024",
                    "ratios": [{"name": "current_ratio", "value": 2.1, "category": "liquidity"}],
                    "scores": [],
                },
            }
        ]

        prompt = rag._build_financial_prompt(
            "What is the current ratio?", "context text", [], relevant_docs
        )
        assert "GRAPH-RETRIEVED FINANCIAL METRICS" in prompt
        assert "current_ratio" in prompt
        assert "FY2024" in prompt

    def test_degrades_to_numpy_when_graph_empty(self):
        """When graph_search returns empty, falls back to numpy."""
        from app_local import SimpleRAG
        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 10
        mock_store = MagicMock()
        mock_store.graph_search.return_value = []

        rag = SimpleRAG.__new__(SimpleRAG)
        rag.documents = [
            {"source": "a.pdf", "content": "Revenue data", "type": "pdf"},
        ]
        rag.embeddings = [[0.1] * 10]
        rag._doc_matrix = np.asarray([[0.1] * 10], dtype=np.float32)
        rag._doc_norms = np.array([np.linalg.norm([0.1] * 10)])
        rag.embedder = mock_embedder
        rag._graph_store = mock_store
        rag._embedding_model_name = "test-model"

        results = rag._semantic_search("test query", top_k=1)
        # Should have fallen through to numpy path
        assert len(results) == 1
        assert results[0]["source"] == "a.pdf"
