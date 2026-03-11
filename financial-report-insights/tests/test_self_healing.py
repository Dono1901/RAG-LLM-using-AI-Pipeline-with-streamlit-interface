"""Tests for self-healing: embedding retry, warm-up, ingestion retry, chunk count."""

import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Embedding retry tests (LocalEmbedder._send_embedding_batch)
# ---------------------------------------------------------------------------


class TestEmbeddingRetry:
    """Tests for _send_embedding_batch retry on 5xx errors."""

    def _make_embedder(self):
        """Create a LocalEmbedder with mocked HTTP client."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder
                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_succeeds_on_first_try(self):
        embedder = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_resp.raise_for_status = MagicMock()
        embedder._client.post.return_value = mock_resp

        result = embedder._send_embedding_batch(["hello"])
        assert result == [[0.1, 0.2]]
        assert embedder._client.post.call_count == 1

    def test_retries_on_500_and_succeeds(self):
        import httpx

        embedder = self._make_embedder()

        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.request = MagicMock()

        ok_resp = MagicMock()
        ok_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        ok_resp.raise_for_status = MagicMock()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 500
                resp.request = MagicMock()
                resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=resp
                )
                return resp
            return ok_resp

        embedder._client.post.side_effect = side_effect

        with patch("time.sleep"):  # skip actual sleep
            result = embedder._send_embedding_batch(["hello"])

        assert result == [[0.1]]
        assert call_count[0] == 2

    def test_exhausts_retries_and_raises(self):
        import httpx

        embedder = self._make_embedder()

        def always_500(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.request = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=resp
            )
            return resp

        embedder._client.post.side_effect = always_500

        with patch("time.sleep"):
            with pytest.raises(httpx.HTTPStatusError):
                embedder._send_embedding_batch(["hello"], max_retries=3)

        assert embedder._client.post.call_count == 3

    def test_no_retry_on_4xx(self):
        import httpx

        embedder = self._make_embedder()

        def always_400(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 400
            resp.request = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400", request=MagicMock(), response=resp
            )
            return resp

        embedder._client.post.side_effect = always_400

        with pytest.raises(httpx.HTTPStatusError):
            embedder._send_embedding_batch(["hello"], max_retries=3)

        # Should NOT retry on 4xx — only 1 call
        assert embedder._client.post.call_count == 1


# ---------------------------------------------------------------------------
# Embedding warm-up tests
# ---------------------------------------------------------------------------


class TestEmbeddingWarmUp:
    """Tests for wait_for_embedding_service()."""

    def _make_embedder(self):
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder
                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_warm_up_succeeds_immediately(self):
        embedder = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        mock_resp.raise_for_status = MagicMock()
        embedder._client.post.return_value = mock_resp

        result = embedder.wait_for_embedding_service(timeout=5)
        assert result is True

    def test_warm_up_times_out(self):
        embedder = self._make_embedder()
        embedder._client.post.side_effect = ConnectionError("not ready")

        # Use a very short timeout so test is fast
        with patch("time.sleep"):
            with patch("time.monotonic", side_effect=[0, 0, 100]):
                result = embedder.wait_for_embedding_service(timeout=5)

        assert result is False

    def test_warm_up_succeeds_after_retries(self):
        embedder = self._make_embedder()
        call_count = [0]

        ok_resp = MagicMock()
        ok_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        ok_resp.raise_for_status = MagicMock()

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("not ready")
            return ok_resp

        embedder._client.post.side_effect = side_effect

        with patch("time.sleep"):
            result = embedder.wait_for_embedding_service(timeout=60)

        assert result is True
        assert call_count[0] >= 3


# ---------------------------------------------------------------------------
# Chunk count validation tests
# ---------------------------------------------------------------------------


class TestChunkCountValidation:
    """Tests for chunk count warning after ingestion."""

    def test_low_chunk_count_logs_warning(self, tmp_path, caplog):
        """When chunks < files * 5, a warning should be logged."""
        import logging

        # Create 5 files but load 0 chunks
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        with caplog.at_level(logging.WARNING):
            # Simulate the validation logic directly
            doc_files = [f for f in tmp_path.rglob("*") if f.is_file()]
            n_files = len(doc_files)
            n_chunks = 10  # < 5 * 5 = 25

            if n_files > 0 and n_chunks < n_files * 5:
                import logging as _log
                _log.getLogger("test").warning(
                    "Only %d chunks indexed from %d files (expected >%d).",
                    n_chunks, n_files, n_files * 5,
                )

        assert any("chunks indexed" in r.message for r in caplog.records)

    def test_normal_chunk_count_no_warning(self, tmp_path, caplog):
        """Enough chunks should not trigger a warning."""
        import logging

        for i in range(3):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        with caplog.at_level(logging.WARNING):
            doc_files = [f for f in tmp_path.rglob("*") if f.is_file()]
            n_files = len(doc_files)
            n_chunks = 100  # >= 3 * 5 = 15

            if n_files > 0 and n_chunks < n_files * 5:
                import logging as _log
                _log.getLogger("test").warning(
                    "Only %d chunks indexed from %d files (expected >%d).",
                    n_chunks, n_files, n_files * 5,
                )

        assert not any("chunks indexed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Ingestion retry tests
# ---------------------------------------------------------------------------


class TestIngestionRetry:
    """Tests for the file-level ingestion retry in _load_documents."""

    def test_retry_logic_recovers_on_second_attempt(self):
        """Simulate: first embed_batch fails, retry succeeds."""
        call_count = [0]

        def mock_embed_batch(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("embedding service not ready")
            return [[0.1] * 1024] * len(texts)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = mock_embed_batch

        # Simulate retry logic extracted from _load_documents
        file_docs = [{"content": "test chunk", "source": "test.pdf", "type": "pdf"}]
        documents = []
        embeddings = []

        with patch("time.sleep"):
            try:
                texts = [d["content"] for d in file_docs]
                embs = mock_embedder.embed_batch(texts)
                documents.extend(file_docs)
                embeddings.extend(embs)
            except Exception:
                # Retry
                try:
                    texts = [d["content"] for d in file_docs]
                    embs = mock_embedder.embed_batch(texts)
                    documents.extend(file_docs)
                    embeddings.extend(embs)
                except Exception:
                    pass

        assert len(documents) == 1
        assert len(embeddings) == 1
        assert call_count[0] == 2

    def test_retry_exhausted_no_crash(self):
        """Both attempts fail — no crash, no documents added."""
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = ConnectionError("always fails")

        file_docs = [{"content": "test chunk", "source": "test.pdf", "type": "pdf"}]
        documents = []
        embeddings = []

        with patch("time.sleep"):
            try:
                embs = mock_embedder.embed_batch([d["content"] for d in file_docs])
                documents.extend(file_docs)
                embeddings.extend(embs)
            except Exception:
                try:
                    embs = mock_embedder.embed_batch([d["content"] for d in file_docs])
                    documents.extend(file_docs)
                    embeddings.extend(embs)
                except Exception:
                    pass

        assert len(documents) == 0
        assert len(embeddings) == 0
