"""
Tests for vector_index.py – NumpyFlatIndex, FAISSIndex, HNSWIndex, and create_index.

NumpyFlatIndex is tested exhaustively because it is the always-available fallback.
FAISSIndex and HNSWIndex are tested via mocked imports so the suite runs on any
environment regardless of whether faiss or hnswlib are installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the package root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_index import (
    NumpyFlatIndex,
    VectorIndex,
    create_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for fast tests


def _unit_vec(values: List[float]) -> List[float]:
    """Return an L2-normalised version of *values*."""
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def _random_vecs(n: int, dim: int = DIM, seed: int = 0) -> List[List[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).tolist()


# ===========================================================================
# NumpyFlatIndex
# ===========================================================================


class TestNumpyFlatIndexBasic:
    """Basic construction and length tests."""

    def test_empty_index_has_zero_length(self):
        idx = NumpyFlatIndex(DIM)
        assert len(idx) == 0

    def test_add_single_vector(self):
        idx = NumpyFlatIndex(DIM)
        idx.add([_random_vecs(1)[0]], [0])
        assert len(idx) == 1

    def test_add_multiple_vectors(self):
        idx = NumpyFlatIndex(DIM)
        vecs = _random_vecs(10)
        ids = list(range(10))
        idx.add(vecs, ids)
        assert len(idx) == 10

    def test_add_incremental_extends_length(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(5), list(range(5)))
        idx.add(_random_vecs(3, seed=1), [10, 11, 12])
        assert len(idx) == 8

    def test_add_empty_list_is_noop(self):
        idx = NumpyFlatIndex(DIM)
        idx.add([], [])
        assert len(idx) == 0

    def test_add_mismatched_lengths_raises(self):
        idx = NumpyFlatIndex(DIM)
        with pytest.raises(ValueError, match="same length"):
            idx.add(_random_vecs(3), [0, 1])  # 3 vecs, 2 ids

    def test_add_wrong_dimension_raises(self):
        idx = NumpyFlatIndex(DIM)
        wrong_dim_vec = [[1.0, 2.0]]  # 2-dim instead of DIM
        with pytest.raises(ValueError, match="dimension"):
            idx.add(wrong_dim_vec, [0])


class TestNumpyFlatIndexSearch:
    """Search correctness tests."""

    def test_search_empty_index_returns_empty(self):
        idx = NumpyFlatIndex(DIM)
        result = idx.search(_random_vecs(1)[0], top_k=3)
        assert result == []

    def test_search_returns_top_k_results(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(20), list(range(20)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        assert len(results) == 5

    def test_search_top_k_clamped_to_index_size(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(3), [0, 1, 2])
        results = idx.search(_random_vecs(1)[0], top_k=10)
        assert len(results) == 3

    def test_search_scores_are_sorted_descending(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(15), list(range(15)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_id_score_tuples(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(5), [10, 20, 30, 40, 50])
        results = idx.search(_random_vecs(1)[0], top_k=3)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            doc_id, score = item
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    def test_search_ids_are_from_added_set(self):
        idx = NumpyFlatIndex(DIM)
        custom_ids = [100, 200, 300]
        idx.add(_random_vecs(3), custom_ids)
        results = idx.search(_random_vecs(1)[0], top_k=3)
        returned_ids = {r[0] for r in results}
        assert returned_ids.issubset(set(custom_ids))

    def test_search_identical_query_scores_highest(self):
        """A query vector identical to an indexed vector should score ~1.0."""
        idx = NumpyFlatIndex(DIM)
        target = _unit_vec([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        others = _random_vecs(9, seed=7)
        idx.add([target] + others, list(range(10)))
        results = idx.search(target, top_k=1)
        assert results[0][0] == 0  # first id (target)
        assert abs(results[0][1] - 1.0) < 1e-4

    def test_search_single_vector_index(self):
        idx = NumpyFlatIndex(DIM)
        vec = _random_vecs(1)[0]
        idx.add([vec], [42])
        results = idx.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 42

    def test_search_zero_query_returns_documents(self):
        """A zero query vector should not raise; cosine sim defaults gracefully."""
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(5), list(range(5)))
        zero_query = [0.0] * DIM
        # Should not raise; may return any ordering
        results = idx.search(zero_query, top_k=3)
        assert len(results) == 3

    def test_search_large_index_partial_sort(self):
        """Trigger the argpartition path (n > top_k * 4)."""
        n = 100
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(n), list(range(n)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        assert len(results) == 5
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestNumpyFlatIndexPersistence:
    """Save and load round-trip tests."""

    def test_save_load_roundtrip(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        vecs = _random_vecs(8)
        ids = list(range(8))
        idx.add(vecs, ids)

        path = str(tmp_path / "test_index")
        idx.save(path)

        idx2 = NumpyFlatIndex(DIM)
        idx2.load(path)
        assert len(idx2) == 8

    def test_save_load_produces_same_results(self, tmp_path):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, DIM)).tolist()
        ids = list(range(10))
        query = rng.standard_normal(DIM).tolist()

        idx = NumpyFlatIndex(DIM)
        idx.add(vecs, ids)
        original = idx.search(query, top_k=3)

        path = str(tmp_path / "roundtrip")
        idx.save(path)

        idx2 = NumpyFlatIndex(DIM)
        idx2.load(path)
        restored = idx2.search(query, top_k=3)

        assert [r[0] for r in original] == [r[0] for r in restored]
        for (_, s1), (_, s2) in zip(original, restored):
            assert abs(s1 - s2) < 1e-5

    def test_load_nonexistent_raises(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        with pytest.raises(FileNotFoundError):
            idx.load(str(tmp_path / "does_not_exist"))

    def test_save_creates_parent_dirs(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(3), [0, 1, 2])
        nested_path = str(tmp_path / "a" / "b" / "c" / "index")
        idx.save(nested_path)
        assert Path(nested_path + ".npz").exists()

    def test_save_empty_index(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        path = str(tmp_path / "empty")
        idx.save(path)  # Should not raise
        assert Path(path + ".npz").exists()

    def test_load_preserves_dimension(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(4), [0, 1, 2, 3])
        path = str(tmp_path / "dim_check")
        idx.save(path)

        idx2 = NumpyFlatIndex(DIM)
        idx2.load(path)
        assert idx2.dimension == DIM


# ===========================================================================
# Protocol conformance
# ===========================================================================


class TestProtocolConformance:
    """Verify NumpyFlatIndex satisfies the VectorIndex Protocol."""

    def test_numpy_flat_index_satisfies_protocol(self):
        idx = NumpyFlatIndex(DIM)
        assert isinstance(idx, VectorIndex)


# ===========================================================================
# create_index factory
# ===========================================================================


class TestCreateIndex:
    """Tests for the create_index factory function."""

    def test_numpy_backend_always_works(self):
        idx = create_index(DIM, backend="numpy")
        assert isinstance(idx, NumpyFlatIndex)

    def test_auto_backend_falls_back_to_numpy_when_no_ann_libs(self):
        """When faiss and hnswlib are both absent, auto must return NumpyFlatIndex."""
        with (
            patch("vector_index._FAISS_AVAILABLE", False),
            patch("vector_index._HNSW_AVAILABLE", False),
        ):
            # Reset cached availability flags so _check_* re-evaluates
            import vector_index as vi
            vi._FAISS_AVAILABLE = False
            vi._HNSW_AVAILABLE = False
            idx = create_index(DIM, backend="auto")
            assert isinstance(idx, NumpyFlatIndex)
            # Restore
            vi._FAISS_AVAILABLE = None
            vi._HNSW_AVAILABLE = None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_index(DIM, backend="bogus")

    def test_numpy_index_is_functional_after_create(self):
        idx = create_index(DIM, backend="numpy")
        vecs = _random_vecs(5)
        idx.add(vecs, list(range(5)))
        results = idx.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_faiss_backend_raises_when_unavailable(self):
        import vector_index as vi
        original = vi._FAISS_AVAILABLE
        vi._FAISS_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="faiss"):
                create_index(DIM, backend="faiss")
        finally:
            vi._FAISS_AVAILABLE = original

    def test_hnswlib_backend_raises_when_unavailable(self):
        import vector_index as vi
        original = vi._HNSW_AVAILABLE
        vi._HNSW_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="hnswlib"):
                create_index(DIM, backend="hnswlib")
        finally:
            vi._HNSW_AVAILABLE = original


# ===========================================================================
# FAISSIndex (mocked)
# ===========================================================================


def _make_faiss_mock():
    """Build a minimal faiss mock that lets FAISSIndex operate correctly."""
    faiss = MagicMock()
    faiss.METRIC_INNER_PRODUCT = 0

    class FakeIndex:
        """Thin numpy-backed faiss.IndexFlatIP substitute."""

        def __init__(self, dim):
            self._dim = dim
            self._matrix = np.empty((0, dim), dtype=np.float32)
            self.is_trained = True

        def add(self, matrix):
            self._matrix = np.vstack([self._matrix, matrix]) if self._matrix.shape[0] else matrix.copy()

        def search(self, query, k):
            if self._matrix.shape[0] == 0:
                return np.array([[-1.0] * k]), np.array([[-1] * k])
            n = self._matrix.shape[0]
            k = min(k, n)
            scores = self._matrix @ query[0]
            top = np.argsort(scores)[-k:][::-1]
            return np.array([scores[top]]), np.array([top])

    def IndexFlatIP(dim):
        return FakeIndex(dim)

    faiss.IndexFlatIP = IndexFlatIP
    faiss.write_index = MagicMock()
    faiss.read_index = MagicMock(return_value=FakeIndex(DIM))
    return faiss


class TestFAISSIndexMocked:
    """FAISSIndex tests using a mocked faiss module."""

    def _get_faiss_index(self, dim=DIM):
        from vector_index import FAISSIndex
        fake_faiss = _make_faiss_mock()
        idx = FAISSIndex.__new__(FAISSIndex)
        idx.dimension = dim
        idx.nprobe = 10
        idx._ids = []
        idx._raw_embeddings = []
        idx._index = None
        idx._faiss = fake_faiss
        return idx

    def test_empty_index_has_zero_length(self):
        idx = self._get_faiss_index()
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = self._get_faiss_index()
        vecs = _random_vecs(5)
        idx.add(vecs, list(range(5)))
        assert len(idx) == 5
        results = idx.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_search_empty_returns_empty(self):
        idx = self._get_faiss_index()
        results = idx.search(_random_vecs(1)[0], top_k=3)
        assert results == []

    def test_scores_sorted_descending(self):
        idx = self._get_faiss_index()
        idx.add(_random_vecs(10), list(range(10)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_mismatched_lengths_raises(self):
        idx = self._get_faiss_index()
        with pytest.raises(ValueError, match="length"):
            idx.add(_random_vecs(3), [0, 1])

    def test_save_calls_write_index(self, tmp_path):
        idx = self._get_faiss_index()
        idx.add(_random_vecs(3), [0, 1, 2])
        idx.save(str(tmp_path / "faiss_test"))
        idx._faiss.write_index.assert_called_once()

    def test_load_calls_read_index(self, tmp_path):
        idx = self._get_faiss_index()
        # Create dummy ids file so load can proceed
        ids_path = tmp_path / "faiss_test.ids.npy"
        np.save(str(ids_path), np.array([0, 1, 2], dtype=np.int64))
        index_path = tmp_path / "faiss_test.faiss"
        index_path.touch()
        idx.load(str(tmp_path / "faiss_test"))
        idx._faiss.read_index.assert_called_once()


# ===========================================================================
# HNSWIndex (mocked)
# ===========================================================================


def _make_hnswlib_mock():
    """Build a minimal hnswlib mock."""
    hnswlib = MagicMock()

    class FakeHNSWIndex:
        def __init__(self, space, dim):
            self._space = space
            self._dim = dim
            self._matrix = np.empty((0, dim), dtype=np.float32)
            self._ids: List[int] = []
            self._max_elements = 0

        def init_index(self, max_elements, M, ef_construction, random_seed=42):
            self._max_elements = max_elements

        def set_ef(self, ef):
            self._ef = ef

        def add_items(self, matrix, ids):
            self._matrix = matrix.astype(np.float32)
            self._ids = list(ids)

        def knn_query(self, query, k):
            n = self._matrix.shape[0]
            if n == 0:
                return np.array([[]], dtype=np.int64), np.array([[]], dtype=np.float32)
            k = min(k, n)
            q = query[0].astype(np.float32)
            # Cosine distance = 1 - dot(a,b) / (|a||b|); use brute force
            norms_m = np.linalg.norm(self._matrix, axis=1)
            norms_m = np.where(norms_m == 0, 1.0, norms_m)
            norm_q = np.linalg.norm(q) or 1.0
            sims = self._matrix @ q / (norms_m * norm_q)
            distances = 1.0 - sims  # cosine distance
            top = np.argsort(distances)[:k]
            return np.array([np.array(self._ids)[top]], dtype=np.int64), np.array([distances[top]], dtype=np.float32)

        def save_index(self, path):
            pass

        def load_index(self, path, max_elements):
            pass

    def Index(space, dim):
        return FakeHNSWIndex(space, dim)

    hnswlib.Index = Index
    return hnswlib


class TestHNSWIndexMocked:
    """HNSWIndex tests using a mocked hnswlib module."""

    def _get_hnsw_index(self, dim=DIM):
        from vector_index import HNSWIndex
        fake_lib = _make_hnswlib_mock()
        idx = HNSWIndex.__new__(HNSWIndex)
        idx.dimension = dim
        idx.m = 16
        idx.ef_construction = 200
        idx.ef = 50
        idx._ids = []
        idx._hnswlib = fake_lib
        idx._index = None
        return idx

    def test_empty_index_has_zero_length(self):
        idx = self._get_hnsw_index()
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = self._get_hnsw_index()
        vecs = _random_vecs(5)
        idx.add(vecs, list(range(5)))
        assert len(idx) == 5
        results = idx.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_search_empty_returns_empty(self):
        idx = self._get_hnsw_index()
        results = idx.search(_random_vecs(1)[0], top_k=3)
        assert results == []

    def test_scores_sorted_descending(self):
        idx = self._get_hnsw_index()
        idx.add(_random_vecs(8), list(range(8)))
        results = idx.search(_random_vecs(1)[0], top_k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_mismatched_lengths_raises(self):
        idx = self._get_hnsw_index()
        with pytest.raises(ValueError, match="length"):
            idx.add(_random_vecs(3), [0, 1])

    def test_save_load_roundtrip(self, tmp_path):
        idx = self._get_hnsw_index()
        idx.add(_random_vecs(4), [0, 1, 2, 3])
        # Create fake index object with save/load
        path = str(tmp_path / "hnsw_test")
        idx.save(path)
        ids_path = Path(path + ".hnsw.ids.npy")
        assert ids_path.exists()

    def test_load_missing_index_file_raises(self, tmp_path):
        idx = self._get_hnsw_index()
        with pytest.raises(FileNotFoundError):
            idx.load(str(tmp_path / "nonexistent"))
