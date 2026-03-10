"""Retrieval quality metrics for RAG evaluation.

All functions accept plain lists of source identifiers (strings) and return
float scores in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import math
from typing import List


def precision_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> float:
    """Fraction of the top-*k* retrieved sources that are relevant.

    Args:
        retrieved_sources: Ordered list of retrieved source identifiers.
        relevant_sources: Set of ground-truth relevant source identifiers.
        k: Cut-off rank.

    Returns:
        Precision@k in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    hits = sum(1 for src in top_k if src in relevant_set)
    return hits / len(top_k)


def recall_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> float:
    """Fraction of relevant sources found in the top-*k* retrieved sources.

    Args:
        retrieved_sources: Ordered list of retrieved source identifiers.
        relevant_sources: Set of ground-truth relevant source identifiers.
        k: Cut-off rank.

    Returns:
        Recall@k in [0.0, 1.0].
    """
    if k <= 0 or not relevant_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    hits = sum(1 for src in top_k if src in relevant_set)
    return hits / len(relevant_set)


def mrr(
    retrieved_sources: List[str],
    relevant_sources: List[str],
) -> float:
    """Mean Reciprocal Rank of the first relevant source.

    Args:
        retrieved_sources: Ordered list of retrieved source identifiers.
        relevant_sources: Set of ground-truth relevant source identifiers.

    Returns:
        MRR in [0.0, 1.0].
    """
    if not retrieved_sources or not relevant_sources:
        return 0.0
    relevant_set = set(relevant_sources)
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at rank *k*.

    Treats relevance as binary (1 if relevant, 0 otherwise).

    Args:
        retrieved_sources: Ordered list of retrieved source identifiers.
        relevant_sources: Set of ground-truth relevant source identifiers.
        k: Cut-off rank.

    Returns:
        NDCG@k in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_sources or not relevant_sources:
        return 0.0

    relevant_set = set(relevant_sources)

    # DCG: sum of 1/log2(rank+1) for each relevant doc in top-k
    dcg = 0.0
    for i, src in enumerate(retrieved_sources[:k]):
        if src in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-based

    # Ideal DCG: best possible ordering
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
