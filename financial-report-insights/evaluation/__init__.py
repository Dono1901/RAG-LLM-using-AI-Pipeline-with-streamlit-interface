"""RAG Evaluation Harness for financial-report-insights."""

from evaluation.golden_qa import GoldenQA, GOLDEN_QA_PAIRS
from evaluation.retrieval_metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k
from evaluation.answer_metrics import faithfulness_score, relevance_score, completeness_score
from evaluation.eval_harness import RAGEvalHarness, EvalReport

__all__ = [
    "GoldenQA",
    "GOLDEN_QA_PAIRS",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness_score",
    "relevance_score",
    "completeness_score",
    "RAGEvalHarness",
    "EvalReport",
]
