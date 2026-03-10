"""
Centralized configuration for the RAG-LLM Financial system.
All hardcoded values are collected here with environment variable overrides.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load ALL env vars from .env (including OLLAMA_HOST for the ollama client)
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE)


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 3
    max_top_k: int = 20

    # Hybrid search (BM25 + semantic)
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    rrf_k: int = 60

    # File upload limits
    max_file_size_mb: int = 200
    max_query_length: int = 2000

    # LLM
    llm_model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2

    # Circuit breaker
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: int = 30

    # Embedding cache
    embedding_cache_dir: str = ".cache/embeddings"

    # LLM response cache
    llm_cache_dir: str = ".cache/llm_responses"
    llm_cache_size_limit_mb: int = 1000
    llm_cache_maxsize: int = 128  # In-memory LRU cache entries

    # Embedding
    embedding_dimension: int = 1024  # mxbai-embed-large; 0 = auto-probe
    embedding_quantization: str = "none"  # "none" | "int8" | "float16"
    embedding_truncation_dim: int = 0  # 0 = no truncation, else target dim
    embedding_batch_size: int = 32  # texts per batch for BatchEmbedder
    embedding_cache_version: str = "1.0"  # bump to invalidate all cached embeddings

    # Excel processing
    max_workbook_rows: int = 500_000

    # Financial analysis
    default_tax_rate: float = 0.25

    # API
    api_port: int = 8504
    cors_origins: str = "http://localhost:8501"  # Comma-separated allowed origins
    max_request_body_bytes: int = 1_048_576  # 1 MB max request body
    max_financial_fields: int = 200  # Max fields in a financial_data dict

    # Export
    export_max_ratios: int = 200  # Max ratios per export
    export_company_name: str = ""  # Default company name for exports

    # Query enhancement
    enable_hyde: bool = True  # HyDE query expansion
    enable_query_decomposition: bool = True  # LLM-based query decomposition
    max_sub_queries: int = 4  # Max sub-queries for decomposition

    # Retrieval enhancement
    enable_reranking: bool = False  # Cross-encoder reranking (default OFF)
    reranking_model: str = "cross-encoder"  # Placeholder model name
    rerank_top_n: int = 20  # Candidates to rerank from initial retrieval
    mmr_lambda: float = 0.7  # MMR diversity parameter (1.0 = pure relevance, 0.0 = pure diversity)
    enable_mmr: bool = True  # Maximal Marginal Relevance diversification
    enable_citations: bool = True  # Citation tracking in responses

    # Semantic cache (Phase 4.3)
    semantic_cache_threshold: float = 0.95
    semantic_cache_max_entries: int = 1000
    enable_semantic_cache: bool = False  # off by default
    enable_chunk_dedup: bool = True
    adaptive_top_k: bool = True

    # Evaluation
    enable_evaluation: bool = False  # RAG evaluation harness

    # Prompt engineering (Phase 2.3)
    prompt_version: str = "2.0"  # Active prompt template version (for A/B testing)

    # Observability
    enable_tracing: bool = True  # Enable request tracing
    metrics_window_size: int = 10000  # Max metrics entries per rolling window

    # Vector index
    vector_backend: str = "auto"  # "auto" | "faiss" | "hnswlib" | "numpy"
    vector_index_path: str = "./data/vector_index"  # Persistence path prefix
    faiss_nprobe: int = 10  # FAISS IVF search parameter
    hnsw_m: int = 16  # HNSW connectivity parameter
    hnsw_ef: int = 50  # HNSW search-time ef parameter

    model_config = {"env_prefix": "RAG_"}


settings = Settings()
