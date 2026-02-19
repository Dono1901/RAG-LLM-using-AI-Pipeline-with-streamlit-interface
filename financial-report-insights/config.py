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
    llm_cache_size_limit_mb: int = 500

    # Excel processing
    max_workbook_rows: int = 500_000

    # Financial analysis
    default_tax_rate: float = 0.25

    # API
    api_port: int = 8504

    model_config = {"env_prefix": "RAG_"}


settings = Settings()
