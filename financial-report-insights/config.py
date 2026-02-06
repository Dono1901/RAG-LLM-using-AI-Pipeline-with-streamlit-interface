"""
Centralized configuration for the RAG-LLM Financial system.
All hardcoded values are collected here with environment variable overrides.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 3
    max_top_k: int = 20

    # File upload limits
    max_file_size_mb: int = 50
    max_query_length: int = 2000

    # LLM
    llm_model: str = "llama3.2"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Embedding cache
    embedding_cache_dir: str = ".cache/embeddings"

    # LLM response cache
    llm_cache_dir: str = ".cache/llm_responses"
    llm_cache_size_limit_mb: int = 500

    # Excel processing
    max_workbook_rows: int = 500_000

    model_config = {"env_prefix": "RAG_"}


settings = Settings()
