"""
Protocol definitions for dependency injection.
Allows swapping LLM and embedding providers without changing consuming code.
"""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for language model providers."""

    def generate(self, prompt: str) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding providers."""

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
