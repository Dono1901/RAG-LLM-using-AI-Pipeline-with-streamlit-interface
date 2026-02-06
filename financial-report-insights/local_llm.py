"""
Local LLM wrapper using Ollama.
Replaces Claude Sonnet with a free, local model.
"""

import hashlib
import logging

import ollama

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Raised when the Ollama backend is unreachable or returns an error."""


class LocalLLM:
    """Wrapper for Ollama local LLM with optional response caching."""

    def __init__(self, model: str = "llama3.2", enable_cache: bool = True):
        """
        Initialize the local LLM.

        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral", "phi3")
            enable_cache: Whether to cache responses in memory
        """
        self.model = model
        self._cache: dict[str, str] = {} if enable_cache else None

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local Ollama model.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response

        Raises:
            LLMConnectionError: If Ollama is unreachable or returns an error.
        """
        # Check cache first
        if self._cache is not None:
            cache_key = self._cache_key(prompt)
            if cache_key in self._cache:
                logger.debug("LLM cache hit")
                return self._cache[cache_key]

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
            )
            result = response.get("response", "")
        except ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama. Is it running? (`ollama serve`)"
            ) from e
        except Exception as e:
            raise LLMConnectionError(f"Ollama error: {e}") from e

        # Store in cache
        if self._cache is not None:
            self._cache[cache_key] = result

        return result

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}:{prompt}".encode()).hexdigest()

    def __call__(self, prompt: str) -> str:
        """Allow calling the instance directly."""
        return self.generate(prompt)


class LocalEmbedder:
    """Local embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedder.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list:
        """
        Generate embeddings for text.

        Args:
            text: Input text to embed

        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
