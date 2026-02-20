"""
Local LLM wrapper using Ollama.
Replaces Claude Sonnet with a free, local model.
"""

import hashlib
import logging
import signal
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum

import ollama

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject immediately
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern for resilient LLM calls.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow one probe request

    Transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After recovery_seconds have passed
    - HALF_OPEN -> CLOSED: On successful probe request
    - HALF_OPEN -> OPEN: On failed probe request
    """

    def __init__(self, failure_threshold: int, recovery_seconds: int):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
            recovery_seconds: Seconds to wait before attempting recovery
        """
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    @property
    def circuit_state(self) -> str:
        """Return the current circuit state as a string."""
        with self._lock:
            return self._state.value

    def allow_request(self):
        """Check if the circuit allows a request to proceed.

        Handles OPEN -> HALF_OPEN transition based on recovery time.
        Used by streaming paths that cannot wrap the entire call in
        circuit_breaker.call().

        Raises:
            LLMConnectionError: If circuit is OPEN and not ready for recovery.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    time_since_failure = time.monotonic() - self._last_failure_time
                    if time_since_failure >= self._recovery_seconds:
                        logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                        self._state = CircuitState.HALF_OPEN
                    else:
                        remaining = self._recovery_seconds - time_since_failure
                        raise LLMConnectionError(
                            f"Circuit breaker open: LLM service unavailable. "
                            f"Retry in {remaining:.0f}s."
                        )
                # Still OPEN after check (no last_failure_time)
                if self._state == CircuitState.OPEN:
                    raise LLMConnectionError(
                        "Circuit breaker open: LLM service unavailable."
                    )

    def call(self, func, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            The result of func(*args, **kwargs)

        Raises:
            LLMConnectionError: If circuit is open
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    time_since_failure = time.monotonic() - self._last_failure_time
                    if time_since_failure >= self._recovery_seconds:
                        logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                        self._state = CircuitState.HALF_OPEN
                    else:
                        remaining = self._recovery_seconds - time_since_failure
                        raise LLMConnectionError(
                            f"Circuit breaker open: LLM service unavailable. "
                            f"Retry in {remaining:.0f}s."
                        )

            # If OPEN and not ready for recovery, reject
            if self._state == CircuitState.OPEN:
                remaining = self._recovery_seconds
                if self._last_failure_time is not None:
                    remaining = self._recovery_seconds - (time.monotonic() - self._last_failure_time)
                raise LLMConnectionError(
                    f"Circuit breaker open: LLM service unavailable. "
                    f"Retry in {remaining:.0f}s."
                )

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except (LLMConnectionError, LLMTimeoutError) as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning to CLOSED (recovery successful)")
                self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning to OPEN (recovery failed)")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    logger.error(
                        "Circuit breaker transitioning to OPEN "
                        f"(failure threshold {self._failure_threshold} reached)"
                    )
                    self._state = CircuitState.OPEN


class LLMConnectionError(Exception):
    """Raised when the Ollama backend is unreachable or returns an error."""


class LLMTimeoutError(LLMConnectionError):
    """Raised when the Ollama backend does not respond within the timeout."""


class LocalLLM:
    """Wrapper for Ollama local LLM with optional response caching and timeout."""

    def __init__(
        self,
        model: str = "llama3.2",
        enable_cache: bool = True,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        circuit_breaker_failure_threshold: int = 3,
        circuit_breaker_recovery_seconds: int = 30,
    ):
        """
        Initialize the local LLM.

        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral", "phi3")
            enable_cache: Whether to cache responses in memory
            timeout_seconds: Max seconds to wait for a response
            max_retries: Number of retries on transient failures
            circuit_breaker_failure_threshold: Failures before opening circuit
            circuit_breaker_recovery_seconds: Seconds before testing recovery
        """
        self.model = model
        self._cache: OrderedDict[str, str] | None = OrderedDict() if enable_cache else None
        try:
            from config import settings as _cfg
            self._cache_maxsize = _cfg.llm_cache_maxsize
        except Exception:
            self._cache_maxsize = 128
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_failure_threshold,
            recovery_seconds=circuit_breaker_recovery_seconds,
        )

    @property
    def circuit_state(self) -> str:
        """Return the current circuit breaker state."""
        return self._circuit_breaker.circuit_state

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local Ollama model.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response

        Raises:
            LLMTimeoutError: If Ollama does not respond within the timeout.
            LLMConnectionError: If Ollama is unreachable, circuit is open, or returns an error.
        """
        # Check cache first
        if self._cache is not None:
            cache_key = self._cache_key(prompt)
            if cache_key in self._cache:
                logger.debug("LLM cache hit")
                return self._cache[cache_key]

        # Execute through circuit breaker (wraps retry logic)
        result = self._circuit_breaker.call(self._generate_with_retry, prompt)

        # Store in cache (bounded LRU)
        if self._cache is not None:
            cache_key = self._cache_key(prompt)
            self._cache[cache_key] = result
            if len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)

        return result

    def generate_stream(self, prompt: str):
        """Generate a streaming response using Ollama.

        Yields text chunks as they arrive. Streaming bypasses the response
        cache but respects the circuit breaker.

        Args:
            prompt: The input prompt

        Yields:
            str: Text chunks as they arrive

        Raises:
            LLMConnectionError: If Ollama is unreachable or circuit is open.
        """
        self._circuit_breaker.allow_request()

        try:
            yield from self._raw_generate_stream(prompt)
            self._circuit_breaker._on_success()
        except (LLMConnectionError, LLMTimeoutError):
            self._circuit_breaker._on_failure()
            raise

    def _raw_generate_stream(self, prompt: str):
        """Raw streaming Ollama call without circuit breaker wrappers."""
        try:
            for chunk in ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
            ):
                text = chunk.get("response", "")
                if text:
                    yield text
        except ConnectionError as e:
            raise LLMConnectionError(
                "Cannot connect to Ollama. Is it running? (`ollama serve`)"
            ) from e
        except Exception as e:
            raise LLMConnectionError(f"Ollama error: {e}") from e

    def _generate_with_retry(self, prompt: str) -> str:
        """
        Generate with retry logic.

        This method is called by the circuit breaker and contains the retry logic.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response

        Raises:
            LLMTimeoutError: If Ollama does not respond within the timeout.
            LLMConnectionError: If Ollama is unreachable or returns an error.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._call_with_timeout(prompt)
                return result
            except LLMTimeoutError:
                raise  # Don't retry timeouts – they'd just timeout again
            except LLMConnectionError as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning("LLM attempt %d/%d failed: %s", attempt, self._max_retries, e)
                    continue
                raise
        # This should be unreachable but satisfies type checker
        raise last_error  # type: ignore[misc]

    def _call_with_timeout(self, prompt: str) -> str:
        """Execute the Ollama call with a timeout guard."""
        future = self._executor.submit(self._raw_generate, prompt)
        try:
            return future.result(timeout=self._timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise LLMTimeoutError(
                f"Ollama did not respond within {self._timeout}s. "
                "The model may be loading or the prompt may be too long."
            ) from None

    def _raw_generate(self, prompt: str) -> str:
        """Raw Ollama call without timeout/retry wrappers."""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
            )
            return response.get("response", "")
        except ConnectionError as e:
            raise LLMConnectionError(
                "Cannot connect to Ollama. Is it running? (`ollama serve`)"
            ) from e
        except Exception as e:
            raise LLMConnectionError(f"Ollama error: {e}") from e

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}:{prompt}".encode()).hexdigest()

    def __call__(self, prompt: str) -> str:
        """Allow calling the instance directly."""
        return self.generate(prompt)


class LocalEmbedder:
    """Local embeddings via OpenAI-compatible API (Docker Model Runner)."""

    def __init__(self, model_name: str = "mxbai-embed-large"):
        """
        Initialize the embedder using the OpenAI-compatible /v1/embeddings endpoint.

        DMR does not support the Ollama /api/embed endpoint, so we use
        the OpenAI-compatible /v1/embeddings API via httpx instead.

        Args:
            model_name: Model name for embeddings
        """
        import os
        import httpx

        self.model_name = model_name
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._url = f"{host.rstrip('/')}/v1/embeddings"
        self._client = httpx.Client(timeout=60.0)
        # Use configured dimension to avoid probe HTTP request
        try:
            from config import settings as _cfg
            cfg_dim = _cfg.embedding_dimension
        except Exception:
            cfg_dim = 0
        if cfg_dim > 0:
            self.dimension = cfg_dim
        else:
            probe = self._request_embeddings(["dimension probe"])
            self.dimension = len(probe[0])

    def _request_embeddings(self, texts: list) -> list:
        """Call the OpenAI-compatible embeddings endpoint."""
        resp = self._client.post(self._url, json={
            "model": self.model_name,
            "input": texts,
        })
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed(self, text: str) -> list:
        """
        Generate embeddings for text.

        Args:
            text: Input text to embed

        Returns:
            List of floats (embedding vector)
        """
        return self._request_embeddings([text])[0]

    def embed_batch(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self._request_embeddings(texts)
