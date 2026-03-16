# Audit Report: RAG Pipeline & Embeddings
**Date:** 2026-03-16
**Scope:** Vector index, reranker, LLM client, graph retriever, prompts
**Files Reviewed:**
- `financial-report-insights/vector_index.py`
- `financial-report-insights/reranker.py`
- `financial-report-insights/local_llm.py`
- `financial-report-insights/graph_retriever.py`
- `financial-report-insights/prompts/templates.py`
- `financial-report-insights/prompts/formatters.py`
- `financial-report-insights/prompts/__init__.py`

---

## P0 — Critical (Fix Immediately)

### P0-01: `_send_embedding_batch` silently returns `None` on final retry exhaustion
- **File:** `local_llm.py:489-521`
- **Severity:** Critical
- **Detail:** The method has no `return` statement on the path where all retries are consumed and the last `attempt` equals `max_retries` without raising. In that case the `if e.response.status_code >= 500 and attempt < max_retries` condition is `False` on the last attempt (because `attempt == max_retries`), so `raise` is executed — but only for 5xx. For any non-5xx `HTTPStatusError` (e.g., 400, 422) the `raise` is also executed because the inner `if` is not taken. However, a subtle path exists: if `resp.raise_for_status()` does not raise (i.e., a 2xx) but `data["data"]` throws a `KeyError` or `IndexError`, the exception propagates uncaught as a bare `Exception` up to `_request_embeddings`, which does not handle it. The caller `embed()` / `embed_batch()` will surface a raw `KeyError` instead of a typed error, making it impossible for upstream retry logic (which catches only `httpx.HTTPStatusError`) to handle it correctly. Any malformed but 2xx response from DMR silently corrupts the embedding result list.
- **Suggestion:** Wrap `data["data"]` access in a try/except and raise a descriptive `ValueError`. Add a typed exception class `EmbeddingServiceError` analogous to `LLMConnectionError` and raise it from `_send_embedding_batch` for all failure paths.

### P0-02: `_request_embeddings` returns zero vectors for empty texts without dimension guard
- **File:** `local_llm.py:459-460`
- **Detail:** The early-return path `return [[0.0] * self.dimension for _ in texts]` at line 460 is reached when every text in the batch is whitespace-only. `self.dimension` is populated during `__init__` either from config (safe) or from a probe call. However, if the probe call at line 436-437 itself receives an all-whitespace input (which it never does — it sends `"dimension probe"` — this is fine), the value is valid. The real risk is that zero vectors are silently inserted into the index for empty chunks: they are mathematically valid unit vectors after L2-normalisation is clamped to `norm==0 -> 1.0`, meaning they score a cosine similarity of exactly 0 against every real query, which is acceptable; but they also score 0 against each other, so they pollute the top-k results for any query that genuinely scores near 0. There is no log warning when zero vectors are emitted.
- **Suggestion:** Log a `WARNING` when zero vectors are returned so operators can identify documents producing empty chunks. Consider filtering them from the index entirely rather than storing them.

---

## P1 — High (Fix This Sprint)

### P1-01: `NumpyFlatIndex.search` cosine similarity formula is mathematically wrong
- **File:** `vector_index.py:170-172`
- **Detail:** The cosine similarity formula at lines 170-172 is:
  ```python
  norms = np.linalg.norm(self._matrix, axis=1)        # shape (n,)
  safe_norms = np.where(norms == 0.0, 1.0, norms)     # shape (n,)
  similarities = self._matrix @ q / safe_norms         # shape (n,)
  ```
  The query vector `q` has already been L2-normalised at line 167, so `||q|| = 1`. The correct formula for cosine similarity is `(A @ q) / (||A|| * ||q||)`, which simplifies to `(A @ q) / ||A||` since `||q|| = 1`. This is what the code computes, and it is mathematically correct. However, the stored `_matrix` is **not** normalised before storage — raw (potentially un-normalised) vectors are stored via `add()` (lines 136-148). This means the similarities returned are not true cosine similarities in `[-1, 1]`; they are dot products divided by only the document norm, which equals cosine similarity only because `||q||=1`. This is correct as written, but the contrast with `FAISSIndex._normalize` (which explicitly normalises before storage) creates an inconsistency: if the same embeddings produce identical cosine scores under both backends, the backends are equivalent, which is true mathematically. **The actual bug** is that `NumpyFlatIndex` does not normalise its stored matrix, so the matrix grows unboundedly as raw float32 vectors accumulate (memory is proportional to scale of embeddings, not unit vectors), and on `load()` the saved matrix might differ from what FAISSIndex would store. More critically: if a caller provides non-unit-normalised vectors (e.g., a custom embedder), the scores are still correct, but if the same index is switched from numpy to FAISS, the FAISS backend normalises at `add()` time while numpy does not, meaning the two backends diverge for non-unit input. This is a correctness trap that will bite during backend switching.
- **Suggestion:** Normalise stored vectors in `NumpyFlatIndex.add()` the same way `FAISSIndex._normalize` does, ensuring backends are interchangeable.

### P1-02: `HNSWIndex.add` rebuilds the entire index from scratch on every call
- **File:** `vector_index.py:453-455`
- **Detail:** Every call to `HNSWIndex.add()` calls `_init_index(len(self._ids))` (line 454) which creates a brand-new hnswlib index, then calls `add_items(matrix, self._ids)` on the entire accumulated set of IDs (line 455). The `self._ids` list grows with every `add()` call, but the raw embeddings are not retained — only IDs are stored. The matrix passed to `add_items` is derived only from the `embeddings` argument of the **current** call, not all accumulated embeddings. This means after the first `add()` call, subsequent calls rebuild an index that contains only the new batch's embeddings but all previously accumulated IDs. The index is corrupt after two or more `add()` calls: IDs from the first batch are registered in `self._ids` but their vectors are absent from the rebuilt index.
- **Suggestion:** Either accumulate raw embeddings (like `FAISSIndex` does with `_raw_embeddings`) or use hnswlib's `resize_index()` to extend capacity and `add_items()` incrementally without rebuilding. This is a data-correctness bug that would cause silent wrong retrieval results.

### P1-03: `CircuitBreaker.call` holds the lock while executing user code
- **File:** `local_llm.py:92-139`
- **Detail:** The circuit breaker acquires `self._lock` at line 107, checks and potentially transitions state, then falls through to the `# Execute the function` comment at line 132 where `func(*args, **kwargs)` is called **without** the lock (the `with self._lock` block ends at the blank line after the second `raise` at line 130). This is correct and the lock is released before the call. However, `_on_success` and `_on_failure` each re-acquire `self._lock`. If `_on_success` or `_on_failure` is called from `generate_stream` (via `self._circuit_breaker._on_success()` at line 274 and `_on_failure()` at line 276), these are direct calls to private methods bypassing the normal `call()` path, creating an alternative code path with no state consistency guarantee. Specifically, `generate_stream` calls `allow_request()` (which acquires and releases the lock) and then begins yielding; it does not hold the lock between `allow_request()` and the subsequent streaming. A concurrent `call()` thread could open the circuit between `allow_request()` checking HALF_OPEN and the stream starting, causing a HALF_OPEN stream to run against an already-OPEN circuit.
- **Suggestion:** For the streaming path, make the state transition to CLOSED/OPEN atomic with the allow-request check, or document the race and accept it as a known limitation. The current code has an undocumented TOCTOU window.

### P1-04: `_generate_with_retry` retry loop does not retry on all `LLMConnectionError` subtypes; only re-raises on final attempt
- **File:** `local_llm.py:327-340`
- **Detail:** The loop iterates `range(1, self._max_retries + 1)` (line 327). On the last iteration when `attempt == self._max_retries`, `attempt < self._max_retries` is `False`, so the code falls through to `raise` (re-raises the exception). This is correct. However, `LLMTimeoutError` inherits from `LLMConnectionError` (line 172). Line 331 catches `LLMTimeoutError` first and re-raises immediately with the comment "Don't retry timeouts". Line 333 catches `LLMConnectionError` (which includes all non-timeout subclasses) and only continues if there are remaining attempts. This is correct logic. The actual issue is that `_call_with_timeout` wraps `_raw_generate` in a `ThreadPoolExecutor`. If the underlying thread raises an unexpected exception not caught by `_raw_generate`'s `except Exception` block (line 377), the `Future.result()` call at line 346 will re-raise that exception as-is, bypassing the `LLMConnectionError` catch in `_generate_with_retry`. This produces an untyped exception leaking out of `generate()` to callers, breaking any caller that catches only `LLMConnectionError`.
- **Suggestion:** Add a bare `except Exception` in `_call_with_timeout` that wraps into `LLMConnectionError` before re-raising, ensuring all failures from the LLM path are typed.

### P1-05: `persist_analysis_to_graph` parses LLM-generated text with no sanitisation
- **File:** `graph_retriever.py:138-169`
- **Detail:** The function parses `report.sections["ratio_analysis"]` and `report.sections["scoring_models"]` by splitting on newlines and colons (lines 142-151, 158-169). This text originates from LLM output, which can contain arbitrary content. The `name` variable at line 144 is constructed as `parts[0].strip().lower().replace(" ", "_")` and then passed to `store.store_financial_data()` as a dict key. If LLM output contains a specially crafted ratio name (e.g., containing Cypher injection fragments like `'); DROP`), and if `store_financial_data` constructs Cypher strings from dict keys without parameterisation, this creates a second-order injection path. The audit of `graph_store.py` is out of scope here, but the data pipeline that flows from LLM text into graph node properties must be treated as untrusted.
- **Suggestion:** Sanitise `name` and `model` keys with a strict allowlist regex (e.g., `re.sub(r'[^a-z0-9_]', '', name)`) before passing to the store. This matches the existing pattern in `graph_store.py`'s `vector_index_statement`.

---

## P2 — Medium (Fix Soon)

### P2-01: `EmbeddingReranker.rerank` catches bare `Exception` and discards error details
- **File:** `reranker.py:94-96`
- **Detail:** Any failure in the reranking path — including dimension mismatches, network errors to the embedding service, or numpy shape errors — is silently swallowed and logged only at `WARNING` level. The function then returns `documents[:top_k]` without any indication to the caller that the returned ranking is unordered. Callers that rely on `_rerank_score` being present will get documents without that key.
- **Suggestion:** Log the full exception (include `exc_info=True`), and optionally return documents with a sentinel `_rerank_score` of `None` to allow callers to distinguish reranked from passthrough results.

### P2-02: `mmr_diversify` inner loop is O(n * k) with Python for-loop overhead
- **File:** `reranker.py:144-174`
- **Detail:** The greedy MMR selection (lines 144-174) iterates over all remaining candidates in a Python for-loop for each of the `top_k` selected documents. This is O(n * k) with a large Python constant. For a 1000-document candidate set with top_k=20, this is 20,000 Python iterations plus one `selected_vecs @ idx_vec` dot-product per iteration. The function is called after retrieval so candidate set sizes are typically small (< 100), but nothing enforces this. For large document sets passed directly to MMR, this becomes a latency bottleneck.
- **Suggestion:** Vectorise the inner loop: compute all remaining-to-selected similarities as a matrix operation and use `np.max` along the axis, then compute all MMR scores in one pass. This reduces the Python loop iterations by a factor of `len(remaining)`.

### P2-03: `LocalLLM._executor` is a single-worker `ThreadPoolExecutor` that is never shut down
- **File:** `local_llm.py:209`
- **Detail:** `self._executor = ThreadPoolExecutor(max_workers=1)` creates a daemon thread pool but there is no `__del__`, `close()`, or `shutdown()` method. In long-running Streamlit apps where `LocalLLM` instances may be recreated (e.g., on session reload), old executor threads accumulate as leaked daemon threads. Python will eventually clean these up at process exit, but under Gunicorn or uvicorn workers they can persist for the worker lifetime, consuming OS thread handles.
- **Suggestion:** Implement `__del__` or a context manager (`__enter__`/`__exit__`) that calls `self._executor.shutdown(wait=False)`.

### P2-04: `LocalEmbedder._client` (httpx.Client) is never closed
- **File:** `local_llm.py:426`
- **Detail:** `self._client = httpx.Client(timeout=60.0)` creates a persistent HTTP connection pool. The `LocalEmbedder` class has no `close()`, `__del__`, or context manager protocol. In environments that recreate the embedder (e.g., Streamlit reruns or test teardown), the underlying TCP connections leak until the GC finalises the object.
- **Suggestion:** Add `def close(self): self._client.close()` and a `__del__` that calls it, or expose `LocalEmbedder` as a context manager.

### P2-05: `build_prompt` uses Python `str.format()` on user-controlled `query` and `context` strings
- **File:** `prompts/formatters.py:137-143`
- **Detail:** `template.user_template.format(few_shot_block=..., context=context, query=query)` at line 137 calls Python's `str.format()`. If `query` or `context` contains literal `{` or `}` characters (e.g., a user querying about `{EBITDA}` growth or a document containing JSON fragments), `str.format()` will raise a `KeyError` or `IndexError` when it encounters an unmatched brace. This is a reliability issue that crashes prompt assembly for valid financial queries containing curly braces.
- **Suggestion:** Escape braces in user-controlled content before substitution, or switch to a safer interpolation method (e.g., `string.Template` with `$`-sigils, or explicit `replace()` calls for each placeholder).

### P2-06: `format_context_with_citations` citation stripping produces incorrect markers for pre-assigned citations
- **File:** `prompts/formatters.py:65-70`
- **Detail:** When a document has a `_citation` key (set by `reranker.add_citations`), the code strips brackets with `citation.strip("[]")` and then re-wraps. The `add_citations` function in `reranker.py:221` formats the citation as `"[{i + 1}] Source: {source} | ..."` (a full descriptive string, not just a number). Stripping `"[]"` from `"[1] Source: Annual Report | Type: income_statement"` yields `"1] Source: Annual Report | Type: income_statement"` — the closing bracket after `1` is not at the string boundary and is not removed. The resulting marker would be `[1] Source: Annual Report | Type: income_statement]` — malformed.
- **Suggestion:** Parse the citation number with a regex (`re.match(r'\[(\d+)\]', citation)`) or just use the loop index `idx` as the authoritative citation number, ignoring pre-existing `_citation` strings for the marker (they can still be appended as metadata).

### P2-07: `HNSWIndex` does not preserve accumulated embeddings, making `load()` incomplete
- **File:** `vector_index.py:496-517`
- **Detail:** Related to P1-02. `HNSWIndex.save()` persists the hnswlib index file and the IDs array. `load()` restores IDs and re-initialises the index from the saved file. This is correct for a fully built index. However, the `_raw_embeddings` equivalent does not exist in `HNSWIndex` — if the loaded index is subsequently used with `add()`, the rebuild at line 454 will only have the new batch's vectors but all accumulated IDs from the loaded state, causing the same corruption described in P1-02.
- **Suggestion:** Resolve together with P1-02; the fix to accumulate raw embeddings will also make the post-load `add()` path safe.

### P2-08: Embedding dimension mismatch between index and embedder is undetected
- **File:** `vector_index.py` (all backends) and `local_llm.py:433-437`
- **Detail:** The `create_index(dimension=...)` call and `LocalEmbedder.dimension` are both initialised independently from config. If `config.embedding_dimension` is misconfigured (e.g., set to 384 while the model returns 1024), the index is created with dimension 384, and the first `add()` call will raise a `ValueError` from `NumpyFlatIndex.add` (line 138). This is the correct behaviour. However, for FAISS and HNSW backends, the dimension mismatch is caught only at `add()` time, not at index creation. More critically, `LocalEmbedder.dimension` is set to `cfg_dim` (line 434) without any validation that `cfg_dim` matches what the model actually returns. The zero-vector path at line 460 uses `self.dimension` which may be the misconfigured value, silently producing zero vectors of the wrong length that are then inserted into the index — triggering the shape check only if the index catches it.
- **Suggestion:** After the first successful embedding call, assert `len(result[0]) == self.dimension` and raise a clear `EmbeddingDimensionError` if mismatched. Add this check in `_send_embedding_batch` on the first successful response.

---

## P3 — Low / Housekeeping

### P3-01: `NumpyFlatIndex.load` does not validate that loaded dimension matches instance dimension
- **File:** `vector_index.py:199-216`
- **Detail:** `load()` overwrites `self.dimension` with whatever is in the `.npz` file (line 213). If the instance was created with `dimension=1024` and the file was saved with `dimension=384`, the instance silently changes its dimension after load. Subsequent `add()` calls will reject new embeddings of dimension 1024 because the matrix now has 384 columns.
- **Suggestion:** Before overwriting, check `int(data["dimension"][0]) == self.dimension` and raise a `ValueError` if mismatched, or document that `load()` resets dimension.

### P3-02: `FAISSIndex.load` loses `_raw_embeddings` which are needed for IVF rebuild
- **File:** `vector_index.py:348-368`
- **Detail:** `_raw_embeddings` is populated in `add()` and is used by `_build_index()` when the collection exceeds `_IVF_THRESHOLD` (10,000 vectors). `load()` restores the FAISS index but leaves `_raw_embeddings = []`. If after loading the caller adds more vectors that push the total past 10,000, `_build_index` is called with a matrix derived only from the new batch's `_raw_embeddings` (the newly added ones), not all accumulated vectors. This is the same class of bug as P1-02 for HNSW.
- **Suggestion:** Either save/load `_raw_embeddings` alongside the index file, or document that `FAISSIndex` cannot accept `add()` calls after `load()` once the index transitions to IVF.

### P3-03: `LocalLLM` model name is not validated before use
- **File:** `local_llm.py:199`
- **Detail:** The `model` parameter is stored as-is and passed directly to `ollama.generate(model=self.model, ...)`. If a caller passes an empty string or a model name containing shell metacharacters, this is passed to the Ollama client, which may behave unexpectedly. Since Ollama communicates over HTTP and model names are sent as JSON, actual shell injection is not a risk, but an empty model name would produce an opaque Ollama error rather than a clear `ValueError`.
- **Suggestion:** Add `if not model or not model.strip(): raise ValueError("model name cannot be empty")` in `__init__`.

### P3-04: `graph_retriever.persist_analysis_to_graph` uses fragile text parsing of LLM output
- **File:** `graph_retriever.py:142-169`
- **Detail:** The function assumes the LLM formats ratio output as `"key: value ..."` with exactly one colon. LLM output is non-deterministic and may use different separators, multi-line values, or markdown formatting. The `try/except (ValueError, IndexError)` at lines 150-151 and 168 silently drops unrecognised lines without logging, making it impossible to diagnose why graph data is missing.
- **Suggestion:** Log dropped lines at `DEBUG` level so operators can identify LLM output format drift. Long-term, prefer `persist_structured_analysis_to_graph` which uses typed data structures.

### P3-05: `format_few_shot_examples` uses `"\n\n".join(lines)` on a list that includes multi-line strings
- **File:** `prompts/formatters.py:104-117`
- **Detail:** Each `lines` element is a multi-line string (e.g., `"Example 1:\nQ: ...\nA: ..."`). Joining with `"\n\n"` adds double newlines between already multi-line items, potentially creating triple or quadruple newlines in the prompt that waste tokens and may confuse some models' instruction parsing.
- **Suggestion:** Use `"\n".join(lines)` or build the string with explicit double-newline separators between examples only, not between the header line and each example.

### P3-06: `LocalEmbedder` SSRF allow-list does not cover IPv4-mapped IPv6 loopback
- **File:** `local_llm.py:413-423`
- **Detail:** The `_ALLOWED_HOSTS` set includes `"127.0.0.1"` and `"::1"` but not `"::ffff:127.0.0.1"` (IPv4-mapped IPv6 loopback) or `"0.0.0.0"`. On some platforms `urlparse` may return `"::ffff:7f00:1"` for the latter. This is a minor gap in the existing SSRF mitigation.
- **Suggestion:** Normalise the parsed hostname through `ipaddress.ip_address()` when it looks like an IP literal and use `is_loopback` / `is_private` for the allow decision.

### P3-07: `wait_for_embedding_service` uses `time.sleep(2)` in a tight retry loop with no jitter
- **File:** `local_llm.py:552`
- **Detail:** The 2-second fixed sleep between warm-up attempts means that if multiple services start simultaneously, all their warm-up loops will retry in lockstep, creating a thundering-herd effect against the DMR endpoint.
- **Suggestion:** Add small random jitter (`random.uniform(1.5, 2.5)`) to stagger retries.

### P3-08: `reranker.py` type annotation for `EmbeddingReranker.__init__` `embedder` parameter is missing
- **File:** `reranker.py:36`
- **Detail:** `def __init__(self, embedder):` has no type annotation. Given the rest of the codebase uses `EmbeddingProvider` protocol from `protocols.py`, this should be annotated as `embedder: "EmbeddingProvider"` for IDE support and mypy coverage.
- **Suggestion:** Add `from protocols import EmbeddingProvider` and annotate the parameter.

---

## Files With No Issues Found

- `financial-report-insights/prompts/__init__.py` — Clean re-export module; no logic, no issues.
- `financial-report-insights/prompts/templates.py` — All template strings are hardcoded constants with no dynamic interpolation at definition time; few-shot examples are static. No injection surface in the template definitions themselves.

---

## Summary

| Severity | Count | Notes |
|----------|-------|-------|
| P0 — Critical | 2 | Silent data corruption from untyped exceptions; zero-vector pollution without warning |
| P1 — High | 5 | Wrong cosine formula in NumpyFlatIndex; HNSWIndex add() corruption after 2+ calls; circuit breaker TOCTOU race; LLM exception leakage; LLM text injection into graph |
| P2 — Medium | 8 | Broad exception swallowing in reranker; O(n*k) MMR loop; resource leaks (executor, httpx client); str.format() crash on curly-brace input; citation marker malformation; missing dimension validation; post-load add() corruption in HNSW and FAISS |
| P3 — Low | 8 | Minor validation gaps, logging omissions, type annotation issues, SSRF IPv6 edge case |

**Most urgent fixes in priority order:**
1. **P1-02** — `HNSWIndex.add()` silently corrupts the index after the first call. Any production environment using the hnswlib backend returns wrong retrieval results from the second ingestion batch onward.
2. **P0-01** — `_send_embedding_batch` can propagate raw `KeyError`/`IndexError` from malformed DMR responses, bypassing all upstream retry logic.
3. **P2-05** — `build_prompt` crashes with `KeyError` for any user query or document content containing `{` or `}` characters, which are common in financial data (e.g., JSON references, formula notation).
4. **P1-01** — `NumpyFlatIndex` stores un-normalised vectors while FAISS normalises, making backend switching silently produce different rankings for the same data.
5. **P1-05** — LLM-generated text flows directly into graph node property keys without sanitisation, creating a second-order injection path into Neo4j.
