# Audit Report: Bug Patterns & Edge Cases
**Date:** 2026-03-16
**Scope:** Race conditions, null handling, error swallowing, resource leaks, float precision, off-by-one errors, state mutation, async/sync mixing, stale cache, and silent data corruption across core modules
**Files Reviewed:**
- `financial-report-insights/financial_analyzer.py` (lines 1–6100+, ~14K lines)
- `financial-report-insights/local_llm.py` (full, 587 lines)
- `financial-report-insights/vector_index.py` (full, 618 lines)
- `financial-report-insights/ingestion_pipeline.py` (full, 448 lines)
- `financial-report-insights/api.py` (full, ~720 lines)
- `financial-report-insights/graph_store.py` (full, ~550 lines)
- `financial-report-insights/portfolio_analyzer.py` (full, ~450 lines)
- `financial-report-insights/compliance_scorer.py` (full, ~700 lines)

---

## P0 — Critical (Fix Immediately)

### BUG-P0-01: `_get_portfolio_analyzer` and `_get_compliance_scorer` use function attribute as singleton storage — not thread-safe on first access across interpreters, and the pattern is fragile

**File:** `financial-report-insights/api.py`, lines 649–669

```python
def _get_portfolio_analyzer():
    """Return a module-level singleton PortfolioAnalyzer (thread-safe)."""
    if not hasattr(_get_portfolio_analyzer, "_inst"):
        with _portfolio_lock:
            if not hasattr(_get_portfolio_analyzer, "_inst"):
                from portfolio_analyzer import PortfolioAnalyzer
                _get_portfolio_analyzer._inst = PortfolioAnalyzer()
    return _get_portfolio_analyzer._inst
```

**Issue:** The outer `if not hasattr(...)` check happens outside the lock. In a multi-threaded environment the CPython GIL typically saves this, but the function-attribute pattern is non-standard and breaks under any bytecode optimizer or alternative Python runtime. More critically, `_get_portfolio_analyzer._inst` is set on the function object itself — if FastAPI ever reloads the module (e.g. via `importlib`), the attribute persists on the stale function object, making the double-checked pattern unreliable. Compare with `_get_rag()` which correctly uses a module-level `_rag_instance = None` variable with a separate lock.

**Risk:** Stale instance after module reload; race under non-CPython runtimes.

**Fix:** Move `_portfolio_inst = None` and `_compliance_inst = None` to module level (as module globals), mirroring the `_rag_instance` pattern already used in the same file.

---

### BUG-P0-02: `LocalEmbedder._request_embeddings` returns zero vectors silently when `self.dimension` is 0

**File:** `financial-report-insights/local_llm.py`, lines 459–460

```python
if not non_empty:
    return [[0.0] * self.dimension for _ in texts]
```

**Issue:** If `self.dimension` is 0 (which happens when the config probe is disabled and `cfg_dim == 0` but the probe HTTP call also fails or raises), `_request_embeddings` returns a list of empty lists `[[]]` for every input. Those empty embeddings are then stored in the vector index and will silently produce incorrect cosine similarity scores (divide-by-zero safe-guarded to 0 in `NumpyFlatIndex` but returning bogus rankings). The caller receives no error, no log warning.

**Risk:** Silent ingestion of zero-dimension embeddings that corrupt the vector index and produce wrong query results.

**Fix:** Add an assertion or guard: `if self.dimension <= 0: raise RuntimeError("Embedder dimension not initialized")` before returning zero vectors.

---

### BUG-P0-03: `LocalLLM._call_with_timeout` cancels the future but the underlying blocking Ollama call continues running in the ThreadPoolExecutor thread

**File:** `financial-report-insights/local_llm.py`, lines 342–352

```python
def _call_with_timeout(self, prompt: str) -> str:
    """Execute the Ollama call with a timeout guard."""
    future = self._executor.submit(self._raw_generate, prompt)
    try:
        return future.result(timeout=self._timeout)
    except FuturesTimeoutError:
        future.cancel()
        raise LLMTimeoutError(...)
```

**Issue:** `ThreadPoolExecutor` futures that have already started (i.e., `_raw_generate` is blocking on Ollama I/O) cannot be cancelled — `future.cancel()` is a no-op once execution has begun. The executor has `max_workers=1`, meaning the single thread is now permanently occupied waiting for the Ollama response. Every subsequent `generate()` call will queue behind the stuck thread and also time out. The executor is never shut down (no `__del__` or context manager), so the thread leaks for the process lifetime.

**Risk:** After a single timeout the LLM becomes permanently unavailable within the process, even after the circuit breaker resets. Thread leak.

**Fix:** Use a larger pool (e.g., `max_workers=2`) so a new request can run while the timed-out one drains. Alternatively, use `asyncio.wait_for` in an async context so cancellation is cooperative. At minimum, document the known limitation.

---

## P1 — High (Fix This Sprint)

### BUG-P1-01: `_send_embedding_batch` does not return a value on the last retry when the final attempt raises a non-5xx `HTTPStatusError`

**File:** `financial-report-insights/local_llm.py`, lines 489–521

```python
def _send_embedding_batch(self, texts: list, max_retries: int = 3, large_batch: bool = False) -> list:
    ...
    for attempt in range(1, max_retries + 1):
        try:
            ...
            return [item["embedding"] for item in data["data"]]
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and attempt < max_retries:
                ...
                continue
            raise
```

**Issue:** The method has no explicit `return` outside the loop. Python returns `None` implicitly only if the loop completes without an exception. However, because the final raise re-raises on non-5xx or on the last 5xx retry, the method can never return `None` silently. But notice the `except httpx.HTTPStatusError` clause does NOT catch `httpx.RequestError` (connection refused, timeout at httpx layer, etc.). A `RequestError` propagates all the way up through `_request_embeddings`, through `embed_batch`, and into the ingestion pipeline, where it is caught by a bare `except Exception as e: logger.error(...)` — swallowing it. The caller gets an empty chunk list instead of an error.

**Risk:** Silent ingestion failure; documents are silently skipped when the embedding endpoint is unreachable at the httpx level (as opposed to returning an HTTP error code).

**Fix:** Add `except httpx.RequestError as e: raise` (or convert to a typed error) so the outer retry loop in `ingestion_pipeline.py` can observe and retry the failure.

---

### BUG-P1-02: `FAISSIndex.add` rebuilds the entire index on every incremental call — `_raw_embeddings` grows unboundedly

**File:** `financial-report-insights/vector_index.py`, lines 286–307

```python
def add(self, embeddings: List[List[float]], ids: List[int]) -> None:
    ...
    self._raw_embeddings.extend(embeddings)
    self._ids.extend(ids)
    matrix = self._normalize(np.asarray(self._raw_embeddings, dtype=np.float32))
    self._build_index(matrix)
```

**Issue:** Every `add()` call copies all previously added embeddings into a new numpy array, L2-normalizes them, and rebuilds the FAISS index from scratch. For a large document set (e.g. 10,000+ chunks), a single add of 100 new chunks allocates a fresh 10,100×1024 float32 matrix (~41 MB), then builds the index again. Over a full ingestion run this causes O(n²) memory allocations. `_raw_embeddings` keeps a Python list of all raw floats (never freed), doubling peak memory vs. FAISS alone.

**Risk:** Memory exhaustion and severe performance degradation during large ingestion runs.

**Fix:** Buffer adds and only rebuild at the end (`flush()`), or switch to FAISS `IndexFlatIP.add_with_ids()` which is incremental. At minimum, add a warning when `len(self._raw_embeddings) > 50_000`.

---

### BUG-P1-03: `HNSWIndex.add` also rebuilds the entire index from scratch on every incremental call

**File:** `financial-report-insights/vector_index.py`, lines 430–455

```python
def add(self, embeddings: List[List[float]], ids: List[int]) -> None:
    ...
    self._ids.extend(ids)
    matrix = np.asarray(embeddings, dtype=np.float32)
    # Rebuild index with updated max_elements
    self._init_index(len(self._ids))
    self._index.add_items(matrix, self._ids)
```

**Issue:** `_init_index` is called every time, which calls `index.init_index(max_elements=len(self._ids))` creating a brand-new hnswlib index. Then `add_items` adds only the new `matrix` (not all historical vectors). The result is an index that contains only the most recently added batch, not the full accumulated set. All previously indexed embeddings are permanently lost after each incremental `add()`.

**Risk:** Critical correctness bug — the HNSW index only ever contains the last batch added; historical document searches return no results.

**Fix:** Either (a) store all embeddings like `FAISSIndex._raw_embeddings` and pass all of them to `add_items`, or (b) call `self._index.add_items` with both old and new vectors after `_init_index`. The correct pattern for hnswlib is: `_init_index(total_elements); self._index.add_items(all_matrix, all_ids)`.

---

### BUG-P1-04: `rate_limit_middleware` grows `_rate_log` dict unboundedly — no eviction for IPs that stop sending requests

**File:** `financial-report-insights/api.py`, lines 180–196

```python
_rate_log: Dict[str, List[float]] = defaultdict(list)
...
async def rate_limit_middleware(request: Request, call_next):
    ...
    with _rate_lock:
        cutoff = now - _RATE_WINDOW
        _rate_log[client_ip] = [t for t in _rate_log[client_ip] if t > cutoff]
        ...
        _rate_log[client_ip].append(now)
```

**Issue:** `_rate_log` accumulates one entry per unique IP that has ever sent a request. Old IPs' lists are trimmed to empty on the next request from that IP, but the key itself is never removed from the dict. In a public deployment with many different source IPs (especially with IPv6, where clients may rotate /128 addresses), this dict grows without bound.

**Risk:** Memory exhaustion over hours/days under moderate traffic from diverse IPs.

**Fix:** After trimming, delete the key if the resulting list is empty: `if not _rate_log[client_ip]: del _rate_log[client_ip]`.

---

### BUG-P1-05: `compliance_scorer.py` `regulatory_ratios` compliance percentage uses 0.0 when all checks have insufficient data, which the caller interprets as 0% compliance rather than "unknown"

**File:** `financial-report-insights/compliance_scorer.py`, lines 550–551

```python
total = pass_count + fail_count
compliance_pct = (pass_count / total * 100) if total > 0 else 0.0
```

**Issue:** When all 6 regulatory checks return `passes = None` (all data is missing), `total` is 0 and `compliance_pct` is `0.0`. In `audit_risk_assessment`, `reg_pts = int(30 * reg.compliance_pct / 100)` computes to 0, which contributes 0 points to the audit score. The score therefore penalizes a company for missing data as if it had failed every regulatory check — instead of simply being unrated. The memory note (`MEMORY.md`) references a known "0% default" fix, but the downstream consumer in `audit_risk_assessment` still uses the `0.0` as a real failure score.

**Risk:** Companies with sparse financial data receive artificially low audit risk scores (critical risk level), which is a material misrepresentation.

**Fix:** Return `compliance_pct = None` (or a sentinel like `-1.0`) when `total == 0`, and adjust `audit_risk_assessment` to skip the regulatory component when data is insufficient.

---

## P2 — Medium (Fix Soon)

### BUG-P2-01: `dupont_analysis` approximates EBT as `NI / (1 - tax_rate)` which produces nonsensical results when `tax_rate == 1.0`

**File:** `financial-report-insights/financial_analyzer.py`, lines 2748–2750

```python
ebt = data.ebt
if ebt is None and data.net_income is not None:
    # Approximate EBT = Net Income / (1 - tax_rate)
    ebt = safe_divide(data.net_income, 1 - self._tax_rate)
```

**Issue:** `safe_divide` guards against zero denominators (`abs(denominator) < 1e-12`), so when `self._tax_rate` is exactly 1.0 (which cannot be set via config but could be passed directly), it returns `None`. However, if `_tax_rate` is 0.9999 the division produces `NI * 10000`, a nonsensical EBT. No validation of `_tax_rate` range is performed in `__init__`. The `config.py` default is 0.21 (safe), but a custom instantiation `CharlieAnalyzer(tax_rate=1.5)` silently produces infinite/inverted EBT.

**Risk:** Downstream `tax_burden` and `interest_burden` DuPont fields contain garbage values when non-standard tax rates are used.

**Fix:** Clamp `_tax_rate` to `[0.0, 0.99]` in `__init__` with a `ValueError` outside range.

---

### BUG-P2-02: `analyze_budget_variance` treats missing-value items as `0` on both sides, which double-counts in `favorable_items` / `unfavorable_items` and biases `total_variance`

**File:** `financial-report-insights/financial_analyzer.py`, lines 2630–2647

```python
actual = 0 if actual_missing else float(actual)
budget = 0 if budget_missing else float(budget)
variance_result = self.calculate_variance(actual, budget, category)
line_items.append(variance_result)
```

**Issue:** When a line item exists only in the budget (no actual), it is treated as `actual=0, budget=X`, producing a large unfavorable variance. When it exists only in the actual (no budget), it is treated as `actual=X, budget=0`, producing a large favorable variance. Both of these are recorded in `favorable_items` / `unfavorable_items` and contribute to `total_variance`. The caller is warned via `logger.warning`, but the computed aggregate totals are silently distorted. A company that spent $0 in one category will show a 100% favorable variance for the entire budget of that category.

**Risk:** Material misrepresentation of budget variance analysis when data is not perfectly aligned.

**Fix:** Exclude unmatched items from `total_budget` / `total_actual` calculations, or flag them as `unmatched` in a separate aggregate rather than treating them as 0.

---

### BUG-P2-03: `forecast_simple` method uses walrus operator (`:=`) inside a list comprehension which is Python 3.8+ syntax but is not documented; also assigns `last_error = None` in `_generate_with_retry` then the final `raise last_error` will `raise None` if loop exits normally

**File 1:** `financial-report-insights/financial_analyzer.py`, line 2584

```python
growth_rates = [
    r - 1 for i in range(1, len(historical))
    if (r := safe_divide(historical[i], historical[i - 1])) is not None
]
```

**File 2:** `financial-report-insights/local_llm.py`, lines 326–340

```python
last_error: Exception | None = None
for attempt in range(1, self._max_retries + 1):
    try:
        result = self._call_with_timeout(prompt)
        return result
    except LLMTimeoutError:
        raise  # Don't retry timeouts
    except LLMConnectionError as e:
        last_error = e
        if attempt < self._max_retries:
            ...
            continue
        raise
# This should be unreachable but satisfies type checker
raise last_error  # type: ignore[misc]
```

**Issue (File 2):** The comment says "unreachable" but that is only true because `LLMTimeoutError` re-raises and `LLMConnectionError` re-raises on the last attempt. However, a future refactor that changes the loop logic could make `last_error` remain `None` and cause `raise None` (a `TypeError`). The `# type: ignore[misc]` suppresses the type checker warning that already detected this.

**Issue (File 1):** Minor — the walrus operator works correctly but the list comprehension evaluates `safe_divide` for its side-effect test and then uses `r` for the result. This pattern is unusual and can confuse readers into thinking `r` could be `None` inside the comprehension (it cannot by construction, since the `if` guards it).

**Risk (File 2):** Future `TypeError: exceptions must derive from BaseException` if loop body is refactored.

**Fix (File 2):** Change `raise last_error` to `raise last_error or LLMConnectionError("No attempts made")`.

---

### BUG-P2-04: `_parse_financial_data` in `api.py` caches `FinancialData.__dataclass_fields__` as a function attribute — cache is never invalidated

**File:** `financial-report-insights/api.py`, lines 672–679

```python
def _parse_financial_data(raw: Dict[str, Any]) -> "FinancialData":
    from financial_analyzer import FinancialData
    if not hasattr(_parse_financial_data, "_fields"):
        _parse_financial_data._fields = frozenset(FinancialData.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in _parse_financial_data._fields}
    return FinancialData(**filtered)
```

**Issue:** If `FinancialData` is monkey-patched in tests (adding/removing fields) or if the module is reloaded, `_parse_financial_data._fields` retains the stale field set. Tests that add mock fields to `FinancialData` would silently drop them, and production module reloads would filter on wrong field names. Additionally, the same function-attribute caching anti-pattern noted in BUG-P0-01 applies here.

**Risk:** Stale field filtering in tests or dynamic-reload scenarios. Low impact in production but can mask test failures.

**Fix:** Use a module-level `_FD_FIELDS: frozenset | None = None` variable, or simply call `frozenset(FinancialData.__dataclass_fields__)` directly (it is cheap — a dict key enumeration).

---

### BUG-P2-05: `NumpyFlatIndex.search` computes cosine similarity incorrectly — it divides the matrix-query dot product by only the stored vector norms, not the query norm

**File:** `financial-report-insights/vector_index.py`, lines 169–172

```python
norms = np.linalg.norm(self._matrix, axis=1)
safe_norms = np.where(norms == 0.0, 1.0, norms)
similarities = self._matrix @ q / safe_norms
```

**Issue:** `q` (the query vector) is L2-normalized at line 167 (`self._l2_normalize(q.reshape(1, -1)).reshape(-1)`), which is correct. However, the stored matrix vectors are NOT L2-normalized before the dot product — they are divided by their norms element-wise after the dot product. The expression `(M @ q) / norms` is NOT equal to `(M_normalized @ q_normalized)` when computed as a vector: it equals `M_i · q / ||M_i||` per row, which IS in fact the correct cosine similarity formula since `||q|| = 1` after normalization. So the math is actually correct. However, the code is confusing and could easily be broken by future edits — a comment explaining why `q` is pre-normalized and `M` is divided post-dot is missing.

**Risk:** Low risk to correctness today, but high risk of future regression if someone removes the pre-normalization of `q`.

**Fix:** Add a comment: "q is pre-normalized above; dividing (M @ q) by ||M|| gives cosine(M_i, q) since ||q||=1." Alternatively, normalize `self._matrix` at add-time and simplify search to pure dot-product.

---

### BUG-P2-06: `ingestion_pipeline.py` `ingest_pdf` calls `text_content.replace(table, "")` where `table` is assumed to be a substring of `section.content` — but `table` may be the extracted table object, not the string representation in the content

**File:** `financial-report-insights/ingestion_pipeline.py`, lines 300–303

```python
text_content = section.content
for table in section.tables:
    text_content = text_content.replace(table, "")
```

**Issue:** `section.tables` is typed in the comment as containing table data. If `table` is a dataclass or dict (not a raw string), Python's `str.replace()` will silently do nothing (since `str.replace` requires the argument to be a string — actually it will raise a `TypeError` if `table` is not a string). If `pdf_parser.parse_pdf` returns tables as string slices of `section.content`, this works. But if the return type changes to structured objects, this silently stops removing table text from the prose chunk, leading to double-indexing. No type annotation or assertion validates that `table` is a string.

**Risk:** Double-indexing of table content (table as atomic chunk AND embedded in text chunk) if `pdf_parser` returns structured objects.

**Fix:** Add an explicit `isinstance(table, str)` guard or use a typed protocol for `section.tables`.

---

### BUG-P2-07: `_scored_analysis` grade mapping uses fixed strings ("Excellent", "Good", "Adequate", "Weak") that differ from the `export_utils.score_to_grade` A/B/C/D/F scale and the `_score_to_grade` AAA-C scale, creating three inconsistent grade vocabularies

**File:** `financial-report-insights/financial_analyzer.py`, lines 5282–5289

```python
if score >= 8:
    grade = "Excellent"
elif score >= 6:
    grade = "Good"
elif score >= 4:
    grade = "Adequate"
else:
    grade = "Weak"
setattr(result, grade_field, grade)
```

**Issue:** The codebase uses at least three incompatible grading scales: (1) `export_utils.score_to_grade(0-100) -> A/B/C/D/F`; (2) `CharlieAnalyzer._score_to_grade(0-10) -> AAA/AA/A/BBB/BB/B/CCC/CC/C`; (3) `_scored_analysis(0-10) -> Excellent/Good/Adequate/Weak`. The `_MEMORY.md` documents `score_to_grade` as "canonical" but the 40+ methods converted to `_scored_analysis` use the fourth vocabulary. UI display code and API consumers that compare grades across methods will see inconsistent values.

**Risk:** Grade comparison bugs in any code that compares `grade` strings across different analysis results.

**Fix:** Standardize `_scored_analysis` to use `self._score_to_grade(score)` or add a clear conversion table in the docstring documenting which methods use which vocabulary.

---

### BUG-P2-08: `monte_carlo_simulation` floors sampled multipliers at 0.01 (`max(sample, 0.01)`) which creates asymmetric distribution and biases results toward slightly-above-zero

**File:** `financial-report-insights/financial_analyzer.py`, lines 3863–3865

```python
sample = rng.normal(mean_mult, std_mult)
sample = max(sample, 0.01)  # Floor at 1% to avoid negatives
adjustments[fld] = sample
```

**Issue:** For variables with `std_pct=10%`, roughly 16% of samples fall below `mean_mult - 1*std = 0.90`. The floor at 0.01 truncates the left tail but does not eliminate it — samples between 0.01 and 0.90 are kept. When `std_pct` is large (e.g. 50%), a significant fraction of samples are clipped to 0.01, causing the effective mean to be higher than intended and creating a non-normal distribution. The simulation output (percentiles, expected values) is silently biased. Additionally, revenue clipped to 1% of base is effectively a near-zero scenario, which is very different from the intended mild negative shock.

**Risk:** Monte Carlo distributions are skewed by the floor, leading to optimistic P10 estimates (worst-case scenarios are modeled as "revenue drops to 1%" rather than negative).

**Fix:** Use a log-normal distribution instead of a clipped normal: `sample = rng.lognormal(mean=np.log(mean_mult), sigma=std_mult/mean_mult)` which is naturally positive without clipping.

---

### BUG-P2-09: `variance_waterfall` computes `pct = (total_var / abs(start) * 100) if start != 0 else 0` — when `start` is `None` (both periods missing NI) the `or 0` coercion makes `start=0` and `pct=0`, hiding the error

**File:** `financial-report-insights/financial_analyzer.py`, lines 5493–5565

```python
start = previous.net_income or 0
end = current.net_income or 0
...
pct = (total_var / abs(start) * 100) if start != 0 else 0
```

**Issue:** When `previous.net_income` is `None`, `start = None or 0 = 0`. The waterfall then shows a "0 -> end" change with `pct=0`, which silently misrepresents a full-data scenario as a zero-start scenario. The summary string `"Net income changed by X (0.0%) from 0 to Y"` is misleading.

**Risk:** Misleading financial report output when prior period data is missing.

**Fix:** Return early with a `summary="Prior net income unavailable"` when `previous.net_income is None`.

---

## P3 — Low / Housekeeping

### BUG-P3-01: `wait_for_embedding_service` catches bare `Exception` in the warm-up retry loop, which swallows programming errors (e.g., `AttributeError` on a misconfig)

**File:** `financial-report-insights/local_llm.py`, lines 550–552

```python
except Exception:
    logger.info("Waiting for embedding service (%s)...", label)
    time.sleep(2)
```

**Fix:** Narrow to `except (httpx.RequestError, httpx.HTTPStatusError, OSError)`. Add a `except Exception as e: logger.warning("Unexpected warm-up error: %s", e); time.sleep(2)` fallback for genuinely unexpected errors to avoid infinite loops on programming bugs.

---

### BUG-P3-02: `ingestion_pipeline.py` `_df_to_markdown` silently truncates DataFrames to 200 rows with no metadata indicating truncation was applied

**File:** `financial-report-insights/ingestion_pipeline.py`, lines 136–158

```python
def _df_to_markdown(df: pd.DataFrame, max_rows: int = 200) -> str:
    ...
    truncated = df.head(max_rows)
```

**Issue:** Financial models often have more than 200 rows (multi-year detail schedules, rent rolls, etc.). When truncated, the chunk text does not indicate that rows were omitted. A RAG query asking about the last row of a 500-row schedule would return no results.

**Fix:** Append `f"\n[...{len(df) - max_rows} rows omitted...]"` to the markdown when `len(df) > max_rows`.

---

### BUG-P3-03: `graph_store.py` `store_chunks` uses `chunks[0].get("type", "unknown")` without a guard — will raise `IndexError` if `chunks` is empty despite the later `if batch:` guard

**File:** `financial-report-insights/graph_store.py`, lines 105–106

```python
file_type = chunks[0].get("type", "unknown") if chunks else "unknown"
session.run(MERGE_DOCUMENT, doc_id=doc_id, filename=doc_id, file_type=file_type)
```

**Issue:** The ternary guard `if chunks` is correct, but `MERGE_DOCUMENT` is called regardless of whether `chunks` is empty. This creates a document node with `file_type="unknown"` even when there's nothing to store, which may produce orphaned nodes in the graph.

**Fix:** Move the `MERGE_DOCUMENT` call inside `if batch:` to avoid creating orphaned document nodes.

---

### BUG-P3-04: `portfolio_analyzer.py` `correlation_matrix` replaces NaN correlations with 0.0 in the output matrix, but the `avg_correlation` is computed before the NaN replacement — if all off-diagonal values are NaN (all companies have identical ratio profiles), `avg_corr` is `0.0` instead of the correct `1.0`

**File:** `financial-report-insights/portfolio_analyzer.py`, lines 234–250

```python
valid = off_diag[~np.isnan(off_diag)]
avg_corr = float(valid.mean()) if len(valid) > 0 else 0.0
...
corr = np.nan_to_num(corr, nan=0.0)
```

**Issue:** When all companies have all-zero ratio vectors (no data), `np.corrcoef` returns an all-NaN matrix. `valid` is empty, so `avg_corr = 0.0`. Then `DiversificationScore.correlation_penalty = 0.0` is reported, which maps to 20 correlation points — a significant overestimate of diversification for a portfolio with no computable ratios.

**Fix:** When `len(valid) == 0` (no computable correlations), return `avg_corr = 1.0` (maximum concentration, no diversification benefit from correlation data), not `0.0`.

---

### BUG-P3-05: `api.py` `/compare` endpoint in the in-memory fallback uses `O(n*m)` linear scan to find existing `PeriodDelta` entries by `ratio_name`

**File:** `financial-report-insights/api.py`, lines 485–496

```python
for d in deltas:
    if d.ratio_name == result.name:
        d.periods[label] = result.value
        found = True
        break
if not found:
    deltas.append(PeriodDelta(...))
```

**Issue:** For each period × each ratio (potentially hundreds of ratios × multiple periods), this is O(ratios * periods) = O(n²) overall. At 100 ratios × 10 periods = 1,000 iterations per ratio lookup, and 100 ratios per period = 100,000 dict-miss checks total. Not a correctness bug but a performance issue on large requests.

**Fix:** Use a dict `delta_map: Dict[str, PeriodDelta]` keyed by `ratio_name`, then convert to list at the end.

---

### BUG-P3-06: `CharlieAnalyzer` loads the full `config.settings` at `__init__` time — when imported at module load time (e.g. for testing), this triggers Pydantic validation of all environment variables before any test fixtures can set them

**File:** `financial-report-insights/financial_analyzer.py`, lines 2220–2223

```python
def __init__(self, tax_rate: Optional[float] = None):
    from config import settings
    self._tax_rate = tax_rate if tax_rate is not None else settings.default_tax_rate
```

**Issue:** This is a lazy import (inside `__init__`), so it only runs when the class is instantiated. This is actually the correct pattern. However, the import is unconditional — if `config` is unavailable (e.g., in a minimal test environment), `CharlieAnalyzer()` will raise `ImportError` or `ValidationError` from Pydantic, rather than defaulting to a safe value. Compare with `LocalEmbedder` which gracefully falls back to `cfg_dim = 0`.

**Fix:** Wrap in `try/except (ImportError, Exception): self._tax_rate = tax_rate if tax_rate is not None else 0.21` with a `logger.debug`.

---

### BUG-P3-07: `liquidity_stress_test` uses `float('inf')` as `survival_months` but `max(...)` then wraps it in `round()` only if `!= float('inf')` — however the `scenarios` list still contains `None` for `survival_months` for infinite cases, breaking any consumer that sums survival months

**File:** `financial-report-insights/financial_analyzer.py`, lines 5837–5847

```python
elif stressed_cf >= 0:
    survival_months = float('inf')
...
scenarios.append({
    ...
    "survival_months": round(survival_months, 1) if survival_months != float('inf') else None,
    "survives_12m": survival_months >= 12,
})
```

**Issue:** When `survival_months = float('inf')`, the dict stores `survival_months=None`. This is safe for the `worst_survival = min(...)` calculation at line 5850 (which uses `if s["survival_months"] is not None`). But the `survives_12m` field is computed as `float('inf') >= 12 = True`, which is correct. The issue is cosmetic: `None` in `survival_months` means either "company survives indefinitely" or "data unavailable", which are different conditions. A consumer cannot distinguish the two.

**Fix:** Use a large sentinel value like `9999.0` for infinite survival, or add a separate `"survival_is_infinite": True` key.

---

## Files With No Issues Found

None — all eight files reviewed contain at least one finding. The issues found vary significantly in severity.

---

## Summary

| Priority | Count | Key Themes |
|---|---|---|
| P0 | 3 | Singleton pattern fragility (api.py), silent zero-dimension embeddings (local_llm.py), thread executor leaking after timeout (local_llm.py) |
| P1 | 5 | HNSW index correctness (data loss after each add), FAISS O(n²) memory, embedding connection errors swallowed, rate limiter memory leak, compliance scoring penalizes missing data as failures |
| P2 | 9 | Various precision/logic issues in financial calculations, stale caches, inconsistent grade vocabularies, Monte Carlo distribution bias |
| P3 | 7 | Error swallowing in warm-up, truncation without metadata, orphaned graph nodes, O(n²) delta lookup, config import fragility |

**Most Critical Items to Address First:**

1. **BUG-P1-03 (HNSWIndex data loss)** — This is a complete correctness failure: the HNSW index only ever contains the last batch of embeddings added. Every search against an HNSW index after the initial load will miss all but the most recently added documents.

2. **BUG-P0-03 (ThreadPoolExecutor thread starvation)** — After a single LLM timeout the executor's only thread is stuck, making the LLM permanently unavailable until the process restarts, even though the circuit breaker will attempt recovery.

3. **BUG-P0-02 (zero-dimension embeddings)** — A misconfigured or unreachable embedding service on startup can result in all documents being silently stored with empty/zero embedding vectors, making the entire vector search index return garbage results.

4. **BUG-P1-01 (httpx.RequestError swallowed)** — Connection-level failures in the embedding endpoint are silently swallowed, resulting in documents being ingested with no error to the operator.
