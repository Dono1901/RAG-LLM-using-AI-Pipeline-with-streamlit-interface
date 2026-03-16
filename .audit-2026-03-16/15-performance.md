# Audit Report: Performance Risks
**Date:** 2026-03-16
**Scope:** Hot paths, memory usage, caching, blocking calls, O(n^2) patterns
**Files Reviewed:**
- `financial-report-insights/financial_analyzer.py` (13,894 lines)
- `financial-report-insights/vector_index.py` (617 lines)
- `financial-report-insights/local_llm.py` (586 lines)
- `financial-report-insights/ingestion_pipeline.py` (447 lines)
- `financial-report-insights/graph_store.py` (741 lines)
- `financial-report-insights/insights_page.py` (7,981 lines)
- `financial-report-insights/ml/semantic_cache.py` (459 lines)
- `financial-report-insights/ml/forecasting.py` (661 lines)
- `financial-report-insights/excel_processor.py` (867 lines)
- `financial-report-insights/document_chunker.py` (419 lines)

---

## P0 — Critical (Fix Immediately)

### P0-1: Monte Carlo — Full Analysis Pipeline Runs Up to 10,000 Times in a Tight Loop
**File:** `financial_analyzer.py`, lines 3858–3879

```python
for _ in range(n_simulations):   # n_simulations capped at 10,000
    adjusted = self._apply_adjustments(data, adjustments)   # deepcopy inside
    health = self.composite_health_score(adjusted)
    z = self.altman_z_score(adjusted)
    f = self.piotroski_f_score(adjusted)
    prof = self.calculate_profitability_ratios(adjusted)
    liq = self.calculate_liquidity_ratios(adjusted)
```

Each iteration calls `_apply_adjustments` (which executes `copy.deepcopy` — see P0-2 below), then five separate analysis methods. With the 10,000 simulation cap that is 50,000 method calls, each of which traverses multiple ratio computations. At even 0.5 ms per iteration this exceeds 5 seconds of synchronous CPU time on the main thread, blocking every Streamlit rerun. The simulation receives no progress feedback and provides no async escape hatch.

**Risk:** Full CPU stall on the Streamlit event thread for multi-second durations; repeated trigger on every widget interaction (see P1-3 for caching gap).

**Recommendation:** Vectorise the stochastic sampling entirely in NumPy — draw all `n_simulations` random multipliers at once as a `(n_simulations, n_vars)` matrix, apply them to the scalar base values without object copies, and compute ratios as array operations. This eliminates both the `deepcopy` overhead and the Python loop entirely. For the interim, lower the default from 1,000 to 200 simulations and run in a background thread with `st.spinner`.

---

### P0-2: `deepcopy` Called Inside Every Monte Carlo Iteration
**File:** `financial_analyzer.py`, line 3580

```python
import copy
adjusted = copy.deepcopy(data)
```

`_apply_adjustments` is called from both `scenario_analysis` and the Monte Carlo loop. The `FinancialData` dataclass carries ~50 float fields and several list/dict fields. While a single `deepcopy` is cheap, at 10,000 iterations it generates 10,000 fully independent Python objects, placing significant pressure on the garbage collector during simulation.

**Recommendation:** Replace with a shallow field copy using `dataclasses.replace(data, **{field: val * mult ...})` for the scalar fields that are actually adjusted. Since `_apply_adjustments` only modifies top-level numeric attributes, a targeted field replacement eliminates the deep copy entirely.

---

### P0-3: `FAISSIndex.add` Rebuilds the Entire FAISS Index on Every Call
**File:** `vector_index.py`, lines 303–307

```python
self._raw_embeddings.extend(embeddings)
self._ids.extend(ids)
matrix = self._normalize(np.asarray(self._raw_embeddings, dtype=np.float32))
self._build_index(matrix)
```

Every call to `add` re-normalises the full accumulated embedding matrix and then calls `faiss.IndexFlatIP.add` (or re-trains and re-builds `IndexIVFFlat`) from scratch. For incremental ingestion of a large document corpus (e.g., 1,000 chunks ingested sheet-by-sheet), this is O(n^2) total work: 1+2+3+...+n normalisations and FAISS rebuilds.

`_raw_embeddings` is a Python `list[list[float]]` that also grows unboundedly — at 1024 dims per vector, 10,000 vectors consume ~160 MB in Python list form before conversion to NumPy.

**Recommendation:** Batch all embeddings in one `add` call during ingestion so the index is built once. For true incremental use, cache the pre-normalised matrix and `np.vstack` only the delta, not the entire history.

---

### P0-4: `HNSWIndex.add` Tears Down and Rebuilds the Entire HNSW Graph on Every Call
**File:** `vector_index.py`, lines 453–455

```python
self._init_index(len(self._ids))
self._index.add_items(matrix, self._ids)
```

The comment at line 452 acknowledges this: "Rebuild index with updated max_elements". Every incremental `add` call destroys the existing HNSW graph and rebuilds it with all vectors including previously added ones. This is O(n log n) per call and O(n^2 log n) total over an ingestion run of n batches. HNSW graph construction is more expensive than FAISS flat index construction per-element.

**Recommendation:** Pre-allocate with a generous `max_elements` at construction time (e.g., 2x estimated corpus size), then call `add_items` once or in large batches. `hnswlib` supports `resize_index` for expansion without full rebuild.

---

## P1 — High (Fix This Sprint)

### P1-1: `NumpyFlatIndex.search` Re-normalises the Entire Stored Matrix on Every Query
**File:** `vector_index.py`, lines 170–172

```python
norms = np.linalg.norm(self._matrix, axis=1)
safe_norms = np.where(norms == 0.0, 1.0, norms)
similarities = self._matrix @ q / safe_norms
```

The row norms are recomputed on every single search query. With a corpus of 10,000 vectors at 1024 dims, this is a 10,000-element `np.linalg.norm` call followed by an element-wise divide — all redundant work since the stored vectors never change between queries.

**Recommendation:** Pre-normalise all vectors at `add` time (the same way `FAISSIndex._normalize` already does) and store the normalised matrix. Then `search` becomes a pure matrix-vector dot product with no per-query normalisation. This eliminates the O(n) norm computation on the hot search path.

---

### P1-2: `SemanticCache.get` Rebuilds the Embedding Matrix on Every Cache Lookup
**File:** `ml/semantic_cache.py`, lines 85–89 and 110–112

```python
def _build_matrix(self) -> Optional[np.ndarray]:
    if not self._store:
        return None
    return np.vstack([emb for emb, _ in self._store.values()])
```

`_build_matrix` is called on every `get`. It iterates the full `OrderedDict` and stacks all stored embeddings into a fresh NumPy matrix each time. With `max_entries=1000`, this vstack allocates a `(1000, 1024)` matrix on every cache lookup — 4 MB of allocation per query at 1024 dims.

Additionally, line 118:
```python
key = list(self._store.keys())[best_idx]
```
This materialises the full keys list (another allocation) just to index by position. Since `OrderedDict` doesn't support integer indexing, this is unavoidable without restructuring, but the matrix rebuild is the dominant cost.

**Recommendation:** Maintain a separate `np.ndarray` matrix as a class member. Append rows at `put` time and evict by row deletion or by maintaining an offset counter when `max_entries` is exceeded. The matrix rebuild is then eliminated from the hot `get` path entirely.

---

### P1-3: `analyze()` Called Unconditionally on Every Streamlit Rerun
**File:** `insights_page.py`, line 258

```python
st.session_state['analysis_results'] = self.analyzer.analyze(df)
```

This line sits inside `render()`, which Streamlit re-executes on every user interaction (tab click, slider move, checkbox toggle). `analyze()` calls `calculate_liquidity_ratios`, `calculate_profitability_ratios`, `calculate_leverage_ratios`, `calculate_efficiency_ratios`, `analyze_cash_flow`, `analyze_working_capital`, `dupont_analysis`, `altman_z_score`, `piotroski_f_score`, `composite_health_score`, and `generate_insights` — all on every rerun.

The existing `st.session_state` assignment does not guard against re-execution: the assignment itself unconditionally overwrites any previously cached value.

**Recommendation:** Gate the `analyze()` call behind a staleness check:
```python
file_key = f"_analysis_{selected_file}_{selected_sheet}"
if file_key not in st.session_state:
    st.session_state[file_key] = self.analyzer.analyze(df)
st.session_state['analysis_results'] = st.session_state[file_key]
```
Alternatively, wrap `analyze` with `@st.cache_data` keyed on the DataFrame hash and sheet name.

---

### P1-4: `EnsembleForecaster.fit` Calls `walk_forward_validate` Three Times Per Model
**File:** `ml/forecasting.py`, lines 552–598

```python
# First call during AR fitting (line 552-556)
metrics = walk_forward_validate(SimpleARModel(...), values, ...)

# Second call during exponential fitting (line 563-567)
metrics = walk_forward_validate(ExponentialSmoother(...), values, ...)

# Third loop during metric aggregation (lines 591-603)
met = walk_forward_validate(m_instance, values, ...)
```

Each `walk_forward_validate` call re-fits the model `test_size` (up to 3) times on expanding windows, including a `scipy.optimize.minimize` (L-BFGS-B) call per fit for the exponential smoother. With two models this is 4 separate `scipy.minimize` optimisations per `EnsembleForecaster.fit` invocation. The third loop at lines 591-603 re-runs the same validation a second time for each model with identical parameters — the results are never different from the first call.

**Recommendation:** Cache the `walk_forward_validate` results from the first call and re-use them in the aggregation loop. Remove the redundant third validation loop entirely.

---

### P1-5: `LocalLLM._executor` Uses a Single-Worker Thread Pool Per Instance
**File:** `local_llm.py`, line 209

```python
self._executor = ThreadPoolExecutor(max_workers=1)
```

The executor is a shared singleton per `LocalLLM` instance, but the instance is recreated in every call to `_get_rag()` (from `insights_page.py`). Each recreation spawns a new `ThreadPoolExecutor` without closing the previous one. The old executor and its background thread are silently abandoned (no `shutdown()` call), leaking OS thread handles.

**Recommendation:** Ensure `LocalLLM` is instantiated once (e.g., as a module-level singleton guarded by `threading.Lock`) and the executor is explicitly shut down on application exit via a registered `atexit` handler or Streamlit session lifecycle hook.

---

### P1-6: `ChunkDeduplicator.deduplicate` Has O(n^2) Complexity with Per-Pair MD5 Hashing
**File:** `ml/semantic_cache.py`, lines 293–304

```python
for i, h_i in enumerate(hashes):
    is_dup = False
    for j in kept_indices:
        sim = _hamming_similarity(h_i, hashes[j], self._NUM_BITS)
        if sim >= self.similarity_threshold:
            is_dup = True
            break
    if not is_dup:
        kept_indices.append(i)
```

The nested loop compares each incoming chunk against all previously retained chunks. For a corpus of n chunks, this is O(n^2) comparisons. Each SimHash also calls `hashlib.md5` for every 3-gram shingle (lines 204–211), making `_simhash` O(m) per chunk where m is text length. For a large ingestion of 5,000 chunks with average 500 characters, the SimHash step alone is ~100M MD5 operations.

**Recommendation:** Replace the per-comparison inner loop with a banding technique: partition the 64-bit SimHash into k bands of b bits each (standard LSH approach). Chunks that share at least one identical band are candidates. This reduces candidate pairs from O(n^2) to O(n) expected with a tunable false-negative rate.

---

## P2 — Medium (Fix Soon)

### P2-1: `_dataframe_to_financial_data` O(n*m) Pattern Matching on Every Analyze Call
**File:** `financial_analyzer.py`, lines 13811–13821

```python
for col in df.columns:
    col_lower = col.lower().strip()
    best_attr = None
    best_len = 0
    for attr, patterns in mappings.items():   # 20 attrs
        for pattern in patterns:              # 3-5 patterns each = ~80 iterations
            if pattern in col_lower and len(pattern) > best_len:
```

With a 200-column Excel file this is 200 × 80 = 16,000 substring checks on every call to `analyze()`. Given that `analyze()` is called on every Streamlit rerun (P1-3), these 16,000 checks fire on every widget interaction.

**Recommendation:** Build the pattern-to-attribute lookup once as a compiled `{pattern: attr}` dict sorted by descending length, then use a single pass over each column name. Cache the resulting `column_map` in `st.session_state` keyed by `tuple(df.columns)`.

---

### P2-2: `_detect_label_column` Performs Nested Pattern Scan on Every `analyze` Fallback
**File:** `financial_analyzer.py`, lines 13847–13858

```python
all_patterns = []
for patterns in mappings.values():
    all_patterns.extend(patterns)      # ~80 patterns rebuilt every call
for col in df.columns:
    if df[col].dtype == object:
        matches = 0
        for val in df[col].dropna():
            val_lower = str(val).lower().strip()
            if any(p in val_lower for p in all_patterns):  # 80 checks per cell
```

The `all_patterns` list is rebuilt from `mappings` on every call. For a 100-row column, this is 100 × 80 = 8,000 substring checks per column. This method also has a sibling `_transpose_financial_df` at line 13869 that calls `df.iterrows()` — a known anti-pattern that is 10–100x slower than vectorised Pandas operations for row iteration.

**Recommendation:** Pre-build `all_patterns` once as a class-level constant. Replace `df.iterrows()` in `_transpose_financial_df` with `df[[label_col, value_col]].apply(...)` or direct vectorised column access.

---

### P2-3: `ingestion_pipeline._find_label_column` Uses `.apply(lambda)` for Type Check
**File:** `ingestion_pipeline.py`, line 125

```python
str_count = col.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 2).sum()
```

Per-element Python lambda calls are 10–100x slower than vectorised Pandas/NumPy operations. This runs for up to 5 columns during sheet type detection, which occurs for every sheet of every uploaded Excel file.

**Recommendation:** Replace with:
```python
str_count = (col.dtype == object) and (col.dropna().astype(str).str.strip().str.len() > 2).sum()
```

---

### P2-4: `graph_store.store_line_items` Opens and Commits Three Separate Sessions
**File:** `graph_store.py`, lines 240–270

```python
with self._driver.session() as session:
    for stmt_type, fields in self._STATEMENT_FIELD_MAP.items():
        ...
        if batch:
            session.run(MERGE_FINANCIAL_STATEMENT, ...)   # query 1 per stmt_type
            session.run(MERGE_LINE_ITEMS_BATCH, ...)       # query 2 per stmt_type
```

Each `session.run` issues a separate round-trip to the Neo4j server over the Bolt connection. With 3 statement types, `store_line_items` issues 6 Bolt round-trips in series. Neo4j Bolt connections have ~1–5 ms round-trip latency even on localhost, adding 6–30 ms of latency.

**Recommendation:** Combine both the `MERGE_FINANCIAL_STATEMENT` and `MERGE_LINE_ITEMS_BATCH` operations into a single parameterised Cypher query using `UNWIND` with nested statement metadata, or pipeline them in an explicit transaction using `session.execute_write`.

---

### P2-5: `ExcelProcessor._clean_dataframe` Calls `pd.to_numeric` on Every Column Every Load
**File:** `excel_processor.py`, lines 399–405

```python
for col in df.columns:
    try:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        if numeric_col.notna().sum() / len(df) > 0.5:
            df[col] = numeric_col
```

`pd.to_numeric` is called for every column in the workbook on every call to `load_workbook`, including the sheet selector populating in the sidebar (which calls `load_workbook` again via `st.session_state` cache miss). For a 50-column workbook, this is 50 `to_numeric` passes on every file load. The `loaded_workbooks` dict at line 194 caches the `WorkbookData` object, but the sidebar re-loads via a separate `cache_key` in `st.session_state` (line 304 of `insights_page.py`), causing two independent `load_workbook` calls per file selection change.

**Recommendation:** Ensure a single shared `load_workbook` call path. The sidebar and the main render path currently cache under different keys, defeating the per-instance `loaded_workbooks` cache.

---

### P2-6: `document_chunker._is_mostly_numeric` Called Redundantly for Both Parent and Child Chunks
**File:** `document_chunker.py`, lines 205–209 and 236–239

```python
# For parent chunk:
if _is_mostly_numeric(parent_text):
    parent_chunk.nl_description = _generate_nl_description(parent_text, ...)

# Then for each child slice of the same parent:
if _is_mostly_numeric(child_text):
    child_chunk.nl_description = _generate_nl_description(child_text, ...)
```

`_is_mostly_numeric` iterates every character in the text twice (once counting digits, once counting alpha). `_generate_nl_description` also splits lines and applies regex twice per call. For a 200-row Excel sheet chunked into 5 parents × 3 children each, this is 20 character-iteration passes over largely overlapping text.

**Recommendation:** Compute `_is_mostly_numeric` once per block and pass the boolean down. For `_generate_nl_description`, memoize the header extraction (lines 98–116) since the section header and source are constant across all chunks of the same sheet.

---

### P2-7: `EnsembleForecaster` Calls `scipy.optimize.minimize` Twice Per Model on Every `fit`
**File:** `ml/forecasting.py`, lines 218–225 (simple), 265–272 (double), 332–343 (triple)

Each `ExponentialSmoother.fit` runs L-BFGS-B optimisation via `scipy.optimize.minimize`. The `EnsembleForecaster.fit` creates and fits a second throw-away instance of each model type inside `walk_forward_validate` for each validation fold. For 3 folds, this means 4 total `minimize` calls per model type (1 production fit + 3 validation folds), each involving multiple gradient evaluations of `_double_smooth` or `_triple_smooth`.

**Recommendation:** Cache optimised parameters from the first full-data fit and warm-start the validation fold fits with those parameters (L-BFGS-B supports `x0`), reducing the number of gradient evaluations per fold significantly.

---

### P2-8: `local_llm._send_embedding_batch` Imports `time` and `httpx` Inside the Hot Path
**File:** `local_llm.py`, lines 497–498

```python
def _send_embedding_batch(self, texts: list, ...) -> list:
    import time
    import httpx
```

Module-level imports inside a function that is called for every embedding batch. While Python caches imports after the first load, the `sys.modules` dictionary lookup still occurs on every call. During bulk ingestion of a 1,000-chunk document, this function is called dozens of times in rapid succession.

**Recommendation:** Move both imports to the module top level or to `__init__`.

---

### P2-9: `NumpyFlatIndex.add` Uses `np.vstack` Which Copies the Entire Matrix
**File:** `vector_index.py`, lines 143–146

```python
if self._matrix.shape[0] == 0:
    self._matrix = new_matrix
else:
    self._matrix = np.vstack([self._matrix, new_matrix])
```

`np.vstack` allocates a new array of size (old_n + new_n) × dim and copies all existing rows. For incremental ingestion in small batches, this is O(n^2) total copy work: each batch copies all previous data. At 10,000 vectors × 1024 dims × 4 bytes = 40 MB, the final append copies 40 MB even when adding just 1 new vector.

**Recommendation:** Pre-allocate with a capacity-doubling strategy (similar to Python's list growth), or collect all embeddings in a list and call `np.vstack` once at query time or after ingestion completes.

---

## P3 — Low / Housekeeping

### P3-1: `LocalLLM._cache_key` Hashes the Full Prompt with SHA-256 on Every Call
**File:** `local_llm.py`, line 381

```python
def _cache_key(self, prompt: str) -> str:
    return hashlib.sha256(f"{self.model}:{prompt}".encode()).hexdigest()
```

SHA-256 is cryptographically strong but ~3x slower than MD5 for this non-security use case. The key is called twice per non-cached `generate` call (pre-check and post-store). For high-throughput scenarios this is minor but unnecessary overhead.

**Recommendation:** Use `hashlib.md5(..., usedforsecurity=False)` for the LRU cache key.

---

### P3-2: `ingestion_pipeline._df_to_markdown` Fallback Uses `iterrows()`
**File:** `ingestion_pipeline.py`, lines 155–157

```python
for _, row in truncated.iterrows():
    vals = [str(v) if pd.notna(v) else "" for v in row]
    lines.append("| " + " | ".join(vals) + " |")
```

The `iterrows()` fallback path (reached when `to_markdown` is unavailable or raises `TypeError`) iterates row-by-row. For a 200-row sheet this is 200 Python iterations. The primary path via `to_markdown` is vectorised and much faster. This is a minor issue since the fallback is rarely triggered in practice, but worth noting.

**Recommendation:** Replace the `iterrows()` fallback with a vectorised approach:
```python
rows = truncated.astype(str).replace("nan", "").values.tolist()
lines += ["| " + " | ".join(r) + " |" for r in rows]
```

---

### P3-3: `graph_store.store_financial_data` Opens a New Session Per Operation
**File:** `graph_store.py`, lines 153–183

Each public write method (`store_financial_data`, `store_line_items`, `store_derived_from_edges`, `store_credit_assessment`, `store_covenant_package`) opens its own `self._driver.session()` context manager. When called sequentially after a single `analyze()` invocation, this is 5 separate Bolt connection handshakes that could be combined into a single atomic transaction.

**Recommendation:** Provide a `store_analysis_batch` method that wraps all post-analysis writes in a single `session.execute_write` transaction. This is both faster and more correct (atomically commits or rolls back all analysis data).

---

### P3-4: `ExcelProcessor.scan_for_excel_files` Calls `rglob` Twice Per Extension
**File:** `excel_processor.py`, lines 138–141

```python
for ext in self.SUPPORTED_EXTENSIONS:
    files.extend(self.documents_path.rglob(f"*{ext}"))
    files.extend(self.documents_path.rglob(f"*{ext.upper()}"))
```

This calls `rglob` 10 times (5 extensions × 2 cases) for a single directory scan. Each `rglob` call performs a full directory tree traversal. On Windows with a large documents folder this is 10 full directory scans where one would suffice.

**Recommendation:** Use a single `rglob("*")` pass and filter by extension using a set membership check in post-processing:
```python
exts = {e.lower() for e in self.SUPPORTED_EXTENSIONS}
files = [p for p in self.documents_path.rglob("*") if p.suffix.lower() in exts]
```

---

### P3-5: `financial_analyzer.analyze` Calls `results.to_dict()` to Pass Back to `generate_insights`
**File:** `financial_analyzer.py`, line 13774

```python
results.insights = self.generate_insights(results.to_dict())
```

`AnalysisResults.to_dict()` serialises the structured result to a plain `Dict[str, Any]` immediately after construction, only for `generate_insights` to re-access fields by string key. This is unnecessary serialisation within the same call stack; `generate_insights` could accept `AnalysisResults` directly and access typed attributes.

**Recommendation:** Add a typed overload of `generate_insights(results: AnalysisResults)` and pass `results` directly, eliminating the `to_dict()` round-trip.

---

### P3-6: `_simhash` in `semantic_cache.py` Creates `num_bits`-Element Python List Per Call
**File:** `ml/semantic_cache.py`, lines 202–218

```python
vector = [0] * num_bits   # 64-element Python list
for i in range(len(text) - 2):
    ...
    for bit in range(num_bits):   # 64 iterations per shingle
        if h & (1 << bit):
            vector[bit] += 1
        else:
            vector[bit] -= 1
```

The inner bit loop runs 64 times per 3-gram shingle. For a 500-character chunk, that is ~500 shingles × 64 bit operations = 32,000 Python integer operations. Replace with a NumPy `int64` packed representation and bitwise population-count (`np.unpackbits` + sum), which would vectorise the inner loop entirely.

**Recommendation:** Use `numpy.unpackbits` or leverage the `popcount` pattern on the full 64-bit integer to compute the vote vector in a single NumPy operation instead of a 64-iteration Python loop.

---

### P3-7: `insights_page.py` Only Has One `@st.cache_data` Annotation
**File:** `insights_page.py`, line 7947

Only `_extract_key_metrics` is decorated with `@st.cache_data`. The much more expensive `analyzer.analyze(df)`, `processor.load_workbook(file_path)`, and `processor.combine_sheets_intelligently(workbook)` calls are not cached at all (see P1-3). This leaves most of the expensive computation unprotected from Streamlit rerun overhead.

**Recommendation:** Audit every method in `render()` that is called unconditionally and apply `@st.cache_data` where inputs are hashable, or use `st.session_state` with explicit cache keys as described in P1-3.

---

## Files With No Issues Found

**`ml/forecasting.py` — Core Model Logic:** `SimpleARModel.fit/predict` and `ExponentialSmoother.predict` use clean NumPy matrix operations with no Python loops in the inner kernels. `compute_prediction_intervals` is properly vectorised. No memory leaks or unbounded allocations detected in the model logic itself (the walk-forward redundancy is called out in P1-4 as an EnsembleForecaster integration issue, not a model issue).

**`graph_store.py` — Write Batching:** All bulk write operations (`store_chunks`, `store_financial_data`, `store_line_items`, `store_derived_from_edges`) correctly use `UNWIND` batching in Cypher rather than per-item N+1 queries. MERGE semantics ensure idempotent upserts. The `graph_search` method correctly batches graph context lookups in a single `GRAPH_CONTEXT_FOR_CHUNKS_BATCH` query rather than issuing per-chunk traversals.

**`document_chunker.py` — Chunking Logic:** The parent-child chunking algorithm is linear O(n) in text length with no nested loops over the full corpus. SHA-256 chunk ID generation is appropriately called once per chunk. No large object accumulation detected.

**`local_llm.py` — Circuit Breaker + Cache:** The `CircuitBreaker` uses `threading.Lock` correctly with no lock inversion risk. The LRU cache uses `OrderedDict` with a bounded `max_entries` eviction. The `ThreadPoolExecutor(max_workers=1)` correctly serialises Ollama calls to prevent concurrent model requests. The retry logic correctly refuses to retry `LLMTimeoutError`.

---

## Summary

| Priority | Count | Theme |
|----------|-------|-------|
| P0 | 4 | Monte Carlo CPU stall, deepcopy in hot loop, FAISS/HNSW full index rebuild on every add |
| P1 | 6 | Per-query matrix re-normalisation, semantic cache matrix rebuild, unconditional re-analysis on Streamlit rerun, duplicate walk-forward validation, thread pool leak, O(n^2) deduplication |
| P2 | 9 | O(n*m) pattern matching, iterrows anti-patterns, triple Neo4j round-trips, duplicate workbook loads, redundant numeric type detection, scipy warm-start opportunities |
| P3 | 7 | Minor: SHA-256 overhead, rglob duplication, serialisation round-trip, SimHash inner loop, cache annotation gaps |

**The most impactful fixes by expected wall-clock improvement, in order:**

1. **P0-1 + P0-2 (Monte Carlo vectorisation + deepcopy elimination):** Eliminates multi-second Streamlit freezes on simulation trigger.
2. **P1-3 (Streamlit rerun caching):** Eliminates re-running all 10 analysis methods on every widget interaction.
3. **P0-3 + P0-4 (FAISS/HNSW incremental add):** Reduces ingestion time from O(n^2) to O(n) for large document corpora.
4. **P1-1 (NumpyFlatIndex pre-normalisation):** Eliminates O(n) norm computation from every vector search query.
5. **P1-2 (SemanticCache matrix caching):** Eliminates 4 MB allocation per cache lookup.
