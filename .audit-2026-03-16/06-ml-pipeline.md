# Audit Report: ML Pipeline
**Date:** 2026-03-16
**Scope:** ML classifiers, clustering, embedding optimizer, feature engineering, forecasting, model registry, semantic cache
**Files Reviewed:**
- `financial-report-insights/ml/__init__.py`
- `financial-report-insights/ml/classifiers.py`
- `financial-report-insights/ml/clustering.py`
- `financial-report-insights/ml/embedding_optimizer.py`
- `financial-report-insights/ml/feature_engineering.py`
- `financial-report-insights/ml/forecasting.py`
- `financial-report-insights/ml/registry.py`
- `financial-report-insights/ml/semantic_cache.py`

---

## P0 — Critical (Fix Immediately)

### 1. Pickle deserialization without any safety controls — `registry.py:152`
`joblib.load(artifact)` deserializes an arbitrary Python object from disk with no validation of the source path, no integrity check (hash/signature), and no type assertion after loading. If an attacker can write to the registry directory (e.g. via a path-traversal bug elsewhere in the system), they can achieve remote code execution. The `model_id` value flows from caller-supplied input at `registry.py:75-76` (`self._root / model_id`) without sanitization, making path traversal possible.

- **File:** `financial-report-insights/ml/registry.py:75-76, 152`
- **Severity:** P0
- **Suggestions:**
  1. Sanitize `model_id` (reject any value containing `/`, `\`, `..`, or non-alphanumeric characters) before constructing paths.
  2. Store a SHA-256 hash of the `.joblib` file in `metadata.json` at registration time and verify it before loading.
  3. Add a type assertion after `joblib.load` (e.g. `assert isinstance(model, (FinancialDistressClassifier, EnsembleForecaster, ...))`) to prevent object substitution.

---

### 2. Path traversal in `ModelRegistry._model_dir` — `registry.py:75-76`
`model_id` is appended directly to `self._root` with no sanitization. A caller passing `model_id = "../../../etc/passwd"` (or similar) will construct a path outside the intended registry directory. The `register`, `load`, `update_status`, `delete`, and `compare` methods all pass caller-controlled `model_id` values into this helper.

- **File:** `financial-report-insights/ml/registry.py:75-76`
- **Severity:** P0
- **Suggestion:** After constructing the candidate path, resolve it and assert it starts with `self._root.resolve()`:
  ```python
  def _model_dir(self, model_id: str) -> Path:
      candidate = (self._root / model_id).resolve()
      if not str(candidate).startswith(str(self._root.resolve())):
          raise ValueError(f"Invalid model_id: {model_id!r}")
      return candidate
  ```

---

## P1 — High (Fix This Sprint)

### 3. `embed_with_cache` silently drops embeddings when a batch call returns fewer vectors than expected — `embedding_optimizer.py:509`
Line 509 filters the `results` list with `[r for r in results if r is not None]`. If the embedder returns fewer embeddings than texts (e.g. the HTTP call fails mid-batch), the `None` slots are silently dropped, returning a list shorter than `texts`. Callers that rely on positional correspondence between input texts and output embeddings will silently receive misaligned results.

- **File:** `financial-report-insights/ml/embedding_optimizer.py:509`
- **Severity:** P1
- **Suggestion:** Replace the final line with an assertion (or raise) that all slots were filled:
  ```python
  if any(r is None for r in results):
      raise RuntimeError("Embedding call returned fewer vectors than expected")
  return results  # type: ignore[return-value]
  ```

---

### 4. `EmbeddingCache` has no size/entry limit — `embedding_optimizer.py:300-406`
The on-disk cache grows without bound. `put()` writes a new `.npz` file for every unique key and `invalidate_all()` is the only bulk cleanup path. There is no LRU eviction, no TTL, and no maximum number of files or total byte cap. In a production deployment processing thousands of documents, this will exhaust disk space.

- **File:** `financial-report-insights/ml/embedding_optimizer.py:348-362`
- **Severity:** P1
- **Suggestion:** Add `max_size_mb` and `max_entries` constructor parameters. In `put()`, after writing the new file check total size/count and evict oldest files (sorted by `stat().st_mtime`) until under the limit.

---

### 5. `FinancialClusterer.fit` does not validate input shape — `clustering.py:62-99`
`np.array(feature_matrix, dtype=np.float64)` will raise a confusing `ValueError` from NumPy if rows have unequal lengths (ragged input). There is no explicit check that `n_samples >= 1` before calling `StandardScaler.fit_transform`, and the code at line 63 does `n_samples, n_features = matrix.shape` which will raise `ValueError: not enough values to unpack` on a 1-D input.

- **File:** `financial-report-insights/ml/clustering.py:62-63`
- **Severity:** P1
- **Suggestion:** Add early validation:
  ```python
  if len(feature_matrix) == 0:
      raise ValueError("feature_matrix must contain at least one sample")
  if matrix.ndim != 2:
      raise ValueError("feature_matrix must be a 2-D array")
  ```

---

### 6. `SimpleARModel.predict` accumulates floating-point drift unboundedly — `forecasting.py:104-115`
Multi-step forecasting appends predicted values back into `history` and uses them as lagged inputs for subsequent steps. For explosive AR processes (sum of coefficients > 1), values will overflow to `inf` within tens of steps; for damped processes values may underflow toward 0. There is no guard against `inf`/`NaN` propagation, no check on coefficient stability, and no cap on `steps`.

- **File:** `financial-report-insights/ml/forecasting.py:104-115`
- **Severity:** P1
- **Suggestion:**
  1. After fitting, log a warning if `|sum(coefficients)| >= 1.0` (unit root / explosive process).
  2. In `predict`, check `math.isfinite(y_hat)` and raise `RuntimeError` (or clip) if it is not.
  3. Add a reasonable maximum for `steps` (e.g. `if steps > 1000: raise ValueError`).

---

### 7. `walk_forward_validate` accesses private attribute `model._seasonal_period` — `forecasting.py:482`
`clone.fit(train, seasonal_period=model._seasonal_period)` reads the private attribute directly, tightly coupling the validation helper to the internal implementation. If `ExponentialSmoother` is ever refactored this will break silently (attribute access on the wrong period would produce meaningless validation metrics rather than an error).

- **File:** `financial-report-insights/ml/forecasting.py:482`
- **Severity:** P1
- **Suggestion:** Expose `seasonal_period` as a public attribute (`self.seasonal_period`) set in `__init__` or add a read-only property, then reference it via the public name.

---

### 8. `SemanticCache.get` key lookup is O(n) on every query — `semantic_cache.py:118`
`list(self._store.keys())[best_idx]` converts the `OrderedDict` keys to a list on every cache hit. Under the lock this is O(n) in the number of cached entries. For `max_entries=1000` this is negligible, but the comment at line 46 says "capped at *max_entries*" without upper-bounding users from setting `max_entries=100000`.

- **File:** `financial-report-insights/ml/semantic_cache.py:118`
- **Severity:** P1 (performance under high `max_entries`)
- **Suggestion:** Maintain a parallel list `_keys` that mirrors `_store` insertion order, or use `next(itertools.islice(self._store.keys(), best_idx, None))` to avoid materializing the full list.

---

## P2 — Medium (Fix Soon)

### 9. `_normalise_text` in `semantic_cache.py` references undefined name `_whitespace_RE` — `semantic_cache.py:181`
Line 181 calls `_whitespace_RE.sub(...)`, but `_whitespace_RE` is not assigned until line 185. This is not caught at import time because the function is only called at runtime. If Python ever changes its module initialization order or if the function is called before the assignment (e.g. during class-body evaluation), this would raise `NameError`.

The current workaround comment ("Work around the forward-slash in the name pattern", line 184) makes no sense — there is no forward slash. The real intent appears to be a late alias to avoid collision with the `re` module name, but the logic is fragile. The simpler and correct fix is to rename the compiled regex to `_WHITESPACE_RE` throughout and call it directly in `_normalise_text`.

- **File:** `financial-report-insights/ml/semantic_cache.py:181, 185`
- **Severity:** P2 (currently works due to module-level execution order, but fragile)
- **Suggestion:** Replace the alias trick with a single, consistently named module-level constant:
  ```python
  _WHITESPACE_RE = re.compile(r"\s+")
  def _normalise_text(text: str) -> str:
      return _WHITESPACE_RE.sub(" ", text.lower().strip())
  ```

---

### 10. `FinancialDistressClassifier._scale` does not validate feature count at inference time — `classifiers.py:221-225`
`_scale` converts the input to a numpy array and calls `self._fitted_scaler.transform(X)` without checking that the column count of `X` matches the number of features the scaler was fitted on. A caller passing a wrong number of features will get a cryptic sklearn `ValueError` rather than an actionable error message. The same issue applies to `predict` and `predict_proba`.

- **File:** `financial-report-insights/ml/classifiers.py:221-225`
- **Severity:** P2
- **Suggestion:** Add a shape check in `_scale`:
  ```python
  expected = self._fitted_scaler.n_features_in_
  if X.shape[1] != expected:
      raise ValueError(f"Expected {expected} features, got {X.shape[1]}")
  ```

---

### 11. `FinancialFeatureExtractor.build_feature_vector` interleaves `_has_` flag columns between feature columns — `feature_engineering.py:375-388`
The `names` list produced by `build_feature_vector` alternates `[feature_0, _has_feature_0, feature_1, _has_feature_1, ...]`. This non-standard layout means any downstream code that wants to split out the mask columns must know the internal structure. More critically, `fit_scaler` applies scaling to the `_has_` binary flags (which are always 0 or 1), distorting the scaler statistics and wasting scale capacity on constant or near-constant values.

- **File:** `financial-report-insights/ml/feature_engineering.py:375-388`
- **Severity:** P2
- **Suggestion:** Separate feature values and missingness flags into two distinct lists, return them together, and skip the flags when fitting the scaler.

---

### 12. `optimal_k` can return `elbow_idx + 2` which exceeds `upper` — `clustering.py:351`
When `silhouettes[best_sil_idx] <= 0.3`, the function returns `elbow_idx + 2`. `elbow_idx` is the index into `second_diffs` which has length `upper - 2` (for `k` values 2..upper, differences have length `upper-1`, second differences `upper-2`). The index correctly maps to a k value via `elbow_idx + 2`, but if `upper == max_k` and `elbow_idx` is the last valid index, the return value could be `max_k` which is fine; however if `n_samples - 1 < max_k` and the elbow falls at the boundary, the returned k could equal `n_samples`, causing KMeans to fail (k cannot equal n_samples). No downstream guard exists.

- **File:** `financial-report-insights/ml/clustering.py:351`
- **Severity:** P2
- **Suggestion:** Clamp the return value: `return min(elbow_idx + 2, upper)`.

---

### 13. `EnsembleForecaster.fit` runs `walk_forward_validate` three times per method — `forecasting.py:550-603`
For each method in `self.methods`, walk-forward validation is run once during weight computation (lines 552-558, 563-568) and a second time during metric aggregation (lines 592-604). For the AR method, a third clone is created at line 594. This means every call to `EnsembleForecaster.fit` performs 2× the necessary validation work, which is O(test_size * len(values)) per method. For long series this is wasteful.

- **File:** `financial-report-insights/ml/forecasting.py:550-604`
- **Severity:** P2 (performance)
- **Suggestion:** Store the metrics computed during weight assignment in a local dict and reuse them for `all_metrics` aggregation, eliminating the second validation loop.

---

### 14. `_StandardScaler.fit` computes population variance (divides by N) — `feature_engineering.py:77`
`var = sum((x - mu) ** 2 for x in col) / len(col)` uses N as the denominator (population variance). sklearn's `StandardScaler` uses N-1 (sample variance) by default. This means `FinancialFeatureExtractor`'s scaler will produce a slightly different scale than sklearn models trained in other parts of the codebase, causing distribution mismatch if vectors scaled here are later fed to sklearn estimators calibrated with a different scaler.

- **File:** `financial-report-insights/ml/feature_engineering.py:77`
- **Severity:** P2
- **Suggestion:** Use `len(col) - 1` as the denominator, or add a `ddof` parameter. Guard against division by zero when `len(col) == 1`.

---

### 15. `registry.py:compare` loads model artifacts unnecessarily — `registry.py:279-280`
`compare` calls `self.load(model_id_a)` and `self.load(model_id_b)`, which deserializes both model objects from disk even though only the metadata (not the model objects) is needed for metric comparison. This is a needless performance cost and, more importantly, an unnecessary invocation of the unsafe `joblib.load` path.

- **File:** `financial-report-insights/ml/registry.py:279-280`
- **Severity:** P2
- **Suggestion:** Call `self._read_metadata(model_id_a)` and `self._read_metadata(model_id_b)` directly (reading only the JSON), and eliminate the model deserialization.

---

### 16. `EmbeddingCache.get` silently swallows all exceptions — `embedding_optimizer.py:344-346`
The broad `except Exception: pass` (with `self._misses += 1; return None`) means disk errors, permission errors, corrupted `.npz` files, and unexpected numpy errors are all treated identically as cache misses. Silent failure prevents operators from detecting disk problems or cache corruption.

- **File:** `financial-report-insights/ml/embedding_optimizer.py:344-346`
- **Severity:** P2
- **Suggestion:** Log the exception at `WARNING` level before returning `None`:
  ```python
  except Exception as exc:
      logger.warning("EmbeddingCache.get failed for key %r: %s", key, exc)
      self._misses += 1
      return None
  ```

---

### 17. `measure_quality_loss` uses queries for `q_proc` but computes against the processed corpus, not reprocessed queries — `embedding_optimizer.py:251`
When `queries` are supplied, `q_proc` is set to `np.asarray(queries, dtype=np.float32)` — the original query embeddings — but `proc_arr` is the already-quantized/truncated corpus. This means the cross-comparison uses unprocessed query vectors against processed corpus vectors, which is the intended real-world usage pattern. However, the function docstring does not make this asymmetry explicit and callers expecting symmetric quality measurement would pass pre-processed queries and get misleading metrics.

- **File:** `financial-report-insights/ml/embedding_optimizer.py:250-254`
- **Severity:** P2 (documentation / correctness risk)
- **Suggestion:** Clarify in the docstring that `queries` should be provided in the *original* (pre-processed) embedding space, and add an assertion that `len(queries[0]) == len(original[0])` to catch dimension mismatches early.

---

## P3 — Low / Housekeeping

### 18. `classifiers.py` never uses `self._estimator` or `self._scaler` after `__init__` — `classifiers.py:63-64`
Both `self._estimator` and `self._scaler` are assigned in `__init__` but the `train()` method re-creates fresh instances via `_create_model()` and `StandardScaler()`. The instance variables are dead code and will confuse future maintainers.

- **File:** `financial-report-insights/ml/classifiers.py:63-64`
- **Severity:** P3

---

### 19. `forecasting.py` uses `assert` in production paths — `forecasting.py:211, 256, 325, 382`
`assert arr is not None` and `assert seasonal is not None` are used as runtime guards in `_fit_simple_optimized`, `_fit_double_optimized`, `_fit_triple_optimized`, and `predict`. Python's `-O` flag disables asserts. These should be replaced with `if arr is None: raise RuntimeError(...)`.

- **File:** `financial-report-insights/ml/forecasting.py:211, 256, 325, 382`
- **Severity:** P3

---

### 20. `_spearman_correlation` overflows for large n — `embedding_optimizer.py:209`
The formula `n * (n ** 2 - 1)` can overflow a 32-bit integer if `n` is large (though Python integers are arbitrary precision, numpy operations may silently overflow if `n` and `d` are numpy scalars rather than Python ints). The `x` and `y` arrays are passed from numpy slices, so `n = len(x)` is a Python int and `d = rx - ry` is float64, making this safe in practice. However, the implementation recomputes rank by double-argsort which is O(n log n) and allocates two extra arrays; `scipy.stats.spearmanr` would be simpler and more robust.

- **File:** `financial-report-insights/ml/embedding_optimizer.py:193-209`
- **Severity:** P3

---

### 21. `FeatureSelector.select_by_correlation` is O(n²) in number of features — `feature_engineering.py:491-498`
The inner double loop over all feature pairs is O(n_features²). For the typical feature counts in this module (< 50) this is fine, but it is worth documenting this limitation so it is not called on high-dimensional embeddings.

- **File:** `financial-report-insights/ml/feature_engineering.py:491-498`
- **Severity:** P3

---

### 22. `registry.py:453` accesses private attribute `model._metrics` — `registry.py:453`
`metrics: dict[str, float] = dict(model._metrics)` reads a private attribute of `EnsembleForecaster`. This is analogous to issue 7 and will break if the attribute is renamed.

- **File:** `financial-report-insights/ml/registry.py:453`
- **Severity:** P3
- **Suggestion:** Add a `get_metrics()` public method to `EnsembleForecaster`.

---

### 23. `AdaptiveTopK` classify-by-keyword logic is defined but never used internally — `semantic_cache.py:339-342`
`_COMPARISON_WORDS`, `_EXPLANATION_WORDS`, `_TREND_WORDS`, and `_RATIO_WORDS` are defined but `compute_k` takes a caller-supplied `query_type` string and never auto-classifies the query from its content. The word sets are dead code.

- **File:** `financial-report-insights/ml/semantic_cache.py:339-342`
- **Severity:** P3
- **Suggestion:** Either implement auto-classification using the word sets, or remove them and note in the docstring that classification is the caller's responsibility.

---

### 24. `__init__.py` is empty — `ml/__init__.py`
The package `__init__.py` exports nothing. Consumers must import from the submodules directly, and there is no `__all__` anywhere in the package. This is acceptable but worth tracking — if the package grows, a curated `__init__.py` exports list prevents accidental use of private helpers.

- **File:** `financial-report-insights/ml/__init__.py:1`
- **Severity:** P3

---

## Files With No Issues Found

No file was entirely issue-free; however, the following files were notably well-implemented:

- **`clustering.py`** — Edge cases for PCA n_components, DBSCAN fallback predict, and IsolationForest contamination are handled thoughtfully. The only material issue is the `optimal_k` boundary condition (issue 12).
- **`feature_engineering.py`** — `safe_divide` is used consistently, NaN/None propagation is handled, and the missingness flag pattern is a reasonable approach despite the ordering issue (issue 11).

---

## Summary

| Priority | Count | Issues |
|----------|-------|--------|
| P0 — Critical | 2 | Pickle deserialization without integrity check; path traversal in model_id |
| P1 — High | 6 | Silent embedding count mismatch; unbounded disk cache; clustering input validation; AR forecast overflow; private attribute access; O(n) key lookup under lock |
| P2 — Medium | 9 | Fragile regex alias; missing feature count validation at inference; scaler interleaves mask columns; elbow index boundary; redundant walk-forward calls; population vs sample variance; unnecessary model load in compare; silent exception swallowing; asymmetric quality loss measurement |
| P3 — Low | 6 | Dead instance variables; assert in production code; Spearman implementation; O(n²) feature selection; private attribute access in registry; dead keyword sets |
| **Total** | **23** | |

**Most urgent work:** Issues 1 and 2 (P0) represent genuine security vulnerabilities — arbitrary code execution via pickle and path traversal via model_id — and must be fixed before this registry is accessible from any API endpoint or multi-tenant environment. Issue 3 (P1) is a correctness bug that silently corrupts embedding alignment.
