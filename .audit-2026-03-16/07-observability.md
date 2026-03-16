# Audit Report: Observability & Monitoring
**Date:** 2026-03-16
**Scope:** Metrics, tracing, system monitor, dashboard data, healthcheck, logging config
**Files Reviewed:**
- `financial-report-insights/observability/__init__.py` (empty file, 0 bytes)
- `financial-report-insights/observability/dashboard_data.py` (172 lines)
- `financial-report-insights/observability/metrics.py` (400 lines)
- `financial-report-insights/observability/system_monitor.py` (515 lines)
- `financial-report-insights/observability/tracing.py` (99 lines)
- `financial-report-insights/healthcheck.py` (167 lines)
- `financial-report-insights/logging_config.py` (80 lines)

---

## P0 — Critical (Fix Immediately)

### P0-01: `_errors_in_window` called without the lock held — data race in `system_monitor.py`

- **File:** `financial-report-insights/observability/system_monitor.py` lines 108–111 and 183–184
- **Severity:** Critical (data race / UB under concurrent load)
- **Detail:** `_errors_in_window()` and `_error_rate_per_min()` iterate directly over `self._errors` (an unbounded `deque`) without holding `self._lock`. The caller at line 183–184 acquires the lock before calling `_error_rate_per_min`, but `_component_statuses` at line 131 also calls `_errors_in_window` inside the lock — however the lock is a non-reentrant `threading.Lock`, so the nested call on line 132 will deadlock when `_component_statuses` is called from `get_health_status` (line 187), which acquires the lock on line 183 and then calls `_component_statuses` on line 187 *after* releasing it for the error-rate call.

  More concretely: `get_health_status` acquires the lock at line 183, computes `error_rate`, then releases it (the `with` block ends), then calls `_component_statuses` at line 187. Inside `_component_statuses` the lock is acquired again at line 131. That is safe for non-reentrant locks, but `_errors_in_window` at line 132 then iterates `self._errors` while a concurrent writer could be appending via `record_error` → the deque is being mutated without the lock in the window helper itself. The root issue: `_errors_in_window` and `_error_rate_per_min` must *either* hold the lock themselves or receive a snapshot list — currently they do neither.

- **Suggestion:** Change `_errors_in_window` to take an already-copied list parameter (or acquire the lock and return a snapshot) so the iteration is always over a stable copy. The same pattern is already used correctly in `_avg_latency_ms` (line 122–124).

---

## P1 — High (Fix This Sprint)

### P1-01: Healthcheck `detail` fields expose raw exception text to API consumers

- **File:** `financial-report-insights/healthcheck.py` lines 25, 50, 61
- **Severity:** High (information disclosure)
- **Detail:** Three `check_*` functions embed the raw exception string directly in the `detail` field returned to callers:
  - Line 25: `f"Cannot connect to Ollama: {e}"` — an `ollama` connection error typically includes the host URL with any embedded credentials or internal hostnames.
  - Line 50: `f"Cannot check models: {e}"` — same risk.
  - Line 61: `f"Cannot create documents folder: {e}"` — an `OSError` includes the full filesystem path.

  The `get_health_status()` return value is forwarded verbatim to the FastAPI `/health` endpoint (`api.py` lines 207–210), which raises an `HTTPException` containing the full `checks` list (including all `detail` strings) in a 503 response body. This means any unauthenticated client that triggers a 503 receives raw exception messages including internal host names and filesystem paths.

  Note: `check_neo4j_connection()` (lines 70–83) correctly suppresses exception detail — it returns a generic string. The same pattern should be applied everywhere.

- **Suggestion:** Log the raw exception at `WARNING` level but return only a generic string in the `detail` field for all error cases. Apply the existing `_redact()` utility from `logging_config.py` at minimum, or use a fixed message like `"Ollama connection failed — see server logs"`.

### P1-02: Trace `summary()` is logged at INFO level for every request, logging full user query data

- **File:** `financial-report-insights/observability/tracing.py` line 97
- **Severity:** High (data volume + potential PII in logs)
- **Detail:** `start_trace` logs `trace.summary()` at `INFO` level on every request completion. The `summary()` dict includes `trace_id`, span names, and token counts. More critically, callers pass user-supplied content as `metadata` kwargs to `start_trace` (see `test_tracing.py` line 149: `start_trace(user="test", query="hello")`). The `summary()` method at lines 68–77 does not include `metadata`, so query text is not directly logged here — but any future metadata passed through could be. Additionally, at `INFO` level in production this generates one log line per RAG request regardless of outcome, which may become high-volume noise. The text-format logger (default) does not redact this output through `_redact()` because that is only applied in `JSONFormatter`.

- **Suggestion:** Lower the trace completion log to `DEBUG` level, or only log at `INFO` when total duration exceeds a threshold. Ensure `_redact()` is applied to the text formatter as well as JSON, or document that trace metadata must never include sensitive values.

### P1-03: Text-format log handler does not apply `_redact()` — secrets can leak in development

- **File:** `financial-report-insights/logging_config.py` lines 70–73
- **Severity:** High (secrets leakage in non-production environments)
- **Detail:** `_redact()` is only invoked inside `JSONFormatter.format()` (line 33 and 41). When `LOG_FORMAT` is `"text"` (the default), the standard `logging.Formatter` is used with no redaction. Any log message containing `password=`, `token=`, `api_key=` etc. will appear in plaintext in development console output. Since the text format is the default, this is the most common runtime configuration.

- **Suggestion:** Create a `TextFormatter(logging.Formatter)` that overrides `format()` and applies `_redact()` to the formatted message string, then use it in the `else` branch of `setup_logging`.

### P1-04: `_errors` and `_latencies` deques are unbounded — memory grows without limit under sustained load

- **File:** `financial-report-insights/observability/system_monitor.py` lines 62, 65
- **Severity:** High (memory leak / OOM under sustained error or high-latency conditions)
- **Detail:** Both deques are created with `deque()` — no `maxlen`. `MetricsCollector` (metrics.py lines 41–44) correctly uses `deque(maxlen=window_size)` for all its rolling windows. `SystemMonitor` has no equivalent cap. Under sustained high error rates or in long-running deployments, `self._errors` and `self._latencies` will grow indefinitely. The 5-minute window filter in `_errors_in_window` only limits *computation*, not *storage*.

- **Suggestion:** Apply `maxlen` to both deques. A safe upper bound aligned with the error window: `maxlen = _ERROR_WINDOW_SECONDS * max_expected_errors_per_second` (e.g., `maxlen=30_000` for 100 errors/s over 300 s). Alternatively mirror `MetricsCollector`'s configurable window approach.

---

## P2 — Medium (Fix Soon)

### P2-01: `_percentile` helper is duplicated verbatim across two modules

- **File:** `financial-report-insights/observability/dashboard_data.py` lines 37–54 and `financial-report-insights/observability/system_monitor.py` lines 497–514
- **Severity:** Medium (maintainability / DRY violation)
- **Detail:** The function body is byte-for-byte identical. The `system_monitor.py` copy even has a comment "mirrors dashboard_data._percentile" (line 493), acknowledging the duplication. If the algorithm needs to change (e.g., to switch to the `numpy` percentile method), it must be updated in two places.

- **Suggestion:** Extract `_percentile` to `observability/utils.py` or directly into `observability/__init__.py` (currently empty) and import from both callers.

### P2-02: `_similarity_histogram` does not guard against `bin_size = 0` when `bins=0`

- **File:** `financial-report-insights/observability/dashboard_data.py` line 67
- **Severity:** Medium (ZeroDivisionError if called with `bins=0`)
- **Detail:** `bin_size = 1.0 / bins` will raise `ZeroDivisionError` if `bins` is 0. The function has no guard. While the default is 10 and callers currently use the default, the parameter is exposed.

- **Suggestion:** Add `if bins <= 0: return []` at the top of the function.

### P2-03: `PerformanceBaseline._data` lists grow without bound

- **File:** `financial-report-insights/observability/system_monitor.py` lines 330–340
- **Severity:** Medium (memory growth over long-lived processes)
- **Detail:** `record_baseline` appends to a plain `list` with no size cap. The docstring mentions "rolling baselines" but the implementation is unbounded. With high-frequency recording (e.g., one entry per query at 10 queries/s over 24 hours = 864,000 entries per metric), memory consumption becomes significant. The statistical computations (mean, std, percentile) also become O(n) in both time and space.

- **Suggestion:** Apply a configurable `maxlen` (default 10,000) to each metric list, or switch to a `deque(maxlen=...)` and sort on demand in `get_baseline`.

### P2-04: `check_ollama_connection` and `check_model_available` accept a `host` that may contain embedded credentials — logged verbatim

- **File:** `financial-report-insights/healthcheck.py` lines 14–50
- **Severity:** Medium (potential credential leakage in logs)
- **Detail:** If `OLLAMA_HOST` contains a URL with embedded credentials (e.g., `http://user:pass@host:11434`), the `host` value is included in the error string at line 25 (`f"Cannot connect to Ollama: {e}"`) and in model names listed at line 46–47. The `run_preflight_checks` function logs these `detail` strings at WARNING/ERROR level (line 140). In the text formatter, `_redact()` is not applied (see P1-03), so the URL with credentials would appear in logs.

- **Suggestion:** Apply `_redact()` to all `detail` strings before logging them in `run_preflight_checks`, and sanitize the host URL before including it in error messages.

### P2-05: `Trace.span()` appends to `self.spans` from a context manager without thread-safety

- **File:** `financial-report-insights/observability/tracing.py` lines 45–53
- **Severity:** Medium (data race if the same `Trace` is used from multiple threads)
- **Detail:** `self.spans.append(s)` at line 49 is unprotected. In the current design, each `Trace` is stored in `threading.local` so a given `Trace` instance is only accessed from one thread in normal use. However, the `Trace` dataclass is a public type — callers can hold a reference and share it. The `add_token_counts` method (lines 55–66) also mutates `self.metadata` without a lock. No documentation warns against sharing `Trace` instances across threads.

- **Suggestion:** Add a docstring note that `Trace` instances are not thread-safe and must not be shared across threads, or add a `threading.Lock` to guard mutations.

### P2-06: `load_baselines` in `PerformanceBaseline` is vulnerable to large-file DoS

- **File:** `financial-report-insights/observability/system_monitor.py` lines 458–489
- **Severity:** Medium (DoS via untrusted baseline file)
- **Detail:** `json.load(fh)` reads the entire file into memory without any size check. A caller passing an adversarially crafted or accidentally oversized JSON file (e.g., a metric list with millions of entries) could exhaust process memory. The file path is accepted as a plain string parameter with no validation.

- **Suggestion:** Check `os.path.getsize(path)` before opening and raise `ValueError` if it exceeds a reasonable maximum (e.g., 50 MB). Validate that each metric value list does not exceed a configured maximum length.

### P2-07: `get_health_status` in healthcheck runs all preflight checks on every `/health` poll — O(n) synchronous I/O on each call

- **File:** `financial-report-insights/healthcheck.py` lines 146–156 and `financial-report-insights/api.py` line 207
- **Severity:** Medium (performance / liveness probe reliability)
- **Detail:** `get_health_status()` calls `run_preflight_checks()` which performs live network I/O (Ollama connection, Neo4j connection) on every invocation. Kubernetes liveness probes typically poll `/health` every 5–10 seconds. A slow Ollama or Neo4j response will delay the HTTP response and may trigger false-positive liveness failures. There is no caching or TTL on health check results.

- **Suggestion:** Cache the result of `run_preflight_checks()` with a short TTL (e.g., 10 seconds) and return the cached result for subsequent calls within the window. Only perform fresh I/O when the cache is stale.

---

## P3 — Low / Housekeeping

### P3-01: `observability/__init__.py` is empty — package surface is undeclared

- **File:** `financial-report-insights/observability/__init__.py` (0 bytes)
- **Detail:** The package exposes no `__all__`, no convenience re-exports, and no version string. Consumers must know the internal module structure to import anything. Consider re-exporting the key public symbols (`get_metrics_collector`, `MetricsCollector`, `SystemMonitor`, `start_trace`, `get_current_trace`, `get_dashboard_data`) so the package has a stable public surface.

### P3-02: `_bucket_by_minute` and cache hit-rate loop in `dashboard_data.py` import `datetime` inside functions

- **File:** `financial-report-insights/observability/dashboard_data.py` lines 26 and 146
- **Detail:** `import datetime` appears twice inside function bodies. While Python caches module imports, the repeated `import` statement adds minor overhead on every call. Move to a top-level import.

### P3-03: `setup_logging` suppresses `urllib3` and `httpx` but not `httpcore` or `ollama`

- **File:** `financial-report-insights/logging_config.py` lines 78–79
- **Detail:** The `ollama` Python client uses `httpx` internally, so suppressing `httpx` at WARNING level is correct. However `httpcore` (a dependency of `httpx`) can also emit DEBUG-level noise. Additionally, the Neo4j driver (`neo4j`) and `asyncio` can be verbose at DEBUG. Consider adding `neo4j` and `httpcore` to the suppression list.

### P3-04: `check_documents_folder` returns absolute path in `detail` on success

- **File:** `financial-report-insights/healthcheck.py` line 67
- **Detail:** `f"Documents folder ready ({file_count} files)"` does not expose the path, but lines 59 and 64 do include the resolved `Path` object in success/error messages. In a containerized environment, internal filesystem paths can reveal deployment structure. This is low risk given the path is operator-controlled, but consider omitting the absolute path from success responses.

### P3-05: `JSONFormatter` includes `module`, `function`, and `line` fields in every log record

- **File:** `financial-report-insights/logging_config.py` lines 35–37
- **Detail:** Exposing source file names and line numbers in production JSON logs is generally acceptable for debugging but can aid an attacker in understanding internal code structure if logs are ever externally accessible. Mark as acknowledged low risk; no action required unless logs are forwarded to an external service without access controls.

### P3-06: `Span.finish()` has no return type annotation; `Trace.span()` context manager yields untyped `Span`

- **File:** `financial-report-insights/observability/tracing.py` lines 28–29 and 45–53
- **Detail:** Minor typing hygiene: `finish(self)` should be annotated `-> None`. The `@contextmanager` generator should be typed as `Generator[Span, None, None]` and the `span` method signature should use `-> contextlib.AbstractContextManager[Span]` for IDE tooling.

### P3-07: `_SECRETS_RE` regex does not match Bearer token patterns or Authorization headers

- **File:** `financial-report-insights/logging_config.py` lines 14–17
- **Detail:** The regex `(password|secret|token|api[_-]?key|credential)[=:]\s*\S+` catches key=value patterns but misses common log patterns such as `Authorization: Bearer <token>` or `Bearer eyJ...`. If any HTTP headers are ever logged (e.g., from httpx in DEBUG mode), bearer tokens would not be redacted.

- **Suggestion:** Extend `_SECRETS_RE` with an alternative branch: `|Bearer\s+[A-Za-z0-9._\-]+`.

---

## Files With No Issues Found

None — all seven files reviewed contain at least one finding. The lowest-risk file is `observability/metrics.py`, which is well-structured, correctly thread-safe (lock-then-copy pattern), and uses `deque(maxlen=...)` throughout. Its only notable weaknesses are inherited from shared design decisions covered under P2 and P3 above.

---

## Summary

| Priority | Count | Key Themes |
|----------|-------|------------|
| P0 | 1 | Data race: `_errors_in_window` iterates unprotected deque |
| P1 | 3 | Info disclosure in healthcheck details; trace logging at INFO; text-format redaction gap |
| P2 | 7 | Unbounded deques in `SystemMonitor`; duplicated `_percentile`; health check polling cost; thread-safety docs on `Trace` |
| P3 | 7 | Typing gaps; import housekeeping; Bearer token regex gap; empty `__init__.py` |

**Most urgent fixes in priority order:**

1. **P0-01** — Lock `_errors_in_window` or pass a snapshot list to prevent data race.
2. **P1-03** — Apply `_redact()` in the text formatter (currently the default format).
3. **P1-01** — Strip raw exception text from healthcheck `detail` fields exposed via `/health`.
4. **P1-04** — Set `maxlen` on `SystemMonitor._errors` and `SystemMonitor._latencies`.
5. **P2-07** — Cache health check results with a short TTL to protect liveness probes.
