# Audit Report: API & Web Layer
**Date:** 2026-03-16
**Scope:** FastAPI API routes, Streamlit UI, visualization utilities
**Files Reviewed:**
- `financial-report-insights/api.py` (867 lines)
- `financial-report-insights/streamlit_app_local.py` (449 lines)
- `financial-report-insights/app_local.py` (read for context on `SimpleRAG`, `LLMConnectionError`)
- `financial-report-insights/insights_page.py` (7,500+ lines, sampled key sections)
- `financial-report-insights/viz_utils.py` (739 lines)
- `financial-report-insights/financial_analyzer.py` (partial, for grade string provenance)
- `financial-report-insights/local_llm.py` (partial, for `LLMConnectionError` message content)
- `financial-report-insights/graph_store.py` (partial, for Neo4j parameterization)

---

## P0 — Critical (Fix Immediately)

### P0-1: `LLMConnectionError` message forwarded verbatim to API clients

**File:** `financial-report-insights/api.py` lines 235-236, 267-268

```python
# Line 235-236
except LLMConnectionError as exc:
    raise HTTPException(status_code=503, detail=str(exc)) from exc

# Lines 267-268
except LLMConnectionError as exc:
    yield {"event": "error", "data": str(exc)}
```

`LLMConnectionError` messages are constructed from internal state including the exact Ollama host address, circuit breaker timing values, and connection error details (e.g., `"Ollama error: {e}"` at `local_llm.py:308` where `e` is the raw `Exception` from the Ollama client). The raw `str(exc)` for `LLMConnectionError` wrapping a generic `Exception` can contain internal hostnames, ports, and stack context from the underlying library.

**Impact:** Information disclosure. An attacker probing the `/query` or `/query-stream` endpoints under load (to trigger the circuit breaker or a connection failure) can learn the internal Ollama host, port, and whether it is a container hostname.

**Suggestion:** Return a fixed-text 503 message to the client. Log the full `str(exc)` server-side only:
```python
except LLMConnectionError as exc:
    logger.warning("LLM error: %s", exc)
    raise HTTPException(status_code=503, detail="LLM service temporarily unavailable.") from exc
```

---

## P1 — High (Fix This Sprint)

### P1-1: Body size middleware is bypassable via chunked transfer encoding

**File:** `financial-report-insights/api.py` lines 162-177

```python
content_length = request.headers.get("content-length")
try:
    if content_length and int(content_length) > settings.max_request_body_bytes:
        return JSONResponse(status_code=413, ...)
```

The guard only triggers when `Content-Length` is present. HTTP clients sending `Transfer-Encoding: chunked` omit `Content-Length`, so the condition `if content_length and ...` evaluates to `False` and the body is passed through unrestricted. A moderately large JSON body (tens of MB) can be streamed in without triggering the 413 response.

**Impact:** Potential resource exhaustion / denial of service through unbounded request body ingestion on `/analyze`, `/export/xlsx`, `/export/pdf`, `/portfolio/analyze`, `/portfolio/correlation`, `/compliance/*`.

**Suggestion:** Read the body and enforce the limit on actual bytes consumed, or use a streaming cap that does not depend on the `Content-Length` header. Alternatively, configure the ASGI server (uvicorn) with `--limit-max-requests` and `--limit-concurrency` and add a Starlette middleware that wraps the receive channel to hard-cap read size.

### P1-2: In-memory rate limiter `_rate_log` grows without bound on unique IPs

**File:** `financial-report-insights/api.py` lines 146, 186-195

```python
_rate_log: Dict[str, List[float]] = defaultdict(list)
...
_rate_log[client_ip] = [t for t in _rate_log[client_ip] if t > cutoff]
if len(_rate_log[client_ip]) >= _RATE_LIMIT:
    ...
_rate_log[client_ip].append(now)
```

The cleanup at line 189 removes expired timestamps for the *current* IP, but the dict itself never shrinks. Each unique client IP that ever makes a request creates a permanent `defaultdict` entry. An attacker spoofing or rotating IPs (typical in IPv6 environments or behind proxies without proper IP extraction) can cause the process heap to grow without bound.

**Impact:** Memory exhaustion / denial of service in long-running deployments.

**Suggestion:** After evicting expired entries, also delete the key if the resulting list is empty:
```python
_rate_log[client_ip] = [t for t in _rate_log[client_ip] if t > cutoff]
if not _rate_log[client_ip]:
    del _rate_log[client_ip]
```
Also consider capping the dict size with `if len(_rate_log) > MAX_TRACKED_IPS: _rate_log.clear()`.

### P1-3: No authentication on any API endpoint

**File:** `financial-report-insights/api.py` (entire file)

The `Authorization` header is listed in `allow_headers` for CORS (line 136) but no authentication dependency (`Depends(oauth2_scheme)`, API key header check, etc.) is applied to any route handler. All endpoints — including `/analyze`, `/export/xlsx`, `/export/pdf`, `/portfolio/analyze`, `/compliance/analyze`, and `/query` — are openly accessible to any caller that can reach the port.

**Impact:** Any network-reachable client can invoke full financial analysis, export, compliance, and portfolio endpoints. This may be acceptable for a local dev tool, but the MEMORY.md notes Docker containers are deployed (`rag-financial-insights` healthy), which implies external reachability.

**Suggestion:** If this is deployment-facing, add a simple API key dependency:
```python
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")
async def verify_api_key(key: str = Depends(api_key_header)):
    if key != settings.api_key:
        raise HTTPException(status_code=403)
```
Document the known accepted risk in code if this is intentionally an unauthenticated local tool.

### P1-4: `ExportRequest` and per-company `financial_data` in `PortfolioRequest` lack field count limits

**File:** `financial-report-insights/api.py` lines 535-537, 610-614

`AnalyzeRequest` at line 48 has a `validate_field_count` validator capping `financial_data` at `settings.max_financial_fields` (200). The `ExportRequest` model accepts `financial_data: Dict[str, Any]` with no such validator, and `PortfolioRequest` limits the number of companies (50) but applies no per-company field count limit — meaning each of up to 50 companies can supply an unbounded dict.

**Impact:** A caller can send a deeply nested or very wide `financial_data` payload to `/export/xlsx`, `/export/pdf`, `/portfolio/analyze`, or `/portfolio/correlation` that bypasses `_parse_financial_data` filtering (which does filter to known fields) but still forces the `FinancialData` constructor and downstream analysis to handle a pathological input.

**Suggestion:** Extract the `validate_field_count` logic into a shared validator and apply it to `ExportRequest.financial_data` and to each value dict inside `PortfolioRequest.companies`:
```python
class ExportRequest(BaseModel):
    financial_data: Dict[str, Any]
    company_name: str = Field(default="", max_length=200)

    @field_validator("financial_data")
    @classmethod
    def validate_field_count(cls, v):
        if len(v) > settings.max_financial_fields:
            raise ValueError(f"Too many fields ({len(v)}); max {settings.max_financial_fields}.")
        return v
```

### P1-5: `category` query parameter on `/graph/ratios/{period_label}` has no length or content validation

**File:** `financial-report-insights/api.py` lines 379, 383-384

```python
async def graph_ratios(period_label: str = Path(..., max_length=100), category: Optional[str] = None):
    ...
    if category:
        ratios = [r for r in ratios if r.get("category", "").lower() == category.lower()]
```

`category` is an unconstrained `Optional[str]` query parameter. There is no max length, no allowlist check, and no sanitization. It is used only for in-memory list filtering (not passed to Neo4j), so there is no direct injection risk, but an arbitrarily long string is accepted and lowercased in the request handler.

**Suggestion:** Add a max length constraint: `category: Optional[str] = Query(default=None, max_length=100)`.

### P1-6: `CompareRequest.period_labels` individual string elements have no length constraint

**File:** `financial-report-insights/api.py` lines 393-394

```python
class CompareRequest(BaseModel):
    period_labels: List[str] = Field(..., min_length=2, max_length=10)
```

The `max_length=10` on the `Field` constrains the *list length* to 10 items, not the *string length* of each element. Each period label string is unconstrained. Period labels are passed to `store.cross_period_ratio_trend(req.period_labels)` as parameters in a Neo4j parameterized query (safe from injection), but an arbitrarily long string is also echoed back in the `CompareResponse.periods_compared` field and in the `summary` string sent to the client.

**Suggestion:** Add per-element length validation:
```python
period_labels: List[constr(max_length=100)] = Field(..., min_length=2, max_length=10)
```

---

## P2 — Medium (Fix Soon)

### P2-1: Missing `Content-Security-Policy` header

**File:** `financial-report-insights/api.py` lines 150-159

The `security_headers_middleware` sets `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, and `Cache-Control`, but does not set a `Content-Security-Policy` header. While this API is primarily JSON, the SSE endpoint (`/query-stream`) streams text data, and the absence of CSP leaves any HTML-rendering browser client without a policy boundary.

**Suggestion:** Add `response.headers["Content-Security-Policy"] = "default-src 'none'"` for the API layer (which never renders HTML).

### P2-2: Missing `Strict-Transport-Security` header

**File:** `financial-report-insights/api.py` lines 150-159

No `Strict-Transport-Security` (HSTS) header is emitted. If TLS is terminated at the ASGI server or a reverse proxy in front of it, browsers that directly contact the API will not be instructed to enforce HTTPS.

**Suggestion:** Add `response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"` conditionally when the deployment is behind HTTPS.

### P2-3: `ExportRequest.company_name` field has no length constraint

**File:** `financial-report-insights/api.py` line 537

```python
company_name: str = ""
```

The field is accepted but not currently forwarded to the exporters (the `export_xlsx` and `export_pdf` handlers never pass `req.company_name` to the exporter). However, the field exists in the model and could be wired in future without a reviewer noticing there is no sanitization. If `company_name` were inserted into an Excel cell title or a PDF cover page, an unsanitized value could cause formula injection in Excel (`=CMD(...)` style) or unexpected rendering in PDF.

**Suggestion:** Add `company_name: str = Field(default="", max_length=200)` now, before the field is wired to the exporters. Also validate that the string contains only printable characters.

### P2-4: `rglob("*")` in Document Manager lists subdirectory contents without a depth cap

**File:** `financial-report-insights/streamlit_app_local.py` lines 116, 354

```python
doc_files = list(docs_path.rglob("*"))
```

`rglob("*")` traverses the full directory tree rooted at `docs_path` to arbitrary depth. If a user (or an upstream process) places a deeply nested directory structure inside `documents/`, this call blocks the Streamlit main thread until the entire tree is enumerated. On Windows, this also follows junctions/symlinks by default, potentially escaping the `documents/` tree if a symlink to a system directory is placed there.

**Impact:** Server-side denial of service (deep tree blocking) and potential path exposure via symlink following.

**Suggestion:** Replace with `docs_path.glob("*")` (non-recursive) unless subdirectory support is genuinely needed. If recursion is needed, add a depth cap and skip symlinks.

### P2-5: Streamlit `st.error(f"Error loading data: {e}")` leaks exception detail to the browser

**File:** `financial-report-insights/insights_page.py` line 287; `financial-report-insights/streamlit_app_local.py` lines 231, 290, 316, 319

```python
# insights_page.py:287
st.error(f"Error loading data: {e}")

# streamlit_app_local.py:231
st.error(f"Failed to load RAG system: {e}")
# streamlit_app_local.py:290
st.error(f"Error: {e}")
```

These `st.error` calls interpolate raw Python exception objects directly into the UI string. For file I/O errors, this can expose full file system paths (e.g., `[Errno 13] Permission denied: '/home/app/documents/...'`). For network errors, it can expose internal hostnames. The Streamlit UI runs server-side and streams HTML to the browser, so these messages are visible to any user of the UI.

**Suggestion:** Log the full exception server-side with `logger.exception(...)` and show a generic message in the UI:
```python
except Exception as e:
    logger.exception("Error loading data")
    st.error("Error loading data. Check server logs for details.")
```
The pattern at `insights_page.py:288` (`logger.exception("Failed to load Excel data")`) is correct; the `st.error` on line 287 should not include `{e}`.

### P2-6: `unsafe_allow_html` usages interpolate `result.*_grade` and `result.*_score` strings from analysis objects

**File:** `financial-report-insights/insights_page.py` — 25 occurrences across lines 2477, 2479, 2598-2600, 2741, 2808-2810, 2896-2898, 3006-3008, 3094-3096, 3186-3188, 3278-3280, 3365-3367, 3440-3442, 3514-3516, 3789-3791, 3855-3857, 3920-3922, 3981-3983, 4208-4210, 4279-4281, 4346-4348, 7170-7172, 7217-7219, 7294-7296, 7374-7376, 7440-7442, 7504-7506

All 25 `unsafe_allow_html` usages interpolate grade and score values derived from `financial_analyzer.py` analysis results (e.g., `result.quality_grade`, `result.manipulation_grade`, `rating.overall_grade`, `result.roic_grade`). These are assigned from hardcoded lookup tables within `CharlieAnalyzer` and its helpers — they are strings like `"High"`, `"Low"`, `"Excellent"`, `"Weak"`, `"Highly Likely"`, etc. They are not derived from raw user-supplied text.

**Current risk:** Low, because the grade strings are controlled entirely by server-side logic.

**Residual risk:** If the analysis result dataclasses were ever extended to include user-supplied values (e.g., a company name or custom metric label) in a field that is then interpolated into one of these HTML strings without review, XSS would result. The pattern is fragile.

**Suggestion:** Replace the grade-display pattern with `st.metric()` or `st.markdown()` (without `unsafe_allow_html`) where possible. Where color-coded HTML is genuinely needed, add a short allowlist check before interpolation:
```python
SAFE_GRADES = frozenset({"A","B","C","D","F","AAA","AA","A","BBB","BB","B","CCC","CC","C",
                          "Excellent","Good","Adequate","Weak","Poor","High","Low","Moderate",
                          "Unlikely","Possible","Likely","Highly Likely","Strong","N/A", ...})
assert result.quality_grade in SAFE_GRADES, f"Unexpected grade value: {result.quality_grade!r}"
```

### P2-7: `_render_upload_section` in `insights_page.py` missing separator check present in `streamlit_app_local.py`

**File:** `financial-report-insights/insights_page.py` lines 374-393 vs `financial-report-insights/streamlit_app_local.py` lines 36-64

`streamlit_app_local.py._sanitize_and_save` explicitly rejects filenames containing `/` or `\` after `os.path.basename` extraction (line 42). The inline upload handler in `insights_page.py._render_upload_section` (lines 377-393) performs only the `os.path.basename` strip and the `resolve().startswith()` containment check but omits the explicit separator reject. While `os.path.basename` on most platforms will strip leading path components, the belt-and-suspenders separator check is absent.

**Suggestion:** Extract `_sanitize_and_save` from `streamlit_app_local.py` into a shared utility module (e.g., `upload_utils.py`) and use it in both upload paths, ensuring identical security behavior.

---

## P3 — Low / Housekeeping

### P3-1: No request ID / correlation ID in API responses

**File:** `financial-report-insights/api.py`

API responses do not include a request correlation ID. When debugging 503 or 422 errors across the middleware chain, it is impossible to correlate a client-visible error with the server log entry that produced it.

**Suggestion:** Generate a UUID per request in `security_headers_middleware` and attach it as `X-Request-ID` to both the response header and the structured log record via `logging.LoggerAdapter`.

### P3-2: `_get_portfolio_analyzer` and `_get_compliance_scorer` use function attribute as singleton store

**File:** `financial-report-insights/api.py` lines 649-669

```python
def _get_portfolio_analyzer():
    if not hasattr(_get_portfolio_analyzer, "_inst"):
        with _portfolio_lock:
            if not hasattr(_get_portfolio_analyzer, "_inst"):
                ...
                _get_portfolio_analyzer._inst = PortfolioAnalyzer()
```

Using a function's `__dict__` as a singleton container is an unusual pattern that can confuse static analysis tools and is not idiomatic. The pattern for `_get_rag()` uses a module-level global with a lock, which is clearer.

**Suggestion:** Use module-level `_portfolio_analyzer_instance = None` with a lock, consistent with `_rag_instance`.

### P3-3: `ExportRequest.company_name` is accepted but silently ignored

**File:** `financial-report-insights/api.py` lines 537, 555-556, 586-587

`req.company_name` is parsed and validated as part of the request model but never forwarded to `FinancialExcelExporter.export_full_report(...)` or `FinancialPDFExporter.export_full_report(...)`. The exporters support a `company_name` parameter on their `export_ratios` method (seen in `export_xlsx.py:158`). This is dead parameter acceptance — callers may set it expecting it to appear in the output.

**Suggestion:** Either wire `company_name` through to the exporter calls, or remove it from the model until the exporters fully support it.

### P3-4: `X-XSS-Protection: 1; mode=block` is a legacy header

**File:** `financial-report-insights/api.py` line 156

`X-XSS-Protection` was deprecated by all major browsers; modern browsers ignore it or treat it as a no-op. Chrome removed support in v78. For a REST API that does not serve HTML, this header has no effect.

**Suggestion:** Remove it and replace with a `Content-Security-Policy: default-src 'none'` header (see P2-1).

### P3-5: `Ollama: {result['detail']}` detail string forwarded to Streamlit sidebar

**File:** `financial-report-insights/streamlit_app_local.py` lines 174-179

```python
result = check_ollama_connection(os.environ.get("OLLAMA_HOST"))
if result["status"] == "ok":
    st.sidebar.success("Ollama: Connected")
else:
    st.sidebar.error(f"Ollama: {result['detail']}")
```

The `detail` field from `check_ollama_connection` may contain internal host information. This is a Streamlit UI (visible to browser users), not a public REST API, so impact is low — but in a multi-user deployment the Ollama backend address should not be surfaced in the UI.

**Suggestion:** Replace with `st.sidebar.error("Ollama: Connection failed. Check server configuration.")` and log the detail server-side.

### P3-6: `ollama_model` selectbox in sidebar has no effect at runtime

**File:** `financial-report-insights/streamlit_app_local.py` lines 103-107

```python
ollama_model = st.selectbox(
    "LLM Model (Ollama)",
    ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"],
    index=0
)
```

The returned value `ollama_model` is never used — the RAG system is loaded via `load_rag_system()` which is decorated `@st.cache_resource` and reads the model from `os.getenv("OLLAMA_MODEL", "llama3.2")`. The sidebar selection has no effect.

**Impact:** No security impact; misleads users into thinking they can switch models via the UI.

**Suggestion:** Either wire the selection to a session state key and invalidate the cache resource on change, or remove the selectbox.

---

## Files With No Issues Found

- `financial-report-insights/viz_utils.py` — Pure data transformation and Plotly figure construction. No file I/O, no user input handling, no HTML generation, no network calls. No issues found.

---

## Summary

| Priority | Count | Most Critical Items |
|----------|-------|---------------------|
| P0 | 1 | LLM error messages leak internal hostnames to API clients |
| P1 | 6 | Body size bypass via chunked encoding; unbounded rate-limiter memory; no auth; missing field count validators on Export/Portfolio; unconstrained query params |
| P2 | 7 | Missing CSP/HSTS headers; exception detail in Streamlit UI; fragile `unsafe_allow_html` pattern; duplicate upload sanitization logic |
| P3 | 6 | Housekeeping: correlation IDs, singleton pattern, dead parameter, deprecated header, detail leakage in sidebar, dead UI control |

**Overall Quality Score: 6.5/10**
**Files Analyzed:** 8
**Issues Found:** 20
**Technical Debt Estimate:** 12 hours (P0: 1h, P1: 5h, P2: 4h, P3: 2h)

The API layer has solid foundational security work: parameterized Neo4j queries, CORS restricted to config-driven origins, per-IP rate limiting, field count validation on the primary endpoint, request body size checks, path traversal guards on file upload, and a full set of standard security response headers. The remaining issues are primarily defence-in-depth gaps rather than fundamental design failures.
