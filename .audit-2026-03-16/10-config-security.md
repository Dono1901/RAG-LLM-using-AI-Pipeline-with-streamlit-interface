# Audit Report: Config, Secrets & Security
**Date:** 2026-03-16
**Scope:** Configuration management, secrets handling, authentication, CORS, rate limiting
**Files Reviewed:**
- financial-report-insights/config.py
- financial-report-insights/protocols.py
- financial-report-insights/.env
- financial-report-insights/.env.example
- financial-report-insights/logging_config.py
- financial-report-insights/healthcheck.py
- financial-report-insights/api.py
- .gitignore
- .githooks/pre-commit

---

## P0 — Critical (Fix Immediately)

### P0-01: Real Neo4j Password Committed in .env
**File:** `financial-report-insights/.env`, line 39
**Finding:** The file contains a live Neo4j credential:
```
NEO4J_PASSWORD=igz*VvYxmfWas4L@giD*CLTQKu_z!eg_
```
The `.gitignore` at the repository root correctly lists `.env` as excluded, which means this file is **not tracked by git right now**. However, several conditions make this P0:
1. If this file was ever staged, committed, or force-added even momentarily, the credential is permanently in git history.
2. Any process that archives the working directory (backup, zip, rsync, IDE sync) will capture it.
3. The credential is identical to what the project memory records as being stored in 1Password — meaning the same password is used in both the secrets manager and the local dev file.
4. The `.env.example` file (line 45) appropriately leaves `NEO4J_PASSWORD` blank with a comment — confirming that placing a real password in `.env` is unintended policy.

**Immediate Action:**
- Rotate the Neo4j password immediately since this file may have been exposed.
- Confirm via `git log --all -- financial-report-insights/.env` and `git log --all -p | grep -i NEO4J_PASSWORD` that the credential was never committed.
- Replace the value in `.env` with a placeholder and fetch from 1Password CLI at process start.

---

## P1 — High (Fix This Sprint)

### P1-01: No Authentication on Any API Endpoint
**File:** `financial-report-insights/api.py`, lines 204-866 (all route handlers)
**Finding:** The FastAPI application has zero authentication on every endpoint including `/analyze`, `/export/xlsx`, `/export/pdf`, `/compliance/*`, `/portfolio/*`, and `/compare`. The `allow_headers` list at line 136 includes `"Authorization"`, but no middleware or FastAPI `Depends` ever enforces it. Any client that can reach port 8504 can exfiltrate full financial analysis, export PDF/Excel reports, and trigger resource-intensive LLM inference without any credential.

### P1-02: `_rate_log` Dict is Unbounded — Memory Exhaustion DoS Vector
**File:** `financial-report-insights/api.py`, lines 146, 187-195
**Finding:** The in-memory rate limiter stores one list per client IP in `_rate_log: Dict[str, List[float]]`. Entries are pruned per-IP only on the next request from that same IP. An attacker sending one request from each of N distinct source IPs will populate N keys that are never cleaned up, growing the dict indefinitely.

### P1-03: `LLMConnectionError` Message Leaked Directly to API Callers
**File:** `financial-report-insights/api.py`, lines 236 and 268
**Finding:** `str(exc)` is passed verbatim to HTTP response bodies. The exception message may contain the OLLAMA_HOST URL, model name, connection refused details, or internal path information. All other error paths correctly use generic sanitized strings; these two paths are inconsistent.

### P1-04: Plain-Text Log Format Does Not Apply Secret Redaction
**File:** `financial-report-insights/logging_config.py`, lines 67-73
**Finding:** The `_redact()` function is applied only inside `JSONFormatter.format()`. When `LOG_FORMAT=text` (the default), logs are emitted using the stock `logging.Formatter` which performs no redaction. Any code that logs connection strings or credentials will emit raw secrets to stdout in text mode.

---

## P2 — Medium (Fix Soon)

### P2-01: OLLAMA_HOST Hostname Allow-List Lives in Two Places and They Differ
**File:** `financial-report-insights/local_llm.py`, lines 413-418; `financial-report-insights/config.py`, lines 141-145
**Finding:** Two independent SSRF-prevention checks exist. `config.validate_settings()` checks scheme only; `LocalEmbedder.__init__()` checks both scheme and hostname. The config-layer check provides false assurance. The LLM client path (Ollama chat API) reads `OLLAMA_HOST` from the environment directly without hostname validation.

### P2-02: healthcheck.py Leaks Ollama Exception Strings to Health Response
**File:** `financial-report-insights/healthcheck.py`, lines 25, 46, 50
**Finding:** Raw exception messages and full model name lists are surfaced to any caller of `/health`. `/health` is explicitly excluded from rate limiting, so this information is freely and unlimitedly available.

### P2-03: Body Size Limit Relies Solely on Content-Length Header
**File:** `financial-report-insights/api.py`, lines 163-177
**Finding:** The `body_size_limit_middleware` only checks the `Content-Length` header. A client using `Transfer-Encoding: chunked` with no `Content-Length` bypasses the 1 MB limit entirely.

### P2-04: No Secret Scanning in Pre-Commit Hook
**File:** `.githooks/pre-commit`
**Finding:** The pre-commit hook does not scan staged files for secrets or credentials. The P0-01 finding would not be caught if `.env` were ever accidentally staged.

### P2-05: NEO4J_URI Not Validated Against SSRF in config.validate_settings()
**File:** `financial-report-insights/config.py`, lines 149-152
**Finding:** `validate_settings()` checks that `NEO4J_URI` has a corresponding `NEO4J_PASSWORD`, but does not validate the URI scheme or hostname.

### P2-06: `Authorization` Header Listed in CORS allow_headers Without Any Auth Implementation
**File:** `financial-report-insights/api.py`, line 136
**Finding:** `allow_headers=["Content-Type", "Authorization"]` allows the header but nothing checks it. Premature and misleading.

---

## P3 — Low / Housekeeping

### P3-01: Missing Content-Security-Policy and HSTS Response Headers
**File:** `financial-report-insights/api.py`, lines 151-159

### P3-02: `export_company_name` Default is Empty String in Settings
**File:** `financial-report-insights/config.py`, line 87

### P3-03: Python Path Hardcoded in Pre-Commit Hook
**File:** `.githooks/pre-commit`, line 52

### P3-04: `check_ollama_connection` Exposes Bare Exception to Detail String
**File:** `financial-report-insights/healthcheck.py`, line 25

### P3-05: `.env` Contains Docker-Internal Hostname Revealing Topology
**File:** `financial-report-insights/.env`, line 37

---

## Files With No Issues Found

- **financial-report-insights/protocols.py** — Clean Protocol interface definitions. No issues.
- **financial-report-insights/.env.example** — Correctly leaves NEO4J_PASSWORD blank. No issues.
- **.gitignore** — `.env` correctly listed under "Secrets & credentials". No issues.

---

## Summary

The codebase shows strong security engineering intent — SSRF hostname allow-lists, request body limits, per-IP rate limiting, secret redaction in logs, sanitized error messages in most paths, and Pydantic field validation are all present. The gap between intent and execution is concentrated in three areas: (1) a real credential present in `.env` that should only contain placeholders, (2) no authentication gate on a service that exposes sensitive financial analysis endpoints, and (3) several information-disclosure paths in health and error responses that bypass the otherwise solid redaction infrastructure.
