# Audit Report: Test Quality & Coverage
**Date:** 2026-03-16
**Scope:** Test infrastructure, coverage analysis, test quality assessment
**Files Reviewed:**
- financial-report-insights/tests/conftest.py
- financial-report-insights/tests/test_api.py
- financial-report-insights/tests/test_security.py
- financial-report-insights/tests/test_config.py
- financial-report-insights/tests/test_config_validation.py
- financial-report-insights/tests/test_healthcheck.py
- financial-report-insights/tests/test_ingestion_pipeline.py
- financial-report-insights/tests/test_vector_index.py
- financial-report-insights/tests/test_graph_store.py
- financial-report-insights/tests/test_self_healing.py
- financial-report-insights/tests/test_edge_cases.py
- financial-report-insights/tests/test_coverage_gaps.py
- financial-report-insights/tests/test_integration.py
- financial-report-insights/tests/test_resilience.py
- financial-report-insights/tests/test_llm_integration.py
- financial-report-insights/tests/test_app_local.py
- financial-report-insights/tests/test_circuit_breaker.py
- Directory listing of all 170+ test files

---

## P0 - Critical (Fix Immediately)

### P0-01: Incorrect Retry-Count Assertion
**File:** test_resilience.py line 103
Comment says "1 initial + 2 retries = 3 total calls" but assertion checks call_count == 2.

### P0-02: Empty Ingestion Retry Section - Stub Tests Never Implemented
**File:** test_self_healing.py lines 236-239
Empty section header with zero test functions. Production _load_documents() backoff path has no direct unit tests.

---

## P1 - High (Fix This Sprint)

### P1-01: Real time.sleep() Calls Create Flaky, Slow Tests
**Files:** test_resilience.py, test_circuit_breaker.py - 10 functions with real sleep totaling 12+ seconds

### P1-02: test_security.py Tests a Local Copy of Sanitisation Logic
**File:** test_security.py lines 20-82 - Tests inline copy, not production streamlit_app_local._sanitize_and_save

### P1-03: Session-Scoped Mutable Fixture in conftest.py
**File:** conftest.py lines 13-32 - sample_financial_data_dict uses scope=session, any mutation corrupts all tests

### P1-04: Overly Broad Exception Assertion
**File:** test_integration.py lines 440-442 - pytest.raises(Exception) with no type specification

### P1-05: insights_page.py Has No Dedicated Test File
**Source:** insights_page.py (7,563 LOC) - Second largest source file with zero direct test coverage

---

## P2 - Medium (Fix Soon)

### P2-01: TestChunkCountValidation Tests Inline Logic
**File:** test_self_healing.py lines 191-233

### P2-02: Mock Target Asymmetry Undocumented
**File:** test_healthcheck.py lines 14-28

### P2-03: TestRateLimiting Mutates Global State
**File:** test_api.py lines 296-326

### P2-04: TestEnvFileWarning Does Not Exercise Warning Path
**File:** test_config_validation.py lines 172-179

### P2-05: Security Tests Test Arithmetic Not Enforcement
**File:** test_security.py lines 89-169

### P2-06: Analyze Endpoint Tests Use MagicMock for Serialisation
**File:** test_api.py lines 135-165

### P2-07: test_integration.py Defines Local is_financial() Inline
**File:** test_integration.py lines 384-399

### P2-08: No End-to-End Ingestion-to-Retrieval Integration Test

---

## P3 - Low / Housekeeping

### P3-01: _make_embedder() Helper Duplicated Four Times
### P3-02: TestIngestionRetry Simulates Retry Logic Inline
### P3-03: conftest.py Defers import pandas Inside Fixtures
### P3-04: FAISS/HNSW Tests Lack Score-Value Correctness Assertion
### P3-05: Permissive Statement-Type Assertion
### P3-06: TestContentHashCacheKey Uses time.sleep for Mtime Separation

---

## Files With No Issues Found
- test_config.py, test_ingestion_pipeline.py, test_vector_index.py, test_edge_cases.py
- test_healthcheck.py, test_self_healing.py (embedding retry), test_resilience.py (circuit breaker)
- test_api.py (security/rate limiting sections)

## Summary
P0: 2 | P1: 5 | P2: 8 | P3: 6
Top actions: Fix retry assertion off-by-one, replace time.sleep with mocked time, create test_insights_page.py
