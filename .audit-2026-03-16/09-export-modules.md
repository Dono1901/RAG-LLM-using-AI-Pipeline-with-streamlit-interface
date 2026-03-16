# Audit Report: Export Modules (PDF/XLSX)
**Date:** 2026-03-16
**Scope:** PDF export, XLSX export, shared export utilities
**Files Reviewed:**
- `financial-report-insights/export_pdf.py` (449 lines)
- `financial-report-insights/export_xlsx.py` (559 lines)
- `financial-report-insights/export_utils.py` (93 lines)
- `financial-report-insights/tests/test_export_pdf.py` (196 lines — read-only reference)
- `financial-report-insights/tests/test_export_xlsx.py` (217 lines — read-only reference)
- `financial-report-insights/tests/test_export_utils.py` (226 lines — read-only reference)

---

## P0 — Critical (Fix Immediately)

_No P0 findings. Neither module writes to disk paths, performs shell calls, nor deserializes untrusted data in dangerous ways. All output is produced into an in-memory `io.BytesIO` buffer._

---

## P1 — High (Fix This Sprint)

### P1-01 — `score_to_grade` silently accepts out-of-range inputs, including floats
**File:** `financial-report-insights/export_utils.py` lines 24–29

The function signature declares `score: int` but does no runtime type or range validation. Callers that pass a `float` (e.g., `78.6`) will silently evaluate correctly due to Python numeric coercion, but callers that pass `None` or a non-numeric type will raise an unhandled `TypeError` inside the comparisons. Passing a value above 100 (e.g., 105) returns `"A"` without complaint; there is no upper-bound guard. Three modules (`underwriting.py`, `compliance_scorer.py`, and `export_pdf.py` via `health.grade`) rely on this function as the canonical grade mapper.

**Suggestion:** Add a guard at the top of the function:
```python
if not isinstance(score, (int, float)):
    raise TypeError(f"score_to_grade expected numeric, got {type(score)}")
score = max(0, min(100, int(score)))
```

### P1-02 — Unbounded PDF page growth with adversarial `analysis_results` dict
**File:** `financial-report-insights/export_pdf.py` lines 108–140

The ratio-page loop iterates over every key in `analysis_results` that is numeric or `None`. There is no cap on how many entries are rendered. A caller with 10,000 ratio keys produces a multi-hundred-page PDF fully in memory before returning bytes. fpdf2 holds all page data in RAM before `pdf.output()` is called (line 149), so this is an unbounded heap allocation. The 50-character truncation on cell values (line 253) mitigates injection width but does not bound the number of rows.

**Suggestion:** Add a guard before the ratio loop. A practical limit of 500 entries is reasonable for financial reports:
```python
if len(numeric) > 500:
    numeric = dict(list(numeric.items())[:500])
    # optionally log a warning
```

### P1-03 — Unbounded XLSX row growth with adversarial `analysis_results` dict
**File:** `financial-report-insights/export_xlsx.py` lines 366–404

Same root cause as P1-02 in the XLSX path. `_write_ratios_sheet` iterates all numeric keys without limit. The `export_scenario_comparison` method (lines 230–265) additionally iterates all ratio keys across multiple scenarios with no cap; a scenario dict with thousands of ratios across dozens of scenarios will produce a correspondingly enormous workbook in memory.

**Suggestion:** Apply the same 500-entry cap per sheet as recommended for P1-02.

### P1-04 — `data.period` and `report.generated_at` are written to PDF without sanitization
**File:** `financial-report-insights/export_pdf.py` lines 319–327

Both `data.period` (line 322) and `report.generated_at` (line 327) are passed directly into `pdf.cell()` as f-string interpolations with no length check. fpdf2 does not interpret embedded markup or execute code from cell text (PDF content streams are inert), so there is no code-injection risk. However, a very long `period` string (e.g., 10,000 characters) will silently overflow the cell because fpdf2 `cell()` clips at the page margin but does not raise an error, making the truncation invisible and the displayed cover page misleading. For fields derived from user-supplied data, a length cap is warranted.

**Suggestion:** Truncate both fields before rendering:
```python
period_display = str(data.period)[:100]
generated_display = str(report.generated_at)[:50]
```

---

## P2 — Medium (Fix Soon)

### P2-01 — `_format_value` formats percentage keys as raw decimals when value >= 1.0
**File:** `financial-report-insights/export_pdf.py` lines 257–277

When `_is_percent_key(key)` returns `True`, the value is formatted with `:.2%` (e.g., `0.15` → `"15.00%"`). This is correct for ratios stored as decimals. However, if a caller stores a margin as `15.0` (already in percentage-point form), the output becomes `"1500.00%"` — a credibly wrong financial figure that no error is raised for. The XLSX module avoids this presentation bug because it passes the raw float to xlsxwriter's number format which is applied at render time, but the PDF `_format_value` performs its own formatting.

**Suggestion:** Document the expected decimal convention in the docstring and/or add a sanity clamp: if a "percentage" key value exceeds 50.0, format it as a plain float rather than multiplying by 100.

### P2-02 — `export_ratios` accepts arbitrary `company_name` string inserted into cell without length check
**File:** `financial-report-insights/export_xlsx.py` lines 168–169

The title string `f"Financial Ratios - {company_name}"` is written directly to the worksheet title cell. xlsxwriter will silently truncate strings that exceed its internal cell limit (32,767 characters), but there is no explicit guard or error for callers who pass a very long or structurally odd string. This is low-severity for injection (xlsxwriter escapes cell content) but can produce a misleading or truncated title with no caller feedback.

**Suggestion:** Add `company_name = str(company_name)[:100]` before use.

### P2-03 — `_add_table` col_widths guard is incomplete
**File:** `financial-report-insights/export_pdf.py` lines 250–251

The cell rendering loop accesses `col_widths[i]` for each column `i`. The fallback `col_widths[-1] if col_widths else 30` protects against an empty list but not against a `col_widths` list that is shorter than the number of header columns and also shorter than `i`. If a caller passes `col_widths=[90]` for a 3-column table and `i` reaches 2, the code returns `col_widths[-1]` (i.e., 90) for all overflow columns — which silently overflows the page width. This path is reachable from `_add_scoring_section` and `_add_health_section` since both call `_add_table` with fixed `col_widths=[90, 60]`.

**Suggestion:** Assert `len(col_widths) == len(headers)` at the top of `_add_table`, or auto-extend `col_widths` to match the header count.

### P2-04 — `data.period` truthiness check masks `"0"` and other falsy-but-valid period strings
**File:** `financial-report-insights/export_pdf.py` line 319

`if data.period:` silently suppresses a period value of `""` (expected) but also `0`, `"0"`, or any other falsy string that could be a legitimate period identifier. A consistent `is not None` check is more defensive.

**Suggestion:** Change to `if data.period is not None:`.

### P2-05 — Same falsy-value pattern on cover page financial lines
**File:** `financial-report-insights/export_pdf.py` lines 303–308

`if data.revenue:`, `if data.total_assets:`, and `if data.net_income:` all skip zero values. A company with `net_income=0` (break-even) or `revenue=0` (pre-revenue startup) will have those lines silently omitted from the cover page, which is misleading.

**Suggestion:** Change each to `if data.revenue is not None:`, etc., matching the outer guard on line 300.

### P2-06 — `_write_summary_sheet` only checks `if value is not None` but `_write_scoring_sheet` checks `isinstance(value, (int, float)) and value is not None`
**File:** `financial-report-insights/export_xlsx.py` lines 500–505

The guard `isinstance(value, (int, float)) and value is not None` is redundant — the `isinstance` check already excludes `None` — but more importantly, the check is inconsistent with the pattern used elsewhere (`if value is not None:`). If a value is a `bool` (e.g., `True` from a scoring dataclass field), `isinstance(value, (int, float))` returns `True` in Python (bool is a subclass of int), and `write_number` will write `1` or `0` instead of a meaningful string. This is cosmetic but will confuse readers of the Scoring sheet.

**Suggestion:** Add `and not isinstance(value, bool)` to the numeric guard in `_write_scoring_sheet` (lines 501 and 546), consistent with the equivalent PDF code at `export_pdf.py` lines 359–362, which already handles this correctly.

### P2-07 — `export_scenario_comparison` calls `ws.freeze_panes` inside the per-scenario loop
**File:** `financial-report-insights/export_xlsx.py` line 238

`ws.freeze_panes(row + 1, 0)` is called once per scenario inside the loop. xlsxwriter silently overwrites prior freeze-pane settings with the last one written. For multi-scenario exports this means only the final scenario's header row is frozen, and earlier scenario header rows are not. This is a UX defect rather than a correctness issue, but in a financial context where users scroll through data, the misaligned freeze is confusing.

**Suggestion:** Set the freeze pane once before the loop, at the known column-header row (which is `row + 1` after the initial title and generated-date rows), or remove it from the loop entirely and set it after the first scenario's header row is established.

### P2-08 — No `try/finally` around `wb.close()` in any XLSX method
**File:** `financial-report-insights/export_xlsx.py` lines 152, 193, 278

All three public XLSX methods call `wb.close()` as the last statement before returning `buf.getvalue()`. If any sheet-writing helper raises an exception between `wb = xlsxwriter.Workbook(buf, ...)` and `wb.close()`, the workbook is left open and the `BytesIO` buffer may contain a partial or corrupt XLSX. xlsxwriter does not implement a context manager in older versions, but the pattern is:
```python
try:
    # ... write sheets
finally:
    wb.close()
```
This ensures the workbook is always properly finalized (or cleanly abandoned) regardless of exceptions.

**Suggestion:** Wrap the body of each public method in `try/finally` with `wb.close()` in the `finally` block.

---

## P3 — Low / Housekeeping

### P3-01 — `safe_divide` is imported but never called in either export module
**File:** `financial-report-insights/export_pdf.py` line 23; `financial-report-insights/export_xlsx.py` line 23

Both modules import `safe_divide` from `financial_analyzer` but neither file uses it. This is dead import noise that confuses readers into thinking division operations exist in these files.

**Suggestion:** Remove `safe_divide` from the import lists in both files.

### P3-02 — `ScenarioResult` is imported in `export_pdf.py` but never used
**File:** `financial-report-insights/export_pdf.py` line 21

`ScenarioResult` is imported but no method in `export_pdf.py` handles scenario comparison — that functionality lives only in `export_xlsx.py`. This is dead code.

**Suggestion:** Remove `ScenarioResult` from `export_pdf.py`'s import list.

### P3-03 — `fields` is imported from `dataclasses` in both files but never called
**File:** `financial-report-insights/export_pdf.py` line 8; `financial-report-insights/export_xlsx.py` line 8

Both files import `fields` alongside `asdict` but only `asdict` is called. `fields` is unused in both.

**Suggestion:** Remove `fields` from the `dataclasses` imports in both files.

### P3-04 — `_PERCENT_KEYWORDS` and `_DOLLAR_KEYWORDS` are imported but not directly used in either module
**File:** `financial-report-insights/export_pdf.py` lines 26–27; `financial-report-insights/export_xlsx.py` lines 26–27

Both modules import the raw tuple constants alongside the helper functions. The constants themselves (`_PERCENT_KEYWORDS`, `_DOLLAR_KEYWORDS`) are never referenced directly in either file; all logic goes through `_is_percent_key` and `_is_dollar_key`. This creates an unnecessary coupling to the internal representation of `export_utils`.

**Suggestion:** Remove the raw tuple imports; keep only `_is_percent_key`, `_is_dollar_key`, `_CATEGORY_MAP`, and `_categorize`.

### P3-05 — `_CATEGORY_MAP` is imported but not called directly in either export module
**File:** `financial-report-insights/export_pdf.py` line 30; `financial-report-insights/export_xlsx.py` line 30

Same pattern as P3-04. Both files use `_categorize(key)`, never `_CATEGORY_MAP[key]` directly.

**Suggestion:** Remove `_CATEGORY_MAP` from both import lists and expose only the function.

### P3-06 — Hard-coded page width 210mm assumes A4 and will be wrong if format is changed
**File:** `financial-report-insights/export_pdf.py` lines 215 and 228

The separator line uses `210 - self.margin` and the column-width calculation uses `210 - 2 * self.margin`, hard-coding A4 width. If `_create_pdf` is ever changed to use a different page format, these values will silently produce off-page elements. The PDF spec is 210mm wide for A4 portrait.

**Suggestion:** Derive the width from `pdf.epw` (effective page width, available in fpdf2) instead of subtracting margin from a hard-coded constant.

### P3-07 — `score_to_grade` is not tested against boundary at 34/35 (one-off risk)
**File:** `financial-report-insights/tests/test_export_utils.py` lines 39–40

The test suite checks `score_to_grade(35) == "D"` and `score_to_grade(34) == "F"`. These are correct per the current thresholds. However, the threshold list is not tested as exhaustive — adding a new threshold tuple to `_GRADE_THRESHOLDS_100` would silently change behaviour without a test failure (e.g., inserting a `(40, "D+")` entry). This is a test-coverage gap, not a code bug.

**Suggestion:** Add a parametrized test that asserts the exact threshold list `[(80, "A"), (65, "B"), (50, "C"), (35, "D")]` against `_GRADE_THRESHOLDS_100` to lock the mapping against inadvertent changes.

### P3-08 — No test exercises `score_to_grade` with a non-integer type (float, None, string)
**File:** `financial-report-insights/tests/test_export_utils.py`

The tests confirm integer behaviour but never verify what happens with `score_to_grade(None)` or `score_to_grade("B")`. Per P1-01, the function raises `TypeError` in these cases, which is appropriate — but without a test, a future refactor that silently swallows the error would not be caught.

**Suggestion:** Add `pytest.raises(TypeError)` tests for `None` and `"B"` inputs once the P1-01 guard is in place.

### P3-09 — `export_executive_summary` does not guard against `report.executive_summary` being `None`
**File:** `financial-report-insights/export_pdf.py` line 164

`report.executive_summary or "No summary available."` correctly handles a falsy but non-`None` value such as `""`. However, if `FinancialReport` is ever changed such that `executive_summary` can be explicitly `None` (e.g., an Optional field), the `or` expression produces the fallback string correctly — so this is safe as written. This is a low-risk observation rather than a defect.

---

## Files With No Issues Found

- No file-system path construction occurs anywhere in these three modules. All output is written to `io.BytesIO()` buffers and returned as bytes. There is no path traversal risk.
- No subprocess calls, `os.system`, `eval`, or `exec` usage exists in any of the three files.
- Encoding: fpdf2 and xlsxwriter both produce binary formats internally. No raw `open(..., encoding=...)` calls are made. No encoding risk identified.
- Thread safety: All three export classes are stateless (no shared mutable class-level state). Each call to a public method creates a fresh `FPDF` or `xlsxwriter.Workbook` object. Multiple threads calling these methods concurrently are safe as written.
- Resource cleanup (PDF): `fpdf2` does not hold file handles; `pdf.output()` returns bytes and no cleanup is needed. The `io.BytesIO` buffer is not explicitly closed but this is harmless in CPython (garbage collected immediately) and correct in all Python implementations.
- `export_utils.py`: No issues found. Score-to-grade thresholds are consistent with the three consuming modules (P1-01 is a robustness concern, not an inconsistency). `_categorize` returns the `.value` string from the `RatioCategory` enum, which is consistently `"Liquidity"`, `"Profitability"`, `"Leverage"`, `"Efficiency"`, or `"Other"`.

---

## Summary

| Severity | Count | Key Themes |
|----------|-------|-----------|
| P0 | 0 | — |
| P1 | 4 | Unbounded memory growth (P1-02, P1-03); missing input validation on `score_to_grade` (P1-01); unsanitized long strings on cover page (P1-04) |
| P2 | 8 | Percentage formatting cliff at ≥1.0 (P2-01); bool-as-int in Scoring sheet (P2-06); missing `try/finally` around `wb.close()` (P2-08); UX bug with misplaced freeze pane (P2-07); zero-value suppression on cover (P2-05) |
| P3 | 9 | Dead imports (P3-01 through P3-05); hard-coded page width (P3-06); test-coverage gaps (P3-07, P3-08); minor defensive coding (P3-09) |

**Overall Quality Score: 7/10**

The modules are well-structured, consistently use in-memory buffers, correctly delegate formatting decisions to `export_utils`, and have good happy-path test coverage. The primary risks are memory-related (unbounded iteration over caller-supplied dicts) and a missing robustness guard on the canonical `score_to_grade` function. No injection or path-traversal risks were identified. The dead imports and inconsistent None-handling are housekeeping items that add noise but do not affect correctness for typical inputs.

**Technical Debt Estimate: 3–4 hours**
- P1 fixes: ~1.5 hours (guards + tests)
- P2 fixes: ~1.5 hours (bool guard, freeze pane, try/finally, zero-suppression)
- P3 cleanup: ~0.5 hours (import pruning, page-width constant)
