# Audit Report: Data Ingestion & Processing
**Date:** 2026-03-16
**Scope:** Excel processing, PDF parsing, document chunking, ingestion pipeline, line item mapping
**Files Reviewed:**
- `financial-report-insights/excel_processor.py` (868 lines)
- `financial-report-insights/pdf_parser.py` (394 lines)
- `financial-report-insights/document_chunker.py` (420 lines)
- `financial-report-insights/ingestion_pipeline.py` (448 lines)
- `financial-report-insights/line_item_mapper.py` (807 lines)

***

## P0 — Critical (Fix Immediately)

### 1. openpyxl workbook not closed on exception — file handle leak with potential OOM
- **File:** `excel_processor.py:202–253`
- **Severity:** Critical
- **Detail:** `wb = openpyxl.load_workbook(...)` is opened at line 202, and `wb.close()` is called at line 253, but there is no `try/finally` or context manager wrapping the sheet-iteration loop (lines 204–252). If any exception is raised while iterating sheets (e.g., a corrupt sheet, a pandas conversion error, or a MemoryError on a huge file), `wb.close()` is never called. openpyxl's `read_only=True` mode opens a ZIP file internally; leaking that handle under memory pressure can prevent the file from being garbage-collected and can exhaust file descriptors in a long-running server.
- **Suggestion:** Wrap the openpyxl section in a `try/finally`:
  ```python
  wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
  try:
      for sheet_name in wb.sheetnames:
          ...
  finally:
      wb.close()
  ```

### 2. xlrd workbook never closed — file handle leak
- **File:** `excel_processor.py:260–299`
- **Severity:** Critical
- **Detail:** `wb = xlrd.open_workbook(file_path)` at line 260 is never explicitly closed. xlrd workbooks hold an open file descriptor for the duration of the object's life. The object is local to `_load_xlrd()` and relies entirely on CPython reference-counting for cleanup. Under PyPy, or in any code path that raises before the function returns (e.g., line 267 `sheet.row_values(i)` on a corrupt sheet), the handle leaks indefinitely.
- **Suggestion:** Add `wb.release_resources()` in a `finally` block, or use `xlrd.open_workbook` with a context variable pattern.

### 3. `ingest_excel` CSV path has no row/size guard — unbounded DataFrame load
- **File:** `ingestion_pipeline.py:186–193`
- **Severity:** Critical
- **Detail:** When the file is a CSV or TSV, `pd.read_csv(file_path, sep=sep)` at line 187 loads the entire file into memory with no row cap. The row limit enforced by `ExcelProcessor` (via `settings.max_workbook_rows`) is completely bypassed in this code path. A 200 MB CSV (allowed by `max_file_size_mb=200`) with narrow columns could contain tens of millions of rows, consuming several gigabytes of RAM before any limit is applied.
- **Suggestion:** Add `nrows=settings.max_workbook_rows` to the `pd.read_csv()` call, or apply `df = df.head(settings.max_workbook_rows)` immediately after loading.

### 4. `ingest_excel` XLSX path also bypasses row guard
- **File:** `ingestion_pipeline.py:195–210`
- **Severity:** Critical
- **Detail:** `pd.read_excel(xls, sheet_name=sheet_name)` at line 199 loads each sheet fully into memory. The `ExcelProcessor.load_workbook()` row-cap logic (lines 164–173 of `excel_processor.py`) is entirely absent in this separate code path. These two ingestion paths are independent — `ingest_excel` does not call `ExcelProcessor` at all.
- **Suggestion:** Add `nrows=settings.max_workbook_rows` to `pd.read_excel()` at line 199.

---

## P1 — High (Fix This Sprint)

### 5. `ingest_file` performs no path validation — arbitrary file read
- **File:** `ingestion_pipeline.py:383–405`
- **Severity:** High
- **Detail:** `ingest_file(file_path)` accepts any `Path` object and immediately dispatches to `ingest_excel`, `ingest_pdf`, or `ingest_text` with no check that the path (a) exists, (b) is a regular file (not a symlink, device file, or FIFO), or (c) is confined to the expected documents directory. Any caller with an arbitrary string can read files outside the intended ingestion folder. The Streamlit upload path (`streamlit_app_local.py`) validates paths before saving, but `ingest_file` is also exposed as a public API entry point; other callers (e.g., API endpoints, background workers) may pass unvalidated paths.
- **Suggestion:** Add at the top of `ingest_file`:
  ```python
  file_path = Path(file_path).resolve()
  if not file_path.is_file():
      logger.error("Path does not exist or is not a file: %s", file_path)
      return []
  ```
  Optionally enforce confinement to a known documents root.

### 6. `parse_pdf` performs no file existence check and no page-count limit
- **File:** `pdf_parser.py:317–393`
- **Severity:** High
- **Detail:** `parse_pdf(file_path)` opens the file with `pymupdf4llm.to_markdown(str(file_path))` (line 336) or `fitz.open(file_path)` (line 346) without first checking that the path exists or is a regular file. A missing file will produce an opaque library exception that is not caught here, propagating as an unhandled error to the caller. More critically, there is no limit on the number of PDF pages processed. A 5,000-page PDF will be fully converted to markdown in-memory before any chunking occurs — this is a reliable OOM vector given `max_file_size_mb=200` (a 200 MB PDF can have thousands of dense pages).
- **Suggestion:** (1) Check `file_path.is_file()` and raise `FileNotFoundError` early. (2) Add a configurable `max_pdf_pages` setting and pass `pages=list(range(max_pdf_pages))` to `pymupdf4llm.to_markdown()`, or process pages in batches in the fitz fallback.

### 7. `ingest_text` reads entire file as UTF-8 with no size guard
- **File:** `ingestion_pipeline.py:355`
- **Severity:** High
- **Detail:** `text = file_path.read_text(encoding="utf-8")` at line 355 reads the complete file into a single Python string. `.txt` and `.md` files are covered by the `max_file_size_mb` check at the upload layer, but no guard exists at the ingestion layer. If called directly (e.g., from a pipeline script or test), a multi-hundred-MB text file will be loaded entirely before the chunker receives it. The fallback for `UnicodeDecodeError` is also absent — a binary file with a `.txt` extension will raise an unhandled exception.
- **Suggestion:** Read in chunks, or at minimum add `errors='replace'` and cap at a byte limit before calling `read_text`.

### 8. `_df_to_markdown` silently swallows all exceptions on fallback
- **File:** `ingestion_pipeline.py:147–158`
- **Severity:** High
- **Detail:** The `except Exception:` at line 149 catches all exceptions from `truncated.to_markdown()` without logging the error. If `to_markdown` fails for a structural reason (e.g., a column with an incompatible type), the fallback pipe-delimited format is used silently. The operator has no visibility into why the primary serialization failed. This pattern also masks programming errors.
- **Suggestion:** Change to `except Exception as exc:` and add `logger.debug("to_markdown failed for sheet, using fallback: %s", exc)`.

### 9. `_load_openpyxl` streaming row limit has a ten-row gap that can load above the intended cap
- **File:** `excel_processor.py:208–212`
- **Severity:** High
- **Detail:** The streaming row read uses `row_limit = settings.max_workbook_rows + 10` as the break condition (line 208). This intentional "+10" margin for header-row search means the in-memory `data` list can hold up to `max_workbook_rows + 10` rows before the loop exits. The per-sheet truncation at line 172–173 is applied only after the DataFrame is constructed. For the default cap of 500,000 rows, this is inconsequential, but it establishes a pattern where the comment says one thing and the code does another, and the margin is hardcoded rather than derived from the header search depth (which also searches only the first 10 rows, line 360).
- **Suggestion:** Either set `row_limit = settings.max_workbook_rows + 10` as an explicit constant tied to the header-search depth (e.g., `HEADER_SEARCH_DEPTH = 10; row_limit = settings.max_workbook_rows + HEADER_SEARCH_DEPTH`), or apply the row cap before the DataFrame is built to avoid temporary over-allocation.

### 10. `chunk_table` chunk ID uses `hash(table_text) % 10000` — high collision probability
- **File:** `document_chunker.py:299`
- **Severity:** High
- **Detail:** The `chunk_id` for table chunks is generated as `_generate_chunk_id(source, f"table:{section_title}", hash(table_text) % 10000)`. `hash()` returns a 64-bit integer modulo 10,000, giving only 10,000 distinct values for the `index` component of the ID. A workbook with many tables sharing the same source and section title has a roughly 1-in-10,000 per-pair collision probability. Colliding chunk IDs cause later chunks to overwrite earlier ones silently in any ID-keyed store (e.g., a vector database upsert). The parent/child text chunks use a monotonically incrementing `block_idx` / `chunk_idx` (lines 191, 221, 252), making this a table-specific inconsistency.
- **Suggestion:** Replace `hash(table_text) % 10000` with a content-hash of the table text, e.g., `int(hashlib.sha256(table_text.encode()).hexdigest()[:8], 16)`, which the `_generate_chunk_id` helper already does for the key as a whole. Alternatively, pass a monotonic index.

---

## P2 — Medium (Fix Soon)

### 11. `ingest_excel` does not close `pd.ExcelFile` — file handle not guaranteed released
- **File:** `ingestion_pipeline.py:195`
- **Severity:** Medium
- **Detail:** `xls = pd.ExcelFile(file_path)` at line 195 is never explicitly closed. `pd.ExcelFile` holds an open file descriptor. It has a `.close()` method, and `__enter__`/`__exit__` for use as a context manager. In CPython with reference counting this is mostly harmless, but the handle leaks on exception and is not idiomatic. The outer `except Exception as e` at line 242 silently discards the error, leaving `xls` unreleased.
- **Suggestion:** Replace with `with pd.ExcelFile(file_path) as xls:`.

### 12. Row-cap check applies to total rows across all sheets, then truncates per-sheet independently
- **File:** `excel_processor.py:164–173`
- **Severity:** Medium
- **Detail:** The guard at line 165 checks `total_rows > settings.max_workbook_rows`. If the workbook has 10 sheets each with 60,000 rows and the cap is 500,000, the total (600,000) exceeds the cap. The fix at lines 171–173 truncates each sheet individually to `max_workbook_rows`, not to `max_workbook_rows / sheet_count`. A workbook with many sheets can therefore retain far more rows than intended. For example, 10 sheets each truncated to 500,000 rows = 5,000,000 total rows in memory.
- **Suggestion:** Compute a per-sheet budget: `per_sheet_limit = max(1, settings.max_workbook_rows // len(sheets))`, then cap each sheet to that.

### 13. `_load_xlrd` loads all rows eagerly before header detection
- **File:** `excel_processor.py:267`
- **Severity:** Medium
- **Detail:** `data = [sheet.row_values(i) for i in range(sheet.nrows)]` at line 267 materialises all rows of every sheet as a list of lists before applying any cap. For a large `.xls` file (which can hold up to 65,536 rows per sheet in the BIFF8 format), this is a full in-memory load. The `max_workbook_rows` cap is applied only after this allocation. The openpyxl path streams rows (line 209–212) correctly; xlrd does not.
- **Suggestion:** Iterate `range(min(sheet.nrows, row_limit))` where `row_limit = settings.max_workbook_rows + HEADER_SEARCH_DEPTH`.

### 14. `parse_pdf` opens `fitz` twice when `pymupdf4llm` is available
- **File:** `pdf_parser.py:336–340`
- **Severity:** Medium
- **Detail:** In the `pymupdf4llm` path (lines 334–341), the PDF is first passed to `pymupdf4llm.to_markdown()` (which internally opens a `fitz.Document`) and then re-opened with `fitz.open(file_path)` solely to read `len(doc)` for page count. This means each PDF is opened and decoded twice. For large PDFs this doubles the I/O and decompression cost. The second `fitz.Document` is also closed inline (line 340) but not in a `finally` block — an exception between lines 338 and 340 would leak it.
- **Suggestion:** Either (a) have `pymupdf4llm` return page count metadata, or (b) expose the page count from the first `fitz.open()` call and reuse it. At minimum, wrap lines 338–340 in `try/finally`.

### 15. `_normalize_label` parenthesis-stripping loop has no depth limit
- **File:** `line_item_mapper.py:691–695`
- **Severity:** Medium
- **Detail:** The `while "(" in label and ")" in label:` loop at line 691 removes innermost parenthetical groups one pass at a time. The loop terminates only when the regex no longer changes the string. For a pathological input with many nested parentheses (e.g., `((((((((label))))))))`) this loop runs O(n) times where n is nesting depth. Labels in financial documents are not attacker-controlled in most deployments, but if documents come from external sources (e.g., uploaded customer files), this can be triggered intentionally. A 10,000-character label with alternating parentheses will execute thousands of regex passes.
- **Suggestion:** Add a maximum iteration count: `for _ in range(50): ...` and break early.

### 16. `_detect_statement_type` scores sections against each other in a winner-take-all vote that ignores confidence gaps
- **File:** `excel_processor.py:490–522`
- **Severity:** Medium
- **Detail:** The detection logic returns the highest-scoring type whenever `max_score > 1` (line 519). A sheet with one "revenue" mention (income_statement score=1) and one "budget" mention (budget score=1) will be returned as `'custom'`, while a sheet with scores of 3 vs 2 will be returned as the type with score 3 regardless of how close the second-place type is. There is no tie-breaking or threshold relative to document length, leading to misclassification of hybrid documents (e.g., a budget with a short income statement summary).
- **Severity downgrade note:** This is a data quality issue, not a security or availability issue, but it directly affects RAG retrieval quality.
- **Suggestion:** Return `None` when the top two scores are equal (a tie), and only classify when the winner leads the runner-up by at least 2.

### 17. `_find_label_column` uses a fixed 5-column scan window
- **File:** `ingestion_pipeline.py:123`
- **Severity:** Medium
- **Detail:** `for col_idx in range(min(5, len(df.columns))):` restricts label column detection to the first five columns. Real-world financial models (especially multi-section workbooks) frequently place labels in column 6 or 7 (e.g., when columns A–E are blank or contain formatting anchors). When no column meets `best_score > 3` in the first 5, `None` is returned and section type detection falls back to sheet name heuristics only.
- **Suggestion:** Extend the scan window to 10 columns, or use a configurable `LABEL_COL_SCAN_DEPTH` constant.

### 18. `chunk_text_content` parent overlap duplicates the last sentence verbatim across adjacent parent chunks
- **File:** `document_chunker.py:179–181`
- **Severity:** Medium
- **Detail:** When a new parent block starts, the last sentence of the previous block is prepended as overlap (lines 179–181). That sentence will appear as the final sentence of one parent chunk and the first sentence of the next, creating literal duplicate text in the vector index. For short financial statements with few but long sentences (e.g., "Total revenue for fiscal year 2024 was $4.2 billion, compared to $3.8 billion in fiscal year 2023"), this can mean a single critical sentence is stored in two parent chunks with identical content, inflating retrieval scores.
- **Suggestion:** The overlap strategy is intentional for context continuity, but child chunks derived from the overlapping sentence should not be re-emitted as new child chunks. Currently, child chunks are generated from `block_sentences` which includes the overlap sentence (line 217). Ensure the first child of each new parent block skips the sentence if it was already a child of the previous block, or document this as a known intentional duplicate.

---

## P3 — Low / Housekeeping

### 19. `_generate_chunk_id` truncates SHA-256 to 16 hex chars (64-bit) — birthday bound is ~4 billion chunks
- **File:** `document_chunker.py:56`
- **Severity:** Low
- **Detail:** `hexdigest()[:16]` produces a 64-bit identifier. The birthday collision probability exceeds 1% at approximately 600 million chunks. For a single deployment this is far beyond practical limits, but it is worth noting for any future distributed or multi-tenant scenario.
- **Suggestion:** Consider 24 chars (96 bits) if chunk counts are expected to grow significantly.

### 20. `_load_with_pandas` fallback uses bare `except Exception` without re-raise
- **File:** `excel_processor.py:351–352`
- **Severity:** Low
- **Detail:** In `_load_with_pandas`, the `except Exception as e:` at line 351 logs the error and returns an empty list. The caller (`_load_xlrd` at line 303) receives an empty list silently when both xlrd and pandas fail, giving the upstream caller no indication that the file could not be loaded at all. The top-level `load_workbook` method would then construct a `WorkbookData` with zero sheets.
- **Suggestion:** Either re-raise or return a sentinel that causes `load_workbook` to raise, so the caller knows the file was unreadable.

### 21. `map_labels_batch` deduplicates on canonical name, silently drops second occurrence
- **File:** `line_item_mapper.py:768–771`
- **Severity:** Low
- **Detail:** `seen_canonical: Set[str]` at line 765 prevents the same canonical field from appearing twice in the batch result. This is intentional to avoid duplicate fields in a financial model, but it means the second row labelled (e.g.) "Net Income" in a table is silently dropped rather than flagged. For sheets with subtotals and parent-level totals that share a canonical name, data is lost without a warning log.
- **Suggestion:** Add a `logger.debug("Duplicate canonical name '%s' for label '%s', skipping", ...)` entry to make the drop visible.

### 22. `ingest_text` hardcodes UTF-8 with no encoding fallback for `.txt` files
- **File:** `ingestion_pipeline.py:355`
- **Severity:** Low
- **Detail:** `file_path.read_text(encoding="utf-8")` will raise `UnicodeDecodeError` on any file saved in a legacy encoding (Windows-1252, Latin-1, etc.). The Excel CSV loader has a multi-encoding fallback (`_load_csv`, `excel_processor.py:309–320`); the text loader does not.
- **Suggestion:** Wrap in a try/except and retry with `encoding='latin-1'` as a fallback, consistent with the Excel path.

### 23. `_split_into_sections` all-caps heading pattern is over-broad
- **File:** `pdf_parser.py:255–258`
- **Severity:** Low
- **Detail:** The heading regex `r"(?:^|\n)(#{1,4}\s+.+|[A-Z][A-Z\s&/\-]{5,}(?:\n|$))"` matches any all-caps line of six or more characters. Financial PDFs frequently contain all-caps table headers, ticker symbols, and company names that are not section boundaries. This produces spurious sections, each emitted as a separate `ParsedSection`, inflating chunk count and fragmenting context.
- **Suggestion:** Require the all-caps pattern to appear at the start of a line after a blank line, reducing false positives: `r"(?:^|\n\n)([A-Z][A-Z\s&/\-]{5,})(?:\n)"`.

### 24. `excel_processor.py` uses f-string logging (log injection via filenames)
- **File:** `excel_processor.py:150, 168, 181`
- **Severity:** Low
- **Detail:** Lines 150, 168, and 181 use f-strings in `logger.info()` / `logger.warning()` / `logger.error()` calls (e.g., `f"Loading workbook: {file_path.name}"`). If a malicious user uploads a file named `"../../etc/passwd\nINFO root – login success"`, the newline-injected string appears in structured logs as a fake log entry. Lazy `%s` formatting (as used correctly in `ingestion_pipeline.py`) avoids this.
- **Suggestion:** Replace f-string logging with `%s`-style: `logger.info("Loading workbook: %s", file_path.name)`.

### 25. `detect_periods` in `line_item_mapper.py` does not normalise column strings before matching
- **File:** `line_item_mapper.py:799`
- **Severity:** Low
- **Detail:** `col_str = str(col).strip()` at line 799 strips leading/trailing whitespace but does not lowercase the string before matching. However, all period patterns use `re.IGNORECASE` where applicable. The `"^(?:FY\s*)?(\d{4})\s*[AaEe]?$"` pattern at line 788 accepts lowercase `a`/`e` for actual/estimate suffixes but the match is already case-insensitive due to no flag — this is fine. No functional bug, but could use a comment to clarify why `lower()` is not called.

---

## Files With No Issues Found

None. All five files have at least one finding. The issues in `line_item_mapper.py` are the least severe (P2–P3 only); the mapper itself is well-structured, with clean longest-pattern-first deduplication and good normalization logic.

---

## Summary

| Priority | Count | Files Affected |
|----------|-------|----------------|
| P0 — Critical | 4 | `excel_processor.py`, `ingestion_pipeline.py` |
| P1 — High | 6 | `excel_processor.py`, `ingestion_pipeline.py`, `pdf_parser.py`, `document_chunker.py` |
| P2 — Medium | 8 | All five files |
| P3 — Low | 7 | `excel_processor.py`, `ingestion_pipeline.py`, `pdf_parser.py`, `document_chunker.py`, `line_item_mapper.py` |
| **Total** | **25** | |

**Estimated Technical Debt:** 18–24 hours

**Highest Risk Areas:**
1. The dual ingestion code paths (`ExcelProcessor` vs. `ingest_excel` in the pipeline) are structurally disconnected. Row caps, encoding fallbacks, and resource cleanup are implemented in `ExcelProcessor` but are partially or fully absent in `ingest_excel`. Consolidating these into a single path would eliminate four findings at once (P0 #3, P0 #4, P2 #13, P3 #11).
2. All three resource-handle leaks (openpyxl workbook, xlrd workbook, `pd.ExcelFile`) share the same root cause: absence of `try/finally` or context managers. A single engineering standard — "all file handles use `with` statements" — enforced by a linter rule would prevent this class of bug.
3. The PDF parser has no page-count limit, which is the only DoS-by-large-file vector not already mitigated by the upload-layer size checks.
