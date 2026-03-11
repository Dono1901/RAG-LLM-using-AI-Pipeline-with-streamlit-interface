# RAG Pipeline Enhancement - Task Plan

## Goal
Build a working end-to-end RAG pipeline: Document ingestion -> Chunking -> Embedding -> Vector storage -> Retrieval -> Generation

## Status: COMPLETE (All phases done)

## Phase 1: Design (COMPLETE)
- [x] Understanding Lock confirmed
- [x] Architecture selected (Approach A: Modular Pipeline)
- [x] Section detection expanded (60+ sections)
- [x] Line item detection: analyzed 20+ real models from user's computer
- [x] Financial modeling skills loaded (xlsx-official, financial-modeling-suite, charlie)
- [x] Complete design review (Skeptic, Constraint Guardian, User Advocate, Arbiter)
- [x] Final design document

## Phase 2: PDF Parser (`pdf_parser.py`) - COMPLETE
- [x] pymupdf4llm integration with fitz fallback
- [x] PDF -> structured markdown with table preservation
- [x] Section boundary detection (30+ section types with 60+ patterns)
- [x] Metadata extraction (company, period, filing type, page numbers)
- [x] Tests (31 tests)

## Phase 3: Document Chunker (`document_chunker.py`) - COMPLETE
- [x] Parent-child chunking strategy (child 250 tokens, parent 1200 tokens)
- [x] Financial-aware section splitting
- [x] Table-as-atomic-chunk handling (never split tables)
- [x] NL descriptions for numeric-heavy chunks (embedding quality boost)
- [x] Excel sheet chunking with header preservation
- [x] Metadata enrichment per chunk
- [x] Tests (26 tests)

## Phase 4: Line Item Mapper (`line_item_mapper.py`) - COMPLETE
- [x] Comprehensive mapping: 75+ canonical fields, 350+ patterns
- [x] Categories: income_statement, balance_sheet, cash_flow, valuation, debt, cannabis, cash_forecast, kpi, saas, construction, fund
- [x] Longest-pattern-first matching (avoids greedy substring matches)
- [x] Nested parenthetical stripping
- [x] Multi-period column detection (years, quarters, months, weeks, YTD/MTD)
- [x] Section context confidence boosting
- [x] Tests (56 tests)

## Phase 5: Ingestion Pipeline (`ingestion_pipeline.py`) - COMPLETE
- [x] Orchestrate: parse -> chunk -> embed -> index
- [x] Support Excel (.xlsx, .xlsm, .xls, .csv, .tsv) + PDF + text/md/docx
- [x] Wire into app_local.py `_load_documents()` (with legacy fallback)
- [x] Sheet section type detection (30+ types from sheet name + content)
- [x] Label column auto-detection (handles column B labels)
- [x] Unnamed column fixing (Excel->CSV export artifacts)
- [x] chunks_to_documents() converter for SimpleRAG compatibility
- [x] Tests (40 tests)

## Phase 6: Retrieval Enhancement - COMPLETE
- [x] Parent chunk expansion at query time (`_expand_parent_chunks()` in app_local.py)
- [x] Cross-encoder reranking via DMR (already wired via `reranker.py` + `EmbeddingReranker`)
- [x] Wire into api.py /documents endpoint (section_type, chunk_level, has_parent fields)
- [x] Config: `enable_parent_expansion` setting (default ON)
- [x] Tests (10 new: 8 parent expansion + 2 API metadata)

## Phase 7: Evaluation - COMPLETE (pre-existing)
- [x] Ground-truth Q&A pairs (20 pairs, 5 query types, 3 difficulty levels)
- [x] Retrieval metrics: MRR, NDCG@k, Precision@k, Recall@k
- [x] Answer metrics: faithfulness, relevance, completeness (word-overlap heuristics)
- [x] RAGEvalHarness: `evaluate_retrieval()`, `evaluate_answers()`, `run_full_eval()`
- [x] Config gated: `enable_evaluation` setting (default OFF)
- [x] Tests (38 tests in test_evaluation.py)
- [ ] Manual test with real uploaded models (requires live documents)

## Decisions Log
| # | Decision | Alternatives | Rationale |
|---|----------|-------------|-----------|
| 1 | Approach A (modular) | LlamaIndex, Hybrid | Fits existing arch, domain-specific chunking |
| 2 | Parent-child chunks | Fixed-size, sentence | Best quality for mixed factual/analytical queries |
| 3 | 60+ section types | Generic 4-type | Real models have cannabis, RE, banking, SaaS sections |
| 4 | Learn from real models | Generic patterns only | User has 20+ real models with non-standard naming |
| 5 | pymupdf4llm for PDF | pdfplumber, PyPDF2 | Best markdown output with table preservation |
| 6 | Local-only LLM | Cloud API | User requirement |
| 7 | NL descriptions for numeric chunks | Embed raw numbers | Numbers alone embed poorly, NL prefix improves retrieval |
| 8 | Ordered hint list for sheet detection | Dict lookup | Avoids "cf" in "DCF" matching cash_flow before dcf |
| 9 | Parent expansion ON by default | OFF | Child chunks too small (250 tok) for LLM context, parent (1200 tok) much better |
| 10 | Dedup by parent_id during expansion | No dedup | Multiple child hits from same parent produce redundant context |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| "ar" in "foobar" false positive | Use longest-pattern-first matching |
| "cf" in "DCF" matching cash_flow | Ordered hint list, dcf before cf |
| Nested parens not fully stripped | Iterative regex until stable |
| Windows PermissionError on temp xlsx | gc.collect() + permissive cleanup |

## Files Created/Modified
- `pdf_parser.py` (NEW, 290 LOC)
- `document_chunker.py` (NEW, 340 LOC)
- `line_item_mapper.py` (NEW, 530 LOC)
- `ingestion_pipeline.py` (NEW, 360 LOC)
- `app_local.py` (MODIFIED - _load_documents() uses pipeline, _expand_parent_chunks())
- `api.py` (MODIFIED - /documents endpoint exposes section_type, chunk_level, has_parent)
- `config.py` (MODIFIED - added enable_parent_expansion setting)
- `tests/test_pdf_parser.py` (NEW, 31 tests)
- `tests/test_line_item_mapper.py` (NEW, 56 tests)
- `tests/test_document_chunker.py` (NEW, 26 tests)
- `tests/test_ingestion_pipeline.py` (NEW, 40 tests)
- `tests/test_app_local.py` (MODIFIED, +8 parent expansion tests)
- `tests/test_api.py` (MODIFIED, +2 /documents metadata tests)
- Total: 163 new tests, all passing
- Full suite: 4,918 tests passing
