# Audit Report: Graph Database Layer
**Date:** 2026-03-16
**Scope:** Neo4j graph store, graph schema, Cypher queries
**Files Reviewed:**
- `financial-report-insights/graph_store.py` (742 lines)
- `financial-report-insights/graph_schema.py` (455 lines)

---

## P0 — Critical (Fix Immediately)

### P0-01 — `vector_index_statement` injects unsanitized `embedding_dim` directly into Cypher string
**File:** `financial-report-insights/graph_schema.py:37,46`

`embedding_dim = int(embedding_dim)` is the only guard, and the result is interpolated
directly into the Cypher DDL string at line 46:
```
f"`vector.dimensions`: {embedding_dim}, "
```
An attacker or misconfigured caller who can control the value after the `int()` cast
cannot inject text (the cast prevents that), but the broader pattern is dangerous: the
entire `vector_index_statement` function produces a non-parameterized DDL string that is
executed verbatim by `ensure_schema` (graph_store.py:76) via `session.run(stmt)` with no
further protection. DDL statements cannot use `$parameter` placeholders in Neo4j; the
current sanitization of `model_name` via regex (line 39) is correct, but `embedding_dim`
relies solely on `int()` conversion. If the call chain is ever refactored and the cast is
removed, injection becomes trivial. The `ensure_schema` call path also executes all of
`CONSTRAINTS` (graph_store.py:75) as raw strings without parameters — acceptable for
static constants, but any future templated constraint will inherit the same risk.

**Severity:** High (injection surface is small but present; pattern is fragile)

**Suggestion:** Wrap `ensure_schema` in a transaction so that a partially-executed schema
setup rolls back cleanly. Document explicitly that `vector_index_statement` is the
**only** place where string-formatting into Cypher is permitted, and add a unit test
asserting that `model_name` values containing Cypher metacharacters (`;`, `}`, `{`) are
fully neutralised by the regex.

---

### P0-02 — Silent swallow of all Neo4j exceptions in every write path
**File:** `financial-report-insights/graph_store.py:128-129, 186-187, 273-274, 329-330, 399-400, 452-453, 483-484, 672-673, 728-729`

Every write method catches `except Exception as exc` and logs `logger.warning(...)`,
returning `0`, `None`, or an empty list. This means:
- A caller that stores chunks, then immediately reads them back, may receive stale or
  absent data with no indication that storage failed.
- A schema mismatch, constraint violation, or authentication expiry is indistinguishable
  from a successful no-op write.
- `store_covenant_package` (line 452) returns `None` on failure; `store_credit_assessment`
  (line 399) also returns `None`. Callers in `api.py` and `graph_retriever.py` that chain
  these calls without checking the return value will silently produce an incomplete graph.

**Severity:** High — data integrity is unverifiable at the call site.

**Suggestion:** Distinguish transient Neo4j errors (connection refused, timeout) from
permanent errors (constraint violation, schema error). Re-raise permanent errors or
return a typed `Result[T, Error]` so callers can decide whether to retry, alert, or abort.
At minimum, emit `logger.error` (not `logger.warning`) for constraint violations and
authentication failures so they appear in production alerting dashboards.

---

## P1 — High (Fix This Sprint)

### P1-01 — `store_line_items` opens one session per statement type in a loop
**File:** `financial-report-insights/graph_store.py:240-270`

The outer `with self._driver.session() as session:` block wraps the entire loop, so this
is technically a single session. However, within that session, there are **two sequential
`session.run()` calls per statement type** (line 263 for `MERGE_FINANCIAL_STATEMENT` and
line 269 for `MERGE_LINE_ITEMS_BATCH`), executed in a `for stmt_type, fields` loop over
three statement types. That is 6 sequential round-trips in the common case — no
pipelining or wrapping transaction.

The deeper issue is that `MERGE_FINANCIAL_STATEMENT` must succeed before
`MERGE_LINE_ITEMS_BATCH` can match the statement node it links to, but there is no
explicit transaction boundary. If Neo4j is in auto-commit mode (the default for
`session.run()` outside an explicit `session.begin_transaction()` block), each call is its
own micro-transaction. A crash between the first and second `session.run()` leaves an
orphaned `FinancialStatement` node with no `LineItem` children, and no error is surfaced.

**Suggestion:** Wrap the two paired `session.run()` calls in an explicit
`with session.begin_transaction() as tx:` block, or consolidate into a single
UNWIND query that both MERGEs the statement and its line items in one round-trip.

---

### P1-02 — `store_financial_data` runs three sequential sessions for what is logically one atomic write
**File:** `financial-report-insights/graph_store.py:153-183`

`MERGE_FISCAL_PERIOD`, `MERGE_RATIOS_BATCH`, and `MERGE_SCORES_BATCH` are issued as
three separate `session.run()` calls inside one `with self._driver.session()` block (lines
154, 168, 183). In Neo4j's default auto-commit mode this means three independent
transactions. If `MERGE_RATIOS_BATCH` succeeds but `MERGE_SCORES_BATCH` fails (e.g.,
constraint violation on a duplicate score_id), the graph is left with ratios but no
scores for that period, and the exception is caught and logged as a warning with no
partial-write indicator.

**Suggestion:** Use an explicit transaction (`session.begin_transaction()`) that covers
all three statements, so a failure on any one rolls back the entire fiscal period write.

---

### P1-03 — `VECTOR_SEARCH` query has no LIMIT clause
**File:** `financial-report-insights/graph_schema.py:178-184`

```cypher
CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
YIELD node, score
RETURN node.chunk_id AS chunk_id, node.content AS content,
       node.source AS source, score
ORDER BY score DESC
```

The `$top_k` parameter controls how many candidates `db.index.vector.queryNodes` yields
internally, but `ORDER BY score DESC` with no `LIMIT` means the result set returned to
the driver is bounded only by the index cardinality. If a caller passes `top_k=100` and
the index has 10,000 nodes, Neo4j will sort all 10,000 yielded candidates and stream them
all to the Python driver unless Neo4j's planner decides to push the sort into the index
call (not guaranteed in all versions). Adding `LIMIT $top_k` after `ORDER BY` is
defensive and self-documenting.

**Suggestion:** Append `LIMIT $top_k` to `VECTOR_SEARCH`.

---

### P1-04 — `GRAPH_CONTEXT_FOR_CHUNKS_BATCH` has unbounded `collect(DISTINCT ...)` with no depth limit
**File:** `financial-report-insights/graph_schema.py:196-205`

```cypher
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[:EXTRACTED_FROM]->(d:Document)
OPTIONAL MATCH (d)-[:PROVIDES_DATA_FOR]->(p:FiscalPeriod)
OPTIONAL MATCH (p)-[:HAS_RATIO]->(r:FinancialRatio)
OPTIONAL MATCH (p)-[:HAS_SCORE]->(s:ScoringResult)
...
collect(DISTINCT {name: r.name, value: r.value, category: r.category}) AS ratios,
collect(DISTINCT {model: s.model, value: s.value, grade: s.grade}) AS scores
```

For a document with many fiscal periods (e.g., 10 years of annual reports), every period
matched via `OPTIONAL MATCH (d)-[:PROVIDES_DATA_FOR]->(p:FiscalPeriod)` fans out to all
its ratios and scores. With `chunk_ids` containing 20 chunk IDs and a document containing
10 periods each with 50 ratios and 5 scores, the Cartesian explosion before the
`collect(DISTINCT ...)` aggregation can reach 20 * 10 * 50 * 5 = 50,000 intermediate
rows per query. There is no `LIMIT` or path length cap on the traversal.

**Suggestion:** Add `LIMIT` to the ratio/score `OPTIONAL MATCH` sub-patterns using `WITH
... LIMIT` or restructure as a subquery with `CALL { ... } IN TRANSACTIONS`. At minimum,
document the known fanout risk and add a `LIMIT 50` guard on each `collect` path.

---

### P1-05 — `CREDIT_ASSESSMENT_BY_COMPANY` and `COMPLIANCE_BY_COMPANY` queries are defined but never referenced from `graph_store.py`
**File:** `financial-report-insights/graph_schema.py:348-362, 447-454`

`CREDIT_ASSESSMENT_BY_COMPANY` and `COMPLIANCE_BY_COMPANY` are Cypher templates with no
corresponding read methods in `graph_store.py`. The data is write-only; there is no
supported path to retrieve stored credit assessments or compliance reports via the graph
layer. This is not a safety issue, but it indicates the read API is incomplete and
callers relying on the graph for these lookups will silently fall back to stale or absent
data.

**Suggestion:** Implement `credit_assessment_by_company(company_name)` and
`compliance_by_company(company_name)` read methods in `graph_store.py`, or remove the
orphaned Cypher constants if the read path is intentionally deferred.

---

## P2 — Medium (Fix Soon)

### P2-01 — `store_chunks` silently ignores missing Document node during batch
**File:** `financial-report-insights/graph_store.py:106,110-124` / `graph_schema.py:107-114`

`MERGE_CHUNKS_BATCH` (graph_schema.py:105-115) uses:
```cypher
MATCH (d:Document {doc_id: row.doc_id})
```
If the preceding `MERGE_DOCUMENT` (line 106) failed (or was never committed due to
auto-commit timing), the `MATCH` finds nothing and the chunk nodes are created but
**never linked to a document**. The `RETURN count(c) AS stored` value will still equal
the batch size because the `MERGE` on `:Chunk` is not gated on the `MATCH` succeeding.
The orphaned chunks will be found by vector search but will return `null` for `source`
in `VECTOR_SEARCH` result rows.

**Suggestion:** Change the `MATCH (d:Document ...)` in `MERGE_CHUNKS_BATCH` to `MERGE`
so the document is created if missing, or add an assertion in `store_chunks` that the
document MERGE succeeded before issuing the batch. Alternatively, use a single combined
Cypher statement that atomically creates both.

---

### P2-02 — `store_derived_from_edges` constructs batch entirely in Python before any session is opened
**File:** `financial-report-insights/graph_store.py:309-320`

The batch is built by iterating `RATIO_CATALOG` and doing SHA-256 lookups against
`field_to_stmt` in Python. If `RATIO_CATALOG` grows (it already has entries for every
financial ratio in the system), this pre-computation loop runs synchronously on the
application thread. For a large catalog this is not a correctness problem, but the
pattern diverges from the rest of the store (which builds batches lazily inside the
session context) and makes it harder to page large batches.

**Suggestion:** Add a comment documenting the expected maximum batch size and consider a
chunked submission (e.g., 500 edges per `session.run()`) if `RATIO_CATALOG` grows past a
few hundred entries.

---

### P2-03 — `vector_search` reconstructs the index name from `model_name` at call time without validation
**File:** `financial-report-insights/graph_store.py:508-509`

```python
safe_model = model_name.replace("-", "_").replace("/", "_")
index_name = f"chunk_embedding_{safe_model}"
```

This uses `str.replace` chaining, which is weaker than the regex sanitization in
`vector_index_statement` (graph_schema.py:39). A `model_name` containing characters such
as `` ` ``, `$`, or `}` would be passed through. The resulting `index_name` is then
supplied as the `$index_name` parameter to `VECTOR_SEARCH` (graph_schema.py:179):
```cypher
CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
```
Because this is a parameterized call, not string interpolation, the risk is limited to
Neo4j raising a "no such index" error rather than a Cypher injection. However, the
inconsistent sanitization between `vector_index_statement` and `vector_search` means the
index created and the index queried may not match if a model name with unusual characters
is introduced.

**Suggestion:** Extract the sanitization logic into a shared helper function
`_safe_index_name(model_name: str) -> str` and use it in both `vector_index_statement`
and `vector_search`.

---

### P2-04 — `close()` silently suppresses all driver-close exceptions
**File:** `financial-report-insights/graph_store.py:740-741`

```python
except Exception:
    logger.debug("Error closing Neo4j driver", exc_info=True)
```

Swallowing close errors at `DEBUG` level means connection-pool exhaustion or leaked
sockets go unnoticed in production. The Neo4j Python driver's `close()` can raise if
there are active connections that cannot be gracefully terminated.

**Suggestion:** Elevate to `logger.warning` and include the exception message string
(not just `exc_info=True`), consistent with the pattern used in every other method in
the file.

---

### P2-05 — `store_portfolio_analysis` calls three `session.run()` statements without an explicit transaction
**File:** `financial-report-insights/graph_store.py:645-663`

`MERGE_PORTFOLIO`, `MERGE_PORTFOLIO_MEMBERSHIP_BATCH`, and `MERGE_PORTFOLIO_RISK` are
issued as three independent auto-commit transactions. If the process crashes between the
second and third call, the portfolio node exists and has company memberships, but has no
`PortfolioRisk` node. Queries that assume every `Portfolio` has exactly one
`HAS_PORTFOLIO_RISK` edge will return empty results or raise null-access errors.

**Suggestion:** Same as P1-02: wrap all three calls in an explicit transaction.

---

### P2-06 — `GRAPH_CONTEXT_FOR_CHUNK` (singular) is defined but never used
**File:** `financial-report-insights/graph_schema.py:186-194`

`GRAPH_CONTEXT_FOR_CHUNK` is a per-chunk query that was presumably superseded by
`GRAPH_CONTEXT_FOR_CHUNKS_BATCH`. It is not imported or used anywhere in `graph_store.py`.
Dead code in the schema module increases maintenance burden and creates confusion about
which is the canonical query.

**Suggestion:** Remove `GRAPH_CONTEXT_FOR_CHUNK`, `MERGE_CHUNK`, `MERGE_LINE_ITEM`,
`MERGE_RATIO`, `MERGE_SCORE`, `MERGE_CREDIT_ASSESSMENT`, and `MERGE_COVENANT_PACKAGE`
(all singular non-batch variants) if they are confirmed unused. Confirmed dead code:
- `graph_schema.py:94-103` (`MERGE_CHUNK`)
- `graph_schema.py:126-133` (`MERGE_LINE_ITEM`)
- `graph_schema.py:135-142` (`MERGE_RATIO`)
- `graph_schema.py:154-161` (`MERGE_SCORE`)
- `graph_schema.py:186-194` (`GRAPH_CONTEXT_FOR_CHUNK`)
- `graph_schema.py:207-212` (`RATIOS_BY_PERIOD`)
- `graph_schema.py:214-218` (`SCORES_BY_PERIOD`)
- `graph_schema.py:290-304` (`MERGE_CREDIT_ASSESSMENT`)
- `graph_schema.py:323-333` (`MERGE_COVENANT_PACKAGE`)

---

### P2-07 — `compliance_id` hash input is sensitive to None values in audit object
**File:** `financial-report-insights/graph_store.py:698-700`

```python
compliance_id = hashlib.sha256(
    f"{company_name}:compliance:{audit.score if audit else 0}".encode()
).hexdigest()
```

If `audit` is not `None` but `audit.score` is `None` (a valid state when the compliance
scorer lacks enough data), the hash key becomes `"{company_name}:compliance:None"`, which
looks like a valid but incorrect key. Subsequent calls with a real score of `0` would
produce a different hash, creating duplicate nodes for the same company instead of
merging. The constraint on `compliance_id` would prevent actual duplicates from being
stored, but the MERGE would target the wrong node.

**Suggestion:** Normalize `None` scores to a sentinel value (e.g., `-1`) with an explicit
comment, or include a richer canonical identifier (e.g., a timestamp or report period
label) in the hash input.

---

## P3 — Low / Housekeeping

### P3-01 — `MERGE_DOCUMENT` returns `d` but the return value is never checked
**File:** `financial-report-insights/graph_schema.py:88-92` / `graph_store.py:106`

`session.run(MERGE_DOCUMENT, ...)` returns a `Result` object. The result is never
consumed (`result.data()`, `result.single()`, etc.), which means Neo4j's result buffer
is not explicitly drained before the next `session.run()` call. The Python driver handles
this transparently in most cases, but it is a subtle resource-management gap. The same
applies to `MERGE_FISCAL_PERIOD`, `MERGE_FINANCIAL_STATEMENT`, `MERGE_COMPANY`,
`MERGE_PORTFOLIO`, and `MERGE_PORTFOLIO_RISK`.

**Suggestion:** Either drop the `RETURN` clauses from write-only MERGE statements
(replacing with `RETURN count(*) AS ok` consumed via `.single()`), or explicitly call
`.consume()` on each result object to drain the buffer.

---

### P3-02 — `MERGE_PORTFOLIO` unconditionally overwrites `created_at` on every call
**File:** `financial-report-insights/graph_schema.py:401-405`

```cypher
MERGE (p:Portfolio {portfolio_id: $portfolio_id})
SET p.name = $name, p.created_at = datetime()
```

`SET` (not `SET ... ON CREATE SET`) resets `created_at` every time the portfolio is
merged, even when it already exists. This silently corrupts the audit trail of when a
portfolio was first created.

**Suggestion:** Change to:
```cypher
MERGE (p:Portfolio {portfolio_id: $portfolio_id})
ON CREATE SET p.created_at = datetime()
SET p.name = $name
```

The same pattern is used correctly for `:Document` nodes (`MERGE_DOCUMENT` at
graph_schema.py:90), so this is an inconsistency.

---

### P3-03 — `MERGE_DOCUMENT` also overwrites `uploaded_at` on re-ingestion
**File:** `financial-report-insights/graph_schema.py:88-92`

```cypher
SET d.filename = $filename, d.file_type = $file_type, d.uploaded_at = datetime()
```

Same issue as P3-02: `uploaded_at` is reset every time the same document is re-ingested
(e.g., on a retry after an embedding failure). The original upload timestamp is lost.

**Suggestion:**
```cypher
ON CREATE SET d.uploaded_at = datetime()
SET d.filename = $filename, d.file_type = $file_type
```

---

### P3-04 — No schema migration strategy exists
**File:** `financial-report-insights/graph_schema.py` (entire file)

All constraints and indexes are created with `IF NOT EXISTS`, which is idempotent for
additions but provides no mechanism to:
- Remove a constraint that has been renamed or replaced.
- Drop and recreate a vector index when `embedding_dim` changes (e.g., switching from
  384-dim to 1024-dim embeddings).
- Evolve a node label or relationship type name.

If `embedding_dim` changes in config, `vector_index_statement` will attempt to create a
new index under the new model-qualified name, leaving the old index intact consuming
memory and storage. There is no version tracking on the schema.

**Suggestion:** Add a `SchemaVersion` node or a dedicated admin script that drops stale
vector indexes before recreating them. Document the current schema version in a comment
at the top of `graph_schema.py`.

---

### P3-05 — `connect()` catches all exceptions during driver creation, including `ImportError`
**File:** `financial-report-insights/graph_store.py:61-63`

```python
except Exception as exc:
    logger.warning("Neo4j connection failed (%s) – falling back to in-memory", exc)
    return None
```

If the `neo4j` package is not installed, `import neo4j` raises `ImportError`, which is
caught here and logs a misleading "connection failed" warning. The real cause
(missing dependency) is buried in the warning message text.

**Suggestion:** Catch `ImportError` separately and log a distinct message:
`"neo4j package not installed; graph store disabled"`.

---

### P3-06 — `assessment_id` hash is not stable across score changes
**File:** `financial-report-insights/graph_store.py:356-358`

```python
assessment_id = hashlib.sha256(
    f"{company_name}:{scorecard.grade}:{scorecard.total_score}".encode()
).hexdigest()
```

Including `grade` and `total_score` in the hash means every re-run of the underwriting
model with even slightly different scores produces a new `CreditAssessment` node rather
than updating the existing one. Over time this creates unbounded historical assessment
nodes for the same company with no explicit versioning or cleanup.

**Suggestion:** If idempotent merge per-run is desired, include a run timestamp or report
period label in the hash. If update-in-place is desired, use only `company_name` as the
MERGE key and store `grade`/`total_score` as mutable `SET` properties.

---

### P3-07 — Unused `MERGE_TEMPORAL_EDGES` produces redundant bidirectional edges
**File:** `financial-report-insights/graph_schema.py:368-375`

```cypher
MERGE (earlier)-[:PRECEDES]->(later)
MERGE (later)-[:FOLLOWS]->(earlier)
```

`PRECEDES` and `FOLLOWS` are semantically redundant: any query that needs to traverse
periods in order can do so using either `PRECEDES` forward or `FOLLOWS` backward. Storing
both doubles the relationship count and the cost of queries that use `MATCH
(p)-[:PRECEDES|:FOLLOWS]->()`. In Neo4j best practice, one directional relationship
type is preferred; the reverse direction is traversed by reversing the arrow in the
MATCH pattern.

**Suggestion:** Remove `MERGE (later)-[:FOLLOWS]->(earlier)` and update any traversal
queries to use `(later)<-[:PRECEDES]-(earlier)`.

---

## Files With No Issues Found

Neither file had issues in the following areas:
- **Cypher injection via string concatenation in parameterized queries**: All
  `session.run()` calls use `$parameter` placeholders for data values. The only
  string-interpolated Cypher is `vector_index_statement`, which is guarded by regex
  sanitization and an `int()` cast.
- **N+1 query patterns**: Every multi-row write operation uses `UNWIND $batch`
  (`MERGE_CHUNKS_BATCH`, `MERGE_RATIOS_BATCH`, `MERGE_SCORES_BATCH`,
  `MERGE_LINE_ITEMS_BATCH`, `MERGE_DERIVED_FROM_BATCH`, `MERGE_TEMPORAL_EDGES`,
  `MERGE_CREDIT_ASSESSMENTS_BATCH`, `MERGE_COVENANT_PACKAGES_BATCH`,
  `MERGE_PORTFOLIO_MEMBERSHIP_BATCH`, `MERGE_COMPLIANCE_REPORT_BATCH`).
- **Driver lifecycle**: `connect()` calls `driver.verify_connectivity()` before
  returning; `close()` exists and is called explicitly. Context managers are used for all
  sessions.
- **Constraint definitions**: All 13 node types with unique IDs have corresponding
  `UNIQUE` constraints defined with `IF NOT EXISTS` (graph_schema.py:12-26).
- **Index coverage for lookup keys**: Every MATCH pattern that filters on a property used
  as a node identity key (e.g., `{chunk_id: ...}`, `{doc_id: ...}`) is backed by a
  uniqueness constraint, which Neo4j automatically indexes.
- **NEO4J_PASSWORD guard**: `connect()` explicitly refuses to connect when the password
  is empty (graph_store.py:52-56), preventing accidental anonymous connections.

---

## Summary

| Priority | Count | Key Theme |
|----------|-------|-----------|
| P0       | 2     | String-interpolated DDL (fragile injection surface); silent swallow of all write errors |
| P1       | 5     | Missing explicit transactions on multi-statement writes; unbounded query results; dead read API |
| P2       | 7     | Orphaned chunk nodes; inconsistent sanitization helpers; `None`-sensitive hash keys; dead schema constants |
| P3       | 7     | Audit-trail overwrite; no schema migration; bidirectional redundant edges; package ImportError masking |

**Total issues:** 21

**Technical debt estimate:** 14–18 hours
- P0 fixes: ~2 h (transaction wrappers, error classification)
- P1 fixes: ~4 h (explicit transactions, LIMIT guards, read method stubs)
- P2 fixes: ~4 h (shared sanitizer, hash normalization, dead-code removal)
- P3 fixes: ~4–8 h (schema migration strategy is the largest item)

**Positive observations:** The UNWIND batching discipline is thorough and consistent
across all 10 batch write operations. Parameterized queries are used correctly everywhere
data values appear. The driver factory pattern (`connect()` returning `None` on
misconfiguration) provides clean optional integration without requiring callers to handle
the absent-Neo4j case. The `NEO4J_PASSWORD` guard is a meaningful security control.
