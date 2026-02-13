"""
Financial entity graph schema for Neo4j.

Node types and relationships for structured financial data storage
with vector-indexed document chunks for hybrid graph+vector retrieval.
"""

# ---------------------------------------------------------------------------
# Constraint & index creation statements (idempotent)
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT period_id IF NOT EXISTS FOR (p:FiscalPeriod) REQUIRE p.period_id IS UNIQUE",
    "CREATE CONSTRAINT stmt_id IF NOT EXISTS FOR (s:FinancialStatement) REQUIRE s.stmt_id IS UNIQUE",
    "CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:LineItem) REQUIRE i.item_id IS UNIQUE",
    "CREATE CONSTRAINT ratio_id IF NOT EXISTS FOR (r:FinancialRatio) REQUIRE r.ratio_id IS UNIQUE",
    "CREATE CONSTRAINT score_id IF NOT EXISTS FOR (s:ScoringResult) REQUIRE s.score_id IS UNIQUE",
]


def vector_index_statement(embedding_dim: int, model_name: str) -> str:
    """Return the Cypher to create a vector index on Chunk nodes.

    Index name encodes the model so that switching models doesn't collide.
    """
    safe_model = model_name.replace("-", "_").replace("/", "_")
    index_name = f"chunk_embedding_{safe_model}"
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {embedding_dim}, "
        f"`vector.similarity_function`: 'cosine'"
        f"}}}}"
    )


# ---------------------------------------------------------------------------
# Node and relationship schema documentation
# ---------------------------------------------------------------------------
#
# (:Document {doc_id, filename, file_type, uploaded_at})
#   -[:PROVIDES_DATA_FOR]-> (:FiscalPeriod {period_id, label, start_date, end_date})
#   -[:HAS_CHUNK]->         (:Chunk {chunk_id, content, embedding, source, chunk_index})
#
# (:Chunk {chunk_id, content, embedding, source, chunk_index})
#   -[:EXTRACTED_FROM]-> (:Document)
#
# (:FiscalPeriod)
#   -[:HAS_STATEMENT]-> (:FinancialStatement {stmt_id, type})   # balance_sheet, income_statement, cash_flow
#
# (:FinancialStatement)
#   -[:CONTAINS]-> (:LineItem {item_id, name, value, unit})
#
# (:LineItem)
#   -[:COMPOSES]-> (:LineItem)   # e.g. Current Assets -> Total Assets
#
# (:FiscalPeriod)
#   -[:HAS_RATIO]-> (:FinancialRatio {ratio_id, name, value, category})
#
# (:FinancialRatio)
#   -[:DERIVED_FROM]-> (:LineItem)
#
# (:FiscalPeriod)
#   -[:HAS_SCORE]-> (:ScoringResult {score_id, model, value, grade, interpretation})
#
# (:ScoringResult)
#   -[:DERIVED_FROM]-> (:FinancialRatio)
#

# ---------------------------------------------------------------------------
# Cypher templates for writes
# ---------------------------------------------------------------------------

MERGE_DOCUMENT = """
MERGE (d:Document {doc_id: $doc_id})
SET d.filename = $filename, d.file_type = $file_type, d.uploaded_at = datetime()
RETURN d
"""

MERGE_CHUNK = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.content = $content, c.embedding = $embedding, c.source = $source,
    c.chunk_index = $chunk_index
WITH c
MATCH (d:Document {doc_id: $doc_id})
MERGE (c)-[:EXTRACTED_FROM]->(d)
MERGE (d)-[:HAS_CHUNK]->(c)
RETURN c
"""

MERGE_FISCAL_PERIOD = """
MERGE (p:FiscalPeriod {period_id: $period_id})
SET p.label = $label
WITH p
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:PROVIDES_DATA_FOR]->(p)
RETURN p
"""

MERGE_LINE_ITEM = """
MERGE (i:LineItem {item_id: $item_id})
SET i.name = $name, i.value = $value, i.unit = $unit
WITH i
MATCH (s:FinancialStatement {stmt_id: $stmt_id})
MERGE (s)-[:CONTAINS]->(i)
RETURN i
"""

MERGE_RATIO = """
MERGE (r:FinancialRatio {ratio_id: $ratio_id})
SET r.name = $name, r.value = $value, r.category = $category
WITH r
MATCH (p:FiscalPeriod {period_id: $period_id})
MERGE (p)-[:HAS_RATIO]->(r)
RETURN r
"""

MERGE_SCORE = """
MERGE (s:ScoringResult {score_id: $score_id})
SET s.model = $model, s.value = $value, s.grade = $grade, s.interpretation = $interpretation
WITH s
MATCH (p:FiscalPeriod {period_id: $period_id})
MERGE (p)-[:HAS_SCORE]->(s)
RETURN s
"""

# ---------------------------------------------------------------------------
# Cypher templates for reads
# ---------------------------------------------------------------------------

VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
YIELD node, score
RETURN node.chunk_id AS chunk_id, node.content AS content,
       node.source AS source, score
ORDER BY score DESC
"""

GRAPH_CONTEXT_FOR_CHUNK = """
MATCH (c:Chunk {chunk_id: $chunk_id})-[:EXTRACTED_FROM]->(d:Document)
OPTIONAL MATCH (d)-[:PROVIDES_DATA_FOR]->(p:FiscalPeriod)
OPTIONAL MATCH (p)-[:HAS_RATIO]->(r:FinancialRatio)
OPTIONAL MATCH (p)-[:HAS_SCORE]->(s:ScoringResult)
RETURN d.filename AS document, p.label AS period,
       collect(DISTINCT {name: r.name, value: r.value, category: r.category}) AS ratios,
       collect(DISTINCT {model: s.model, value: s.value, grade: s.grade}) AS scores
"""

RATIOS_BY_PERIOD = """
MATCH (p:FiscalPeriod)-[:HAS_RATIO]->(r:FinancialRatio)
WHERE p.period_id = $period_id
RETURN r.name AS name, r.value AS value, r.category AS category
ORDER BY r.category, r.name
"""

SCORES_BY_PERIOD = """
MATCH (p:FiscalPeriod)-[:HAS_SCORE]->(s:ScoringResult)
WHERE p.period_id = $period_id
RETURN s.model AS model, s.value AS value, s.grade AS grade, s.interpretation AS interpretation
"""
