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
    "CREATE CONSTRAINT assessment_id IF NOT EXISTS FOR (a:CreditAssessment) REQUIRE a.assessment_id IS UNIQUE",
    "CREATE CONSTRAINT package_id IF NOT EXISTS FOR (p:CovenantPackage) REQUIRE p.package_id IS UNIQUE",
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT portfolio_id IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.portfolio_id IS UNIQUE",
    "CREATE CONSTRAINT portfolio_risk_id IF NOT EXISTS FOR (r:PortfolioRisk) REQUIRE r.risk_id IS UNIQUE",
    "CREATE CONSTRAINT compliance_id IF NOT EXISTS FOR (c:ComplianceReport) REQUIRE c.compliance_id IS UNIQUE",
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

MERGE_CHUNKS_BATCH = """
UNWIND $batch AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.content = row.content, c.embedding = row.embedding, c.source = row.source,
    c.chunk_index = row.chunk_index
WITH c, row
MATCH (d:Document {doc_id: row.doc_id})
MERGE (c)-[:EXTRACTED_FROM]->(d)
MERGE (d)-[:HAS_CHUNK]->(c)
RETURN count(c) AS stored
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

MERGE_RATIOS_BATCH = """
UNWIND $batch AS row
MERGE (r:FinancialRatio {ratio_id: row.ratio_id})
SET r.name = row.name, r.value = row.value, r.category = row.category
WITH r, row
MATCH (p:FiscalPeriod {period_id: row.period_id})
MERGE (p)-[:HAS_RATIO]->(r)
RETURN count(r) AS stored
"""

MERGE_SCORE = """
MERGE (s:ScoringResult {score_id: $score_id})
SET s.model = $model, s.value = $value, s.grade = $grade, s.interpretation = $interpretation
WITH s
MATCH (p:FiscalPeriod {period_id: $period_id})
MERGE (p)-[:HAS_SCORE]->(s)
RETURN s
"""

MERGE_SCORES_BATCH = """
UNWIND $batch AS row
MERGE (s:ScoringResult {score_id: row.score_id})
SET s.model = row.model, s.value = row.value, s.grade = row.grade,
    s.interpretation = row.interpretation
WITH s, row
MATCH (p:FiscalPeriod {period_id: row.period_id})
MERGE (p)-[:HAS_SCORE]->(s)
RETURN count(s) AS stored
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

GRAPH_CONTEXT_FOR_CHUNKS_BATCH = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[:EXTRACTED_FROM]->(d:Document)
OPTIONAL MATCH (d)-[:PROVIDES_DATA_FOR]->(p:FiscalPeriod)
OPTIONAL MATCH (p)-[:HAS_RATIO]->(r:FinancialRatio)
OPTIONAL MATCH (p)-[:HAS_SCORE]->(s:ScoringResult)
RETURN cid AS chunk_id, d.filename AS document, p.label AS period,
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

# ---------------------------------------------------------------------------
# Phase 2: Structured financial data population
# ---------------------------------------------------------------------------

MERGE_FINANCIAL_STATEMENT = """
MERGE (s:FinancialStatement {stmt_id: $stmt_id})
SET s.type = $stmt_type
WITH s
MATCH (p:FiscalPeriod {period_id: $period_id})
MERGE (p)-[:HAS_STATEMENT]->(s)
RETURN s
"""

MERGE_LINE_ITEMS_BATCH = """
UNWIND $batch AS row
MERGE (i:LineItem {item_id: row.item_id})
SET i.name = row.name, i.value = row.value, i.unit = row.unit
WITH i, row
MATCH (s:FinancialStatement {stmt_id: row.stmt_id})
MERGE (s)-[:CONTAINS]->(i)
RETURN count(i) AS stored
"""

MERGE_DERIVED_FROM_BATCH = """
UNWIND $batch AS row
MATCH (r:FinancialRatio {ratio_id: row.ratio_id})
MATCH (i:LineItem {item_id: row.item_id})
MERGE (r)-[:DERIVED_FROM {role: row.role}]->(i)
RETURN count(*) AS linked
"""

# ---------------------------------------------------------------------------
# Phase 2/3: Period-label-based reads (no period_id required from caller)
# ---------------------------------------------------------------------------

RATIOS_BY_PERIOD_LABEL = """
MATCH (p:FiscalPeriod)-[:HAS_RATIO]->(r:FinancialRatio)
WHERE p.label = $period_label
RETURN r.name AS name, r.value AS value, r.category AS category
ORDER BY r.category, r.name
"""

SCORES_BY_PERIOD_LABEL = """
MATCH (p:FiscalPeriod)-[:HAS_SCORE]->(s:ScoringResult)
WHERE p.label = $period_label
RETURN s.model AS model, s.value AS value, s.grade AS grade, s.interpretation AS interpretation
"""

# ---------------------------------------------------------------------------
# Phase 3: Credit assessment and covenant package nodes
# ---------------------------------------------------------------------------
#
# (:Company {name})
#   -[:HAS_CREDIT_ASSESSMENT]-> (:CreditAssessment {assessment_id, total_score, grade,
#                                                    recommendation, category_scores,
#                                                    strengths, weaknesses,
#                                                    max_additional_debt, current_leverage})
#
# (:CreditAssessment)
#   -[:REQUIRES_COVENANTS]-> (:CovenantPackage {package_id, covenant_tier,
#                                               financial_covenants,
#                                               reporting_requirements,
#                                               events_of_default})
#

MERGE_COMPANY = """
MERGE (c:Company {name: $name})
RETURN c
"""

MERGE_CREDIT_ASSESSMENT = """
MERGE (a:CreditAssessment {assessment_id: $assessment_id})
SET a.total_score = $total_score,
    a.grade = $grade,
    a.recommendation = $recommendation,
    a.category_scores = $category_scores,
    a.strengths = $strengths,
    a.weaknesses = $weaknesses,
    a.max_additional_debt = $max_additional_debt,
    a.current_leverage = $current_leverage
WITH a
MATCH (c:Company {name: $company_name})
MERGE (c)-[:HAS_CREDIT_ASSESSMENT]->(a)
RETURN a
"""

MERGE_CREDIT_ASSESSMENTS_BATCH = """
UNWIND $batch AS row
MERGE (a:CreditAssessment {assessment_id: row.assessment_id})
SET a.total_score = row.total_score,
    a.grade = row.grade,
    a.recommendation = row.recommendation,
    a.category_scores = row.category_scores,
    a.strengths = row.strengths,
    a.weaknesses = row.weaknesses,
    a.max_additional_debt = row.max_additional_debt,
    a.current_leverage = row.current_leverage
WITH a, row
MATCH (c:Company {name: row.company_name})
MERGE (c)-[:HAS_CREDIT_ASSESSMENT]->(a)
RETURN count(a) AS stored
"""

MERGE_COVENANT_PACKAGE = """
MERGE (p:CovenantPackage {package_id: $package_id})
SET p.covenant_tier = $covenant_tier,
    p.financial_covenants = $financial_covenants,
    p.reporting_requirements = $reporting_requirements,
    p.events_of_default = $events_of_default
WITH p
MATCH (a:CreditAssessment {assessment_id: $assessment_id})
MERGE (a)-[:REQUIRES_COVENANTS]->(p)
RETURN p
"""

MERGE_COVENANT_PACKAGES_BATCH = """
UNWIND $batch AS row
MERGE (p:CovenantPackage {package_id: row.package_id})
SET p.covenant_tier = row.covenant_tier,
    p.financial_covenants = row.financial_covenants,
    p.reporting_requirements = row.reporting_requirements,
    p.events_of_default = row.events_of_default
WITH p, row
MATCH (a:CreditAssessment {assessment_id: row.assessment_id})
MERGE (a)-[:REQUIRES_COVENANTS]->(p)
RETURN count(p) AS stored
"""

CREDIT_ASSESSMENT_BY_COMPANY = """
MATCH (c:Company {name: $company_name})-[:HAS_CREDIT_ASSESSMENT]->(a:CreditAssessment)
OPTIONAL MATCH (a)-[:REQUIRES_COVENANTS]->(p:CovenantPackage)
RETURN a.assessment_id AS assessment_id, a.total_score AS total_score,
       a.grade AS grade, a.recommendation AS recommendation,
       a.category_scores AS category_scores,
       a.strengths AS strengths, a.weaknesses AS weaknesses,
       a.max_additional_debt AS max_additional_debt,
       a.current_leverage AS current_leverage,
       p.covenant_tier AS covenant_tier,
       p.financial_covenants AS financial_covenants,
       p.reporting_requirements AS reporting_requirements,
       p.events_of_default AS events_of_default
ORDER BY a.total_score DESC
"""

# ---------------------------------------------------------------------------
# Phase 4: Temporal edges and cross-period queries
# ---------------------------------------------------------------------------

MERGE_TEMPORAL_EDGES = """
UNWIND $pairs AS pair
MATCH (earlier:FiscalPeriod {period_id: pair.earlier_id})
MATCH (later:FiscalPeriod {period_id: pair.later_id})
MERGE (earlier)-[:PRECEDES]->(later)
MERGE (later)-[:FOLLOWS]->(earlier)
RETURN count(*) AS linked
"""

CROSS_PERIOD_RATIO_TREND = """
UNWIND $period_labels AS lbl
MATCH (p:FiscalPeriod {label: lbl})-[:HAS_RATIO]->(r:FinancialRatio)
RETURN p.label AS period, r.name AS ratio_name, r.value AS value, r.category AS category
ORDER BY r.name, p.label
"""

# ---------------------------------------------------------------------------
# Phase 7: Portfolio and compliance graph nodes
# ---------------------------------------------------------------------------
#
# (:Portfolio {portfolio_id, name, created_at})
#   -[:CONTAINS_COMPANY]-> (:Company)
#   -[:HAS_PORTFOLIO_RISK]-> (:PortfolioRisk {risk_id, avg_health, min_health,
#                                              max_health, risk_level, distress_count,
#                                              diversification_score, risk_flags})
#
# (:Company)
#   -[:HAS_COMPLIANCE_REPORT]-> (:ComplianceReport {compliance_id, sox_risk,
#                                                    sox_score, sec_score,
#                                                    regulatory_pct, audit_risk,
#                                                    audit_score, going_concern})
#

MERGE_PORTFOLIO = """
MERGE (p:Portfolio {portfolio_id: $portfolio_id})
SET p.name = $name, p.created_at = datetime()
RETURN p
"""

MERGE_PORTFOLIO_MEMBERSHIP_BATCH = """
UNWIND $company_names AS cname
MERGE (c:Company {name: cname})
WITH c
MATCH (p:Portfolio {portfolio_id: $portfolio_id})
MERGE (p)-[:CONTAINS_COMPANY]->(c)
RETURN count(c) AS linked
"""

MERGE_PORTFOLIO_RISK = """
MERGE (r:PortfolioRisk {risk_id: $risk_id})
SET r.avg_health = $avg_health,
    r.min_health = $min_health,
    r.max_health = $max_health,
    r.risk_level = $risk_level,
    r.distress_count = $distress_count,
    r.diversification_score = $diversification_score,
    r.risk_flags = $risk_flags
WITH r
MATCH (p:Portfolio {portfolio_id: $portfolio_id})
MERGE (p)-[:HAS_PORTFOLIO_RISK]->(r)
RETURN r
"""

MERGE_COMPLIANCE_REPORT_BATCH = """
UNWIND $batch AS row
MERGE (cr:ComplianceReport {compliance_id: row.compliance_id})
SET cr.sox_risk = row.sox_risk,
    cr.sox_score = row.sox_score,
    cr.sec_score = row.sec_score,
    cr.regulatory_pct = row.regulatory_pct,
    cr.audit_risk = row.audit_risk,
    cr.audit_score = row.audit_score,
    cr.going_concern = row.going_concern
WITH cr, row
MATCH (c:Company {name: row.company_name})
MERGE (c)-[:HAS_COMPLIANCE_REPORT]->(cr)
RETURN count(cr) AS stored
"""

COMPLIANCE_BY_COMPANY = """
MATCH (c:Company {name: $company_name})-[:HAS_COMPLIANCE_REPORT]->(cr:ComplianceReport)
RETURN cr.compliance_id AS compliance_id, cr.sox_risk AS sox_risk,
       cr.sox_score AS sox_score, cr.sec_score AS sec_score,
       cr.regulatory_pct AS regulatory_pct, cr.audit_risk AS audit_risk,
       cr.audit_score AS audit_score, cr.going_concern AS going_concern
ORDER BY cr.audit_score DESC
"""
