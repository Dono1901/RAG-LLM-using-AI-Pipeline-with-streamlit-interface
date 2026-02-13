"""
Generic parameterized ratio analysis framework.

Replaces 300+ repetitive ratio methods with a declarative configuration-driven approach.
Each ratio is defined once with scoring thresholds and adjustments, then computed generically.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum

from financial_analyzer import FinancialData, safe_divide


class Operator(Enum):
    """Comparison operators for ratio adjustments."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="


@dataclass
class Adjustment:
    """Defines a scoring adjustment based on a secondary ratio."""
    field: str  # Field name to check (e.g., "current_ratio")
    operator: Operator
    threshold: float
    delta: float  # Score adjustment amount (can be negative)
    description: str = ""


@dataclass
class RatioDefinition:
    """Declarative definition of a financial ratio with scoring rules."""
    name: str
    description: str
    numerator_field: str  # Field name on FinancialData
    denominator_field: str
    higher_is_better: bool = True
    scoring_thresholds: List[Tuple[float, float]] = field(default_factory=list)  # [(threshold, score), ...] descending
    adjustments: List[Adjustment] = field(default_factory=list)
    grade_map: Dict[Tuple[float, float], str] = field(default_factory=lambda: {
        (8.0, 10.0): "Excellent",
        (6.0, 8.0): "Good",
        (4.0, 6.0): "Adequate",
        (0.0, 4.0): "Weak"
    })
    unit: str = ""  # e.g., "%", "x", "days"

    def get_grade(self, score: float) -> str:
        """Map a score to a grade label."""
        for (low, high), grade in self.grade_map.items():
            if low <= score < high:
                return grade
        # Handle edge case for perfect score
        if score >= 10.0:
            return "Excellent"
        return "Weak"


@dataclass
class RatioResult:
    """Result of computing a single ratio."""
    name: str
    value: Optional[float]
    score: float
    grade: str
    summary: str
    secondary_ratios: Dict[str, Optional[float]] = field(default_factory=dict)


def _get_field_value(data: FinancialData, field_name: str) -> Optional[float]:
    """Safely extract a field value from FinancialData."""
    return getattr(data, field_name, None)


def _apply_operator(value: float, operator: Operator, threshold: float) -> bool:
    """Apply comparison operator."""
    if operator == Operator.GT:
        return value > threshold
    elif operator == Operator.GTE:
        return value >= threshold
    elif operator == Operator.LT:
        return value < threshold
    elif operator == Operator.LTE:
        return value <= threshold
    elif operator == Operator.EQ:
        return abs(value - threshold) < 1e-9
    return False


def _compute_base_score(
    ratio_value: float,
    scoring_thresholds: List[Tuple[float, float]],
    higher_is_better: bool
) -> float:
    """
    Compute base score from ratio value and thresholds.

    Args:
        ratio_value: The computed ratio value
        scoring_thresholds: List of (threshold, score) tuples in DESCENDING order
        higher_is_better: If False, invert the comparison logic

    Returns:
        Base score (0-10 range before adjustments)
    """
    if not scoring_thresholds:
        return 5.0  # Default mid-range score

    # Sort thresholds descending by threshold value
    sorted_thresholds = sorted(scoring_thresholds, key=lambda x: x[0], reverse=True)

    if higher_is_better:
        # Standard logic: higher ratio = higher score
        for threshold, score in sorted_thresholds:
            if ratio_value >= threshold:
                return score
        # Below all thresholds
        return sorted_thresholds[-1][1] - 1.0
    else:
        # Inverted logic: lower ratio = higher score
        for threshold, score in sorted_thresholds:
            if ratio_value <= threshold:
                return score
        # Above all thresholds
        return sorted_thresholds[-1][1] - 1.0


def _apply_adjustments(
    base_score: float,
    adjustments: List[Adjustment],
    data: FinancialData,
    computed_ratios: Dict[str, Optional[float]]
) -> float:
    """
    Apply scoring adjustments based on secondary ratios.

    Args:
        base_score: Initial score before adjustments
        adjustments: List of Adjustment rules
        data: FinancialData instance
        computed_ratios: Dict of already-computed secondary ratios

    Returns:
        Adjusted score (clamped to 0-10)
    """
    score = base_score

    for adj in adjustments:
        # Try to get value from computed_ratios first, then from data fields
        value = computed_ratios.get(adj.field)
        if value is None:
            value = _get_field_value(data, adj.field)

        if value is not None and _apply_operator(value, adj.operator, adj.threshold):
            score += adj.delta

    # Clamp to valid range
    return max(0.0, min(10.0, score))


def _build_summary(
    name: str,
    value: Optional[float],
    score: float,
    grade: str,
    unit: str,
    description: str
) -> str:
    """Build a human-readable summary string."""
    if value is None:
        return f"{name}: Insufficient data for calculation"

    value_str = f"{value:.2f}{unit}" if unit else f"{value:.2f}"
    return f"{name}: {value_str} | Score: {score:.1f}/10 ({grade}) | {description}"


def compute_ratio(
    data: FinancialData,
    definition: RatioDefinition,
    computed_ratios: Optional[Dict[str, Optional[float]]] = None
) -> RatioResult:
    """
    Generic ratio computation engine.

    Implements the complete scoring pipeline:
    1. Extract numerator/denominator from FinancialData
    2. Compute ratio using safe_divide
    3. Apply scoring thresholds
    4. Apply adjustments based on secondary ratios
    5. Assign grade
    6. Build summary

    Args:
        data: FinancialData instance
        definition: RatioDefinition configuration
        computed_ratios: Optional dict of pre-computed ratios for adjustments

    Returns:
        RatioResult with value, score, grade, and summary
    """
    computed_ratios = computed_ratios or {}

    # Step 1: Extract fields
    numerator = _get_field_value(data, definition.numerator_field)
    denominator = _get_field_value(data, definition.denominator_field)

    # Step 2: Compute ratio
    ratio_value = safe_divide(numerator, denominator)

    if ratio_value is None:
        return RatioResult(
            name=definition.name,
            value=None,
            score=0.0,
            grade="Insufficient Data",
            summary=f"{definition.name}: Insufficient data",
            secondary_ratios={}
        )

    # Step 3: Base score from thresholds
    base_score = _compute_base_score(
        ratio_value,
        definition.scoring_thresholds,
        definition.higher_is_better
    )

    # Step 4: Apply adjustments
    final_score = _apply_adjustments(
        base_score,
        definition.adjustments,
        data,
        computed_ratios
    )

    # Step 5: Assign grade
    grade = definition.get_grade(final_score)

    # Step 6: Build summary
    summary = _build_summary(
        definition.name,
        ratio_value,
        final_score,
        grade,
        definition.unit,
        definition.description
    )

    # Track secondary ratios used in adjustments
    secondary = {
        adj.field: computed_ratios.get(adj.field) or _get_field_value(data, adj.field)
        for adj in definition.adjustments
    }

    return RatioResult(
        name=definition.name,
        value=ratio_value,
        score=final_score,
        grade=grade,
        summary=summary,
        secondary_ratios=secondary
    )


# ============================================================================
# RATIO CATALOG: Canonical Financial Ratios
# ============================================================================

RATIO_CATALOG: Dict[str, RatioDefinition] = {
    # --- PROFITABILITY RATIOS ---
    "roa": RatioDefinition(
        name="Return on Assets (ROA)",
        description="Measures how efficiently assets generate profit",
        numerator_field="net_income",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),  # Excellent: >15%
            (0.10, 8.0),   # Good: 10-15%
            (0.05, 6.0),   # Adequate: 5-10%
            (0.02, 4.0),   # Weak: 2-5%
        ],
        adjustments=[
            Adjustment("operating_income", Operator.GT, 0, 0.5, "Positive operating income"),
            Adjustment("total_debt", Operator.LT, 0.5, 0.5, "Low leverage"),
        ],
        unit="%"
    ),

    "roe": RatioDefinition(
        name="Return on Equity (ROE)",
        description="Measures return generated on shareholders' equity",
        numerator_field="net_income",
        denominator_field="total_equity",
        higher_is_better=True,
        scoring_thresholds=[
            (0.20, 10.0),  # Excellent: >20%
            (0.15, 8.0),   # Good: 15-20%
            (0.10, 6.0),   # Adequate: 10-15%
            (0.05, 4.0),   # Weak: 5-10%
        ],
        adjustments=[
            Adjustment("total_debt", Operator.GT, 2.0, -0.5, "High leverage inflates ROE"),
        ],
        unit="%"
    ),

    "roic": RatioDefinition(
        name="Return on Invested Capital (ROIC)",
        description="Measures return on total invested capital",
        numerator_field="ebit",
        denominator_field="total_assets",  # Simplified; should be invested capital
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),
            (0.10, 8.0),
            (0.07, 6.0),
            (0.05, 4.0),
        ],
        unit="%"
    ),

    "gross_margin": RatioDefinition(
        name="Gross Profit Margin",
        description="Measures profitability after direct costs",
        numerator_field="gross_profit",
        denominator_field="revenue",
        higher_is_better=True,
        scoring_thresholds=[
            (0.50, 10.0),  # >50%
            (0.35, 8.0),   # 35-50%
            (0.25, 6.0),   # 25-35%
            (0.15, 4.0),   # 15-25%
        ],
        unit="%"
    ),

    "operating_margin": RatioDefinition(
        name="Operating Profit Margin",
        description="Measures profitability from operations",
        numerator_field="operating_income",
        denominator_field="revenue",
        higher_is_better=True,
        scoring_thresholds=[
            (0.20, 10.0),  # >20%
            (0.15, 8.0),
            (0.10, 6.0),
            (0.05, 4.0),
        ],
        unit="%"
    ),

    "net_margin": RatioDefinition(
        name="Net Profit Margin",
        description="Measures bottom-line profitability",
        numerator_field="net_income",
        denominator_field="revenue",
        higher_is_better=True,
        scoring_thresholds=[
            (0.15, 10.0),  # >15%
            (0.10, 8.0),
            (0.05, 6.0),
            (0.02, 4.0),
        ],
        unit="%"
    ),

    # --- LIQUIDITY RATIOS ---
    "current_ratio": RatioDefinition(
        name="Current Ratio",
        description="Measures ability to meet short-term obligations",
        numerator_field="current_assets",
        denominator_field="current_liabilities",
        higher_is_better=True,
        scoring_thresholds=[
            (2.0, 10.0),   # Excellent: >2.0
            (1.5, 8.0),    # Good: 1.5-2.0
            (1.0, 6.0),    # Adequate: 1.0-1.5
            (0.75, 4.0),   # Weak: 0.75-1.0
        ],
        adjustments=[
            Adjustment("cash_and_equivalents", Operator.GT, 0.3, 0.5, "Strong cash position"),
        ],
        unit="x"
    ),

    "quick_ratio": RatioDefinition(
        name="Quick Ratio (Acid Test)",
        description="Measures liquidity excluding inventory",
        numerator_field="current_assets",  # Simplified; should subtract inventory
        denominator_field="current_liabilities",
        higher_is_better=True,
        scoring_thresholds=[
            (1.5, 10.0),
            (1.0, 8.0),
            (0.75, 6.0),
            (0.5, 4.0),
        ],
        unit="x"
    ),

    "cash_ratio": RatioDefinition(
        name="Cash Ratio",
        description="Most conservative liquidity measure",
        numerator_field="cash_and_equivalents",
        denominator_field="current_liabilities",
        higher_is_better=True,
        scoring_thresholds=[
            (0.75, 10.0),
            (0.50, 8.0),
            (0.30, 6.0),
            (0.15, 4.0),
        ],
        unit="x"
    ),

    # --- LEVERAGE RATIOS ---
    "debt_to_equity": RatioDefinition(
        name="Debt-to-Equity Ratio",
        description="Measures financial leverage",
        numerator_field="total_debt",
        denominator_field="total_equity",
        higher_is_better=False,  # Lower is better
        scoring_thresholds=[
            (0.3, 10.0),   # <0.3 Excellent
            (0.5, 8.0),    # 0.3-0.5 Good
            (1.0, 6.0),    # 0.5-1.0 Adequate
            (2.0, 4.0),    # 1.0-2.0 Weak
        ],
        adjustments=[
            Adjustment("ebitda", Operator.GT, 0, 0.5, "Positive cash generation"),
        ],
        unit="x"
    ),

    "debt_to_ebitda": RatioDefinition(
        name="Debt-to-EBITDA Ratio",
        description="Measures debt coverage by earnings",
        numerator_field="total_debt",
        denominator_field="ebitda",
        higher_is_better=False,
        scoring_thresholds=[
            (2.0, 10.0),   # <2x Excellent
            (3.0, 8.0),    # 2-3x Good
            (4.0, 6.0),    # 3-4x Adequate
            (5.0, 4.0),    # 4-5x Weak
        ],
        unit="x"
    ),

    "interest_coverage": RatioDefinition(
        name="Interest Coverage Ratio",
        description="Measures ability to pay interest",
        numerator_field="ebit",
        denominator_field="interest_expense",
        higher_is_better=True,
        scoring_thresholds=[
            (8.0, 10.0),   # >8x Excellent
            (5.0, 8.0),    # 5-8x Good
            (2.5, 6.0),    # 2.5-5x Adequate
            (1.5, 4.0),    # 1.5-2.5x Weak
        ],
        unit="x"
    ),

    # --- EFFICIENCY RATIOS ---
    "asset_turnover": RatioDefinition(
        name="Asset Turnover Ratio",
        description="Measures efficiency of asset utilization",
        numerator_field="revenue",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (2.0, 10.0),   # >2x Excellent
            (1.5, 8.0),
            (1.0, 6.0),
            (0.5, 4.0),
        ],
        unit="x"
    ),

    "inventory_turnover": RatioDefinition(
        name="Inventory Turnover",
        description="Measures how quickly inventory is sold",
        numerator_field="cogs",
        denominator_field="inventory",
        higher_is_better=True,
        scoring_thresholds=[
            (12.0, 10.0),  # >12x (monthly turnover)
            (8.0, 8.0),
            (6.0, 6.0),
            (4.0, 4.0),
        ],
        unit="x"
    ),

    "receivables_turnover": RatioDefinition(
        name="Receivables Turnover",
        description="Measures collection efficiency",
        numerator_field="revenue",
        denominator_field="accounts_receivable",
        higher_is_better=True,
        scoring_thresholds=[
            (12.0, 10.0),  # >12x
            (10.0, 8.0),
            (8.0, 6.0),
            (6.0, 4.0),
        ],
        unit="x"
    ),

    # --- CASH FLOW RATIOS ---
    "fcf_yield": RatioDefinition(
        name="Free Cash Flow Yield",
        description="FCF as % of enterprise value (simplified using assets)",
        numerator_field="operating_cash_flow",
        denominator_field="total_assets",
        higher_is_better=True,
        scoring_thresholds=[
            (0.10, 10.0),  # >10%
            (0.07, 8.0),
            (0.05, 6.0),
            (0.03, 4.0),
        ],
        unit="%"
    ),

    "ocf_to_ni": RatioDefinition(
        name="Operating Cash Flow to Net Income",
        description="Measures earnings quality",
        numerator_field="operating_cash_flow",
        denominator_field="net_income",
        higher_is_better=True,
        scoring_thresholds=[
            (1.2, 10.0),   # OCF > 120% of NI (excellent quality)
            (1.0, 8.0),    # OCF = NI
            (0.8, 6.0),    # OCF < NI (some concerns)
            (0.6, 4.0),
        ],
        unit="x"
    ),

    "cash_conversion_cycle": RatioDefinition(
        name="Cash Conversion Cycle",
        description="Days to convert operations to cash (simplified)",
        numerator_field="inventory",
        denominator_field="revenue",  # Simplified CCC proxy
        higher_is_better=False,  # Lower is better
        scoring_thresholds=[
            (0.05, 10.0),  # <5% of revenue
            (0.10, 8.0),
            (0.15, 6.0),
            (0.20, 4.0),
        ],
        unit="days"
    ),
}


def run_all_ratios(data: FinancialData) -> Dict[str, RatioResult]:
    """
    Compute all ratios in the catalog.

    Handles dependencies by computing in multiple passes:
    1. First pass: compute all ratios
    2. Adjustments can reference previously computed ratios

    Args:
        data: FinancialData instance

    Returns:
        Dict mapping ratio name to RatioResult
    """
    results: Dict[str, RatioResult] = {}
    computed_values: Dict[str, Optional[float]] = {}

    # First pass: compute all ratio values
    for ratio_key, definition in RATIO_CATALOG.items():
        result = compute_ratio(data, definition, computed_values)
        results[ratio_key] = result
        computed_values[ratio_key] = result.value

    # Second pass: recompute with full adjustment context
    # (Optional optimization if adjustments reference other catalog ratios)
    for ratio_key, definition in RATIO_CATALOG.items():
        if definition.adjustments:
            result = compute_ratio(data, definition, computed_values)
            results[ratio_key] = result

    return results


def get_ratio_by_category() -> Dict[str, List[str]]:
    """
    Group ratios by category for organized display.

    Returns:
        Dict mapping category name to list of ratio keys
    """
    categories = {
        "Profitability": ["roa", "roe", "roic", "gross_margin", "operating_margin", "net_margin"],
        "Liquidity": ["current_ratio", "quick_ratio", "cash_ratio"],
        "Leverage": ["debt_to_equity", "debt_to_ebitda", "interest_coverage"],
        "Efficiency": ["asset_turnover", "inventory_turnover", "receivables_turnover"],
        "Cash Flow": ["fcf_yield", "ocf_to_ni", "cash_conversion_cycle"],
    }
    return categories


def compute_category(data: FinancialData, category: str) -> Dict[str, RatioResult]:
    """
    Compute all ratios in a specific category.

    Args:
        data: FinancialData instance
        category: Category name (e.g., "Profitability", "Liquidity")

    Returns:
        Dict mapping ratio name to RatioResult for the category
    """
    categories = get_ratio_by_category()
    ratio_keys = categories.get(category, [])

    computed_values: Dict[str, Optional[float]] = {}
    results: Dict[str, RatioResult] = {}

    for ratio_key in ratio_keys:
        definition = RATIO_CATALOG.get(ratio_key)
        if definition:
            result = compute_ratio(data, definition, computed_values)
            results[ratio_key] = result
            computed_values[ratio_key] = result.value

    return results
