"""
Charlie-Style Financial Intelligence Module.
CFO-grade financial analysis with automatic metric extraction and insight generation.
Named after Charlie Munger who embodied the principles of rigorous financial analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely divide two numbers, returning *default* when division is impossible.

    Handles ``None`` inputs, zero denominators, and near-zero denominators
    (``abs(denominator) < 1e-12``) to avoid ``ZeroDivisionError`` and ``inf``
    results across all financial ratio calculations.
    """
    if numerator is None or denominator is None:
        return default
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


@dataclass
class VarianceResult:
    """Result of a variance calculation."""
    actual: float
    budget: float
    variance: float
    variance_percent: float
    favorable: bool
    category: str
    explanation: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    metric_name: str
    values: List[float]
    periods: List[str]
    yoy_growth: Optional[float]
    qoq_growth: Optional[float]
    mom_growth: Optional[float]
    cagr: Optional[float]
    moving_avg_3: List[float]
    moving_avg_12: List[float]
    trend_direction: str  # 'up', 'down', 'stable'
    seasonality_detected: bool


@dataclass
class CashFlowAnalysis:
    """Cash flow analysis results."""
    operating_cf: Optional[float]
    investing_cf: Optional[float]
    financing_cf: Optional[float]
    free_cash_flow: Optional[float]
    dso: Optional[float]  # Days Sales Outstanding
    dio: Optional[float]  # Days Inventory Outstanding
    dpo: Optional[float]  # Days Payables Outstanding
    cash_conversion_cycle: Optional[float]


@dataclass
class WorkingCapitalAnalysis:
    """Working capital analysis results."""
    current_assets: Optional[float]
    current_liabilities: Optional[float]
    net_working_capital: Optional[float]
    working_capital_ratio: Optional[float]
    working_capital_turnover: Optional[float]


@dataclass
class BudgetAnalysis:
    """Comprehensive budget analysis."""
    total_budget: float
    total_actual: float
    total_variance: float
    total_variance_percent: float
    line_items: List[VarianceResult]
    favorable_items: List[VarianceResult]
    unfavorable_items: List[VarianceResult]
    largest_variances: List[VarianceResult]


@dataclass
class Insight:
    """A generated insight."""
    category: str  # liquidity, profitability, efficiency, risk, trend
    severity: str  # info, warning, critical
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class Anomaly:
    """A detected anomaly."""
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    z_score: float
    description: str


@dataclass
class Forecast:
    """Simple forecast result."""
    metric_name: str
    historical_values: List[float]
    forecasted_values: List[float]
    forecast_periods: List[str]
    method: str
    confidence_interval: Tuple[float, float]


@dataclass
class DuPontAnalysis:
    """DuPont decomposition of Return on Equity.

    3-factor: ROE = Net Margin × Asset Turnover × Equity Multiplier
    5-factor: adds Tax Burden (NI/EBT) and Interest Burden (EBT/EBIT)
    """
    roe: Optional[float] = None
    net_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    equity_multiplier: Optional[float] = None
    # 5-factor extensions
    tax_burden: Optional[float] = None       # NI / EBT
    interest_burden: Optional[float] = None  # EBT / EBIT
    # Diagnosis
    primary_driver: Optional[str] = None  # which factor drives ROE most
    interpretation: Optional[str] = None


@dataclass
class AltmanZScore:
    """Altman Z-Score bankruptcy prediction model.

    Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
    Where:
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Value of Equity / Total Liabilities (book value used as proxy)
      X5 = Sales / Total Assets
    """
    z_score: Optional[float] = None
    zone: str = "unknown"  # 'safe', 'grey', 'distress'
    components: Dict[str, Optional[float]] = field(default_factory=dict)
    interpretation: Optional[str] = None


@dataclass
class PiotroskiFScore:
    """Piotroski F-Score financial strength model (0-9 scale).

    Profitability (4 points): ROA>0, Operating CF>0, Delta ROA>0, Accruals
    Leverage/Liquidity (3 points): Delta Leverage<0, Delta Current Ratio>0, No dilution
    Operating Efficiency (2 points): Delta Gross Margin>0, Delta Asset Turnover>0
    """
    score: int = 0
    max_score: int = 9
    criteria: Dict[str, bool] = field(default_factory=dict)
    interpretation: Optional[str] = None


@dataclass
class CompositeHealthScore:
    """Weighted 0-100 composite financial health score with letter grade.

    Components (100 total):
      - Z-Score zone:     25 pts
      - F-Score:          25 pts
      - Profitability:    20 pts
      - Liquidity:        15 pts
      - Leverage:         15 pts
    """
    score: int = 0
    grade: str = "F"  # A, B, C, D, F
    component_scores: Dict[str, int] = field(default_factory=dict)
    interpretation: Optional[str] = None


@dataclass
class PeriodComparison:
    """Result of comparing two financial periods."""
    current_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    prior_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    improvements: List[str] = field(default_factory=list)
    deteriorations: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of a what-if scenario analysis comparing base vs adjusted data."""
    scenario_name: str = ""
    adjustments: Dict[str, float] = field(default_factory=dict)
    base_health: Optional[CompositeHealthScore] = None
    scenario_health: Optional[CompositeHealthScore] = None
    base_z_score: Optional[float] = None
    scenario_z_score: Optional[float] = None
    base_f_score: Optional[int] = None
    scenario_f_score: Optional[int] = None
    base_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    scenario_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    impact_summary: str = ""


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis across a range of adjustments."""
    variable_name: str = ""
    variable_labels: List[str] = field(default_factory=list)
    variable_multipliers: List[float] = field(default_factory=list)
    metric_results: Dict[str, List[Optional[float]]] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo simulation on financial metrics."""
    n_simulations: int = 0
    variable_assumptions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metric_distributions: Dict[str, List[float]] = field(default_factory=dict)
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: str = ""


@dataclass
class CashFlowForecast:
    """Multi-period cash flow projection."""
    periods: List[str] = field(default_factory=list)
    revenue_forecast: List[float] = field(default_factory=list)
    expense_forecast: List[float] = field(default_factory=list)
    net_cash_flow: List[float] = field(default_factory=list)
    cumulative_cash: List[float] = field(default_factory=list)
    fcf_forecast: List[float] = field(default_factory=list)
    dcf_value: Optional[float] = None
    terminal_value: Optional[float] = None
    discount_rate: float = 0.10
    growth_rates: List[float] = field(default_factory=list)


@dataclass
class TornadoDriver:
    """A single driver's impact on a target metric."""
    variable: str = ""
    low_value: float = 0.0
    high_value: float = 0.0
    base_value: float = 0.0
    spread: float = 0.0  # high_value - low_value


@dataclass
class TornadoResult:
    """Result of tornado/driver ranking analysis."""
    target_metric: str = ""
    base_metric_value: float = 0.0
    drivers: List[TornadoDriver] = field(default_factory=list)
    top_driver: str = ""


@dataclass
class BreakevenResult:
    """Breakeven analysis result."""
    breakeven_revenue: Optional[float] = None
    current_revenue: Optional[float] = None
    margin_of_safety: Optional[float] = None  # (current - breakeven) / current
    fixed_costs: Optional[float] = None
    variable_cost_ratio: Optional[float] = None
    contribution_margin_ratio: Optional[float] = None


@dataclass
class CovenantCheck:
    """Result of a single covenant/KPI check."""
    name: str = ""
    current_value: Optional[float] = None
    threshold: float = 0.0
    direction: str = "above"  # 'above' = good if current >= threshold
    status: str = "unknown"   # 'pass', 'warning', 'breach'
    headroom: Optional[float] = None  # how far above/below threshold


@dataclass
class CovenantMonitorResult:
    """Result of covenant monitoring across all KPIs."""
    checks: List[CovenantCheck] = field(default_factory=list)
    passes: int = 0
    warnings: int = 0
    breaches: int = 0
    summary: str = ""


@dataclass
class WorkingCapitalResult:
    """Working capital efficiency metrics."""
    dso: Optional[float] = None   # Days Sales Outstanding
    dio: Optional[float] = None   # Days Inventory Outstanding
    dpo: Optional[float] = None   # Days Payables Outstanding
    ccc: Optional[float] = None   # Cash Conversion Cycle (DSO + DIO - DPO)
    net_working_capital: Optional[float] = None
    working_capital_ratio: Optional[float] = None
    insights: List[str] = field(default_factory=list)


@dataclass
class NarrativeReport:
    """Auto-generated narrative intelligence from financial analysis."""
    headline: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class RegressionForecast:
    """Result of regression-based trend forecasting."""
    metric_name: str = ""
    historical_values: List[float] = field(default_factory=list)
    forecast_values: List[float] = field(default_factory=list)
    r_squared: Optional[float] = None
    slope: Optional[float] = None
    intercept: Optional[float] = None
    method: str = "linear"  # linear, exponential, polynomial
    confidence_upper: List[float] = field(default_factory=list)
    confidence_lower: List[float] = field(default_factory=list)
    periods_ahead: int = 0


@dataclass
class BenchmarkComparison:
    """Single metric comparison against industry benchmark."""
    metric_name: str = ""
    company_value: Optional[float] = None
    industry_median: Optional[float] = None
    industry_p25: Optional[float] = None
    industry_p75: Optional[float] = None
    percentile_rank: Optional[float] = None  # 0-100, where company falls
    rating: str = ""  # "above average", "average", "below average"


@dataclass
class IndustryBenchmarkResult:
    """Full industry benchmarking result."""
    industry_name: str = ""
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    overall_percentile: Optional[float] = None
    summary: str = ""


@dataclass
class CustomKPIDefinition:
    """Definition of a user-defined KPI formula."""
    name: str = ""
    formula: str = ""  # e.g. "revenue / total_assets"
    description: str = ""
    target_min: Optional[float] = None
    target_max: Optional[float] = None


@dataclass
class CustomKPIResult:
    """Result of evaluating a custom KPI."""
    name: str = ""
    formula: str = ""
    value: Optional[float] = None
    meets_target: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class CustomKPIReport:
    """Collection of custom KPI evaluation results."""
    results: List[CustomKPIResult] = field(default_factory=list)
    summary: str = ""


@dataclass
class PeerCompanyData:
    """Financial data for a single peer in comparison."""
    name: str = ""
    data: Optional['FinancialData'] = None


@dataclass
class PeerMetricComparison:
    """A single metric compared across peers."""
    metric_name: str = ""
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    best_performer: str = ""
    worst_performer: str = ""
    average: Optional[float] = None
    median: Optional[float] = None


@dataclass
class PeerComparisonReport:
    """Full peer comparison analysis."""
    peer_names: List[str] = field(default_factory=list)
    comparisons: List[PeerMetricComparison] = field(default_factory=list)
    rankings: Dict[str, int] = field(default_factory=dict)
    summary: str = ""


@dataclass
class RatioDecompositionNode:
    """A node in the ratio decomposition tree."""
    name: str = ""
    value: Optional[float] = None
    formula: str = ""
    children: List['RatioDecompositionNode'] = field(default_factory=list)


@dataclass
class RatioDecompositionTree:
    """Full ratio decomposition from ROE down to individual drivers."""
    root: Optional[RatioDecompositionNode] = None
    summary: str = ""


@dataclass
class RatingCategory:
    """A scored category in the financial rating."""
    name: str = ""
    score: float = 0.0
    max_score: float = 10.0
    grade: str = ""
    details: str = ""


@dataclass
class FinancialRating:
    """Comprehensive financial rating scorecard."""
    overall_score: float = 0.0
    overall_grade: str = ""
    categories: List[RatingCategory] = field(default_factory=list)
    summary: str = ""


@dataclass
class WaterfallItem:
    """A single item in a variance waterfall."""
    label: str = ""
    value: float = 0.0
    cumulative: float = 0.0
    item_type: str = "delta"  # "start", "delta", "total"


@dataclass
class VarianceWaterfall:
    """Waterfall breakdown of variance between two periods."""
    start_value: float = 0.0
    end_value: float = 0.0
    items: List[WaterfallItem] = field(default_factory=list)
    total_variance: float = 0.0
    summary: str = ""


@dataclass
class EarningsQualityResult:
    """Earnings quality analysis result."""
    accrual_ratio: Optional[float] = None
    cash_to_income_ratio: Optional[float] = None
    quality_score: float = 0.0
    quality_grade: str = ""
    indicators: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class CapitalEfficiencyResult:
    """Capital efficiency and value creation analysis."""
    roic: Optional[float] = None
    invested_capital: Optional[float] = None
    nopat: Optional[float] = None
    eva: Optional[float] = None
    wacc_estimate: float = 0.10
    capital_turnover: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    summary: str = ""


@dataclass
class LiquidityStressResult:
    """Liquidity stress test result."""
    current_cash: float = 0.0
    monthly_burn: float = 0.0
    months_of_cash: Optional[float] = None
    stressed_quick_ratio: Optional[float] = None
    stress_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = ""  # "Low", "Moderate", "High", "Critical"
    summary: str = ""


@dataclass
class DebtServiceResult:
    """Debt service coverage analysis result."""
    dscr: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    total_debt: float = 0.0
    annual_debt_service: Optional[float] = None
    free_cash_after_service: Optional[float] = None
    risk_level: str = ""  # "Low", "Moderate", "High", "Critical"
    summary: str = ""


@dataclass
class HealthDimension:
    """A single dimension of the composite health score."""
    name: str = ""
    score: float = 0.0
    max_score: float = 100.0
    weight: float = 0.0
    status: str = ""  # "green", "yellow", "red"
    detail: str = ""


@dataclass
class ComprehensiveHealthResult:
    """Comprehensive financial health score aggregating all analyses."""
    overall_score: float = 0.0
    grade: str = ""  # A+ through F
    dimensions: List[HealthDimension] = field(default_factory=list)
    summary: str = ""


@dataclass
class OperatingLeverageResult:
    """Operating leverage and break-even analysis result."""
    degree_of_operating_leverage: Optional[float] = None
    contribution_margin: Optional[float] = None
    contribution_margin_ratio: Optional[float] = None
    estimated_fixed_costs: Optional[float] = None
    estimated_variable_costs: Optional[float] = None
    break_even_revenue: Optional[float] = None
    margin_of_safety: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None
    cost_structure: str = ""  # "High Fixed", "Balanced", "High Variable"
    summary: str = ""


@dataclass
class CashFlowQualityResult:
    """Cash flow quality and free cash flow analysis result."""
    fcf: Optional[float] = None
    fcf_yield: Optional[float] = None
    fcf_margin: Optional[float] = None
    ocf_to_net_income: Optional[float] = None
    capex_to_revenue: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    cash_conversion_efficiency: Optional[float] = None
    quality_grade: str = ""  # "Strong", "Adequate", "Weak", "Poor"
    indicators: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class AssetEfficiencyResult:
    """Asset efficiency and turnover analysis result."""
    total_asset_turnover: Optional[float] = None
    fixed_asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    payables_turnover: Optional[float] = None
    working_capital_turnover: Optional[float] = None
    equity_turnover: Optional[float] = None
    efficiency_score: float = 0.0  # 0-10
    efficiency_grade: str = ""  # "Excellent", "Good", "Average", "Below Average"
    summary: str = ""


@dataclass
class ProfitabilityDecompResult:
    """Profitability decomposition and return analysis result."""
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    invested_capital: Optional[float] = None
    nopat: Optional[float] = None
    spread: Optional[float] = None  # ROIC - cost of capital proxy
    economic_profit: Optional[float] = None  # Spread × Invested Capital
    asset_turnover: Optional[float] = None
    financial_leverage: Optional[float] = None  # TA / Equity
    tax_efficiency: Optional[float] = None  # NI / EBT
    capital_intensity: Optional[float] = None  # TA / Revenue
    profitability_score: float = 0.0  # 0-10
    profitability_grade: str = ""  # "Elite", "Strong", "Adequate", "Poor"
    summary: str = ""


@dataclass
class RiskAdjustedResult:
    """Risk-adjusted performance metrics result."""
    return_on_risk: Optional[float] = None  # ROE / leverage ratio
    risk_adjusted_roe: Optional[float] = None  # ROE adjusted for financial risk
    risk_adjusted_roa: Optional[float] = None  # ROA adjusted for operating risk
    debt_adjusted_return: Optional[float] = None  # NI / (Equity + Debt)
    volatility_proxy: Optional[float] = None  # Margin variability proxy
    downside_risk: Optional[float] = None  # Probability of loss given leverage
    margin_of_safety: Optional[float] = None  # (EBIT - Interest) / Interest
    return_per_unit_risk: Optional[float] = None  # Return / Risk ratio
    risk_score: float = 0.0  # 0-10
    risk_grade: str = ""  # "Superior", "Favorable", "Neutral", "Elevated"
    summary: str = ""


@dataclass
class CapitalStructureResult:
    """Capital structure analysis result."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    equity_multiplier: Optional[float] = None  # TA / Equity
    interest_coverage: Optional[float] = None  # EBIT / Interest
    debt_service_coverage: Optional[float] = None  # OCF / (Interest + Debt payments)
    net_debt: Optional[float] = None  # Total Debt - Cash
    net_debt_to_ebitda: Optional[float] = None
    capitalization_rate: Optional[float] = None  # Debt / (Debt + Equity)
    equity_ratio: Optional[float] = None  # Equity / TA
    wacc_estimate: Optional[float] = None  # Simplified WACC
    optimal_leverage_distance: Optional[float] = None  # Distance from optimal D/E
    capital_score: float = 0.0  # 0-10
    capital_grade: str = ""  # "Conservative", "Balanced", "Aggressive", "Distressed"
    summary: str = ""


@dataclass
class ValuationIndicatorsResult:
    """Valuation indicators and intrinsic value estimates."""
    earnings_yield: Optional[float] = None  # NI / Total Equity (proxy for E/P)
    book_value_per_share_proxy: Optional[float] = None  # Equity value
    ev_proxy: Optional[float] = None  # Equity + Debt - Cash
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    ev_to_ebit: Optional[float] = None
    price_to_book_proxy: Optional[float] = None  # (Equity + premium) / Book value
    return_on_invested_capital: Optional[float] = None
    graham_number_proxy: Optional[float] = None  # sqrt(22.5 * EPS_proxy * BV_proxy)
    intrinsic_value_proxy: Optional[float] = None  # DCF-lite estimate
    margin_of_safety_pct: Optional[float] = None  # (intrinsic - book) / intrinsic
    valuation_score: float = 0.0  # 0-10
    valuation_grade: str = ""  # "Undervalued", "Fair Value", "Fully Valued", "Overvalued"
    summary: str = ""


@dataclass
class SustainableGrowthResult:
    """Sustainability and growth capacity analysis result."""
    sustainable_growth_rate: Optional[float] = None  # ROE × retention ratio
    internal_growth_rate: Optional[float] = None  # ROA × retention / (1 - ROA × retention)
    retention_ratio: Optional[float] = None  # 1 - payout ratio
    payout_ratio: Optional[float] = None  # Dividends / NI
    plowback_capacity: Optional[float] = None  # Retained earnings / NI
    roe: Optional[float] = None
    roa: Optional[float] = None
    asset_growth_rate: Optional[float] = None  # Growth in TA if prior available
    equity_growth_rate: Optional[float] = None  # Retained earnings / Equity
    reinvestment_rate: Optional[float] = None  # Capex / Depreciation
    growth_profitability_balance: Optional[float] = None  # SGR / cost of equity proxy
    growth_score: float = 0.0  # 0-10
    growth_grade: str = ""  # "High Growth", "Sustainable", "Moderate", "Constrained"
    summary: str = ""


@dataclass
class ConcentrationRiskResult:
    """Concentration and structural risk analysis result."""
    revenue_asset_intensity: Optional[float] = None  # Revenue / TA (asset turnover)
    operating_dependency: Optional[float] = None  # OI / Revenue
    asset_composition_current: Optional[float] = None  # CA / TA
    asset_composition_fixed: Optional[float] = None  # (TA - CA) / TA
    liability_structure_current: Optional[float] = None  # CL / TL
    earnings_retention_ratio: Optional[float] = None  # NI / EBITDA
    working_capital_intensity: Optional[float] = None  # NWC / Revenue
    capex_intensity: Optional[float] = None  # Capex / Revenue
    interest_burden: Optional[float] = None  # Interest / EBIT
    cash_conversion_efficiency: Optional[float] = None  # OCF / NI
    fixed_asset_ratio: Optional[float] = None  # Fixed assets / Equity
    debt_concentration: Optional[float] = None  # Total debt / TL
    concentration_score: float = 0.0  # 0-10
    concentration_grade: str = ""  # "Well Diversified", "Balanced", "Concentrated", "Highly Concentrated"
    summary: str = ""


@dataclass
class MarginOfSafetyResult:
    """Margin of safety and intrinsic value cushion analysis result."""
    earnings_yield: Optional[float] = None  # NI / Market Cap or EPS/Price
    book_value_per_share: Optional[float] = None  # Equity / Shares
    price_to_book: Optional[float] = None  # Price / BV per share
    book_value_discount: Optional[float] = None  # 1 - P/B (positive = trading below BV)
    intrinsic_value_estimate: Optional[float] = None  # NI / discount rate (capitalized earnings)
    market_cap: Optional[float] = None  # Shares * Price
    intrinsic_margin: Optional[float] = None  # (IV - MC) / IV
    tangible_bv: Optional[float] = None  # Equity - Intangibles (approx)
    liquidation_value: Optional[float] = None  # Tangible BV * discount
    net_current_asset_value: Optional[float] = None  # CA - TL (Graham NCAV)
    ncav_per_share: Optional[float] = None  # NCAV / shares
    safety_score: float = 0.0  # 0-10
    safety_grade: str = ""  # "Wide Margin", "Adequate", "Thin", "No Margin"
    summary: str = ""


@dataclass
class FinancialFlexibilityResult:
    """Financial flexibility and adaptive capacity analysis result."""
    cash_to_assets: Optional[float] = None  # Cash / TA
    cash_to_debt: Optional[float] = None  # Cash / Total Debt
    cash_to_revenue: Optional[float] = None  # Cash / Revenue
    fcf_to_revenue: Optional[float] = None  # FCF / Revenue
    fcf_margin: Optional[float] = None  # FCF / Revenue (alias)
    spare_borrowing_capacity: Optional[float] = None  # (TA * 0.50 - Debt) / TA
    unencumbered_assets: Optional[float] = None  # TA - TL
    financial_slack: Optional[float] = None  # (Cash + FCF) / Total Debt
    debt_headroom: Optional[float] = None  # Max additional debt at 3x EBITDA - current debt
    retained_earnings_ratio: Optional[float] = None  # RE / Equity
    flexibility_score: float = 0.0  # 0-10
    flexibility_grade: str = ""  # "Highly Flexible", "Flexible", "Constrained", "Rigid"
    summary: str = ""


@dataclass
class AltmanZScoreResult:
    """Altman Z-Score bankruptcy prediction result."""
    # Component ratios
    working_capital_to_assets: Optional[float] = None  # (CA - CL) / TA
    retained_earnings_to_assets: Optional[float] = None  # RE / TA
    ebit_to_assets: Optional[float] = None  # EBIT / TA
    equity_to_liabilities: Optional[float] = None  # Equity / TL
    revenue_to_assets: Optional[float] = None  # Revenue / TA
    # Weighted components
    x1_weighted: Optional[float] = None  # 1.2 * WC/TA
    x2_weighted: Optional[float] = None  # 1.4 * RE/TA
    x3_weighted: Optional[float] = None  # 3.3 * EBIT/TA
    x4_weighted: Optional[float] = None  # 0.6 * Equity/TL
    x5_weighted: Optional[float] = None  # 1.0 * Rev/TA
    z_score: Optional[float] = None  # Total Z-Score
    z_zone: str = ""  # "Safe", "Gray", "Distress"
    altman_score: float = 0.0  # 0-10 normalized score
    altman_grade: str = ""  # "Strong", "Adequate", "Watch", "Critical"
    summary: str = ""


@dataclass
class PiotroskiFScoreResult:
    """Piotroski F-Score value investing screen (adapted for single-period)."""
    # Profitability signals
    roa_positive: Optional[bool] = None  # NI / TA > 0
    ocf_positive: Optional[bool] = None  # OCF > 0
    accruals_negative: Optional[bool] = None  # OCF > NI (cash > accrual)
    # Leverage / Liquidity signals
    current_ratio_above_1: Optional[bool] = None  # CA / CL > 1
    low_leverage: Optional[bool] = None  # Debt/TA < 0.5
    # Efficiency signals
    gross_margin_healthy: Optional[bool] = None  # Gross Margin > 0.20
    asset_turnover_adequate: Optional[bool] = None  # Rev/TA > 0.5
    # Computed metrics
    roa: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_assets: Optional[float] = None
    gross_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    # Score
    f_score: int = 0  # 0-7 (adapted; original is 0-9 with prior-period deltas)
    f_score_max: int = 7  # Max possible
    piotroski_grade: str = ""  # "Strong Value", "Moderate Value", "Weak", "Avoid"
    summary: str = ""


@dataclass
class InterestCoverageResult:
    """Interest coverage and debt capacity analysis result."""
    ebit_coverage: Optional[float] = None  # EBIT / Interest
    ebitda_coverage: Optional[float] = None  # EBITDA / Interest
    debt_to_ebitda: Optional[float] = None  # Total Debt / EBITDA
    ocf_to_debt: Optional[float] = None  # OCF / Total Debt
    fcf_to_debt: Optional[float] = None  # FCF / Total Debt
    fixed_charge_coverage: Optional[float] = None  # (EBIT + Lease) / (Interest + Lease)
    max_debt_capacity: Optional[float] = None  # 3x EBITDA (theoretical)
    spare_debt_capacity: Optional[float] = None  # Max - Current Debt
    interest_to_revenue: Optional[float] = None  # Interest / Revenue
    debt_to_equity: Optional[float] = None  # Total Debt / Equity
    coverage_score: float = 0.0  # 0-10
    coverage_grade: str = ""  # "Excellent", "Adequate", "Strained", "Critical"
    summary: str = ""


@dataclass
class WACCResult:
    """Weighted Average Cost of Capital estimation from financial statements."""
    cost_of_debt: Optional[float] = None  # Interest / Debt
    after_tax_cost_of_debt: Optional[float] = None  # Rd * (1 - T)
    implied_cost_of_equity: Optional[float] = None  # Estimated from ROE/earnings yield
    effective_tax_rate: Optional[float] = None  # Tax proxy
    debt_weight: Optional[float] = None  # D / (D + E)
    equity_weight: Optional[float] = None  # E / (D + E)
    wacc: Optional[float] = None  # Blended cost of capital
    debt_to_total_capital: Optional[float] = None  # Debt / (Debt + Equity)
    equity_to_total_capital: Optional[float] = None  # Equity / (Debt + Equity)
    total_capital: Optional[float] = None  # Debt + Equity
    wacc_score: float = 0.0  # 0-10
    wacc_grade: str = ""  # "Excellent", "Good", "Fair", "Expensive"
    summary: str = ""


@dataclass
class EVAResult:
    """Economic Value Added analysis."""
    nopat: Optional[float] = None  # Net Operating Profit After Tax
    invested_capital: Optional[float] = None  # Total Assets - Current Liabilities (or Debt + Equity)
    wacc_used: Optional[float] = None  # WACC estimate used
    capital_charge: Optional[float] = None  # Invested Capital * WACC
    eva: Optional[float] = None  # NOPAT - Capital Charge
    eva_margin: Optional[float] = None  # EVA / Revenue
    roic: Optional[float] = None  # NOPAT / Invested Capital
    roic_wacc_spread: Optional[float] = None  # ROIC - WACC
    eva_score: float = 0.0  # 0-10
    eva_grade: str = ""  # "Value Creator", "Adequate", "Marginal", "Value Destroyer"
    summary: str = ""


@dataclass
class FCFYieldResult:
    """Free Cash Flow yield and quality metrics."""
    fcf: Optional[float] = None  # OCF - CapEx
    fcf_margin: Optional[float] = None  # FCF / Revenue
    fcf_to_net_income: Optional[float] = None  # FCF / NI (conversion quality)
    fcf_yield_on_capital: Optional[float] = None  # FCF / Invested Capital
    fcf_yield_on_equity: Optional[float] = None  # FCF / Equity
    fcf_to_debt: Optional[float] = None  # FCF / Total Debt
    capex_to_ocf: Optional[float] = None  # CapEx / OCF (reinvestment rate)
    capex_to_revenue: Optional[float] = None  # CapEx / Revenue (capital intensity)
    fcf_score: float = 0.0  # 0-10
    fcf_grade: str = ""  # "Strong", "Healthy", "Weak", "Negative"
    summary: str = ""


@dataclass
class CashConversionResult:
    """Phase 41: Cash Conversion Efficiency."""
    dso: Optional[float] = None  # Days Sales Outstanding
    dio: Optional[float] = None  # Days Inventory Outstanding
    dpo: Optional[float] = None  # Days Payable Outstanding
    ccc: Optional[float] = None  # Cash Conversion Cycle (DSO+DIO-DPO)
    cash_to_revenue: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    ocf_to_ebitda: Optional[float] = None
    cash_conversion_score: float = 0.0  # 0-10
    cash_conversion_grade: str = ""  # "Excellent", "Good", "Fair", "Poor"
    summary: str = ""


@dataclass
class ProfitRetentionPowerResult:
    """Phase 356: Profit Retention Power Analysis."""
    prp_ratio: Optional[float] = None
    re_to_equity: Optional[float] = None
    re_to_revenue: Optional[float] = None
    re_growth_capacity: Optional[float] = None
    retention_rate: Optional[float] = None
    prp_spread: Optional[float] = None
    prp_score: float = 0.0
    prp_grade: str = ""
    summary: str = ""


@dataclass
class EarningsToDebtResult:
    """Phase 353: Earnings To Debt Analysis."""
    etd_ratio: Optional[float] = None
    ni_to_interest: Optional[float] = None
    ni_to_liabilities: Optional[float] = None
    earnings_yield_on_debt: Optional[float] = None
    debt_years_from_earnings: Optional[float] = None
    etd_spread: Optional[float] = None
    etd_score: float = 0.0
    etd_grade: str = ""
    summary: str = ""


@dataclass
class RevenueGrowthResult:
    """Phase 350: Revenue Growth Capacity Analysis."""
    rg_capacity: Optional[float] = None
    roe: Optional[float] = None
    plowback: Optional[float] = None
    sustainable_growth: Optional[float] = None
    revenue_per_asset: Optional[float] = None
    rg_spread: Optional[float] = None
    rg_score: float = 0.0
    rg_grade: str = ""
    summary: str = ""


@dataclass
class OperatingMarginResult:
    """Phase 349: Operating Margin Analysis."""
    operating_margin: Optional[float] = None
    oi_to_revenue: Optional[float] = None
    ebit_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    margin_trend: Optional[float] = None
    opm_spread: Optional[float] = None
    opm_score: float = 0.0
    opm_grade: str = ""
    summary: str = ""


@dataclass
class DebtToEquityResult:
    """Phase 348: Debt To Equity Analysis."""
    dte_ratio: Optional[float] = None
    td_to_te: Optional[float] = None
    lt_debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    equity_multiplier: Optional[float] = None
    dte_spread: Optional[float] = None
    dte_score: float = 0.0
    dte_grade: str = ""
    summary: str = ""


@dataclass
class CashFlowToDebtResult:
    """Phase 347: Cash Flow To Debt Analysis."""
    cf_to_debt: Optional[float] = None
    ocf_to_td: Optional[float] = None
    fcf_to_td: Optional[float] = None
    debt_payback_years: Optional[float] = None
    ocf_to_interest: Optional[float] = None
    cf_debt_spread: Optional[float] = None
    cfd_score: float = 0.0
    cfd_grade: str = ""
    summary: str = ""


@dataclass
class NetWorthGrowthResult:
    """Phase 346: Net Worth Growth Analysis."""
    nw_growth_ratio: Optional[float] = None
    re_to_equity: Optional[float] = None
    equity_to_assets: Optional[float] = None
    ni_to_equity: Optional[float] = None
    plowback_rate: Optional[float] = None
    nw_spread: Optional[float] = None
    nwg_score: float = 0.0
    nwg_grade: str = ""
    summary: str = ""


@dataclass
class AssetLightnessResult:
    """Phase 341: Asset Lightness Analysis."""
    lightness_ratio: Optional[float] = None
    ca_to_ta: Optional[float] = None
    revenue_to_assets: Optional[float] = None
    intangible_intensity: Optional[float] = None
    fixed_asset_ratio: Optional[float] = None
    lightness_spread: Optional[float] = None
    alt_score: float = 0.0
    alt_grade: str = ""
    summary: str = ""


@dataclass
class InternalGrowthRateResult:
    """Phase 337: Internal Growth Rate Analysis."""
    igr: Optional[float] = None
    roa: Optional[float] = None
    retention_ratio: Optional[float] = None
    roa_times_b: Optional[float] = None
    sustainable_growth: Optional[float] = None
    growth_capacity: Optional[float] = None
    igr_score: float = 0.0
    igr_grade: str = ""
    summary: str = ""


@dataclass
class OperatingExpenseRatioResult:
    """Phase 330: Operating Expense Ratio Analysis."""
    opex_ratio: Optional[float] = None
    opex_per_revenue: Optional[float] = None
    opex_to_gross_profit: Optional[float] = None
    opex_to_ebitda: Optional[float] = None
    opex_coverage: Optional[float] = None
    efficiency_gap: Optional[float] = None
    oer_score: float = 0.0
    oer_grade: str = ""
    summary: str = ""


@dataclass
class NoncurrentAssetRatioResult:
    """Phase 327: Noncurrent Asset Ratio Analysis."""
    nca_ratio: Optional[float] = None
    current_asset_ratio: Optional[float] = None
    nca_to_equity: Optional[float] = None
    nca_to_debt: Optional[float] = None
    asset_structure_spread: Optional[float] = None
    liquidity_complement: Optional[float] = None
    nar_score: float = 0.0
    nar_grade: str = ""
    summary: str = ""


@dataclass
class PayoutResilienceResult:
    """Phase 317: Payout Resilience Analysis."""
    div_to_ni: Optional[float] = None
    div_to_ocf: Optional[float] = None
    div_to_revenue: Optional[float] = None
    div_to_ebitda: Optional[float] = None
    payout_ratio: Optional[float] = None
    resilience_buffer: Optional[float] = None
    prs_score: float = 0.0
    prs_grade: str = ""
    summary: str = ""


@dataclass
class DebtBurdenIndexResult:
    """Phase 314: Debt Burden Index Analysis."""
    debt_to_ebitda: Optional[float] = None
    debt_to_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_revenue: Optional[float] = None
    debt_ratio: Optional[float] = None
    burden_intensity: Optional[float] = None
    dbi_score: float = 0.0
    dbi_grade: str = ""
    summary: str = ""


@dataclass
class InventoryCoverageResult:
    """Phase 309: Inventory Coverage Analysis."""
    inventory_to_revenue: Optional[float] = None
    inventory_to_cogs: Optional[float] = None
    inventory_to_assets: Optional[float] = None
    inventory_to_current_assets: Optional[float] = None
    inventory_days: Optional[float] = None
    inventory_buffer: Optional[float] = None
    icv_score: float = 0.0
    icv_grade: str = ""
    summary: str = ""


@dataclass
class CapexToRevenueResult:
    """Phase 307: CapEx to Revenue Analysis."""
    capex_to_revenue: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    capex_to_ebitda: Optional[float] = None
    capex_to_assets: Optional[float] = None
    investment_intensity: Optional[float] = None
    capex_yield: Optional[float] = None
    ctr_score: float = 0.0
    ctr_grade: str = ""
    summary: str = ""


@dataclass
class InventoryHoldingCostResult:
    """Phase 294: Inventory Holding Cost Analysis."""
    inventory_to_revenue: Optional[float] = None
    inventory_to_current_assets: Optional[float] = None
    inventory_to_total_assets: Optional[float] = None
    inventory_days: Optional[float] = None
    inventory_carrying_cost: Optional[float] = None
    inventory_intensity: Optional[float] = None
    ihc_score: float = 0.0
    ihc_grade: str = ""
    summary: str = ""


@dataclass
class FundingMixBalanceResult:
    """Phase 293: Funding Mix Balance Analysis."""
    equity_to_total_capital: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_total_capital: Optional[float] = None
    equity_multiplier: Optional[float] = None
    leverage_headroom: Optional[float] = None
    funding_stability: Optional[float] = None
    fmb_score: float = 0.0
    fmb_grade: str = ""
    summary: str = ""


@dataclass
class ExpenseRatioDisciplineResult:
    """Phase 292: Expense Ratio Discipline Analysis."""
    opex_to_revenue: Optional[float] = None
    cogs_to_revenue: Optional[float] = None
    sga_to_revenue: Optional[float] = None
    total_expense_ratio: Optional[float] = None
    operating_margin: Optional[float] = None
    expense_efficiency: Optional[float] = None
    erd_score: float = 0.0
    erd_grade: str = ""
    summary: str = ""


@dataclass
class RevenueCashRealizationResult:
    """Phase 291: Revenue Cash Realization Analysis."""
    cash_to_revenue: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    collection_rate: Optional[float] = None
    revenue_cash_gap: Optional[float] = None
    cash_conversion_speed: Optional[float] = None
    revenue_quality_ratio: Optional[float] = None
    rcr_score: float = 0.0
    rcr_grade: str = ""
    summary: str = ""


@dataclass
class NetDebtPositionResult:
    """Phase 286: Net Debt Position Analysis."""
    net_debt: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    net_debt_to_equity: Optional[float] = None
    net_debt_to_assets: Optional[float] = None
    cash_to_debt: Optional[float] = None
    net_debt_to_ocf: Optional[float] = None
    ndp_score: float = 0.0
    ndp_grade: str = ""
    summary: str = ""


@dataclass
class LiabilityCoverageStrengthResult:
    """Phase 281: Liability Coverage Strength Analysis."""
    ocf_to_liabilities: Optional[float] = None
    ebitda_to_liabilities: Optional[float] = None
    assets_to_liabilities: Optional[float] = None
    equity_to_liabilities: Optional[float] = None
    liability_to_revenue: Optional[float] = None
    liability_burden: Optional[float] = None
    lcs_score: float = 0.0
    lcs_grade: str = ""
    summary: str = ""


@dataclass
class CapitalAdequacyResult:
    """Phase 279: Capital Adequacy Analysis."""
    equity_ratio: Optional[float] = None
    equity_to_debt: Optional[float] = None
    retained_to_equity: Optional[float] = None
    equity_to_liabilities: Optional[float] = None
    tangible_equity_ratio: Optional[float] = None
    capital_buffer: Optional[float] = None
    caq_score: float = 0.0
    caq_grade: str = ""
    summary: str = ""


@dataclass
class OperatingIncomeQualityResult:
    """Phase 275: Operating Income Quality Analysis."""
    oi_to_revenue: Optional[float] = None
    oi_to_ebitda: Optional[float] = None
    oi_to_ocf: Optional[float] = None
    oi_to_total_assets: Optional[float] = None
    operating_spread: Optional[float] = None
    oi_cash_backing: Optional[float] = None
    oiq_score: float = 0.0
    oiq_grade: str = ""
    summary: str = ""


@dataclass
class EbitdaToDebtCoverageResult:
    """Phase 274: EBITDA-to-Debt Coverage Analysis."""
    ebitda_to_debt: Optional[float] = None
    ebitda_to_interest: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    ebitda_to_total_liabilities: Optional[float] = None
    debt_service_buffer: Optional[float] = None
    leverage_headroom: Optional[float] = None
    etdc_score: float = 0.0
    etdc_grade: str = ""
    summary: str = ""


@dataclass
class DebtQualityResult:
    """Phase 267: Debt Quality Assessment."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    long_term_debt_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_cost: Optional[float] = None
    dq_score: float = 0.0
    dq_grade: str = ""
    summary: str = ""


@dataclass
class FixedAssetProductivityResult:
    """Phase 263: Fixed Asset Productivity Analysis."""
    fixed_asset_turnover: Optional[float] = None
    revenue_per_fixed_asset: Optional[float] = None
    fixed_to_total_assets: Optional[float] = None
    capex_to_fixed_assets: Optional[float] = None
    depreciation_to_fixed: Optional[float] = None
    net_fixed_asset_intensity: Optional[float] = None
    fap_score: float = 0.0
    fap_grade: str = ""
    summary: str = ""


@dataclass
class DepreciationBurdenResult:
    """Phase 259: Depreciation Burden Analysis."""
    dep_to_revenue: Optional[float] = None
    dep_to_assets: Optional[float] = None
    dep_to_ebitda: Optional[float] = None
    dep_to_gross_profit: Optional[float] = None
    ebitda_to_ebit_spread: Optional[float] = None
    asset_age_proxy: Optional[float] = None
    db_score: float = 0.0
    db_grade: str = ""
    summary: str = ""


@dataclass
class DebtToCapitalResult:
    """Phase 258: Debt-to-Capital Analysis."""
    debt_to_capital: Optional[float] = None
    debt_to_equity: Optional[float] = None
    long_term_debt_to_capital: Optional[float] = None
    equity_ratio: Optional[float] = None
    net_debt_to_capital: Optional[float] = None
    financial_risk_index: Optional[float] = None
    dtc_score: float = 0.0
    dtc_grade: str = ""
    summary: str = ""


@dataclass
class DividendPayoutResult:
    """Phase 251: Dividend Payout Ratio Analysis."""
    div_to_ni: Optional[float] = None
    retention_ratio: Optional[float] = None
    div_to_ocf: Optional[float] = None
    div_to_fcf: Optional[float] = None
    div_to_revenue: Optional[float] = None
    div_coverage: Optional[float] = None
    dpr_score: float = 0.0
    dpr_grade: str = ""
    summary: str = ""


@dataclass
class OperatingCashFlowRatioResult:
    """Phase 249: Operating Cash Flow Ratio Analysis."""
    ocf_to_cl: Optional[float] = None
    ocf_to_tl: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    ocf_to_ni: Optional[float] = None
    ocf_to_debt: Optional[float] = None
    ocf_margin: Optional[float] = None
    ocfr_score: float = 0.0
    ocfr_grade: str = ""
    summary: str = ""


@dataclass
class CashConversionCycleResult:
    """Phase 248: Cash Conversion Cycle Analysis."""
    dso: Optional[float] = None
    dio: Optional[float] = None
    dpo: Optional[float] = None
    ccc: Optional[float] = None
    ccc_to_revenue: Optional[float] = None
    working_cap_days: Optional[float] = None
    ccc_score: float = 0.0
    ccc_grade: str = ""
    summary: str = ""


@dataclass
class InventoryTurnoverResult:
    """Phase 247: Inventory Turnover Analysis."""
    cogs_to_inv: Optional[float] = None
    dio: Optional[float] = None
    inv_to_ca: Optional[float] = None
    inv_to_ta: Optional[float] = None
    inv_to_revenue: Optional[float] = None
    inv_velocity: Optional[float] = None
    ito_score: float = 0.0
    ito_grade: str = ""
    summary: str = ""


@dataclass
class PayablesTurnoverResult:
    """Phase 246: Payables Turnover Analysis."""
    cogs_to_ap: Optional[float] = None
    dpo: Optional[float] = None
    ap_to_cl: Optional[float] = None
    ap_to_tl: Optional[float] = None
    ap_to_cogs: Optional[float] = None
    payment_velocity: Optional[float] = None
    pto_score: float = 0.0
    pto_grade: str = ""
    summary: str = ""


@dataclass
class ReceivablesTurnoverResult:
    """Phase 245: Receivables Turnover Analysis."""
    rev_to_ar: Optional[float] = None
    dso: Optional[float] = None
    ar_to_ca: Optional[float] = None
    ar_to_ta: Optional[float] = None
    ar_to_revenue: Optional[float] = None
    collection_efficiency: Optional[float] = None
    rto_score: float = 0.0
    rto_grade: str = ""
    summary: str = ""


@dataclass
class CashConversionEfficiencyResult:
    """Phase 237: Cash Conversion Efficiency Analysis."""
    ocf_to_oi: Optional[float] = None
    ocf_to_ni: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    ocf_to_ebitda: Optional[float] = None
    fcf_to_oi: Optional[float] = None
    cash_to_oi: Optional[float] = None
    cce_score: float = 0.0
    cce_grade: str = ""
    summary: str = ""


@dataclass
class FixedCostLeverageRatioResult:
    """Phase 236: Fixed Cost Leverage Ratio Analysis."""
    dol: Optional[float] = None
    contribution_margin: Optional[float] = None
    oi_to_revenue: Optional[float] = None
    cogs_to_revenue: Optional[float] = None
    opex_to_revenue: Optional[float] = None
    breakeven_proxy: Optional[float] = None
    fclr_score: float = 0.0
    fclr_grade: str = ""
    summary: str = ""


@dataclass
class RevenueQualityIndexResult:
    """Phase 232: Revenue Quality Index Analysis."""
    ocf_to_revenue: Optional[float] = None
    gross_margin: Optional[float] = None
    ni_to_revenue: Optional[float] = None
    ebitda_to_revenue: Optional[float] = None
    ar_to_revenue: Optional[float] = None
    cash_to_revenue: Optional[float] = None
    rqi_score: float = 0.0
    rqi_grade: str = ""
    summary: str = ""


@dataclass
class FixedAssetUtilizationResult:
    """Phase 221: Fixed Asset Utilization Analysis."""
    fixed_asset_turnover: Optional[float] = None
    depreciation_to_revenue: Optional[float] = None
    capex_to_depreciation: Optional[float] = None
    capex_to_total_assets: Optional[float] = None
    fixed_to_total_assets: Optional[float] = None
    depreciation_to_total_assets: Optional[float] = None
    fau_score: float = 0.0
    fau_grade: str = ""
    summary: str = ""


@dataclass
class CostControlResult:
    """Phase 215: Cost Control Analysis."""
    opex_to_revenue: Optional[float] = None
    cogs_to_revenue: Optional[float] = None
    sga_to_revenue: Optional[float] = None
    operating_margin: Optional[float] = None
    opex_to_gross_profit: Optional[float] = None
    ebitda_margin: Optional[float] = None
    cc_score: float = 0.0
    cc_grade: str = ""
    summary: str = ""


@dataclass
class ValuationSignalResult:
    """Phase 212: Valuation Signal Analysis."""
    ev_to_ebitda: Optional[float] = None
    price_to_earnings: Optional[float] = None
    price_to_book: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    earnings_yield: Optional[float] = None
    fcf_yield: Optional[float] = None
    vsg_score: float = 0.0
    vsg_grade: str = ""
    summary: str = ""


@dataclass
class CapitalDisciplineResult:
    """Phase 211: Capital Discipline Analysis."""
    retained_to_equity: Optional[float] = None
    retained_to_assets: Optional[float] = None
    dividend_payout: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    debt_to_equity: Optional[float] = None
    ocf_to_debt: Optional[float] = None
    cd_score: float = 0.0
    cd_grade: str = ""
    summary: str = ""


@dataclass
class ResourceOptimizationResult:
    """Phase 210: Resource Optimization Analysis."""
    fcf_to_revenue: Optional[float] = None
    ocf_to_revenue: Optional[float] = None
    capex_to_revenue: Optional[float] = None
    ocf_to_assets: Optional[float] = None
    fcf_to_assets: Optional[float] = None
    dividend_payout_ratio: Optional[float] = None
    ro_score: float = 0.0
    ro_grade: str = ""
    summary: str = ""


@dataclass
class FinancialProductivityResult:
    """Phase 205: Financial Productivity Analysis."""
    revenue_per_asset: Optional[float] = None
    revenue_per_equity: Optional[float] = None
    ebitda_per_employee_proxy: Optional[float] = None
    operating_income_per_asset: Optional[float] = None
    net_income_per_revenue: Optional[float] = None
    cash_flow_per_asset: Optional[float] = None
    fp_score: float = 0.0
    fp_grade: str = ""
    summary: str = ""


@dataclass
class EquityPreservationResult:
    """Phase 198: Equity Preservation Analysis."""
    equity_to_assets: Optional[float] = None
    retained_to_equity: Optional[float] = None
    equity_growth_capacity: Optional[float] = None
    equity_to_liabilities: Optional[float] = None
    tangible_equity_ratio: Optional[float] = None
    equity_per_revenue: Optional[float] = None
    ep_score: float = 0.0
    ep_grade: str = ""
    summary: str = ""


@dataclass
class DebtManagementResult:
    """Phase 197: Debt Management Analysis."""
    debt_to_operating_income: Optional[float] = None
    debt_to_ocf: Optional[float] = None
    interest_to_revenue: Optional[float] = None
    debt_to_gross_profit: Optional[float] = None
    net_debt_ratio: Optional[float] = None
    debt_coverage_ratio: Optional[float] = None
    dm_score: float = 0.0
    dm_grade: str = ""
    summary: str = ""


@dataclass
class IncomeRetentionResult:
    """Phase 196: Income Retention Analysis."""
    net_to_gross_ratio: Optional[float] = None
    net_to_operating_ratio: Optional[float] = None
    net_to_ebitda_ratio: Optional[float] = None
    retention_rate: Optional[float] = None
    income_to_asset_generation: Optional[float] = None
    after_tax_margin: Optional[float] = None
    ir_score: float = 0.0
    ir_grade: str = ""
    summary: str = ""


@dataclass
class OperationalEfficiencyResult:
    """Phase 195: Operational Efficiency Analysis."""
    oi_margin: Optional[float] = None
    revenue_to_assets: Optional[float] = None
    gross_profit_per_asset: Optional[float] = None
    opex_efficiency: Optional[float] = None
    asset_utilization: Optional[float] = None
    income_per_liability: Optional[float] = None
    oe_score: float = 0.0
    oe_grade: str = ""
    summary: str = ""


@dataclass
class OperatingMomentumResult:
    """Phase 191: Operating Momentum Analysis."""
    ebitda_margin: Optional[float] = None
    ebit_margin: Optional[float] = None
    ocf_margin: Optional[float] = None
    gross_to_operating_conversion: Optional[float] = None
    operating_cash_conversion: Optional[float] = None
    overhead_absorption: Optional[float] = None
    om_score: float = 0.0
    om_grade: str = ""
    summary: str = ""


@dataclass
class PayoutDisciplineResult:
    """Phase 185: Payout Discipline Analysis."""
    cash_dividend_coverage: Optional[float] = None
    payout_ratio: Optional[float] = None
    retention_ratio: Optional[float] = None
    dividend_to_ocf: Optional[float] = None
    capex_priority: Optional[float] = None
    free_cash_after_dividends: Optional[float] = None
    pd_score: float = 0.0
    pd_grade: str = ""
    summary: str = ""


@dataclass
class IncomeResilienceResult:
    """Phase 184: Income Resilience Analysis."""
    operating_income_stability: Optional[float] = None
    ebit_coverage: Optional[float] = None
    net_margin_resilience: Optional[float] = None
    depreciation_buffer: Optional[float] = None
    tax_interest_drag: Optional[float] = None
    ebitda_cushion: Optional[float] = None
    ir_score: float = 0.0
    ir_grade: str = ""
    summary: str = ""


@dataclass
class StructuralStrengthResult:
    """Phase 182: Structural Strength Analysis."""
    equity_multiplier: Optional[float] = None
    debt_to_equity: Optional[float] = None
    liability_composition: Optional[float] = None
    equity_cushion: Optional[float] = None
    fixed_asset_coverage: Optional[float] = None
    financial_leverage_ratio: Optional[float] = None
    ss_score: float = 0.0
    ss_grade: str = ""
    summary: str = ""


@dataclass
class ProfitConversionResult:
    """Phase 179: Profit Conversion Analysis."""
    gross_conversion: Optional[float] = None
    operating_conversion: Optional[float] = None
    net_conversion: Optional[float] = None
    ebitda_conversion: Optional[float] = None
    cash_conversion: Optional[float] = None
    profit_to_cash_ratio: Optional[float] = None
    pc_score: float = 0.0
    pc_grade: str = ""
    summary: str = ""


@dataclass
class AssetDeploymentEfficiencyResult:
    """Phase 172: Asset Deployment Efficiency Analysis."""
    asset_turnover: Optional[float] = None
    fixed_asset_leverage: Optional[float] = None
    asset_income_yield: Optional[float] = None
    asset_cash_yield: Optional[float] = None
    inventory_velocity: Optional[float] = None
    receivables_velocity: Optional[float] = None
    ade_score: float = 0.0
    ade_grade: str = ""
    summary: str = ""


@dataclass
class ProfitSustainabilityResult:
    """Phase 171: Profit Sustainability Analysis."""
    profit_cash_backing: Optional[float] = None
    profit_margin_depth: Optional[float] = None
    profit_reinvestment: Optional[float] = None
    profit_to_asset: Optional[float] = None
    profit_stability_proxy: Optional[float] = None
    profit_leverage: Optional[float] = None
    ps_score: float = 0.0
    ps_grade: str = ""
    summary: str = ""


@dataclass
class DebtDisciplineResult:
    """Phase 170: Debt Discipline Analysis."""
    debt_prudence_ratio: Optional[float] = None
    debt_servicing_power: Optional[float] = None
    debt_coverage_spread: Optional[float] = None
    debt_to_equity_leverage: Optional[float] = None
    interest_absorption: Optional[float] = None
    debt_repayment_capacity: Optional[float] = None
    dd_score: float = 0.0
    dd_grade: str = ""
    summary: str = ""


@dataclass
class CapitalPreservationResult:
    """Phase 168: Capital Preservation Analysis."""
    retained_earnings_power: Optional[float] = None
    capital_erosion_rate: Optional[float] = None
    asset_integrity_ratio: Optional[float] = None
    operating_capital_ratio: Optional[float] = None
    net_worth_growth_proxy: Optional[float] = None
    capital_buffer: Optional[float] = None
    cp_score: float = 0.0
    cp_grade: str = ""
    summary: str = ""


@dataclass
class ObligationCoverageResult:
    """Phase 160: Obligation Coverage Analysis."""
    ebitda_interest_coverage: Optional[float] = None
    cash_interest_coverage: Optional[float] = None
    debt_amortization_capacity: Optional[float] = None
    fixed_charge_coverage: Optional[float] = None
    debt_burden_ratio: Optional[float] = None
    interest_to_revenue: Optional[float] = None
    oc_score: float = 0.0
    oc_grade: str = ""
    summary: str = ""


@dataclass
class InternalGrowthCapacityResult:
    """Phase 159: Internal Growth Capacity Analysis."""
    sustainable_growth_rate: Optional[float] = None
    internal_growth_rate: Optional[float] = None
    plowback_ratio: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    growth_financing_ratio: Optional[float] = None
    equity_growth_rate: Optional[float] = None
    igc_score: float = 0.0
    igc_grade: str = ""
    summary: str = ""


@dataclass
class LiabilityManagementResult:
    """Phase 146: Liability Management Analysis."""
    liability_to_assets: Optional[float] = None
    liability_to_equity: Optional[float] = None
    current_liability_ratio: Optional[float] = None
    liability_coverage: Optional[float] = None
    liability_to_revenue: Optional[float] = None
    net_liability: Optional[float] = None
    lm_score: float = 0.0
    lm_grade: str = ""
    summary: str = ""


@dataclass
class RevenuePredictabilityResult:
    """Phase 142: Revenue Predictability Analysis."""
    revenue_to_assets: Optional[float] = None
    revenue_to_equity: Optional[float] = None
    revenue_to_debt: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    rp_score: float = 0.0
    rp_grade: str = ""
    summary: str = ""


@dataclass
class EquityReinvestmentResult:
    """Phase 139: Equity Reinvestment Analysis."""
    retention_ratio: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    equity_growth_proxy: Optional[float] = None
    plowback_to_assets: Optional[float] = None
    internal_growth_rate: Optional[float] = None
    dividend_coverage: Optional[float] = None
    er_score: float = 0.0
    er_grade: str = ""
    summary: str = ""


@dataclass
class FixedAssetEfficiencyResult:
    """Phase 138: Fixed Asset Efficiency Analysis."""
    fixed_asset_ratio: Optional[float] = None
    fixed_asset_turnover: Optional[float] = None
    fixed_to_equity: Optional[float] = None
    fixed_asset_coverage: Optional[float] = None
    depreciation_to_fixed: Optional[float] = None
    capex_to_fixed: Optional[float] = None
    fae_score: float = 0.0
    fae_grade: str = ""
    summary: str = ""


@dataclass
class IncomeStabilityResult:
    """Phase 134: Income Stability Analysis."""
    net_income_margin: Optional[float] = None
    retained_earnings_ratio: Optional[float] = None
    operating_income_cushion: Optional[float] = None
    net_to_gross_ratio: Optional[float] = None
    ebitda_margin: Optional[float] = None
    income_resilience: Optional[float] = None
    is_score: float = 0.0
    is_grade: str = ""
    summary: str = ""


@dataclass
class DefensivePostureResult:
    """Phase 133: Defensive Posture Analysis."""
    defensive_interval: Optional[float] = None
    cash_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_flow_coverage: Optional[float] = None
    equity_buffer: Optional[float] = None
    debt_shield: Optional[float] = None
    dp_score: float = 0.0
    dp_grade: str = ""
    summary: str = ""


@dataclass
class FundingEfficiencyResult:
    """Phase 131: Funding Efficiency Analysis."""
    debt_to_capitalization: Optional[float] = None
    equity_multiplier: Optional[float] = None
    interest_coverage_ebitda: Optional[float] = None
    cost_of_debt: Optional[float] = None
    weighted_funding_cost: Optional[float] = None
    funding_spread: Optional[float] = None
    fe_score: float = 0.0
    fe_grade: str = ""
    summary: str = ""


@dataclass
class CashFlowStabilityResult:
    """Phase 125: Cash Flow Stability Analysis."""
    ocf_margin: Optional[float] = None
    ocf_to_ebitda: Optional[float] = None
    ocf_to_debt_service: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    dividend_coverage: Optional[float] = None
    cash_flow_sufficiency: Optional[float] = None
    cfs_score: float = 0.0
    cfs_grade: str = ""
    summary: str = ""


@dataclass
class IncomeQualityResult:
    """Phase 124: Income Quality Analysis."""
    ocf_to_net_income: Optional[float] = None
    accruals_ratio: Optional[float] = None
    cash_earnings_ratio: Optional[float] = None
    non_cash_ratio: Optional[float] = None
    earnings_persistence: Optional[float] = None
    operating_income_ratio: Optional[float] = None
    iq_score: float = 0.0
    iq_grade: str = ""
    summary: str = ""


@dataclass
class ReceivablesManagementResult:
    """Phase 114: Receivables Management Analysis."""
    dso: Optional[float] = None
    ar_to_revenue: Optional[float] = None
    ar_to_current_assets: Optional[float] = None
    receivables_turnover: Optional[float] = None
    collection_effectiveness: Optional[float] = None
    ar_concentration: Optional[float] = None
    rm_score: float = 0.0
    rm_grade: str = ""
    summary: str = ""


@dataclass
class SolvencyDepthResult:
    """Phase 109: Solvency Depth Analysis."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    equity_to_assets: Optional[float] = None
    interest_coverage_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    financial_leverage: Optional[float] = None
    sd_score: float = 0.0
    sd_grade: str = ""
    summary: str = ""


@dataclass
class OperationalLeverageDepthResult:
    """Phase 105: Operational Leverage Depth Analysis."""
    fixed_cost_ratio: Optional[float] = None
    variable_cost_ratio: Optional[float] = None
    contribution_margin: Optional[float] = None
    dol_proxy: Optional[float] = None
    breakeven_coverage: Optional[float] = None
    cost_flexibility: Optional[float] = None
    old_score: float = 0.0
    old_grade: str = ""
    summary: str = ""


@dataclass
class ProfitabilityDepthResult:
    """Phase 103: Profitability Depth Analysis."""
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    net_margin: Optional[float] = None
    margin_spread: Optional[float] = None
    profit_retention_ratio: Optional[float] = None
    pd_score: float = 0.0
    pd_grade: str = ""
    summary: str = ""


@dataclass
class RevenueEfficiencyResult:
    """Phase 102: Revenue Efficiency Analysis."""
    revenue_per_asset: Optional[float] = None
    cash_conversion_efficiency: Optional[float] = None
    gross_margin_efficiency: Optional[float] = None
    operating_leverage_ratio: Optional[float] = None
    revenue_to_equity: Optional[float] = None
    net_revenue_retention: Optional[float] = None
    rev_eff_score: float = 0.0
    rev_eff_grade: str = ""
    summary: str = ""


@dataclass
class DebtCompositionResult:
    """Phase 101: Debt Composition Analysis."""
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    long_term_debt_ratio: Optional[float] = None
    interest_burden: Optional[float] = None
    debt_cost_ratio: Optional[float] = None
    debt_coverage_margin: Optional[float] = None
    dco_score: float = 0.0
    dco_grade: str = ""
    summary: str = ""


@dataclass
class OperationalRiskResult:
    """Phase 92: Operational Risk Analysis."""
    operating_leverage: Optional[float] = None
    cost_rigidity: Optional[float] = None
    breakeven_ratio: Optional[float] = None
    margin_of_safety: Optional[float] = None
    cash_burn_ratio: Optional[float] = None
    risk_buffer: Optional[float] = None
    or_score: float = 0.0
    or_grade: str = ""
    summary: str = ""


@dataclass
class FinancialHealthScoreResult:
    """Phase 90: Financial Health Score Analysis."""
    profitability_score: Optional[float] = None
    liquidity_score: Optional[float] = None
    solvency_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    coverage_score: Optional[float] = None
    composite_score: Optional[float] = None
    fh_score: float = 0.0
    fh_grade: str = ""
    summary: str = ""


@dataclass
class AssetQualityResult:
    """Phase 86: Asset Quality Analysis."""
    tangible_asset_ratio: Optional[float] = None       # (TA - Intangibles) / TA
    fixed_asset_ratio: Optional[float] = None          # Fixed Assets / TA (approx TA - CA)
    current_asset_ratio: Optional[float] = None        # CA / TA
    cash_to_current_assets: Optional[float] = None     # Cash / CA
    receivables_to_assets: Optional[float] = None      # AR / TA
    inventory_to_assets: Optional[float] = None        # Inv / TA
    aq_score: float = 0.0
    aq_grade: str = ""
    summary: str = ""


@dataclass
class FinancialResilienceResult:
    """Phase 82: Financial Resilience Analysis."""
    cash_to_assets: Optional[float] = None              # Cash / Total Assets
    cash_to_debt: Optional[float] = None                # Cash / Total Debt
    operating_cash_coverage: Optional[float] = None     # OCF / Total Liabilities
    interest_coverage_cash: Optional[float] = None      # OCF / Interest Expense
    free_cash_margin: Optional[float] = None            # FCF / Revenue
    resilience_buffer: Optional[float] = None           # (Cash + OCF) / Annual OpEx
    fr_score: float = 0.0
    fr_grade: str = ""
    summary: str = ""


@dataclass
class EquityMultiplierResult:
    """Phase 81: Equity Multiplier Analysis."""
    equity_multiplier: Optional[float] = None        # TA / Equity
    debt_ratio: Optional[float] = None               # TL / TA
    equity_ratio: Optional[float] = None             # Equity / TA
    financial_leverage_index: Optional[float] = None  # ROE / ROA
    dupont_roe: Optional[float] = None               # Margin * Turnover * EM
    leverage_spread: Optional[float] = None          # ROA - Cost of Debt
    em_score: float = 0.0
    em_grade: str = ""
    summary: str = ""


@dataclass
class DefensiveIntervalResult:
    """Phase 80: Defensive Interval Analysis."""
    defensive_interval_days: Optional[float] = None   # Liquid Assets / Daily OpEx
    cash_interval_days: Optional[float] = None        # Cash / Daily OpEx
    liquid_assets_ratio: Optional[float] = None       # Liquid Assets / Total Assets
    days_cash_on_hand: Optional[float] = None         # Cash / (Annual OpEx / 365)
    liquid_reserve_adequacy: Optional[float] = None   # Liquid Assets / CL
    operating_expense_coverage: Optional[float] = None  # Liquid Assets / Annual OpEx
    di_score: float = 0.0
    di_grade: str = ""
    summary: str = ""


@dataclass
class CashBurnResult:
    """Phase 79: Cash Burn Analysis."""
    ocf_margin: Optional[float] = None              # OCF / Revenue
    capex_intensity: Optional[float] = None         # CapEx / Revenue
    fcf_margin: Optional[float] = None              # (OCF - CapEx) / Revenue
    cash_self_sufficiency: Optional[float] = None   # OCF / (CapEx + Div)
    cash_runway_months: Optional[float] = None      # Cash / Monthly Burn (if burning)
    net_cash_position: Optional[float] = None       # Cash - Total Debt
    cb_score: float = 0.0
    cb_grade: str = ""
    summary: str = ""


@dataclass
class ProfitRetentionResult:
    """Phase 78: Profit Retention Analysis."""
    retention_ratio: Optional[float] = None         # (NI - Div) / NI
    payout_ratio: Optional[float] = None            # Div / NI
    re_to_equity: Optional[float] = None            # Retained Earnings / Equity
    sustainable_growth_rate: Optional[float] = None  # ROE * retention
    internal_growth_rate: Optional[float] = None     # ROA * retention / (1 - ROA * retention)
    plowback_amount: Optional[float] = None          # NI - Dividends
    pr_score: float = 0.0
    pr_grade: str = ""
    summary: str = ""


@dataclass
class DebtServiceCoverageResult:
    """Phase 76: Debt Service Coverage Analysis."""
    dscr: Optional[float] = None                   # EBITDA / Total Debt Service
    ocf_to_debt_service: Optional[float] = None    # OCF / Total Debt Service
    ebitda_to_interest: Optional[float] = None     # EBITDA / IE (interest coverage)
    fcf_to_debt_service: Optional[float] = None    # FCF / Total Debt Service
    debt_service_to_revenue: Optional[float] = None  # Total Debt Service / Revenue
    coverage_cushion: Optional[float] = None       # DSCR - 1.0 (excess coverage)
    dsc_score: float = 0.0
    dsc_grade: str = ""
    summary: str = ""


@dataclass
class CapitalAllocationResult:
    """Phase 73: Capital Allocation Analysis."""
    capex_to_revenue: Optional[float] = None          # CapEx / Revenue
    capex_to_ocf: Optional[float] = None              # CapEx / OCF (reinvestment rate)
    shareholder_return_ratio: Optional[float] = None   # (Div+BB) / NI
    reinvestment_rate: Optional[float] = None          # CapEx / Depreciation
    fcf_yield: Optional[float] = None                  # FCF / Revenue
    total_payout_to_fcf: Optional[float] = None        # (Div+BB) / FCF
    ca_score: float = 0.0
    ca_grade: str = ""
    summary: str = ""


@dataclass
class TaxEfficiencyResult:
    """Phase 70: Tax Efficiency Analysis."""
    effective_tax_rate: Optional[float] = None        # TaxExp / PreTaxIncome
    tax_to_revenue: Optional[float] = None            # TaxExp / Revenue
    tax_to_ebitda: Optional[float] = None             # TaxExp / EBITDA
    after_tax_margin: Optional[float] = None          # NI / Revenue
    tax_shield_ratio: Optional[float] = None          # (IE * ETR) / NI
    pretax_to_ebit: Optional[float] = None            # PreTaxIncome / EBIT
    te_score: float = 0.0
    te_grade: str = ""
    summary: str = ""


@dataclass
class ROICResult:
    """Phase 56: Return on Invested Capital Analysis."""
    roic_pct: Optional[float] = None
    nopat: Optional[float] = None
    invested_capital: Optional[float] = None
    roic_roa_spread: Optional[float] = None
    roic_wacc_spread: Optional[float] = None
    capital_efficiency: Optional[float] = None
    roic_score: float = 0.0
    roic_grade: str = ""
    summary: str = ""


@dataclass
class ROAQualityResult:
    """Phase 55: Return on Assets Quality Analysis."""
    roa_pct: Optional[float] = None
    operating_roa_pct: Optional[float] = None
    cash_roa_pct: Optional[float] = None
    asset_turnover: Optional[float] = None
    fixed_asset_turnover: Optional[float] = None
    capital_intensity: Optional[float] = None
    roa_score: float = 0.0
    roa_grade: str = ""
    summary: str = ""


@dataclass
class ROEAnalysisResult:
    """Phase 54: Return on Equity Analysis."""
    roe_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    asset_turnover: Optional[float] = None
    equity_multiplier: Optional[float] = None
    roa_pct: Optional[float] = None
    retention_ratio: Optional[float] = None
    sustainable_growth_rate: Optional[float] = None
    roe_score: float = 0.0
    roe_grade: str = ""
    summary: str = ""


@dataclass
class NetProfitMarginResult:
    """Phase 53: Net Profit Margin Analysis."""
    net_margin_pct: Optional[float] = None          # NI / Revenue
    ebitda_margin_pct: Optional[float] = None       # EBITDA / Revenue
    ebit_margin_pct: Optional[float] = None         # EBIT / Revenue
    tax_burden: Optional[float] = None              # NI / EBT (1 - effective tax rate)
    interest_burden: Optional[float] = None         # EBT / EBIT
    net_to_ebitda: Optional[float] = None           # NI / EBITDA (retention ratio)
    npm_score: float = 0.0                          # 0-10 (higher = better)
    npm_grade: str = ""                             # "Excellent", "Good", "Adequate", "Weak"
    summary: str = ""


@dataclass
class EbitdaMarginQualityResult:
    """Phase 52: EBITDA Margin Quality Analysis."""
    ebitda_margin_pct: Optional[float] = None       # EBITDA / Revenue
    operating_margin_pct: Optional[float] = None    # OI / Revenue
    da_intensity: Optional[float] = None            # D&A / Revenue
    ebitda_oi_spread: Optional[float] = None        # EBITDA margin - OI margin (= D&A/Rev)
    ebitda_to_gp: Optional[float] = None            # EBITDA / Gross Profit
    ebitda_score: float = 0.0                       # 0-10 (higher = better)
    ebitda_grade: str = ""                          # "Excellent", "Good", "Adequate", "Weak"
    summary: str = ""


@dataclass
class GrossMarginStabilityResult:
    """Phase 51: Gross Margin Stability Analysis."""
    gross_margin_pct: Optional[float] = None       # GP / Revenue
    cogs_ratio: Optional[float] = None             # COGS / Revenue
    operating_margin_pct: Optional[float] = None   # Operating Income / Revenue
    margin_spread: Optional[float] = None          # Gross Margin - Operating Margin
    opex_coverage: Optional[float] = None          # GP / Operating Expenses
    margin_buffer: Optional[float] = None          # Distance to breakeven (= gross margin)
    gm_stability_score: float = 0.0                # 0-10 (higher = better)
    gm_stability_grade: str = ""                   # "Excellent", "Good", "Adequate", "Weak"
    summary: str = ""


@dataclass
class BeneishMScoreResult:
    """Phase 43: Beneish M-Score (Earnings Manipulation Detection)."""
    m_score: Optional[float] = None
    dsri: Optional[float] = None  # Days Sales in Receivables Index
    gmi: Optional[float] = None   # Gross Margin Index
    aqi: Optional[float] = None   # Asset Quality Index
    sgi: Optional[float] = None   # Sales Growth Index
    depi: Optional[float] = None  # Depreciation Index
    sgai: Optional[float] = None  # SGA Expense Index
    lvgi: Optional[float] = None  # Leverage Index
    tata: Optional[float] = None  # Total Accruals to Total Assets
    manipulation_score: float = 0.0  # 0-10 (higher = less likely manipulated)
    manipulation_grade: str = ""  # "Unlikely", "Possible", "Likely", "Highly Likely"
    summary: str = ""


@dataclass
class FinancialReport:
    """Structured financial analysis report."""
    sections: Dict[str, str] = field(default_factory=dict)
    executive_summary: str = ""
    generated_at: str = ""


@dataclass
class FinancialData:
    """Container for financial data used in analysis."""
    # Balance Sheet items
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    cash: Optional[float] = None
    inventory: Optional[float] = None
    accounts_receivable: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    accounts_payable: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None

    # Income Statement items
    revenue: Optional[float] = None
    cogs: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    operating_expenses: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    net_income: Optional[float] = None
    ebt: Optional[float] = None  # Earnings Before Tax
    tax_expense: Optional[float] = None  # Income tax expense
    retained_earnings: Optional[float] = None
    depreciation: Optional[float] = None

    # Cash Flow items
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    # Shareholder returns
    dividends_paid: Optional[float] = None
    share_buybacks: Optional[float] = None
    shares_outstanding: Optional[float] = None
    share_price: Optional[float] = None

    # Derived/Previous period
    avg_inventory: Optional[float] = None
    avg_receivables: Optional[float] = None
    avg_payables: Optional[float] = None
    avg_total_assets: Optional[float] = None

    # Time series data
    time_series: Optional[pd.DataFrame] = None
    period: Optional[str] = None


class CharlieAnalyzer:
    """
    CFO-grade financial analysis engine inspired by Charlie Munger's
    analytical framework: focus on fundamentals, cash flow, and
    sustainable competitive advantages.
    """

    def __init__(self, tax_rate: Optional[float] = None):
        from config import settings
        self._tax_rate = tax_rate if tax_rate is not None else settings.default_tax_rate
        self.ratio_definitions = self._load_ratio_definitions()
        self.insight_templates = self._load_insight_templates()

    def _load_ratio_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load standard financial ratio definitions."""
        return {
            'current_ratio': {
                'name': 'Current Ratio',
                'formula': 'Current Assets / Current Liabilities',
                'benchmark': {'good': 1.5, 'warning': 1.0},
                'interpretation': 'Measures ability to pay short-term obligations'
            },
            'quick_ratio': {
                'name': 'Quick Ratio (Acid Test)',
                'formula': '(Current Assets - Inventory) / Current Liabilities',
                'benchmark': {'good': 1.0, 'warning': 0.5},
                'interpretation': 'Measures ability to pay short-term obligations without selling inventory'
            },
            'cash_ratio': {
                'name': 'Cash Ratio',
                'formula': 'Cash / Current Liabilities',
                'benchmark': {'good': 0.5, 'warning': 0.2},
                'interpretation': 'Most conservative liquidity measure'
            },
            'gross_margin': {
                'name': 'Gross Margin',
                'formula': '(Revenue - COGS) / Revenue',
                'benchmark': {'varies': True},
                'interpretation': 'Profitability after direct costs'
            },
            'operating_margin': {
                'name': 'Operating Margin',
                'formula': 'Operating Income / Revenue',
                'benchmark': {'good': 0.15, 'warning': 0.05},
                'interpretation': 'Profitability from core operations'
            },
            'net_margin': {
                'name': 'Net Profit Margin',
                'formula': 'Net Income / Revenue',
                'benchmark': {'good': 0.10, 'warning': 0.02},
                'interpretation': 'Bottom-line profitability'
            },
            'roe': {
                'name': 'Return on Equity',
                'formula': 'Net Income / Shareholders Equity',
                'benchmark': {'good': 0.15, 'warning': 0.08},
                'interpretation': 'Return generated on shareholder investment'
            },
            'roa': {
                'name': 'Return on Assets',
                'formula': 'Net Income / Total Assets',
                'benchmark': {'good': 0.05, 'warning': 0.02},
                'interpretation': 'Efficiency in using assets to generate profit'
            },
            'debt_to_equity': {
                'name': 'Debt-to-Equity Ratio',
                'formula': 'Total Debt / Total Equity',
                'benchmark': {'good': 1.0, 'warning': 2.0},
                'interpretation': 'Financial leverage and risk'
            },
            'interest_coverage': {
                'name': 'Interest Coverage Ratio',
                'formula': 'EBIT / Interest Expense',
                'benchmark': {'good': 3.0, 'warning': 1.5},
                'interpretation': 'Ability to pay interest on debt'
            },
            'asset_turnover': {
                'name': 'Asset Turnover',
                'formula': 'Revenue / Average Total Assets',
                'benchmark': {'varies': True},
                'interpretation': 'Efficiency in using assets to generate sales'
            }
        }

    def _load_insight_templates(self) -> Dict[str, str]:
        """Load insight generation templates."""
        return {
            'high_liquidity': "Strong liquidity position with {ratio_name} of {value:.2f}. The company can comfortably meet short-term obligations.",
            'low_liquidity': "Liquidity concern: {ratio_name} of {value:.2f} is below the recommended threshold of {threshold}. Consider improving working capital management.",
            'improving_margin': "Profitability improving: {metric} increased from {old_value:.1%} to {new_value:.1%}, indicating better operational efficiency.",
            'declining_margin': "Margin pressure: {metric} declined from {old_value:.1%} to {new_value:.1%}. Investigate cost structure and pricing.",
            'high_leverage': "Elevated leverage: Debt-to-equity of {value:.2f} exceeds typical comfort levels. Monitor debt service capacity.",
            'strong_cash_generation': "Robust cash generation: Free cash flow of {value:,.0f} provides strategic flexibility.",
            'cash_conversion_concern': "Cash conversion cycle of {value:.0f} days may tie up working capital. Consider optimizing AR/AP terms.",
            'favorable_variance': "Favorable budget variance in {category}: {percent:.1%} under budget, saving {amount:,.0f}.",
            'unfavorable_variance': "Budget overrun in {category}: {percent:.1%} over budget. Review spending controls."
        }

    # ===== LIQUIDITY RATIOS =====

    def calculate_liquidity_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate liquidity ratios.
        - Current Ratio = Current Assets / Current Liabilities
        - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        - Cash Ratio = Cash & Equivalents / Current Liabilities
        """
        ratios = {}

        ratios['current_ratio'] = safe_divide(data.current_assets, data.current_liabilities)

        quick_numerator = (
            (data.current_assets - data.inventory)
            if data.current_assets is not None and data.inventory is not None
            else None
        )
        ratios['quick_ratio'] = safe_divide(quick_numerator, data.current_liabilities)

        ratios['cash_ratio'] = safe_divide(data.cash, data.current_liabilities)

        return ratios

    # ===== PROFITABILITY RATIOS =====

    def calculate_profitability_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate profitability ratios.
        """
        ratios = {}

        # Gross Margin
        gross_numerator = None
        if data.revenue is not None and data.cogs is not None:
            gross_numerator = data.revenue - data.cogs
        elif data.gross_profit is not None:
            gross_numerator = data.gross_profit
        ratios['gross_margin'] = safe_divide(gross_numerator, data.revenue)

        # Operating Margin
        ratios['operating_margin'] = safe_divide(data.operating_income, data.revenue)

        # Net Margin
        ratios['net_margin'] = safe_divide(data.net_income, data.revenue)

        # ROE
        ratios['roe'] = safe_divide(data.net_income, data.total_equity)

        # ROA
        roa = safe_divide(data.net_income, data.total_assets)
        if roa is None:
            roa = safe_divide(data.net_income, data.avg_total_assets)
        ratios['roa'] = roa

        # ROIC (Return on Invested Capital)
        invested_capital = (
            (data.total_equity + data.total_debt)
            if data.total_equity is not None and data.total_debt is not None
            else None
        )
        nopat = (
            data.operating_income * (1 - self._tax_rate)
            if data.operating_income is not None
            else None
        )
        ratios['roic'] = safe_divide(nopat, invested_capital)

        return ratios

    # ===== LEVERAGE RATIOS =====

    def calculate_leverage_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate leverage/solvency ratios.
        """
        ratios = {}

        ratios['debt_to_equity'] = safe_divide(data.total_debt, data.total_equity)
        ratios['debt_to_assets'] = safe_divide(data.total_debt, data.total_assets)

        interest_cov = safe_divide(data.ebit, data.interest_expense)
        if interest_cov is None:
            interest_cov = safe_divide(data.operating_income, data.interest_expense)
        ratios['interest_coverage'] = interest_cov

        return ratios

    # ===== EFFICIENCY RATIOS =====

    def calculate_efficiency_ratios(self, data: FinancialData) -> Dict[str, Optional[float]]:
        """
        Calculate efficiency/activity ratios.
        """
        ratios = {}

        asset_turn = safe_divide(data.revenue, data.avg_total_assets)
        if asset_turn is None:
            asset_turn = safe_divide(data.revenue, data.total_assets)
        ratios['asset_turnover'] = asset_turn

        inv_turn = safe_divide(data.cogs, data.avg_inventory)
        if inv_turn is None:
            inv_turn = safe_divide(data.cogs, data.inventory)
        ratios['inventory_turnover'] = inv_turn

        recv_turn = safe_divide(data.revenue, data.avg_receivables)
        if recv_turn is None:
            recv_turn = safe_divide(data.revenue, data.accounts_receivable)
        ratios['receivables_turnover'] = recv_turn

        pay_turn = safe_divide(data.cogs, data.avg_payables)
        if pay_turn is None:
            pay_turn = safe_divide(data.cogs, data.accounts_payable)
        ratios['payables_turnover'] = pay_turn

        return ratios

    # ===== TREND ANALYSIS =====

    def analyze_trends(self, time_series: pd.DataFrame, metric_column: str,
                      period_column: Optional[str] = None) -> TrendAnalysis:
        """
        Analyze trends in a time series.
        """
        values = time_series[metric_column].dropna().tolist()

        if period_column and period_column in time_series.columns:
            periods = time_series[period_column].tolist()
        else:
            periods = [f"Period {i+1}" for i in range(len(values))]

        # Calculate growth rates
        yoy_growth = None
        qoq_growth = None
        mom_growth = None

        if len(values) >= 2:
            ratio = safe_divide(values[-1], values[-2])
            mom_growth = (ratio - 1) if ratio is not None else None

        if len(values) >= 4:
            ratio = safe_divide(values[-1], values[-4])
            qoq_growth = (ratio - 1) if ratio is not None else None

        if len(values) >= 12:
            ratio = safe_divide(values[-1], values[-12])
            yoy_growth = (ratio - 1) if ratio is not None else None

        # Calculate CAGR - requires positive start and end values
        cagr = None
        if len(values) >= 2 and values[0] is not None and values[-1] is not None:
            if values[0] > 0 and values[-1] > 0:
                n_periods = len(values) - 1
                cagr = (values[-1] / values[0]) ** (1 / n_periods) - 1

        # Moving averages
        ma_3 = self._calculate_moving_average(values, 3)
        ma_12 = self._calculate_moving_average(values, 12)

        # Trend direction
        if len(values) >= 3:
            recent_avg = np.mean(values[-3:])
            older_avg = np.mean(values[:3])
            if recent_avg > older_avg * 1.05:
                trend_direction = 'up'
            elif recent_avg < older_avg * 0.95:
                trend_direction = 'down'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'

        # Simple seasonality detection
        seasonality_detected = self._detect_seasonality(values)

        return TrendAnalysis(
            metric_name=metric_column,
            values=values,
            periods=periods,
            yoy_growth=yoy_growth,
            qoq_growth=qoq_growth,
            mom_growth=mom_growth,
            cagr=cagr,
            moving_avg_3=ma_3,
            moving_avg_12=ma_12,
            trend_direction=trend_direction,
            seasonality_detected=seasonality_detected
        )

    def _calculate_moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average using pandas rolling for O(n) performance."""
        if len(values) < window:
            return list(values)

        s = pd.Series(values)
        return s.rolling(window=window, min_periods=1).mean().tolist()

    def _detect_seasonality(self, values: List[float]) -> bool:
        """Simple seasonality detection."""
        if len(values) < 24:  # Need at least 2 years of monthly data
            return False

        # Check for repeating patterns
        try:
            first_year = values[:12]
            second_year = values[12:24]
            correlation = np.corrcoef(first_year, second_year)[0, 1]
            return correlation > 0.7
        except (ValueError, IndexError, TypeError):
            return False

    def calculate_variance(self, actual: float, budget: float,
                          category: str = "General") -> VarianceResult:
        """
        Calculate budget variance.
        """
        variance = actual - budget
        variance_percent = safe_divide(variance, abs(budget), default=0.0 if variance == 0 else float('inf'))

        # For expenses, under budget is favorable
        # For revenue, over budget is favorable
        is_expense = category.lower() in ['expense', 'cost', 'cogs', 'opex', 'sg&a']
        favorable = (variance < 0) if is_expense else (variance > 0)

        return VarianceResult(
            actual=actual,
            budget=budget,
            variance=variance,
            variance_percent=variance_percent,
            favorable=favorable,
            category=category
        )

    def forecast_simple(self, historical: List[float], periods: int = 3,
                       method: str = 'linear') -> Forecast:
        """
        Simple forecasting methods.
        """
        if not historical or len(historical) < 2:
            return Forecast(
                metric_name="unknown",
                historical_values=historical,
                forecasted_values=[],
                forecast_periods=[],
                method=method,
                confidence_interval=(0, 0)
            )

        forecasted = []

        if method == 'linear':
            # Linear trend extrapolation
            x = np.arange(len(historical))
            coeffs = np.polyfit(x, historical, 1)
            for i in range(periods):
                forecasted.append(coeffs[0] * (len(historical) + i) + coeffs[1])

        elif method == 'moving_average':
            # Moving average projection
            window = min(3, len(historical))
            ma = np.mean(historical[-window:])
            forecasted = [ma] * periods

        elif method == 'growth_rate':
            # Apply average growth rate
            growth_rates = [
                r - 1 for i in range(1, len(historical))
                if (r := safe_divide(historical[i], historical[i - 1])) is not None
            ]
            if growth_rates:
                avg_growth = np.mean(growth_rates)
                last_value = historical[-1]
                for _ in range(periods):
                    last_value = last_value * (1 + avg_growth)
                    forecasted.append(last_value)
            else:
                forecasted = [historical[-1]] * periods

        # Calculate confidence interval (simple std-based)
        std = np.std(historical) if len(historical) > 1 else 0
        mean_forecast = np.mean(forecasted) if forecasted else 0
        ci = (mean_forecast - 2 * std, mean_forecast + 2 * std)

        forecast_periods = [f"Forecast {i+1}" for i in range(periods)]

        return Forecast(
            metric_name="unknown",
            historical_values=historical,
            forecasted_values=forecasted,
            forecast_periods=forecast_periods,
            method=method,
            confidence_interval=ci
        )

    # ===== BUDGET VS ACTUAL ANALYSIS =====

    def analyze_budget_variance(self, actual_df: pd.DataFrame, budget_df: pd.DataFrame,
                               item_column: str, actual_column: str,
                               budget_column: str) -> BudgetAnalysis:
        """
        Comprehensive budget analysis.
        """
        line_items = []

        # Merge on item column
        merged = actual_df.merge(budget_df, on=item_column, how='outer', suffixes=('_actual', '_budget'))

        for _, row in merged.iterrows():
            category = row[item_column]
            actual = row.get(actual_column, row.get(f'{actual_column}_actual', 0)) or 0
            budget = row.get(budget_column, row.get(f'{budget_column}_budget', 0)) or 0

            variance_result = self.calculate_variance(actual, budget, category)
            line_items.append(variance_result)

        total_actual = sum(item.actual for item in line_items)
        total_budget = sum(item.budget for item in line_items)
        total_variance = total_actual - total_budget
        total_variance_percent = safe_divide(total_variance, total_budget, default=0.0)

        favorable_items = [item for item in line_items if item.favorable]
        unfavorable_items = [item for item in line_items if not item.favorable]

        # Get largest variances by absolute value
        largest_variances = sorted(line_items, key=lambda x: abs(x.variance), reverse=True)[:5]

        return BudgetAnalysis(
            total_budget=total_budget,
            total_actual=total_actual,
            total_variance=total_variance,
            total_variance_percent=total_variance_percent,
            line_items=line_items,
            favorable_items=favorable_items,
            unfavorable_items=unfavorable_items,
            largest_variances=largest_variances
        )

    # ===== CASH FLOW ANALYSIS =====

    def analyze_cash_flow(self, data: FinancialData) -> CashFlowAnalysis:
        """
        Analyze cash flow metrics.
        """
        # Free Cash Flow = Operating CF - CapEx
        fcf = None
        if data.operating_cash_flow is not None and data.capex is not None:
            fcf = data.operating_cash_flow - abs(data.capex)
        elif data.operating_cash_flow is not None:
            fcf = data.operating_cash_flow

        # Days Sales Outstanding (DSO)
        dso_ratio = safe_divide(data.accounts_receivable, data.revenue)
        dso = dso_ratio * 365 if dso_ratio is not None else None

        # Days Inventory Outstanding (DIO)
        dio_ratio = safe_divide(data.inventory, data.cogs)
        dio = dio_ratio * 365 if dio_ratio is not None else None

        # Days Payables Outstanding (DPO)
        dpo_ratio = safe_divide(data.accounts_payable, data.cogs)
        dpo = dpo_ratio * 365 if dpo_ratio is not None else None

        # Cash Conversion Cycle = DSO + DIO - DPO
        ccc = None
        if dso is not None and dio is not None and dpo is not None:
            ccc = dso + dio - dpo

        return CashFlowAnalysis(
            operating_cf=data.operating_cash_flow,
            investing_cf=data.investing_cash_flow,
            financing_cf=data.financing_cash_flow,
            free_cash_flow=fcf,
            dso=dso,
            dio=dio,
            dpo=dpo,
            cash_conversion_cycle=ccc
        )

    # ===== WORKING CAPITAL ANALYSIS =====

    def analyze_working_capital(self, data: FinancialData) -> WorkingCapitalAnalysis:
        """
        Analyze working capital.
        """
        nwc = None
        if data.current_assets is not None and data.current_liabilities is not None:
            nwc = data.current_assets - data.current_liabilities

        wc_ratio = safe_divide(data.current_assets, data.current_liabilities)
        wc_turnover = safe_divide(data.revenue, nwc)

        return WorkingCapitalAnalysis(
            current_assets=data.current_assets,
            current_liabilities=data.current_liabilities,
            net_working_capital=nwc,
            working_capital_ratio=wc_ratio,
            working_capital_turnover=wc_turnover
        )

    # ===== ADVANCED SCORING MODELS =====

    def dupont_analysis(self, data: FinancialData) -> DuPontAnalysis:
        """DuPont decomposition of ROE into component drivers.

        3-factor: ROE = Net Margin × Asset Turnover × Equity Multiplier
        5-factor: ROE = Tax Burden × Interest Burden × EBIT Margin × Asset Turnover × Equity Multiplier
        """
        net_margin = safe_divide(data.net_income, data.revenue)
        asset_turnover = safe_divide(data.revenue, data.total_assets)
        equity_multiplier = safe_divide(data.total_assets, data.total_equity)

        # 3-factor ROE
        roe = None
        if all(v is not None for v in [net_margin, asset_turnover, equity_multiplier]):
            roe = net_margin * asset_turnover * equity_multiplier

        # 5-factor extensions
        ebt = data.ebt
        if ebt is None and data.net_income is not None:
            # Approximate EBT = Net Income / (1 - tax_rate)
            ebt = safe_divide(data.net_income, 1 - self._tax_rate)

        tax_burden = safe_divide(data.net_income, ebt)
        interest_burden = safe_divide(ebt, data.ebit or data.operating_income)

        # Identify primary driver
        primary_driver = None
        if all(v is not None for v in [net_margin, asset_turnover, equity_multiplier]):
            drivers = {
                'net_margin': abs(net_margin) if net_margin else 0,
                'asset_turnover': abs(asset_turnover) if asset_turnover else 0,
                'equity_multiplier': abs(equity_multiplier - 1) if equity_multiplier else 0,
            }
            primary_driver = max(drivers, key=drivers.get)

        # Generate interpretation
        interpretation = None
        if roe is not None:
            parts = []
            if net_margin is not None:
                parts.append(f"net margin {net_margin:.1%}")
            if asset_turnover is not None:
                parts.append(f"asset turnover {asset_turnover:.2f}x")
            if equity_multiplier is not None:
                parts.append(f"equity multiplier {equity_multiplier:.2f}x")
            interpretation = f"ROE of {roe:.1%} driven by {', '.join(parts)}."

        return DuPontAnalysis(
            roe=roe,
            net_margin=net_margin,
            asset_turnover=asset_turnover,
            equity_multiplier=equity_multiplier,
            tax_burden=tax_burden,
            interest_burden=interest_burden,
            primary_driver=primary_driver,
            interpretation=interpretation,
        )

    def altman_z_score(self, data: FinancialData) -> AltmanZScore:
        """Altman Z-Score bankruptcy prediction model.

        Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
        Zones: Safe (>2.99), Grey (1.81-2.99), Distress (<1.81)
        """
        if data.total_assets is None or data.total_assets == 0:
            return AltmanZScore(interpretation="Insufficient data: total assets required.")

        # X1: Working Capital / Total Assets
        working_capital = None
        if data.current_assets is not None and data.current_liabilities is not None:
            working_capital = data.current_assets - data.current_liabilities
        x1 = safe_divide(working_capital, data.total_assets)

        # X2: Retained Earnings / Total Assets
        x2 = safe_divide(data.retained_earnings, data.total_assets)

        # X3: EBIT / Total Assets
        ebit = data.ebit or data.operating_income
        x3 = safe_divide(ebit, data.total_assets)

        # X4: Book Value of Equity / Total Liabilities (proxy for market value)
        x4 = safe_divide(data.total_equity, data.total_liabilities)

        # X5: Sales / Total Assets
        x5 = safe_divide(data.revenue, data.total_assets)

        components = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

        # Calculate Z-Score only with available components
        available = {k: v for k, v in components.items() if v is not None}
        if not available:
            return AltmanZScore(
                components=components,
                interpretation="Insufficient data for Z-Score calculation."
            )

        weights = {'x1': 1.2, 'x2': 1.4, 'x3': 3.3, 'x4': 0.6, 'x5': 1.0}
        z = sum(weights[k] * v for k, v in available.items())

        # Classify zone
        if len(available) == 5:
            if z > 2.99:
                zone = 'safe'
                interp = f"Z-Score of {z:.2f} indicates low bankruptcy risk (safe zone >2.99)."
            elif z >= 1.81:
                zone = 'grey'
                interp = f"Z-Score of {z:.2f} is in the grey zone (1.81-2.99). Monitor closely."
            else:
                zone = 'distress'
                interp = f"Z-Score of {z:.2f} signals financial distress (<1.81). Immediate attention required."
        else:
            zone = 'partial'
            interp = (f"Partial Z-Score of {z:.2f} (using {len(available)}/5 components). "
                      "Interpret with caution.")

        return AltmanZScore(
            z_score=z,
            zone=zone,
            components=components,
            interpretation=interp,
        )

    def piotroski_f_score(self, data: FinancialData,
                          prior_data: Optional[FinancialData] = None) -> PiotroskiFScore:
        """Piotroski F-Score: 9-criteria financial strength model.

        Profitability (4 pts):
          1. ROA > 0
          2. Operating Cash Flow > 0
          3. Delta ROA > 0 (improving)
          4. Accruals: Operating CF > Net Income (quality of earnings)

        Leverage/Liquidity (3 pts):
          5. Delta Leverage < 0 (decreasing)
          6. Delta Current Ratio > 0 (improving)
          7. No new shares issued (no dilution)

        Operating Efficiency (2 pts):
          8. Delta Gross Margin > 0 (improving)
          9. Delta Asset Turnover > 0 (improving)
        """
        criteria = {}
        score = 0

        # --- Profitability (4 points) ---

        # 1. ROA > 0
        roa = safe_divide(data.net_income, data.total_assets)
        criteria['positive_roa'] = roa is not None and roa > 0
        if criteria['positive_roa']:
            score += 1

        # 2. Operating Cash Flow > 0
        criteria['positive_ocf'] = (data.operating_cash_flow is not None
                                     and data.operating_cash_flow > 0)
        if criteria['positive_ocf']:
            score += 1

        # 3. Delta ROA > 0 (improving profitability)
        if prior_data is not None:
            prior_roa = safe_divide(prior_data.net_income, prior_data.total_assets)
            criteria['improving_roa'] = (roa is not None and prior_roa is not None
                                          and roa > prior_roa)
        else:
            criteria['improving_roa'] = False  # Can't assess without prior data
        if criteria['improving_roa']:
            score += 1

        # 4. Accruals: OCF > Net Income (quality of earnings)
        criteria['quality_earnings'] = (data.operating_cash_flow is not None
                                         and data.net_income is not None
                                         and data.operating_cash_flow > data.net_income)
        if criteria['quality_earnings']:
            score += 1

        # --- Leverage / Liquidity (3 points) ---

        # 5. Delta Leverage < 0 (decreasing debt-to-assets)
        leverage = safe_divide(data.total_debt, data.total_assets)
        if prior_data is not None:
            prior_leverage = safe_divide(prior_data.total_debt, prior_data.total_assets)
            criteria['decreasing_leverage'] = (leverage is not None and prior_leverage is not None
                                                and leverage < prior_leverage)
        else:
            criteria['decreasing_leverage'] = False
        if criteria['decreasing_leverage']:
            score += 1

        # 6. Delta Current Ratio > 0 (improving liquidity)
        current_ratio = safe_divide(data.current_assets, data.current_liabilities)
        if prior_data is not None:
            prior_cr = safe_divide(prior_data.current_assets, prior_data.current_liabilities)
            criteria['improving_liquidity'] = (current_ratio is not None and prior_cr is not None
                                                and current_ratio > prior_cr)
        else:
            criteria['improving_liquidity'] = False
        if criteria['improving_liquidity']:
            score += 1

        # 7. No dilution (no new shares - heuristic: equity didn't jump without income)
        # Without share count, approximate: equity growth < net income suggests no dilution
        if prior_data is not None and data.total_equity is not None and prior_data.total_equity is not None:
            equity_change = data.total_equity - prior_data.total_equity
            criteria['no_dilution'] = (data.net_income is not None
                                        and equity_change <= (data.net_income or 0) * 1.1)
        else:
            criteria['no_dilution'] = True  # Assume no dilution if unknown
        if criteria['no_dilution']:
            score += 1

        # --- Operating Efficiency (2 points) ---

        # 8. Delta Gross Margin > 0
        gross_margin = safe_divide(
            (data.revenue - data.cogs) if data.revenue and data.cogs else data.gross_profit,
            data.revenue
        )
        if prior_data is not None:
            prior_gm = safe_divide(
                (prior_data.revenue - prior_data.cogs) if prior_data.revenue and prior_data.cogs else prior_data.gross_profit,
                prior_data.revenue
            )
            criteria['improving_gross_margin'] = (gross_margin is not None and prior_gm is not None
                                                   and gross_margin > prior_gm)
        else:
            criteria['improving_gross_margin'] = False
        if criteria['improving_gross_margin']:
            score += 1

        # 9. Delta Asset Turnover > 0
        at = safe_divide(data.revenue, data.total_assets)
        if prior_data is not None:
            prior_at = safe_divide(prior_data.revenue, prior_data.total_assets)
            criteria['improving_asset_turnover'] = (at is not None and prior_at is not None
                                                     and at > prior_at)
        else:
            criteria['improving_asset_turnover'] = False
        if criteria['improving_asset_turnover']:
            score += 1

        # Interpretation
        if score >= 8:
            interp = f"F-Score {score}/9: Very strong financial position."
        elif score >= 6:
            interp = f"F-Score {score}/9: Healthy fundamentals."
        elif score >= 4:
            interp = f"F-Score {score}/9: Mixed signals - review weak criteria."
        else:
            interp = f"F-Score {score}/9: Weak financial health. Multiple concerns detected."

        return PiotroskiFScore(
            score=score,
            max_score=9,
            criteria=criteria,
            interpretation=interp,
        )

    # ===== COMPOSITE HEALTH SCORE =====

    def composite_health_score(self, data: FinancialData,
                                prior_data: Optional[FinancialData] = None) -> CompositeHealthScore:
        """Compute a weighted 0-100 composite financial health score with letter grade.

        Components (100 total):
          - Z-Score zone:     25 pts
          - F-Score:          25 pts
          - Profitability:    20 pts
          - Liquidity:        15 pts
          - Leverage:         15 pts
        """
        component_scores: Dict[str, int] = {}

        # 1. Z-Score component (25 points)
        z = self.altman_z_score(data)
        if z.zone == 'safe':
            z_pts = 25
        elif z.zone == 'grey':
            z_pts = 15
        elif z.zone == 'partial':
            z_pts = 10
        elif z.zone == 'distress':
            z_pts = 0
        else:
            z_pts = 5
        component_scores['z_score'] = z_pts

        # 2. F-Score component (25 points)
        f = self.piotroski_f_score(data, prior_data)
        f_pts = round(f.score / f.max_score * 25)
        component_scores['f_score'] = f_pts

        # 3. Profitability component (20 points)
        prof = self.calculate_profitability_ratios(data)
        prof_pts = 0

        nm = prof.get('net_margin')
        if nm is not None:
            if nm > 0.15:
                prof_pts += 8
            elif nm > 0.05:
                prof_pts += 5
            elif nm > 0:
                prof_pts += 3

        roa_val = prof.get('roa')
        if roa_val is not None:
            if roa_val > 0.05:
                prof_pts += 6
            elif roa_val > 0.02:
                prof_pts += 4
            elif roa_val > 0:
                prof_pts += 2

        roe_val = prof.get('roe')
        if roe_val is not None:
            if roe_val > 0.15:
                prof_pts += 6
            elif roe_val > 0.08:
                prof_pts += 4
            elif roe_val > 0:
                prof_pts += 2

        component_scores['profitability'] = prof_pts

        # 4. Liquidity component (15 points)
        liq = self.calculate_liquidity_ratios(data)
        cr = liq.get('current_ratio')
        if cr is not None:
            if cr >= 2.0:
                liq_pts = 15
            elif cr >= 1.5:
                liq_pts = 12
            elif cr >= 1.0:
                liq_pts = 8
            elif cr >= 0.5:
                liq_pts = 4
            else:
                liq_pts = 0
        else:
            liq_pts = 0
        component_scores['liquidity'] = liq_pts

        # 5. Leverage component (15 points)
        lev = self.calculate_leverage_ratios(data)
        de = lev.get('debt_to_equity')
        if de is not None:
            if de <= 0.5:
                lev_pts = 15
            elif de <= 1.0:
                lev_pts = 12
            elif de <= 2.0:
                lev_pts = 8
            elif de <= 3.0:
                lev_pts = 4
            else:
                lev_pts = 0
        else:
            lev_pts = 0
        component_scores['leverage'] = lev_pts

        total = sum(component_scores.values())

        # Letter grade
        if total >= 80:
            grade = 'A'
        elif total >= 65:
            grade = 'B'
        elif total >= 50:
            grade = 'C'
        elif total >= 35:
            grade = 'D'
        else:
            grade = 'F'

        strongest = max(component_scores, key=component_scores.get)
        weakest = min(component_scores, key=component_scores.get)
        interpretation = (
            f"Composite score {total}/100 (Grade {grade}). "
            f"Strongest: {strongest.replace('_', ' ')}. "
            f"Weakest: {weakest.replace('_', ' ')}."
        )

        return CompositeHealthScore(
            score=total,
            grade=grade,
            component_scores=component_scores,
            interpretation=interpretation,
        )

    # ===== MULTI-PERIOD COMPARISON =====

    def compare_periods(self, current: FinancialData,
                         prior: FinancialData) -> PeriodComparison:
        """Compare two periods and compute deltas for all financial metrics."""
        current_ratios: Dict[str, Optional[float]] = {}
        prior_ratios: Dict[str, Optional[float]] = {}

        for category_name, method in [
            ('liquidity', self.calculate_liquidity_ratios),
            ('profitability', self.calculate_profitability_ratios),
            ('leverage', self.calculate_leverage_ratios),
            ('efficiency', self.calculate_efficiency_ratios),
        ]:
            c = method(current)
            p = method(prior)
            for k, v in c.items():
                current_ratios[f"{category_name}_{k}"] = v
            for k, v in p.items():
                prior_ratios[f"{category_name}_{k}"] = v

        # Scoring models
        c_z = self.altman_z_score(current)
        p_z = self.altman_z_score(prior)
        current_ratios['altman_z_score'] = c_z.z_score
        prior_ratios['altman_z_score'] = p_z.z_score

        c_f = self.piotroski_f_score(current, prior)
        p_f = self.piotroski_f_score(prior)
        current_ratios['piotroski_f_score'] = float(c_f.score)
        prior_ratios['piotroski_f_score'] = float(p_f.score)

        # Compute deltas
        deltas: Dict[str, float] = {}
        improvements: List[str] = []
        deteriorations: List[str] = []

        lower_is_better = {'leverage_debt_to_equity', 'leverage_debt_to_assets'}

        for key in current_ratios:
            cv = current_ratios.get(key)
            pv = prior_ratios.get(key)
            if cv is not None and pv is not None:
                delta = cv - pv
                deltas[key] = delta

                if key in lower_is_better:
                    if delta < -1e-6:
                        improvements.append(key)
                    elif delta > 1e-6:
                        deteriorations.append(key)
                else:
                    if delta > 1e-6:
                        improvements.append(key)
                    elif delta < -1e-6:
                        deteriorations.append(key)

        return PeriodComparison(
            current_ratios=current_ratios,
            prior_ratios=prior_ratios,
            deltas=deltas,
            improvements=improvements,
            deteriorations=deteriorations,
        )

    # ===== REPORT GENERATION =====

    def generate_report(self, data: FinancialData,
                         prior_data: Optional[FinancialData] = None) -> FinancialReport:
        """Generate a structured financial analysis report.

        Returns a FinancialReport with sections for executive summary,
        ratio analysis, scoring models, risk assessment, and recommendations.
        """
        sections: Dict[str, str] = {}

        results = self.analyze(data)
        health = self.composite_health_score(data, prior_data)

        # --- Executive Summary ---
        lines = [f"Overall Financial Health: {health.grade} ({health.score}/100)"]
        z = results['altman_z_score']
        if z.z_score is not None:
            lines.append(f"Bankruptcy Risk: {z.zone.title()} (Z-Score: {z.z_score:.2f})")
        f_result = results['piotroski_f_score']
        lines.append(f"Financial Strength: {f_result.score}/{f_result.max_score} (Piotroski F-Score)")

        prof = results['profitability_ratios']
        if prof.get('net_margin') is not None:
            lines.append(f"Net Margin: {prof['net_margin']:.1%}")
        if prof.get('roe') is not None:
            lines.append(f"Return on Equity: {prof['roe']:.1%}")

        sections['executive_summary'] = '\n'.join(lines)

        # --- Ratio Analysis ---
        ratio_lines: List[str] = []
        for category, ratios in [
            ('Liquidity', results['liquidity_ratios']),
            ('Profitability', results['profitability_ratios']),
            ('Leverage', results['leverage_ratios']),
            ('Efficiency', results['efficiency_ratios']),
        ]:
            available = {k: v for k, v in ratios.items() if v is not None}
            if available:
                ratio_lines.append(f"\n{category}:")
                for k, v in available.items():
                    label = k.replace('_', ' ').title()
                    if any(x in k for x in ('margin', 'roe', 'roa', 'roic')):
                        ratio_lines.append(f"  {label}: {v:.1%}")
                    else:
                        ratio_lines.append(f"  {label}: {v:.2f}")

        sections['ratio_analysis'] = '\n'.join(ratio_lines) if ratio_lines else 'Insufficient data for ratio analysis.'

        # --- Scoring Models ---
        scoring_lines: List[str] = []
        dupont = results['dupont']
        if dupont.roe is not None:
            scoring_lines.append(f"DuPont ROE: {dupont.roe:.1%}")
            if dupont.net_margin is not None:
                scoring_lines.append(f"  Net Margin: {dupont.net_margin:.1%}")
            if dupont.asset_turnover is not None:
                scoring_lines.append(f"  Asset Turnover: {dupont.asset_turnover:.2f}x")
            if dupont.equity_multiplier is not None:
                scoring_lines.append(f"  Equity Multiplier: {dupont.equity_multiplier:.2f}x")

        if z.z_score is not None:
            scoring_lines.append(f"\nAltman Z-Score: {z.z_score:.2f} ({z.zone})")
        scoring_lines.append(f"Piotroski F-Score: {f_result.score}/{f_result.max_score}")
        scoring_lines.append(f"Composite Health: {health.score}/100 (Grade {health.grade})")

        sections['scoring_models'] = '\n'.join(scoring_lines)

        # --- Risk Assessment ---
        risk_lines: List[str] = []
        insights = results['insights']
        warnings = [i for i in insights if i.severity in ('warning', 'critical')]
        if warnings:
            for w in warnings:
                risk_lines.append(f"[{w.severity.upper()}] {w.message}")
        else:
            risk_lines.append("No significant risk factors identified.")
        sections['risk_assessment'] = '\n'.join(risk_lines)

        # --- Recommendations ---
        recs = [i.recommendation for i in insights if i.recommendation]
        sections['recommendations'] = (
            '\n'.join(f"- {r}" for r in recs)
            if recs
            else 'No specific recommendations at this time.'
        )

        # --- Period Comparison (if prior data available) ---
        if prior_data is not None:
            comparison = self.compare_periods(data, prior_data)
            comp_lines: List[str] = []
            if comparison.improvements:
                comp_lines.append("Improvements:")
                for m in comparison.improvements[:5]:
                    delta = comparison.deltas[m]
                    comp_lines.append(f"  + {m.replace('_', ' ').title()}: {delta:+.4f}")
            if comparison.deteriorations:
                comp_lines.append("Deteriorations:")
                for m in comparison.deteriorations[:5]:
                    delta = comparison.deltas[m]
                    comp_lines.append(f"  - {m.replace('_', ' ').title()}: {delta:+.4f}")
            sections['period_comparison'] = (
                '\n'.join(comp_lines)
                if comp_lines
                else 'No significant changes detected.'
            )

        return FinancialReport(
            sections=sections,
            executive_summary=sections['executive_summary'],
            generated_at=datetime.now().isoformat(),
        )

    # ===== INSIGHT GENERATION =====

    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[Insight]:
        """
        Generate natural language insights using Charlie Munger principles.
        """
        insights = []

        # Liquidity insights
        ratios = analysis_results.get('liquidity_ratios', {})
        if ratios.get('current_ratio') is not None:
            cr = ratios['current_ratio']
            if cr >= 1.5:
                insights.append(Insight(
                    category='liquidity',
                    severity='info',
                    message=f"Strong liquidity: Current ratio of {cr:.2f} indicates healthy ability to meet short-term obligations.",
                    metric_name='current_ratio',
                    metric_value=cr
                ))
            elif cr < 1.0:
                insights.append(Insight(
                    category='liquidity',
                    severity='warning',
                    message=f"Liquidity concern: Current ratio of {cr:.2f} below 1.0 suggests potential difficulty meeting short-term obligations.",
                    metric_name='current_ratio',
                    metric_value=cr,
                    recommendation="Review working capital management and consider improving collection cycles or negotiating extended payment terms."
                ))

        # Profitability insights
        prof_ratios = analysis_results.get('profitability_ratios', {})
        if prof_ratios.get('net_margin') is not None:
            margin = prof_ratios['net_margin']
            if margin > 0.15:
                insights.append(Insight(
                    category='profitability',
                    severity='info',
                    message=f"Strong profitability: Net margin of {margin:.1%} indicates excellent bottom-line performance.",
                    metric_name='net_margin',
                    metric_value=margin
                ))
            elif margin < 0.02:
                insights.append(Insight(
                    category='profitability',
                    severity='warning',
                    message=f"Thin margins: Net margin of {margin:.1%} leaves little room for error.",
                    metric_name='net_margin',
                    metric_value=margin,
                    recommendation="Analyze cost structure and pricing strategy. Consider operational efficiency improvements."
                ))

        # ROE insights (Charlie Munger focus)
        if prof_ratios.get('roe') is not None:
            roe = prof_ratios['roe']
            if roe > 0.20:
                insights.append(Insight(
                    category='profitability',
                    severity='info',
                    message=f"Excellent capital efficiency: ROE of {roe:.1%} suggests strong competitive advantages.",
                    metric_name='roe',
                    metric_value=roe,
                    recommendation="Investigate sources of high returns - sustainable competitive moats or temporary factors?"
                ))

        # Leverage insights
        lev_ratios = analysis_results.get('leverage_ratios', {})
        if lev_ratios.get('debt_to_equity') is not None:
            de = lev_ratios['debt_to_equity']
            if de > 2.0:
                insights.append(Insight(
                    category='risk',
                    severity='warning',
                    message=f"High leverage: Debt-to-equity of {de:.2f} increases financial risk.",
                    metric_name='debt_to_equity',
                    metric_value=de,
                    recommendation="Monitor debt service coverage. Consider deleveraging if cash flows are volatile."
                ))

        # Cash flow insights (Charlie Munger emphasis)
        cf_analysis = analysis_results.get('cash_flow', None)
        if cf_analysis and cf_analysis.free_cash_flow is not None:
            fcf = cf_analysis.free_cash_flow
            if fcf > 0:
                insights.append(Insight(
                    category='cash_flow',
                    severity='info',
                    message=f"Positive free cash flow of {fcf:,.0f} provides strategic flexibility.",
                    metric_name='free_cash_flow',
                    metric_value=fcf,
                    recommendation="Strong cash generation enables reinvestment, debt reduction, or shareholder returns."
                ))
            else:
                insights.append(Insight(
                    category='cash_flow',
                    severity='warning',
                    message=f"Negative free cash flow of {fcf:,.0f} requires external financing or asset monetization.",
                    metric_name='free_cash_flow',
                    metric_value=fcf,
                    recommendation="Investigate cash burn drivers. Assess sustainability of current capital structure."
                ))

        # Cash conversion cycle
        if cf_analysis and cf_analysis.cash_conversion_cycle is not None:
            ccc = cf_analysis.cash_conversion_cycle
            if ccc > 60:
                insights.append(Insight(
                    category='efficiency',
                    severity='warning',
                    message=f"Extended cash conversion cycle of {ccc:.0f} days ties up working capital.",
                    metric_name='cash_conversion_cycle',
                    metric_value=ccc,
                    recommendation="Optimize inventory management, accelerate collections, and negotiate better payment terms."
                ))

        # DuPont decomposition insights
        dupont = analysis_results.get('dupont')
        if dupont and dupont.roe is not None:
            insights.append(Insight(
                category='profitability',
                severity='info',
                message=dupont.interpretation or f"DuPont ROE: {dupont.roe:.1%}",
                metric_name='dupont_roe',
                metric_value=dupont.roe,
                recommendation=(
                    f"Primary ROE driver is {dupont.primary_driver.replace('_', ' ')}. "
                    "Focus improvement efforts there."
                ) if dupont.primary_driver else None
            ))

        # Altman Z-Score insights
        z_result = analysis_results.get('altman_z_score')
        if z_result and z_result.z_score is not None:
            severity = 'info'
            if z_result.zone == 'distress':
                severity = 'critical'
            elif z_result.zone == 'grey':
                severity = 'warning'

            rec = None
            if z_result.zone == 'distress':
                rec = "Urgent: Review capital structure, reduce leverage, and improve profitability immediately."
            elif z_result.zone == 'grey':
                rec = "Monitor closely. Strengthen working capital and profitability to move into the safe zone."

            insights.append(Insight(
                category='risk',
                severity=severity,
                message=z_result.interpretation or f"Altman Z-Score: {z_result.z_score:.2f}",
                metric_name='altman_z_score',
                metric_value=z_result.z_score,
                recommendation=rec
            ))

        # Piotroski F-Score insights
        f_result = analysis_results.get('piotroski_f_score')
        if f_result and f_result.score is not None:
            severity = 'info'
            if f_result.score <= 3:
                severity = 'warning'

            # Find weak criteria
            weak = [k.replace('_', ' ') for k, v in f_result.criteria.items() if not v]
            rec = None
            if weak:
                rec = f"Weak areas: {', '.join(weak[:3])}. Address these to strengthen financial position."

            insights.append(Insight(
                category='risk',
                severity=severity,
                message=f_result.interpretation or f"Piotroski F-Score: {f_result.score}/9",
                metric_name='piotroski_f_score',
                metric_value=float(f_result.score),
                recommendation=rec
            ))

        return insights

    def detect_anomalies(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'zscore', threshold: float = 2.0) -> List[Anomaly]:
        """Flag unusual patterns using z-score or IQR method.

        Args:
            data: DataFrame containing numeric data to analyze.
            columns: Specific columns to check. Defaults to all numeric columns.
            method: Detection method - 'zscore' (default) or 'iqr'.
            threshold: For zscore: number of std devs (default 2.0).
                       For IQR: multiplier of IQR (default 1.5, but uses threshold param).
        """
        anomalies = []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            series = data[col].dropna()
            if len(series) < 3:
                continue

            if method == 'iqr':
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                iqr_multiplier = threshold if threshold != 2.0 else 1.5
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr

                mean = series.mean()
                std = series.std() or 1.0  # avoid div by zero for z_score calc

                for idx, value in series.items():
                    if value < lower or value > upper:
                        z = (value - mean) / std
                        anomalies.append(Anomaly(
                            metric_name=col,
                            value=value,
                            expected_range=(lower, upper),
                            z_score=z,
                            description=(f"IQR anomaly in {col}: {value:.2f} outside "
                                         f"[{lower:.2f}, {upper:.2f}]")
                        ))
            else:
                # Default z-score method
                mean = series.mean()
                std = series.std()
                if std == 0:
                    continue

                for idx, value in series.items():
                    z_score = (value - mean) / std
                    if abs(z_score) > threshold:
                        anomalies.append(Anomaly(
                            metric_name=col,
                            value=value,
                            expected_range=(mean - threshold*std, mean + threshold*std),
                            z_score=z_score,
                            description=(f"Unusual value in {col}: {value:.2f} is "
                                         f"{abs(z_score):.1f} standard deviations from mean")
                        ))

        return anomalies

    # ===== SCENARIO / WHAT-IF ANALYSIS =====

    def _apply_adjustments(self, data: FinancialData,
                           adjustments: Dict[str, float]) -> FinancialData:
        """Create a copy of FinancialData with percentage adjustments applied.

        Args:
            data: Base financial data.
            adjustments: Dict mapping field names to multipliers.
                         e.g. {'revenue': 1.10} means +10% revenue.
        """
        import copy
        adjusted = copy.deepcopy(data)
        for field_name, multiplier in adjustments.items():
            current_val = getattr(adjusted, field_name, None)
            if current_val is not None:
                setattr(adjusted, field_name, current_val * multiplier)
        return adjusted

    def scenario_analysis(self, data: FinancialData,
                          adjustments: Dict[str, float],
                          scenario_name: str = "Custom Scenario") -> ScenarioResult:
        """Run what-if analysis by applying adjustments and comparing all metrics.

        Args:
            data: Base FinancialData.
            adjustments: Field-name-to-multiplier map (e.g. {'revenue': 1.10}).
            scenario_name: Human-readable label for the scenario.

        Returns:
            ScenarioResult comparing base vs scenario across all key metrics.
        """
        adjusted = self._apply_adjustments(data, adjustments)

        # Base metrics
        base_health = self.composite_health_score(data)
        base_z = self.altman_z_score(data)
        base_f = self.piotroski_f_score(data)
        base_liq = self.calculate_liquidity_ratios(data)
        base_prof = self.calculate_profitability_ratios(data)
        base_lev = self.calculate_leverage_ratios(data)

        # Scenario metrics
        scen_health = self.composite_health_score(adjusted)
        scen_z = self.altman_z_score(adjusted)
        scen_f = self.piotroski_f_score(adjusted)
        scen_liq = self.calculate_liquidity_ratios(adjusted)
        scen_prof = self.calculate_profitability_ratios(adjusted)
        scen_lev = self.calculate_leverage_ratios(adjusted)

        # Collect key ratios
        base_ratios: Dict[str, Optional[float]] = {}
        scenario_ratios: Dict[str, Optional[float]] = {}
        for prefix, base_r, scen_r in [
            ('', base_liq, scen_liq),
            ('', base_prof, scen_prof),
            ('', base_lev, scen_lev),
        ]:
            for k, v in base_r.items():
                base_ratios[k] = v
            for k, v in scen_r.items():
                scenario_ratios[k] = v

        base_ratios['health_score'] = float(base_health.score)
        scenario_ratios['health_score'] = float(scen_health.score)
        base_ratios['z_score'] = base_z.z_score
        scenario_ratios['z_score'] = scen_z.z_score

        # Build impact summary
        health_delta = scen_health.score - base_health.score
        z_delta = ((scen_z.z_score or 0) - (base_z.z_score or 0))
        adj_strs = [f"{k} {'+'if v>=1 else ''}{(v-1)*100:+.0f}%"
                    for k, v in adjustments.items()]
        summary_parts = [
            f"Scenario '{scenario_name}': {', '.join(adj_strs)}.",
            f"Health score: {base_health.score} -> {scen_health.score} ({health_delta:+d}).",
            f"Z-Score: {base_z.z_score:.2f} -> {scen_z.z_score:.2f} ({z_delta:+.2f})." if base_z.z_score and scen_z.z_score else "",
            f"Grade: {base_health.grade} -> {scen_health.grade}.",
        ]

        return ScenarioResult(
            scenario_name=scenario_name,
            adjustments=adjustments,
            base_health=base_health,
            scenario_health=scen_health,
            base_z_score=base_z.z_score,
            scenario_z_score=scen_z.z_score,
            base_f_score=base_f.score,
            scenario_f_score=scen_f.score,
            base_ratios=base_ratios,
            scenario_ratios=scenario_ratios,
            impact_summary=' '.join(p for p in summary_parts if p),
        )

    def sensitivity_analysis(self, data: FinancialData, variable: str,
                             pct_range: Optional[List[float]] = None) -> SensitivityResult:
        """Vary a single financial variable and measure impact on key metrics.

        Args:
            data: Base FinancialData.
            variable: Field name to vary (e.g. 'revenue').
            pct_range: List of percentage changes to test.
                       Defaults to [-20, -15, -10, -5, 0, +5, +10, +15, +20].

        Returns:
            SensitivityResult with metric values at each variation point.
        """
        if pct_range is None:
            pct_range = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

        labels = [f"{p:+d}%" for p in pct_range]
        multipliers = [1 + p / 100.0 for p in pct_range]

        metrics: Dict[str, List[Optional[float]]] = {
            'health_score': [],
            'z_score': [],
            'f_score': [],
            'current_ratio': [],
            'net_margin': [],
            'debt_to_equity': [],
            'roe': [],
        }

        for mult in multipliers:
            adjusted = self._apply_adjustments(data, {variable: mult})
            health = self.composite_health_score(adjusted)
            z = self.altman_z_score(adjusted)
            f = self.piotroski_f_score(adjusted)
            liq = self.calculate_liquidity_ratios(adjusted)
            prof = self.calculate_profitability_ratios(adjusted)
            lev = self.calculate_leverage_ratios(adjusted)

            metrics['health_score'].append(float(health.score))
            metrics['z_score'].append(z.z_score)
            metrics['f_score'].append(float(f.score))
            metrics['current_ratio'].append(liq.get('current_ratio'))
            metrics['net_margin'].append(prof.get('net_margin'))
            metrics['debt_to_equity'].append(lev.get('debt_to_equity'))
            metrics['roe'].append(prof.get('roe'))

        return SensitivityResult(
            variable_name=variable,
            variable_labels=labels,
            variable_multipliers=multipliers,
            metric_results=metrics,
        )

    # ===== MONTE CARLO SIMULATION =====

    def monte_carlo_simulation(
        self,
        data: FinancialData,
        assumptions: Optional[Dict[str, Dict[str, float]]] = None,
        n_simulations: int = 1000,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation to model uncertainty in financial outcomes.

        Args:
            data: Base FinancialData.
            assumptions: Dict mapping field names to distribution params.
                         Each entry: {'mean_pct': 0.0, 'std_pct': 10.0}
                         meaning the variable is normally distributed around
                         base value with std of 10% of base.
                         Defaults to revenue/cogs/operating_expenses at 10% std.
            n_simulations: Number of random scenarios to run (default 1000).

        Returns:
            MonteCarloResult with distributions and percentiles.
        """
        if assumptions is None:
            assumptions = {}
            for fld in ['revenue', 'cogs', 'operating_expenses']:
                if getattr(data, fld, None) is not None:
                    assumptions[fld] = {'mean_pct': 0.0, 'std_pct': 10.0}

        if not assumptions:
            return MonteCarloResult(
                n_simulations=0,
                variable_assumptions=assumptions,
                summary="No variables with data to simulate.",
            )

        rng = np.random.default_rng(seed=42)

        metrics_collected: Dict[str, List[float]] = {
            'health_score': [],
            'z_score': [],
            'f_score': [],
            'net_margin': [],
            'current_ratio': [],
            'roe': [],
        }

        for _ in range(n_simulations):
            adjustments: Dict[str, float] = {}
            for fld, params in assumptions.items():
                mean_mult = 1.0 + params.get('mean_pct', 0.0) / 100.0
                std_mult = params.get('std_pct', 10.0) / 100.0
                sample = rng.normal(mean_mult, std_mult)
                sample = max(sample, 0.01)  # Floor at 1% to avoid negatives
                adjustments[fld] = sample

            adjusted = self._apply_adjustments(data, adjustments)
            health = self.composite_health_score(adjusted)
            z = self.altman_z_score(adjusted)
            f = self.piotroski_f_score(adjusted)
            prof = self.calculate_profitability_ratios(adjusted)
            liq = self.calculate_liquidity_ratios(adjusted)

            metrics_collected['health_score'].append(float(health.score))
            metrics_collected['z_score'].append(z.z_score if z.z_score is not None else 0.0)
            metrics_collected['f_score'].append(float(f.score))
            metrics_collected['net_margin'].append(prof.get('net_margin') or 0.0)
            metrics_collected['current_ratio'].append(liq.get('current_ratio') or 0.0)
            metrics_collected['roe'].append(prof.get('roe') or 0.0)

        # Compute percentiles
        percentiles: Dict[str, Dict[str, float]] = {}
        for metric, values in metrics_collected.items():
            arr = np.array(values)
            percentiles[metric] = {
                'p10': float(np.percentile(arr, 10)),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
            }

        # Summary
        hs = percentiles.get('health_score', {})
        summary = (
            f"Monte Carlo ({n_simulations} simulations): "
            f"Health Score median={hs.get('p50', 0):.0f}, "
            f"range=[{hs.get('p10', 0):.0f}, {hs.get('p90', 0):.0f}] (10th-90th). "
            f"Variables: {', '.join(assumptions.keys())}."
        )

        return MonteCarloResult(
            n_simulations=n_simulations,
            variable_assumptions=assumptions,
            metric_distributions=metrics_collected,
            percentiles=percentiles,
            summary=summary,
        )

    # ===== CASH FLOW FORECASTING =====

    def forecast_cashflow(
        self,
        data: FinancialData,
        periods: int = 12,
        revenue_growth: float = 0.05,
        expense_ratio: Optional[float] = None,
        capex_ratio: float = 0.05,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.02,
    ) -> CashFlowForecast:
        """Forecast cash flows over multiple periods with DCF valuation.

        Args:
            data: Base FinancialData with current financials.
            periods: Number of forecast periods (default 12).
            revenue_growth: Annual revenue growth rate (default 5%).
            expense_ratio: Operating expenses as fraction of revenue.
                          If None, derived from current data.
            capex_ratio: CapEx as fraction of revenue (default 5%).
            discount_rate: WACC / required return (default 10%).
            terminal_growth: Long-term growth for terminal value (default 2%).

        Returns:
            CashFlowForecast with period-by-period projections and DCF value.
        """
        base_revenue = data.revenue or 0.0
        if base_revenue <= 0:
            return CashFlowForecast(periods=[])

        # Derive expense ratio from current data if not provided
        if expense_ratio is None:
            total_expenses = (data.cogs or 0) + (data.operating_expenses or 0)
            expense_ratio = safe_divide(total_expenses, base_revenue, default=0.70) or 0.70

        period_labels = []
        rev_forecast = []
        exp_forecast = []
        net_cf = []
        cum_cash = []
        fcf_list = []
        growth_rates = []

        current_rev = base_revenue
        cumulative = 0.0
        pv_fcf_sum = 0.0

        for i in range(1, periods + 1):
            current_rev = current_rev * (1 + revenue_growth)
            expenses = current_rev * expense_ratio
            capex = current_rev * capex_ratio
            fcf = current_rev - expenses - capex
            net = current_rev - expenses
            cumulative += net

            # PV of FCF for DCF
            pv_fcf = fcf / ((1 + discount_rate) ** i)
            pv_fcf_sum += pv_fcf

            period_labels.append(f"Period {i}")
            rev_forecast.append(round(current_rev, 2))
            exp_forecast.append(round(expenses, 2))
            net_cf.append(round(net, 2))
            cum_cash.append(round(cumulative, 2))
            fcf_list.append(round(fcf, 2))
            growth_rates.append(round(revenue_growth * 100, 2))

        # Terminal value (Gordon Growth Model)
        if discount_rate > terminal_growth:
            terminal_fcf = fcf_list[-1] * (1 + terminal_growth)
            tv = terminal_fcf / (discount_rate - terminal_growth)
            pv_tv = tv / ((1 + discount_rate) ** periods)
        else:
            tv = 0.0
            pv_tv = 0.0

        dcf_value = pv_fcf_sum + pv_tv

        return CashFlowForecast(
            periods=period_labels,
            revenue_forecast=rev_forecast,
            expense_forecast=exp_forecast,
            net_cash_flow=net_cf,
            cumulative_cash=cum_cash,
            fcf_forecast=fcf_list,
            dcf_value=round(dcf_value, 2),
            terminal_value=round(tv, 2),
            discount_rate=discount_rate,
            growth_rates=growth_rates,
        )

    # ===== TORNADO / DRIVER ANALYSIS =====

    def tornado_analysis(
        self,
        data: FinancialData,
        target_metric: str = "health_score",
        variables: Optional[List[str]] = None,
        pct_swing: float = 10.0,
    ) -> TornadoResult:
        """Rank which financial variables have the greatest impact on a target metric.

        For each variable, applies +/- pct_swing% and measures the resulting
        change in the target metric, then ranks by spread (high - low).

        Args:
            data: Base FinancialData.
            target_metric: Metric to measure impact on. One of:
                'health_score', 'z_score', 'f_score', 'net_margin',
                'current_ratio', 'roe', 'debt_to_equity'.
            variables: List of FinancialData field names to vary.
                       If None, auto-detects non-None numeric fields.
            pct_swing: Percentage swing to apply (+/- this value).

        Returns:
            TornadoResult with drivers ranked by spread (descending).
        """
        # Auto-detect variables if not provided
        if variables is None:
            candidate_fields = [
                'revenue', 'cogs', 'operating_expenses', 'net_income',
                'total_assets', 'total_liabilities', 'total_equity',
                'current_assets', 'current_liabilities', 'total_debt',
                'ebit', 'ebitda', 'interest_expense', 'depreciation',
                'inventory', 'accounts_receivable', 'accounts_payable',
                'capex', 'operating_cash_flow', 'retained_earnings',
            ]
            variables = [f for f in candidate_fields
                         if getattr(data, f, None) is not None]

        if not variables:
            return TornadoResult(target_metric=target_metric)

        def _get_metric_value(d: FinancialData) -> float:
            """Extract a single metric value from FinancialData."""
            if target_metric == 'health_score':
                return float(self.composite_health_score(d).score)
            elif target_metric == 'z_score':
                return self.altman_z_score(d).z_score or 0.0
            elif target_metric == 'f_score':
                return float(self.piotroski_f_score(d).score)
            elif target_metric == 'net_margin':
                return self.calculate_profitability_ratios(d).get('net_margin') or 0.0
            elif target_metric == 'current_ratio':
                return self.calculate_liquidity_ratios(d).get('current_ratio') or 0.0
            elif target_metric == 'roe':
                return self.calculate_profitability_ratios(d).get('roe') or 0.0
            elif target_metric == 'debt_to_equity':
                return self.calculate_leverage_ratios(d).get('debt_to_equity') or 0.0
            return 0.0

        base_value = _get_metric_value(data)
        low_mult = 1 - pct_swing / 100.0
        high_mult = 1 + pct_swing / 100.0

        drivers: List[TornadoDriver] = []
        for var in variables:
            low_data = self._apply_adjustments(data, {var: low_mult})
            high_data = self._apply_adjustments(data, {var: high_mult})
            low_val = _get_metric_value(low_data)
            high_val = _get_metric_value(high_data)
            spread = abs(high_val - low_val)
            drivers.append(TornadoDriver(
                variable=var,
                low_value=round(low_val, 4),
                high_value=round(high_val, 4),
                base_value=round(base_value, 4),
                spread=round(spread, 4),
            ))

        # Sort by spread descending (most impactful first)
        drivers.sort(key=lambda d: d.spread, reverse=True)
        top_driver = drivers[0].variable if drivers else ""

        return TornadoResult(
            target_metric=target_metric,
            base_metric_value=round(base_value, 4),
            drivers=drivers,
            top_driver=top_driver,
        )

    # ===== BREAKEVEN ANALYSIS =====

    def breakeven_analysis(self, data: FinancialData) -> BreakevenResult:
        """Calculate breakeven revenue using contribution margin approach.

        Estimates fixed and variable costs from the financial data, then
        computes the revenue needed to cover all fixed costs (zero profit).

        Args:
            data: FinancialData with revenue, cogs, and operating_expenses.

        Returns:
            BreakevenResult with breakeven revenue and margin of safety.
        """
        revenue = data.revenue or 0.0
        if revenue <= 0:
            return BreakevenResult()

        cogs = data.cogs or 0.0
        opex = data.operating_expenses or 0.0

        # Estimate variable costs = COGS (scales with revenue)
        # Estimate fixed costs = operating expenses (relatively fixed)
        variable_cost_ratio = safe_divide(cogs, revenue, default=0.0) or 0.0
        contribution_margin_ratio = 1.0 - variable_cost_ratio

        if contribution_margin_ratio <= 0:
            return BreakevenResult(
                current_revenue=revenue,
                fixed_costs=opex,
                variable_cost_ratio=round(variable_cost_ratio, 4),
                contribution_margin_ratio=0.0,
            )

        # Fixed costs include opex + interest expense
        fixed_costs = opex + (data.interest_expense or 0.0)

        breakeven_rev = safe_divide(fixed_costs, contribution_margin_ratio, default=None)
        margin_of_safety = None
        if breakeven_rev is not None and revenue > 0:
            margin_of_safety = safe_divide(
                revenue - breakeven_rev, revenue, default=None
            )

        return BreakevenResult(
            breakeven_revenue=round(breakeven_rev, 2) if breakeven_rev is not None else None,
            current_revenue=round(revenue, 2),
            margin_of_safety=round(margin_of_safety, 4) if margin_of_safety is not None else None,
            fixed_costs=round(fixed_costs, 2),
            variable_cost_ratio=round(variable_cost_ratio, 4),
            contribution_margin_ratio=round(contribution_margin_ratio, 4),
        )

    # ===== COVENANT / KPI MONITORING =====

    def covenant_monitor(
        self,
        data: FinancialData,
        covenants: Optional[List[Dict[str, Any]]] = None,
    ) -> CovenantMonitorResult:
        """Check financial data against a set of covenant/KPI thresholds.

        Each covenant specifies a metric, a threshold, and a direction
        ('above' means passing if current >= threshold, 'below' if <=).

        A 'warning' is triggered when the value is within 10% of the threshold.

        Args:
            data: FinancialData to evaluate.
            covenants: List of covenant dicts, each with keys:
                - 'name': Display name
                - 'metric': One of 'current_ratio', 'debt_to_equity',
                  'net_margin', 'roe', 'interest_coverage', 'health_score',
                  'dscr' (debt service coverage ratio).
                - 'threshold': Numeric threshold value
                - 'direction': 'above' or 'below' (default 'above')
                If None, uses standard default covenants.

        Returns:
            CovenantMonitorResult with per-covenant checks and summary counts.
        """
        if covenants is None:
            covenants = [
                {'name': 'Current Ratio', 'metric': 'current_ratio',
                 'threshold': 1.5, 'direction': 'above'},
                {'name': 'Debt-to-Equity', 'metric': 'debt_to_equity',
                 'threshold': 2.0, 'direction': 'below'},
                {'name': 'Net Margin', 'metric': 'net_margin',
                 'threshold': 0.05, 'direction': 'above'},
                {'name': 'Interest Coverage', 'metric': 'interest_coverage',
                 'threshold': 3.0, 'direction': 'above'},
                {'name': 'Health Score', 'metric': 'health_score',
                 'threshold': 50.0, 'direction': 'above'},
            ]

        # Gather all metric values
        liq = self.calculate_liquidity_ratios(data)
        prof = self.calculate_profitability_ratios(data)
        lev = self.calculate_leverage_ratios(data)
        health = self.composite_health_score(data)

        # DSCR = EBITDA / (interest_expense + debt repayment)
        # Simplified: EBITDA / interest_expense if no repayment data
        dscr = safe_divide(data.ebitda, data.interest_expense, default=None)

        metric_values = {
            'current_ratio': liq.get('current_ratio'),
            'quick_ratio': liq.get('quick_ratio'),
            'debt_to_equity': lev.get('debt_to_equity'),
            'debt_to_assets': lev.get('debt_to_assets'),
            'net_margin': prof.get('net_margin'),
            'roe': prof.get('roe'),
            'roa': prof.get('roa'),
            'interest_coverage': lev.get('interest_coverage'),
            'health_score': float(health.score),
            'dscr': dscr,
        }

        checks: List[CovenantCheck] = []
        passes = 0
        warnings = 0
        breaches = 0

        for cov in covenants:
            name = cov.get('name', cov.get('metric', 'Unknown'))
            metric_key = cov.get('metric', '')
            threshold = float(cov.get('threshold', 0))
            direction = cov.get('direction', 'above')

            current = metric_values.get(metric_key)

            if current is None:
                checks.append(CovenantCheck(
                    name=name, current_value=None,
                    threshold=threshold, direction=direction,
                    status='unknown', headroom=None,
                ))
                continue

            # Determine pass/warning/breach
            warning_zone = threshold * 0.10  # 10% of threshold

            if direction == 'above':
                headroom = current - threshold
                if current >= threshold:
                    if current < threshold + warning_zone:
                        status = 'warning'
                        warnings += 1
                    else:
                        status = 'pass'
                        passes += 1
                else:
                    status = 'breach'
                    breaches += 1
            else:  # 'below'
                headroom = threshold - current
                if current <= threshold:
                    if current > threshold - warning_zone:
                        status = 'warning'
                        warnings += 1
                    else:
                        status = 'pass'
                        passes += 1
                else:
                    status = 'breach'
                    breaches += 1

            checks.append(CovenantCheck(
                name=name,
                current_value=round(current, 4),
                threshold=threshold,
                direction=direction,
                status=status,
                headroom=round(headroom, 4),
            ))

        # Build summary
        total = len(checks)
        unknown = total - passes - warnings - breaches
        parts = []
        if breaches > 0:
            parts.append(f"{breaches} BREACH(ES)")
        if warnings > 0:
            parts.append(f"{warnings} warning(s)")
        if passes > 0:
            parts.append(f"{passes} passing")
        if unknown > 0:
            parts.append(f"{unknown} unknown")
        summary = f"Covenant Monitor: {', '.join(parts)}." if parts else "No covenants checked."

        return CovenantMonitorResult(
            checks=checks,
            passes=passes,
            warnings=warnings,
            breaches=breaches,
            summary=summary,
        )

    # ===== WORKING CAPITAL ANALYSIS =====

    def working_capital_analysis(self, data: FinancialData) -> WorkingCapitalResult:
        """Analyze working capital efficiency using days metrics and CCC.

        Calculates DSO, DIO, DPO, Cash Conversion Cycle, and generates
        optimization insights based on industry benchmarks.

        Args:
            data: FinancialData with revenue, COGS, AR, inventory, AP.

        Returns:
            WorkingCapitalResult with days metrics and actionable insights.
        """
        revenue = data.revenue or 0.0
        cogs = data.cogs or 0.0
        ar = data.accounts_receivable or data.avg_receivables
        inv = data.inventory or data.avg_inventory
        ap = data.accounts_payable or data.avg_payables
        current_assets = data.current_assets
        current_liabilities = data.current_liabilities

        insights: List[str] = []

        # DSO = Accounts Receivable / Revenue * 365
        dso = None
        if ar is not None and revenue > 0:
            dso = round(ar / revenue * 365, 1)
            if dso > 60:
                insights.append(f"DSO of {dso:.0f} days is high; consider tightening payment terms or improving collections.")
            elif dso > 45:
                insights.append(f"DSO of {dso:.0f} days is moderate; monitor for deterioration.")
            else:
                insights.append(f"DSO of {dso:.0f} days is healthy.")

        # DIO = Inventory / COGS * 365
        dio = None
        if inv is not None and cogs > 0:
            dio = round(inv / cogs * 365, 1)
            if dio > 90:
                insights.append(f"DIO of {dio:.0f} days indicates slow inventory turnover; review SKU performance.")
            elif dio > 60:
                insights.append(f"DIO of {dio:.0f} days is moderate.")
            else:
                insights.append(f"DIO of {dio:.0f} days shows efficient inventory management.")

        # DPO = Accounts Payable / COGS * 365
        dpo = None
        if ap is not None and cogs > 0:
            dpo = round(ap / cogs * 365, 1)
            if dpo < 30:
                insights.append(f"DPO of {dpo:.0f} days is low; negotiate longer payment terms with suppliers.")
            elif dpo > 90:
                insights.append(f"DPO of {dpo:.0f} days is very long; verify supplier relationships are healthy.")
            else:
                insights.append(f"DPO of {dpo:.0f} days is within normal range.")

        # CCC = DSO + DIO - DPO
        ccc = None
        if dso is not None and dio is not None and dpo is not None:
            ccc = round(dso + dio - dpo, 1)
            if ccc < 0:
                insights.append(f"Negative CCC of {ccc:.0f} days: company generates cash before paying suppliers. Excellent.")
            elif ccc < 30:
                insights.append(f"CCC of {ccc:.0f} days is efficient.")
            elif ccc < 60:
                insights.append(f"CCC of {ccc:.0f} days is moderate; look for optimization opportunities.")
            else:
                insights.append(f"CCC of {ccc:.0f} days is high; cash is tied up in the operating cycle.")

        # Net Working Capital
        nwc = None
        if current_assets is not None and current_liabilities is not None:
            nwc = round(current_assets - current_liabilities, 2)

        # Working Capital Ratio (same as current ratio but included here for context)
        wc_ratio = safe_divide(current_assets, current_liabilities, default=None)
        if wc_ratio is not None:
            wc_ratio = round(wc_ratio, 4)

        return WorkingCapitalResult(
            dso=dso,
            dio=dio,
            dpo=dpo,
            ccc=ccc,
            net_working_capital=nwc,
            working_capital_ratio=wc_ratio,
            insights=insights,
        )

    # ===== NARRATIVE INTELLIGENCE =====

    def generate_narrative(self, data: FinancialData) -> NarrativeReport:
        """Auto-generate a SWOT-style narrative from financial analysis.

        Synthesizes all major metrics into human-readable strengths, weaknesses,
        opportunities, risks, and a one-line recommendation.

        Args:
            data: FinancialData with sufficient metrics for analysis.

        Returns:
            NarrativeReport with categorized insights and recommendation.
        """
        strengths: List[str] = []
        weaknesses: List[str] = []
        opportunities: List[str] = []
        risks: List[str] = []

        # Health score
        health = self.composite_health_score(data)
        score = health.score
        grade = health.grade

        if score >= 70:
            headline = f"Strong financial position (Score: {score}/100, Grade: {grade})"
            strengths.append(f"Overall health score of {score}/100 ({grade}) indicates robust fundamentals.")
        elif score >= 50:
            headline = f"Moderate financial position (Score: {score}/100, Grade: {grade})"
            opportunities.append("Room to improve financial health through targeted interventions.")
        else:
            headline = f"Weak financial position (Score: {score}/100, Grade: {grade})"
            risks.append(f"Low health score of {score}/100 signals potential financial distress.")

        # Z-Score
        z = self.altman_z_score(data)
        if z.zone == 'safe':
            strengths.append(f"Altman Z-Score of {z.z_score:.2f} places company in the safe zone (low bankruptcy risk).")
        elif z.zone == 'grey':
            risks.append(f"Z-Score of {z.z_score:.2f} is in the grey zone; monitor for deterioration.")
        elif z.zone == 'distress':
            risks.append(f"Z-Score of {z.z_score:.2f} indicates financial distress; immediate attention needed.")

        # Profitability
        prof = self.calculate_profitability_ratios(data)
        nm = prof.get('net_margin')
        if nm is not None:
            if nm > 0.15:
                strengths.append(f"Net margin of {nm*100:.1f}% demonstrates strong profitability.")
            elif nm > 0.05:
                strengths.append(f"Net margin of {nm*100:.1f}% is adequate.")
            elif nm > 0:
                weaknesses.append(f"Thin net margin of {nm*100:.1f}% leaves little room for error.")
            else:
                weaknesses.append(f"Negative net margin of {nm*100:.1f}% indicates the company is unprofitable.")

        roe = prof.get('roe')
        if roe is not None:
            if roe > 0.15:
                strengths.append(f"ROE of {roe*100:.1f}% shows efficient use of shareholder equity.")
            elif roe < 0:
                weaknesses.append(f"Negative ROE of {roe*100:.1f}% means equity is being destroyed.")

        # Liquidity
        liq = self.calculate_liquidity_ratios(data)
        cr = liq.get('current_ratio')
        if cr is not None:
            if cr >= 2.0:
                strengths.append(f"Current ratio of {cr:.2f} provides strong short-term liquidity.")
            elif cr >= 1.0:
                pass  # Acceptable, no specific call-out
            else:
                weaknesses.append(f"Current ratio of {cr:.2f} below 1.0 raises liquidity concerns.")

        # Leverage
        lev = self.calculate_leverage_ratios(data)
        dte = lev.get('debt_to_equity')
        if dte is not None:
            if dte > 2.0:
                risks.append(f"High debt-to-equity of {dte:.2f} increases financial risk and interest burden.")
            elif dte < 0.5:
                strengths.append(f"Conservative leverage (D/E: {dte:.2f}) provides financial flexibility.")

        ic = lev.get('interest_coverage')
        if ic is not None:
            if ic > 5:
                strengths.append(f"Strong interest coverage of {ic:.1f}x comfortably services debt obligations.")
            elif ic < 2:
                risks.append(f"Low interest coverage of {ic:.1f}x; at risk of debt servicing difficulties.")

        # Working Capital
        wc = self.working_capital_analysis(data)
        if wc.ccc is not None:
            if wc.ccc < 0:
                strengths.append(f"Negative cash conversion cycle ({wc.ccc:.0f} days) means the business generates cash quickly.")
            elif wc.ccc > 60:
                opportunities.append(f"CCC of {wc.ccc:.0f} days can be improved through better AR/AP/inventory management.")

        # Breakeven
        be = self.breakeven_analysis(data)
        if be.margin_of_safety is not None:
            if be.margin_of_safety > 0.3:
                strengths.append(f"Margin of safety of {be.margin_of_safety*100:.0f}% above breakeven revenue.")
            elif be.margin_of_safety > 0:
                opportunities.append(f"Operating {be.margin_of_safety*100:.0f}% above breakeven; work to widen this buffer.")
            else:
                risks.append("Operating below breakeven revenue level.")

        # Generate recommendation
        if score >= 70 and len(risks) == 0:
            recommendation = "Company is in strong financial health. Maintain current trajectory and look for growth opportunities."
        elif score >= 50:
            top_weakness = weaknesses[0] if weaknesses else "overall efficiency"
            recommendation = f"Address key weakness: {top_weakness} Focus on improving profitability and reducing leverage."
        else:
            recommendation = "Urgent attention needed. Prioritize cash preservation, debt restructuring, and operational efficiency."

        return NarrativeReport(
            headline=headline,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            risks=risks,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------ #
    #  Phase 9 – Regression Forecasting & Industry Benchmarking           #
    # ------------------------------------------------------------------ #

    def regression_forecast(
        self,
        values: List[float],
        periods_ahead: int = 3,
        method: str = "linear",
        metric_name: str = "metric",
        confidence_level: float = 0.90,
    ) -> RegressionForecast:
        """Fit a regression model and project forward with confidence bands.

        Args:
            values: Historical data points (at least 3).
            periods_ahead: Number of future periods to forecast.
            method: 'linear', 'exponential', or 'polynomial'.
            metric_name: Label for reporting.
            confidence_level: Confidence band width (0-1).

        Returns:
            RegressionForecast with fitted params, projected values, and bands.
        """
        result = RegressionForecast(
            metric_name=metric_name,
            historical_values=list(values),
            periods_ahead=periods_ahead,
            method=method,
        )
        if len(values) < 3:
            return result

        n = len(values)
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)

        if method == "exponential" and np.all(y > 0):
            log_y = np.log(y)
            coeffs = np.polyfit(x, log_y, 1)
            slope_log, intercept_log = coeffs
            fitted = np.exp(intercept_log + slope_log * x)
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            future_x = np.arange(n, n + periods_ahead, dtype=float)
            forecast = np.exp(intercept_log + slope_log * future_x)
            result.slope = float(slope_log)
            result.intercept = float(intercept_log)
            result.r_squared = float(max(0.0, r2))
            result.forecast_values = forecast.tolist()
            # Confidence bands: use residual std on log scale
            se = np.sqrt(ss_res / max(n - 2, 1))
            z = 1.645 if confidence_level >= 0.90 else 1.28
            result.confidence_upper = np.exp(intercept_log + slope_log * future_x + z * se).tolist()
            result.confidence_lower = np.exp(np.maximum(intercept_log + slope_log * future_x - z * se, -20)).tolist()
        elif method == "polynomial":
            deg = min(2, n - 1)
            coeffs = np.polyfit(x, y, deg)
            poly = np.poly1d(coeffs)
            fitted = poly(x)
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            future_x = np.arange(n, n + periods_ahead, dtype=float)
            forecast = poly(future_x)
            result.slope = float(coeffs[-2]) if len(coeffs) > 1 else 0.0
            result.intercept = float(coeffs[-1])
            result.r_squared = float(max(0.0, r2))
            result.forecast_values = forecast.tolist()
            se = np.sqrt(ss_res / max(n - deg - 1, 1))
            z = 1.645 if confidence_level >= 0.90 else 1.28
            result.confidence_upper = (forecast + z * se).tolist()
            result.confidence_lower = (forecast - z * se).tolist()
        else:
            # Default linear regression
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            fitted = intercept + slope * x
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            future_x = np.arange(n, n + periods_ahead, dtype=float)
            forecast = intercept + slope * future_x
            result.slope = float(slope)
            result.intercept = float(intercept)
            result.r_squared = float(max(0.0, r2))
            result.forecast_values = forecast.tolist()
            se = np.sqrt(ss_res / max(n - 2, 1))
            z = 1.645 if confidence_level >= 0.90 else 1.28
            result.confidence_upper = (forecast + z * se).tolist()
            result.confidence_lower = (forecast - z * se).tolist()

        return result

    # Industry benchmark medians (representative defaults)
    INDUSTRY_BENCHMARKS: Dict[str, Dict[str, Dict[str, float]]] = {
        "general": {
            "current_ratio":      {"p25": 1.2, "median": 1.8, "p75": 2.5},
            "quick_ratio":        {"p25": 0.8, "median": 1.2, "p75": 1.8},
            "gross_margin":       {"p25": 0.25, "median": 0.40, "p75": 0.55},
            "net_margin":         {"p25": 0.03, "median": 0.08, "p75": 0.15},
            "roe":                {"p25": 0.06, "median": 0.12, "p75": 0.20},
            "roa":                {"p25": 0.03, "median": 0.06, "p75": 0.12},
            "debt_to_equity":     {"p25": 0.3, "median": 0.8, "p75": 1.5},
            "interest_coverage":  {"p25": 2.0, "median": 5.0, "p75": 10.0},
            "asset_turnover":     {"p25": 0.4, "median": 0.8, "p75": 1.2},
        },
        "technology": {
            "current_ratio":      {"p25": 1.5, "median": 2.5, "p75": 4.0},
            "gross_margin":       {"p25": 0.50, "median": 0.65, "p75": 0.80},
            "net_margin":         {"p25": 0.05, "median": 0.15, "p75": 0.25},
            "roe":                {"p25": 0.08, "median": 0.18, "p75": 0.30},
            "roa":                {"p25": 0.04, "median": 0.10, "p75": 0.18},
            "debt_to_equity":     {"p25": 0.1, "median": 0.3, "p75": 0.7},
        },
        "manufacturing": {
            "current_ratio":      {"p25": 1.0, "median": 1.5, "p75": 2.0},
            "gross_margin":       {"p25": 0.20, "median": 0.30, "p75": 0.40},
            "net_margin":         {"p25": 0.02, "median": 0.05, "p75": 0.10},
            "roe":                {"p25": 0.05, "median": 0.10, "p75": 0.15},
            "debt_to_equity":     {"p25": 0.5, "median": 1.0, "p75": 2.0},
            "asset_turnover":     {"p25": 0.6, "median": 1.0, "p75": 1.5},
        },
        "retail": {
            "current_ratio":      {"p25": 0.8, "median": 1.3, "p75": 1.8},
            "gross_margin":       {"p25": 0.25, "median": 0.35, "p75": 0.50},
            "net_margin":         {"p25": 0.01, "median": 0.04, "p75": 0.08},
            "roe":                {"p25": 0.08, "median": 0.15, "p75": 0.25},
            "debt_to_equity":     {"p25": 0.4, "median": 1.0, "p75": 2.0},
            "asset_turnover":     {"p25": 1.0, "median": 1.8, "p75": 2.5},
        },
        "healthcare": {
            "current_ratio":      {"p25": 1.2, "median": 1.8, "p75": 3.0},
            "gross_margin":       {"p25": 0.30, "median": 0.45, "p75": 0.60},
            "net_margin":         {"p25": 0.02, "median": 0.08, "p75": 0.15},
            "roe":                {"p25": 0.05, "median": 0.12, "p75": 0.20},
            "debt_to_equity":     {"p25": 0.3, "median": 0.7, "p75": 1.5},
        },
    }

    def industry_benchmark(
        self,
        data: FinancialData,
        industry: str = "general",
    ) -> IndustryBenchmarkResult:
        """Compare company metrics against industry percentile benchmarks.

        Args:
            data: Company financial data.
            industry: Industry name (general, technology, manufacturing, retail, healthcare).

        Returns:
            IndustryBenchmarkResult with per-metric comparison and overall percentile.
        """
        benchmarks = self.INDUSTRY_BENCHMARKS.get(
            industry.lower(), self.INDUSTRY_BENCHMARKS["general"]
        )

        # Gather company metrics
        liquidity = self.calculate_liquidity_ratios(data)
        profitability = self.calculate_profitability_ratios(data)
        leverage = self.calculate_leverage_ratios(data)
        efficiency = self.calculate_efficiency_ratios(data)

        company_metrics: Dict[str, Optional[float]] = {}
        company_metrics["current_ratio"] = liquidity.get("current_ratio")
        company_metrics["quick_ratio"] = liquidity.get("quick_ratio")
        company_metrics["gross_margin"] = profitability.get("gross_margin")
        company_metrics["net_margin"] = profitability.get("net_margin")
        company_metrics["roe"] = profitability.get("roe")
        company_metrics["roa"] = profitability.get("roa")
        company_metrics["debt_to_equity"] = leverage.get("debt_to_equity")
        company_metrics["interest_coverage"] = leverage.get("interest_coverage")
        company_metrics["asset_turnover"] = efficiency.get("asset_turnover")

        comparisons: List[BenchmarkComparison] = []
        percentile_ranks: List[float] = []

        for metric_name, bench in benchmarks.items():
            val = company_metrics.get(metric_name)
            if val is None:
                continue

            p25 = bench["p25"]
            median = bench["median"]
            p75 = bench["p75"]

            # Estimate percentile rank (linear interpolation)
            # For debt_to_equity, lower is better (inverted)
            inverted = metric_name in ("debt_to_equity",)

            if inverted:
                if val <= p75:
                    pct = 75 + 25 * (p75 - val) / max(p75 - p25, 1e-9) * (50 / 25)
                    pct = min(pct, 99)
                elif val <= median:
                    pct = 50 + 25 * (median - val) / max(median - p25, 1e-9)
                elif val <= p25:
                    pct = 25 + 25 * (p25 - val) / max(p25, 1e-9)
                else:
                    # Higher than p25 (bad for inverted)
                    pct = max(5, 25 * (1 - (val - p25) / max(p25, 1e-9)))
            else:
                if val >= p75:
                    pct = min(75 + 25 * (val - p75) / max(p75 - median, 1e-9), 99)
                elif val >= median:
                    pct = 50 + 25 * (val - median) / max(p75 - median, 1e-9)
                elif val >= p25:
                    pct = 25 + 25 * (val - p25) / max(median - p25, 1e-9)
                else:
                    pct = max(1, 25 * val / max(p25, 1e-9))

            pct = float(np.clip(pct, 1, 99))

            if pct >= 65:
                rating = "above average"
            elif pct >= 35:
                rating = "average"
            else:
                rating = "below average"

            comparisons.append(BenchmarkComparison(
                metric_name=metric_name,
                company_value=val,
                industry_median=median,
                industry_p25=p25,
                industry_p75=p75,
                percentile_rank=pct,
                rating=rating,
            ))
            percentile_ranks.append(pct)

        overall = float(np.mean(percentile_ranks)) if percentile_ranks else None

        above = sum(1 for c in comparisons if c.rating == "above average")
        below = sum(1 for c in comparisons if c.rating == "below average")
        total = len(comparisons)

        if overall is not None and overall >= 60:
            summary = (f"Company ranks in the {overall:.0f}th percentile overall vs {industry} peers. "
                       f"{above}/{total} metrics above average.")
        elif overall is not None and overall >= 40:
            summary = (f"Company performs near industry median ({overall:.0f}th percentile). "
                       f"{below}/{total} metrics need attention.")
        elif overall is not None:
            summary = (f"Company underperforms {industry} peers ({overall:.0f}th percentile). "
                       f"{below}/{total} metrics below average.")
        else:
            summary = "Insufficient data for industry comparison."

        return IndustryBenchmarkResult(
            industry_name=industry,
            comparisons=comparisons,
            overall_percentile=overall,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    #  Phase 10 – Custom KPI Builder                                      #
    # ------------------------------------------------------------------ #

    # Fields that can be referenced in custom KPI formulas
    _KPI_ALLOWED_FIELDS = frozenset({
        'revenue', 'cogs', 'gross_profit', 'operating_income', 'operating_expenses',
        'ebit', 'ebitda', 'interest_expense', 'net_income', 'ebt',
        'retained_earnings', 'depreciation',
        'total_assets', 'current_assets', 'cash', 'inventory',
        'accounts_receivable', 'total_liabilities', 'current_liabilities',
        'accounts_payable', 'total_debt', 'total_equity',
        'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow', 'capex',
    })

    def evaluate_custom_kpis(
        self,
        data: FinancialData,
        kpi_definitions: List[CustomKPIDefinition],
    ) -> CustomKPIReport:
        """Evaluate user-defined KPI formulas against financial data.

        Formulas can reference FinancialData field names and use basic arithmetic:
            ``+``, ``-``, ``*``, ``/``, ``()``, and numeric literals.

        Examples:
            - ``"revenue / total_assets"``  (asset turnover)
            - ``"(revenue - cogs) / revenue"``  (gross margin)
            - ``"ebitda - capex"``  (free cash proxy)

        Args:
            data: Company financial data providing variable values.
            kpi_definitions: List of KPI definitions to evaluate.

        Returns:
            CustomKPIReport with per-KPI results and summary.
        """
        results: List[CustomKPIResult] = []

        # Build namespace from FinancialData fields
        namespace: Dict[str, float] = {}
        for field_name in self._KPI_ALLOWED_FIELDS:
            val = getattr(data, field_name, None)
            if val is not None:
                namespace[field_name] = float(val)

        for kpi in kpi_definitions:
            r = CustomKPIResult(name=kpi.name, formula=kpi.formula)

            if not kpi.formula.strip():
                r.error = "Empty formula"
                results.append(r)
                continue

            # Validate formula tokens (security: only allow field names, numbers, operators)
            sanitized = kpi.formula
            # Remove valid tokens to check for leftovers
            temp = sanitized
            for fname in sorted(self._KPI_ALLOWED_FIELDS, key=len, reverse=True):
                temp = temp.replace(fname, '')
            # Remove numbers, operators, whitespace, parentheses, dots
            temp = re.sub(r'[\d+\-*/().\s]', '', temp)
            if temp:
                r.error = f"Invalid tokens in formula: {temp}"
                results.append(r)
                continue

            try:
                value = eval(sanitized, {"__builtins__": {}}, namespace)  # noqa: S307
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    r.error = "Formula produced invalid result"
                else:
                    r.value = float(value)
                    if kpi.target_min is not None and kpi.target_max is not None:
                        r.meets_target = kpi.target_min <= r.value <= kpi.target_max
                    elif kpi.target_min is not None:
                        r.meets_target = r.value >= kpi.target_min
                    elif kpi.target_max is not None:
                        r.meets_target = r.value <= kpi.target_max
            except ZeroDivisionError:
                r.error = "Division by zero"
            except NameError as e:
                r.error = f"Unknown variable: {e}"
            except Exception as e:
                r.error = f"Evaluation error: {e}"

            results.append(r)

        met = sum(1 for r in results if r.meets_target is True)
        total_with_target = sum(1 for r in results if r.meets_target is not None)
        errors = sum(1 for r in results if r.error)

        if errors:
            summary = f"{len(results)} KPIs evaluated, {errors} errors."
        elif total_with_target:
            summary = f"{met}/{total_with_target} KPIs meeting targets."
        else:
            summary = f"{len(results)} KPIs evaluated."

        return CustomKPIReport(results=results, summary=summary)

    # ------------------------------------------------------------------
    # Phase 11: Peer Comparison & Ratio Decomposition
    # ------------------------------------------------------------------

    def peer_comparison(
        self,
        peers: List[PeerCompanyData],
        metrics: Optional[List[str]] = None,
    ) -> PeerComparisonReport:
        """Compare financial metrics across multiple peers.

        Parameters
        ----------
        peers : list of PeerCompanyData
            Each peer has a name and FinancialData.
        metrics : list of str, optional
            Metric names to compare. Defaults to a standard set.

        Returns
        -------
        PeerComparisonReport
        """
        if not peers:
            return PeerComparisonReport(summary="No peers provided.")

        if metrics is None:
            metrics = [
                "current_ratio", "gross_margin", "net_margin", "roe",
                "roa", "debt_to_equity", "asset_turnover",
                "operating_margin", "interest_coverage",
            ]

        peer_names = [p.name or f"Peer {i+1}" for i, p in enumerate(peers)]
        comparisons: List[PeerMetricComparison] = []

        for metric_name in metrics:
            values: Dict[str, Optional[float]] = {}
            for peer in peers:
                name = peer.name or f"Peer {peers.index(peer)+1}"
                val = self._compute_peer_metric(peer.data, metric_name)
                values[name] = val

            valid = {k: v for k, v in values.items() if v is not None}
            if valid:
                avg = sum(valid.values()) / len(valid)
                sorted_v = sorted(valid.items(), key=lambda x: x[1], reverse=True)
                # For debt_to_equity, lower is better
                higher_is_better = metric_name != "debt_to_equity"
                best = sorted_v[0][0] if higher_is_better else sorted_v[-1][0]
                worst = sorted_v[-1][0] if higher_is_better else sorted_v[0][0]
                vals_list = sorted(valid.values())
                median_val = vals_list[len(vals_list) // 2]
            else:
                avg = None
                best = ""
                worst = ""
                median_val = None

            comparisons.append(PeerMetricComparison(
                metric_name=metric_name,
                values=values,
                best_performer=best,
                worst_performer=worst,
                average=avg,
                median=median_val,
            ))

        # Compute overall ranking: count how many metrics each peer is best at
        best_counts: Dict[str, int] = {n: 0 for n in peer_names}
        for comp in comparisons:
            if comp.best_performer and comp.best_performer in best_counts:
                best_counts[comp.best_performer] += 1

        # Rank by best counts (higher = better rank = lower number)
        sorted_peers = sorted(best_counts.items(), key=lambda x: x[1], reverse=True)
        rankings = {name: rank + 1 for rank, (name, _) in enumerate(sorted_peers)}

        top = sorted_peers[0][0] if sorted_peers else ""
        summary = (
            f"Compared {len(peer_names)} peers across {len(metrics)} metrics. "
            f"{top} leads in {best_counts.get(top, 0)} metrics."
        )

        return PeerComparisonReport(
            peer_names=peer_names,
            comparisons=comparisons,
            rankings=rankings,
            summary=summary,
        )

    def _compute_peer_metric(
        self, data: Optional[FinancialData], metric_name: str
    ) -> Optional[float]:
        """Compute a single named metric from FinancialData."""
        if data is None:
            return None
        m = metric_name.lower()
        if m == "current_ratio":
            return safe_divide(data.current_assets, data.current_liabilities)
        elif m == "gross_margin":
            return safe_divide(data.gross_profit, data.revenue)
        elif m == "net_margin":
            return safe_divide(data.net_income, data.revenue)
        elif m == "operating_margin":
            return safe_divide(data.operating_income, data.revenue)
        elif m == "roe":
            return safe_divide(data.net_income, data.total_equity)
        elif m == "roa":
            return safe_divide(data.net_income, data.total_assets)
        elif m == "debt_to_equity":
            return safe_divide(data.total_debt, data.total_equity)
        elif m == "asset_turnover":
            return safe_divide(data.revenue, data.total_assets)
        elif m == "interest_coverage":
            return safe_divide(data.ebit, data.interest_expense)
        elif m == "ebitda_margin":
            return safe_divide(data.ebitda, data.revenue)
        return None

    def ratio_decomposition(self, data: FinancialData) -> RatioDecompositionTree:
        """Build a DuPont-based ratio decomposition tree from ROE.

        Decomposes ROE -> Net Margin × Asset Turnover × Equity Multiplier,
        then further decomposes each factor into its constituent components.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        RatioDecompositionTree
        """
        # Level 0: ROE
        roe_val = safe_divide(data.net_income, data.total_equity)

        # Level 1: DuPont 3-factor
        net_margin = safe_divide(data.net_income, data.revenue)
        asset_turnover = safe_divide(data.revenue, data.total_assets)
        equity_multiplier = safe_divide(data.total_assets, data.total_equity)

        # Level 2: Net Margin decomposition
        gross_margin = safe_divide(data.gross_profit, data.revenue)
        opex_ratio = safe_divide(data.operating_expenses, data.revenue)
        interest_burden = safe_divide(data.interest_expense, data.revenue)
        tax_burden = None
        if data.net_income is not None and data.ebt is not None and data.ebt:
            tax_burden = safe_divide(data.net_income, data.ebt)

        net_margin_children = [
            RatioDecompositionNode(
                name="Gross Margin",
                value=gross_margin,
                formula="gross_profit / revenue",
            ),
            RatioDecompositionNode(
                name="OpEx Ratio",
                value=opex_ratio,
                formula="operating_expenses / revenue",
            ),
            RatioDecompositionNode(
                name="Interest Burden",
                value=interest_burden,
                formula="interest_expense / revenue",
            ),
        ]
        if tax_burden is not None:
            net_margin_children.append(RatioDecompositionNode(
                name="Tax Retention",
                value=tax_burden,
                formula="net_income / ebt",
            ))

        # Level 2: Asset Turnover decomposition
        receivables_turnover = safe_divide(data.revenue, data.accounts_receivable)
        inventory_turnover = safe_divide(data.cogs, data.inventory)
        fixed_asset_ratio = None
        if data.total_assets is not None and data.current_assets is not None:
            fixed_assets = data.total_assets - data.current_assets
            fixed_asset_ratio = safe_divide(data.revenue, fixed_assets if fixed_assets > 0 else None)

        at_children = [
            RatioDecompositionNode(
                name="Receivables Turnover",
                value=receivables_turnover,
                formula="revenue / accounts_receivable",
            ),
            RatioDecompositionNode(
                name="Inventory Turnover",
                value=inventory_turnover,
                formula="cogs / inventory",
            ),
        ]
        if fixed_asset_ratio is not None:
            at_children.append(RatioDecompositionNode(
                name="Fixed Asset Turnover",
                value=fixed_asset_ratio,
                formula="revenue / fixed_assets",
            ))

        # Level 2: Equity Multiplier decomposition
        debt_ratio = safe_divide(data.total_liabilities, data.total_assets)
        debt_to_equity = safe_divide(data.total_debt, data.total_equity)

        em_children = [
            RatioDecompositionNode(
                name="Debt Ratio",
                value=debt_ratio,
                formula="total_liabilities / total_assets",
            ),
            RatioDecompositionNode(
                name="Debt-to-Equity",
                value=debt_to_equity,
                formula="total_debt / total_equity",
            ),
        ]

        # Build tree
        root = RatioDecompositionNode(
            name="ROE",
            value=roe_val,
            formula="net_income / total_equity",
            children=[
                RatioDecompositionNode(
                    name="Net Profit Margin",
                    value=net_margin,
                    formula="net_income / revenue",
                    children=net_margin_children,
                ),
                RatioDecompositionNode(
                    name="Asset Turnover",
                    value=asset_turnover,
                    formula="revenue / total_assets",
                    children=at_children,
                ),
                RatioDecompositionNode(
                    name="Equity Multiplier",
                    value=equity_multiplier,
                    formula="total_assets / total_equity",
                    children=em_children,
                ),
            ],
        )

        # Summary
        parts = []
        if roe_val is not None:
            parts.append(f"ROE: {roe_val:.1%}")
        if net_margin is not None:
            parts.append(f"Net Margin: {net_margin:.1%}")
        if asset_turnover is not None:
            parts.append(f"Asset Turnover: {asset_turnover:.2f}x")
        if equity_multiplier is not None:
            parts.append(f"Equity Multiplier: {equity_multiplier:.2f}x")
        summary = " | ".join(parts) if parts else "Insufficient data for decomposition."

        return RatioDecompositionTree(root=root, summary=summary)

    # ------------------------------------------------------------------
    # Phase 12: Financial Rating Scorecard & Variance Waterfall
    # ------------------------------------------------------------------

    _GRADE_THRESHOLDS = [
        (9.0, "AAA"), (8.0, "AA"), (7.0, "A"),
        (6.0, "BBB"), (5.0, "BB"), (4.0, "B"),
        (3.0, "CCC"), (2.0, "CC"), (0.0, "C"),
    ]

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert a numeric score (0-10) to a letter grade."""
        for threshold, grade in CharlieAnalyzer._GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "C"

    def financial_rating(self, data: FinancialData) -> FinancialRating:
        """Generate a comprehensive letter-grade financial rating.

        Scores 5 categories (0-10 each) and computes a weighted average:
        - Liquidity (20%)
        - Profitability (25%)
        - Leverage (20%)
        - Efficiency (20%)
        - Cash Flow (15%)

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        FinancialRating
        """
        categories = []

        # 1. Liquidity (20%)
        cr = safe_divide(data.current_assets, data.current_liabilities)
        qr = safe_divide(
            (data.current_assets or 0) - (data.inventory or 0),
            data.current_liabilities,
        )
        liq_score = 0.0
        liq_details = []
        if cr is not None:
            if cr >= 2.0:
                liq_score += 5.0
            elif cr >= 1.5:
                liq_score += 4.0
            elif cr >= 1.0:
                liq_score += 2.5
            else:
                liq_score += 1.0
            liq_details.append(f"Current Ratio: {cr:.2f}")
        if qr is not None:
            if qr >= 1.5:
                liq_score += 5.0
            elif qr >= 1.0:
                liq_score += 4.0
            elif qr >= 0.5:
                liq_score += 2.5
            else:
                liq_score += 1.0
            liq_details.append(f"Quick Ratio: {qr:.2f}")
        categories.append(RatingCategory(
            name="Liquidity", score=liq_score, grade=self._score_to_grade(liq_score),
            details="; ".join(liq_details) if liq_details else "Insufficient data",
        ))

        # 2. Profitability (25%)
        gm = safe_divide(data.gross_profit, data.revenue)
        nm = safe_divide(data.net_income, data.revenue)
        roe = safe_divide(data.net_income, data.total_equity)
        prof_score = 0.0
        prof_details = []
        if gm is not None:
            s = min(3.3, max(0, gm * 10))
            prof_score += s
            prof_details.append(f"Gross Margin: {gm:.1%}")
        if nm is not None:
            s = min(3.3, max(0, nm * 20))
            prof_score += s
            prof_details.append(f"Net Margin: {nm:.1%}")
        if roe is not None:
            s = min(3.4, max(0, roe * 20))
            prof_score += s
            prof_details.append(f"ROE: {roe:.1%}")
        categories.append(RatingCategory(
            name="Profitability", score=min(10, prof_score),
            grade=self._score_to_grade(min(10, prof_score)),
            details="; ".join(prof_details) if prof_details else "Insufficient data",
        ))

        # 3. Leverage (20%)
        de = safe_divide(data.total_debt, data.total_equity)
        ic = safe_divide(data.ebit, data.interest_expense)
        lev_score = 0.0
        lev_details = []
        if de is not None:
            # Lower is better
            if de <= 0.5:
                lev_score += 5.0
            elif de <= 1.0:
                lev_score += 4.0
            elif de <= 2.0:
                lev_score += 2.5
            else:
                lev_score += 1.0
            lev_details.append(f"D/E: {de:.2f}")
        if ic is not None:
            if ic >= 8.0:
                lev_score += 5.0
            elif ic >= 4.0:
                lev_score += 4.0
            elif ic >= 2.0:
                lev_score += 2.5
            else:
                lev_score += 1.0
            lev_details.append(f"Interest Coverage: {ic:.1f}x")
        categories.append(RatingCategory(
            name="Leverage", score=lev_score,
            grade=self._score_to_grade(lev_score),
            details="; ".join(lev_details) if lev_details else "Insufficient data",
        ))

        # 4. Efficiency (20%)
        at = safe_divide(data.revenue, data.total_assets)
        inv_t = safe_divide(data.cogs, data.inventory)
        eff_score = 0.0
        eff_details = []
        if at is not None:
            s = min(5.0, max(0, at * 5))
            eff_score += s
            eff_details.append(f"Asset Turnover: {at:.2f}x")
        if inv_t is not None:
            s = min(5.0, max(0, inv_t / 2))
            eff_score += s
            eff_details.append(f"Inventory Turnover: {inv_t:.1f}x")
        categories.append(RatingCategory(
            name="Efficiency", score=min(10, eff_score),
            grade=self._score_to_grade(min(10, eff_score)),
            details="; ".join(eff_details) if eff_details else "Insufficient data",
        ))

        # 5. Cash Flow (15%)
        cf_score = 0.0
        cf_details = []
        ocf = data.operating_cash_flow
        fcf = None
        if ocf is not None and data.capex is not None:
            fcf = ocf - abs(data.capex)
        if ocf is not None and data.revenue:
            cf_ratio = ocf / data.revenue
            s = min(5.0, max(0, cf_ratio * 20))
            cf_score += s
            cf_details.append(f"OCF/Revenue: {cf_ratio:.1%}")
        if fcf is not None and fcf > 0:
            cf_score += 5.0
            cf_details.append(f"FCF: {fcf:,.0f} (positive)")
        elif fcf is not None:
            cf_score += 1.0
            cf_details.append(f"FCF: {fcf:,.0f} (negative)")
        categories.append(RatingCategory(
            name="Cash Flow", score=min(10, cf_score),
            grade=self._score_to_grade(min(10, cf_score)),
            details="; ".join(cf_details) if cf_details else "Insufficient data",
        ))

        # Weighted average
        weights = [0.20, 0.25, 0.20, 0.20, 0.15]
        overall = sum(c.score * w for c, w in zip(categories, weights))
        overall_grade = self._score_to_grade(overall)

        # Summary
        strengths = [c.name for c in categories if c.score >= 7.0]
        weaknesses = [c.name for c in categories if c.score < 4.0]
        parts = [f"Overall: {overall_grade} ({overall:.1f}/10)"]
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths)}")
        if weaknesses:
            parts.append(f"Weaknesses: {', '.join(weaknesses)}")

        return FinancialRating(
            overall_score=overall,
            overall_grade=overall_grade,
            categories=categories,
            summary=" | ".join(parts),
        )

    def variance_waterfall(
        self,
        current: FinancialData,
        previous: FinancialData,
    ) -> VarianceWaterfall:
        """Build a waterfall showing what drove the change in net income.

        Decomposes net income variance into revenue, COGS, OpEx,
        interest, and other components.

        Parameters
        ----------
        current : FinancialData
        previous : FinancialData

        Returns
        -------
        VarianceWaterfall
        """
        start = previous.net_income or 0
        end = current.net_income or 0
        items: List[WaterfallItem] = []
        cumulative = start

        items.append(WaterfallItem(
            label="Prior Net Income",
            value=start,
            cumulative=start,
            item_type="start",
        ))

        # Revenue change
        rev_delta = (current.revenue or 0) - (previous.revenue or 0)
        cumulative += rev_delta
        items.append(WaterfallItem(
            label="Revenue Change",
            value=rev_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # COGS change (negative = favorable if COGS decreased)
        cogs_delta = -((current.cogs or 0) - (previous.cogs or 0))
        cumulative += cogs_delta
        items.append(WaterfallItem(
            label="COGS Impact",
            value=cogs_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # OpEx change (negative = favorable if OpEx decreased)
        opex_delta = -((current.operating_expenses or 0) - (previous.operating_expenses or 0))
        cumulative += opex_delta
        items.append(WaterfallItem(
            label="OpEx Impact",
            value=opex_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # Interest change (negative = favorable if interest decreased)
        int_delta = -((current.interest_expense or 0) - (previous.interest_expense or 0))
        cumulative += int_delta
        items.append(WaterfallItem(
            label="Interest Impact",
            value=int_delta,
            cumulative=cumulative,
            item_type="delta",
        ))

        # Other / residual
        residual = end - cumulative
        if abs(residual) > 0.01:
            cumulative += residual
            items.append(WaterfallItem(
                label="Other / Tax",
                value=residual,
                cumulative=cumulative,
                item_type="delta",
            ))

        items.append(WaterfallItem(
            label="Current Net Income",
            value=end,
            cumulative=end,
            item_type="total",
        ))

        total_var = end - start
        pct = (total_var / abs(start) * 100) if start != 0 else 0
        summary = f"Net income changed by {total_var:,.0f} ({pct:+.1f}%) from {start:,.0f} to {end:,.0f}."

        return VarianceWaterfall(
            start_value=start,
            end_value=end,
            items=items,
            total_variance=total_var,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 13 – Earnings Quality & Capital Efficiency
    # ------------------------------------------------------------------

    def earnings_quality(self, data: FinancialData) -> EarningsQualityResult:
        """Assess the quality of reported earnings.

        Evaluates how well net income is supported by actual cash flows
        using the accrual ratio and cash-to-income ratio.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        EarningsQualityResult
        """
        ni = data.net_income
        ocf = data.operating_cash_flow
        ta = data.total_assets

        # Guard: need at least net_income to score
        if ni is None and ocf is None:
            return EarningsQualityResult(
                quality_score=0.0,
                quality_grade="N/A",
                summary="Insufficient data to assess earnings quality.",
            )

        indicators: List[str] = []
        score = 5.0  # start at mid-point

        # --- Accrual ratio ---
        accrual_ratio = None
        if ni is not None and ocf is not None and ta and abs(ta) > 1e-12:
            accrual_ratio = (ni - ocf) / ta
            if accrual_ratio < -0.05:
                score += 2.0
                indicators.append("Very low accruals – earnings strongly backed by cash")
            elif accrual_ratio < 0.05:
                score += 1.0
                indicators.append("Low accruals – good earnings quality")
            elif accrual_ratio < 0.15:
                indicators.append("Moderate accruals – monitor working capital changes")
            else:
                score -= 2.0
                indicators.append("High accruals – earnings may not be sustainable")

        # --- Cash-to-income ratio ---
        cash_to_income = None
        if ni is not None and ocf is not None and abs(ni) > 1e-12:
            cash_to_income = ocf / ni
            if ni > 0:
                if cash_to_income >= 1.2:
                    score += 1.5
                    indicators.append("Cash flow exceeds net income – strong cash generation")
                elif cash_to_income >= 0.8:
                    score += 0.5
                    indicators.append("Cash flow approximates net income – adequate quality")
                elif cash_to_income >= 0.5:
                    indicators.append("Cash flow trails net income – some non-cash items")
                else:
                    score -= 1.5
                    indicators.append("Cash flow well below net income – investigate accruals")
            else:
                # Negative NI: if OCF is positive, credit the company
                if ocf > 0:
                    score += 1.0
                    indicators.append("Positive cash flow despite net loss – operationally sound")
                else:
                    score -= 1.0
                    indicators.append("Negative cash flow and net loss – poor quality")

        # --- Operating cash flow positive ---
        if ocf is not None:
            if ocf > 0:
                score += 0.5
                indicators.append("Operating cash flow is positive")
            else:
                score -= 1.0
                indicators.append("Operating cash flow is negative")

        # Clamp score
        score = max(0.0, min(10.0, score))

        # Grade
        if score >= 8.0:
            grade = "High"
        elif score >= 5.0:
            grade = "Moderate"
        else:
            grade = "Low"

        parts = [f"Earnings Quality: {grade} ({score:.1f}/10)"]
        if accrual_ratio is not None:
            parts.append(f"Accrual ratio: {accrual_ratio:.3f}")
        if cash_to_income is not None:
            parts.append(f"Cash-to-income: {cash_to_income:.2f}")

        return EarningsQualityResult(
            accrual_ratio=accrual_ratio,
            cash_to_income_ratio=cash_to_income,
            quality_score=score,
            quality_grade=grade,
            indicators=indicators,
            summary=" | ".join(parts),
        )

    def capital_efficiency(
        self,
        data: FinancialData,
        cost_of_capital: float = 0.10,
    ) -> CapitalEfficiencyResult:
        """Evaluate capital efficiency and economic value creation.

        Computes ROIC, EVA, capital turnover, and reinvestment rate.

        Parameters
        ----------
        data : FinancialData
        cost_of_capital : float
            Weighted average cost of capital (default 10%).

        Returns
        -------
        CapitalEfficiencyResult
        """
        equity = data.total_equity or 0
        debt = data.total_debt or 0
        revenue = data.revenue

        # Invested capital = equity + debt
        invested_capital = equity + debt

        if invested_capital < 1e-12:
            return CapitalEfficiencyResult(
                wacc_estimate=cost_of_capital,
                summary="Insufficient data to compute capital efficiency.",
            )

        # Estimate NOPAT: EBIT * (1 - tax_rate)
        # Estimate tax rate from net_income / ebit if possible
        ebit = data.ebit or data.operating_income
        ni = data.net_income
        tax_rate = 0.25  # default
        if ebit and ni is not None and ebit > 1e-12:
            implied_tax = 1.0 - (ni / ebit) if ebit > 0 else 0.25
            if 0.0 <= implied_tax <= 0.60:
                tax_rate = implied_tax

        nopat = None
        if ebit is not None:
            nopat = ebit * (1.0 - tax_rate)

        # ROIC
        roic = safe_divide(nopat, invested_capital)

        # EVA
        eva = None
        if nopat is not None:
            eva = nopat - (cost_of_capital * invested_capital)

        # Capital turnover
        cap_turnover = safe_divide(revenue, invested_capital)

        # Reinvestment rate
        reinvest = None
        capex = data.capex
        if capex is not None and nopat is not None and abs(nopat) > 1e-12:
            reinvest = capex / nopat

        parts = []
        if roic is not None:
            parts.append(f"ROIC: {roic:.1%}")
        if eva is not None:
            parts.append(f"EVA: {eva:,.0f}")
        if cap_turnover is not None:
            parts.append(f"Capital Turnover: {cap_turnover:.2f}x")
        if reinvest is not None:
            parts.append(f"Reinvestment Rate: {reinvest:.1%}")
        summary = " | ".join(parts) if parts else "Insufficient data."

        return CapitalEfficiencyResult(
            roic=roic,
            invested_capital=invested_capital,
            nopat=nopat,
            eva=eva,
            wacc_estimate=cost_of_capital,
            capital_turnover=cap_turnover,
            reinvestment_rate=reinvest,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 14 – Liquidity Stress Test & Debt Service Analysis
    # ------------------------------------------------------------------

    def liquidity_stress_test(
        self,
        data: FinancialData,
        revenue_shocks: Optional[List[float]] = None,
    ) -> LiquidityStressResult:
        """Run a liquidity stress test under various revenue-shock scenarios.

        Parameters
        ----------
        data : FinancialData
        revenue_shocks : list of float, optional
            Revenue decline percentages to simulate (e.g. [0.10, 0.25, 0.50]).
            Defaults to [0.10, 0.25, 0.50].

        Returns
        -------
        LiquidityStressResult
        """
        if revenue_shocks is None:
            revenue_shocks = [0.10, 0.25, 0.50]

        ca = data.current_assets or 0
        cl = data.current_liabilities or 0
        inv = data.inventory or 0
        ocf = data.operating_cash_flow
        revenue = data.revenue or 0
        opex = data.operating_expenses or 0
        cogs = data.cogs or 0

        # Estimate monthly burn from operating data
        monthly_revenue = revenue / 12 if revenue > 0 else 0
        monthly_costs = (opex + cogs) / 12 if (opex + cogs) > 0 else 0

        # If OCF available, use it; otherwise estimate from revenue - costs
        if ocf is not None:
            monthly_net_cf = ocf / 12
        else:
            monthly_net_cf = monthly_revenue - monthly_costs

        # Cash proxy: current_assets - inventory
        cash_proxy = max(ca - inv, 0)

        # Months of cash
        if monthly_costs > 0 and monthly_net_cf < 0:
            months_cash = cash_proxy / abs(monthly_net_cf) if abs(monthly_net_cf) > 1e-12 else None
        elif monthly_costs > 0:
            # Positive OCF => estimate how long cash lasts if revenue stops
            months_cash = cash_proxy / monthly_costs if monthly_costs > 1e-12 else None
        else:
            months_cash = None

        # Stressed quick ratio (remove inventory)
        quick_assets = ca - inv
        stressed_qr = safe_divide(quick_assets, cl) if cl > 0 else None

        # Run stress scenarios
        scenarios = []
        for shock in sorted(revenue_shocks):
            stressed_rev = monthly_revenue * (1.0 - shock)
            stressed_cf = stressed_rev - monthly_costs
            if stressed_cf < 0 and cash_proxy > 0:
                survival_months = cash_proxy / abs(stressed_cf)
            elif stressed_cf >= 0:
                survival_months = float('inf')
            else:
                survival_months = 0.0

            scenarios.append({
                "shock_pct": shock,
                "label": f"{shock:.0%} revenue decline",
                "monthly_cash_flow": round(stressed_cf, 2),
                "survival_months": round(survival_months, 1) if survival_months != float('inf') else None,
                "survives_12m": survival_months >= 12,
            })

        # Risk level
        worst_survival = min(
            (s["survival_months"] for s in scenarios if s["survival_months"] is not None),
            default=None,
        )
        if worst_survival is None:
            risk = "Low"
        elif worst_survival >= 12:
            risk = "Low"
        elif worst_survival >= 6:
            risk = "Moderate"
        elif worst_survival >= 3:
            risk = "High"
        else:
            risk = "Critical"

        parts = [f"Liquidity Risk: {risk}"]
        if months_cash is not None:
            parts.append(f"Months of cash: {months_cash:.1f}")
        if stressed_qr is not None:
            parts.append(f"Quick ratio: {stressed_qr:.2f}")

        return LiquidityStressResult(
            current_cash=cash_proxy,
            monthly_burn=round(monthly_costs, 2),
            months_of_cash=round(months_cash, 1) if months_cash is not None else None,
            stressed_quick_ratio=stressed_qr,
            stress_scenarios=scenarios,
            risk_level=risk,
            summary=" | ".join(parts),
        )

    def debt_service_analysis(self, data: FinancialData) -> DebtServiceResult:
        """Analyze debt service capacity.

        Computes DSCR, interest coverage, debt-to-EBITDA, and free cash
        after debt service.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        DebtServiceResult
        """
        ebitda = data.ebitda
        ebit = data.ebit or data.operating_income
        interest = data.interest_expense or 0
        total_debt = data.total_debt or 0
        ocf = data.operating_cash_flow
        capex = data.capex or 0

        # Interest coverage = EBIT / interest
        int_coverage = safe_divide(ebit, interest) if interest > 0 else None

        # Debt-to-EBITDA
        debt_ebitda = safe_divide(total_debt, ebitda) if ebitda and ebitda > 0 else None

        # Estimate annual debt service: interest + principal estimate
        # Principal estimate: total_debt / 10 (rough 10-year amortization)
        principal_estimate = total_debt / 10 if total_debt > 0 else 0
        annual_service = interest + principal_estimate

        # DSCR = OCF / annual_debt_service (or EBITDA / service as proxy)
        if annual_service > 0:
            if ocf is not None:
                dscr = ocf / annual_service
            elif ebitda is not None:
                dscr = ebitda / annual_service
            else:
                dscr = None
        else:
            dscr = None

        # Free cash after service
        fcf_after = None
        if ocf is not None:
            fcf_after = ocf - capex - annual_service

        # Risk level based on DSCR
        if dscr is None:
            risk = "N/A"
        elif dscr >= 2.0:
            risk = "Low"
        elif dscr >= 1.25:
            risk = "Moderate"
        elif dscr >= 1.0:
            risk = "High"
        else:
            risk = "Critical"

        parts = []
        if dscr is not None:
            parts.append(f"DSCR: {dscr:.2f}x")
        if int_coverage is not None:
            parts.append(f"Interest Coverage: {int_coverage:.2f}x")
        if debt_ebitda is not None:
            parts.append(f"Debt/EBITDA: {debt_ebitda:.2f}x")
        parts.append(f"Debt Service Risk: {risk}")
        summary = " | ".join(parts) if parts else "Insufficient data."

        return DebtServiceResult(
            dscr=dscr,
            interest_coverage=int_coverage,
            debt_to_ebitda=debt_ebitda,
            total_debt=total_debt,
            annual_debt_service=annual_service if annual_service > 0 else None,
            free_cash_after_service=fcf_after,
            risk_level=risk,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 15 – Composite Financial Health Score
    # ------------------------------------------------------------------

    _HEALTH_GRADES = [
        (90, "A+"), (80, "A"), (70, "B+"), (60, "B"),
        (50, "C+"), (40, "C"), (30, "D"), (0, "F"),
    ]

    @staticmethod
    def _health_grade(score: float) -> str:
        """Map a 0-100 score to a letter grade."""
        for threshold, grade in CharlieAnalyzer._HEALTH_GRADES:
            if score >= threshold:
                return grade
        return "F"

    @staticmethod
    def _traffic_light(score: float) -> str:
        """Map a 0-100 score to a traffic-light status."""
        if score >= 70:
            return "green"
        elif score >= 40:
            return "yellow"
        return "red"

    def comprehensive_health_score(self, data: FinancialData) -> ComprehensiveHealthResult:
        """Compute a comprehensive financial health score (0-100).

        Aggregates 7 dimensions with configurable weights:
        - Profitability (20%)
        - Liquidity (15%)
        - Leverage/Solvency (15%)
        - Efficiency (10%)
        - Cash Flow Quality (15%)
        - Capital Efficiency (10%)
        - Debt Service (15%)

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ComprehensiveHealthResult
        """
        dimensions: List[HealthDimension] = []

        # --- 1. Profitability (20%) ---
        prof_score = 50.0
        gm = safe_divide(data.gross_profit, data.revenue)
        nm = safe_divide(data.net_income, data.revenue)
        roe = safe_divide(data.net_income, data.total_equity)
        if gm is not None:
            if gm >= 0.50:
                prof_score += 20
            elif gm >= 0.30:
                prof_score += 10
            elif gm < 0.10:
                prof_score -= 20
        if nm is not None:
            if nm >= 0.15:
                prof_score += 15
            elif nm >= 0.05:
                prof_score += 5
            elif nm < 0:
                prof_score -= 20
        if roe is not None:
            if roe >= 0.15:
                prof_score += 15
            elif roe >= 0.08:
                prof_score += 5
            elif roe < 0:
                prof_score -= 10
        prof_score = max(0, min(100, prof_score))
        dimensions.append(HealthDimension(
            name="Profitability", score=prof_score, weight=0.20,
            status=self._traffic_light(prof_score),
            detail=f"GM={gm:.1%}" if gm is not None else "N/A",
        ))

        # --- 2. Liquidity (15%) ---
        liq_score = 50.0
        cr = safe_divide(data.current_assets, data.current_liabilities)
        qr = safe_divide(
            (data.current_assets or 0) - (data.inventory or 0),
            data.current_liabilities,
        ) if data.current_liabilities and data.current_liabilities > 0 else None
        if cr is not None:
            if cr >= 2.0:
                liq_score += 25
            elif cr >= 1.5:
                liq_score += 15
            elif cr >= 1.0:
                liq_score += 5
            else:
                liq_score -= 25
        if qr is not None:
            if qr >= 1.5:
                liq_score += 15
            elif qr >= 1.0:
                liq_score += 5
            elif qr < 0.5:
                liq_score -= 15
        liq_score = max(0, min(100, liq_score))
        dimensions.append(HealthDimension(
            name="Liquidity", score=liq_score, weight=0.15,
            status=self._traffic_light(liq_score),
            detail=f"CR={cr:.2f}" if cr is not None else "N/A",
        ))

        # --- 3. Leverage / Solvency (15%) ---
        lev_score = 50.0
        de = safe_divide(data.total_debt, data.total_equity)
        da = safe_divide(data.total_liabilities, data.total_assets)
        if de is not None:
            if de < 0.5:
                lev_score += 25
            elif de < 1.0:
                lev_score += 10
            elif de < 2.0:
                lev_score -= 5
            else:
                lev_score -= 25
        if da is not None:
            if da < 0.40:
                lev_score += 15
            elif da < 0.60:
                lev_score += 5
            elif da > 0.80:
                lev_score -= 15
        lev_score = max(0, min(100, lev_score))
        dimensions.append(HealthDimension(
            name="Leverage", score=lev_score, weight=0.15,
            status=self._traffic_light(lev_score),
            detail=f"D/E={de:.2f}" if de is not None else "N/A",
        ))

        # --- 4. Efficiency (10%) ---
        eff_score = 50.0
        at = safe_divide(data.revenue, data.total_assets)
        if at is not None:
            if at >= 1.0:
                eff_score += 25
            elif at >= 0.5:
                eff_score += 10
            elif at < 0.2:
                eff_score -= 15
        eff_score = max(0, min(100, eff_score))
        dimensions.append(HealthDimension(
            name="Efficiency", score=eff_score, weight=0.10,
            status=self._traffic_light(eff_score),
            detail=f"AT={at:.2f}x" if at is not None else "N/A",
        ))

        # --- 5. Cash Flow Quality (15%) ---
        cf_score = 50.0
        ocf = data.operating_cash_flow
        ni = data.net_income
        if ocf is not None:
            if ocf > 0:
                cf_score += 15
            else:
                cf_score -= 20
        if ocf is not None and ni is not None and abs(ni) > 1e-12:
            ratio = ocf / ni
            if ni > 0:
                if ratio >= 1.2:
                    cf_score += 20
                elif ratio >= 0.8:
                    cf_score += 10
                elif ratio < 0.5:
                    cf_score -= 15
        if data.capex is not None and ocf is not None:
            fcf = ocf - data.capex
            if fcf > 0:
                cf_score += 10
            else:
                cf_score -= 10
        cf_score = max(0, min(100, cf_score))
        dimensions.append(HealthDimension(
            name="Cash Flow", score=cf_score, weight=0.15,
            status=self._traffic_light(cf_score),
            detail=f"OCF={'pos' if ocf and ocf > 0 else 'neg/NA'}",
        ))

        # --- 6. Capital Efficiency (10%) ---
        cap_score = 50.0
        ce = self.capital_efficiency(data)
        if ce.roic is not None:
            if ce.roic >= 0.15:
                cap_score += 25
            elif ce.roic >= 0.10:
                cap_score += 15
            elif ce.roic >= 0.05:
                cap_score += 5
            elif ce.roic < 0:
                cap_score -= 25
        if ce.eva is not None:
            if ce.eva > 0:
                cap_score += 15
            else:
                cap_score -= 10
        cap_score = max(0, min(100, cap_score))
        dimensions.append(HealthDimension(
            name="Capital Efficiency", score=cap_score, weight=0.10,
            status=self._traffic_light(cap_score),
            detail=f"ROIC={ce.roic:.1%}" if ce.roic is not None else "N/A",
        ))

        # --- 7. Debt Service (15%) ---
        ds_score = 50.0
        ds = self.debt_service_analysis(data)
        if ds.dscr is not None:
            if ds.dscr >= 2.0:
                ds_score += 30
            elif ds.dscr >= 1.25:
                ds_score += 15
            elif ds.dscr >= 1.0:
                ds_score -= 5
            else:
                ds_score -= 30
        if ds.interest_coverage is not None:
            if ds.interest_coverage >= 5.0:
                ds_score += 15
            elif ds.interest_coverage >= 3.0:
                ds_score += 5
            elif ds.interest_coverage < 1.5:
                ds_score -= 15
        ds_score = max(0, min(100, ds_score))
        dimensions.append(HealthDimension(
            name="Debt Service", score=ds_score, weight=0.15,
            status=self._traffic_light(ds_score),
            detail=f"DSCR={ds.dscr:.2f}x" if ds.dscr is not None else "N/A",
        ))

        # --- Weighted overall ---
        overall = sum(d.score * d.weight for d in dimensions)
        overall = max(0, min(100, overall))
        grade = self._health_grade(overall)

        green = sum(1 for d in dimensions if d.status == "green")
        yellow = sum(1 for d in dimensions if d.status == "yellow")
        red = sum(1 for d in dimensions if d.status == "red")
        summary = f"Overall: {grade} ({overall:.0f}/100) | {green} green, {yellow} yellow, {red} red"

        return ComprehensiveHealthResult(
            overall_score=overall,
            grade=grade,
            dimensions=dimensions,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 16 – Operating Leverage & Break-Even Analysis
    # ------------------------------------------------------------------

    def operating_leverage_analysis(self, data: FinancialData) -> OperatingLeverageResult:
        """Analyze operating leverage, cost structure, and break-even point.

        Estimates fixed vs variable cost split from COGS/OpEx composition.
        Calculates contribution margin, DOL, break-even revenue, and margin of safety.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        OperatingLeverageResult
        """
        revenue = data.revenue or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        oi = data.operating_income
        depreciation = data.depreciation or 0

        # --- Estimate fixed vs variable costs ---
        # Heuristic: depreciation + portion of opex are fixed; COGS is mostly variable
        # We treat COGS as variable, depreciation as fixed, and split remaining opex 50/50
        variable_costs = cogs
        non_dep_opex = max(0, opex - depreciation)
        fixed_costs = depreciation + non_dep_opex * 0.5
        variable_costs += non_dep_opex * 0.5

        total_costs = fixed_costs + variable_costs

        # --- Contribution margin ---
        contribution_margin = revenue - variable_costs if revenue > 0 else None
        cm_ratio = safe_divide(contribution_margin, revenue) if contribution_margin is not None else None

        # --- Break-even revenue ---
        break_even = safe_divide(fixed_costs, cm_ratio) if cm_ratio and cm_ratio > 0 else None

        # --- Margin of safety ---
        mos = (revenue - break_even) if break_even is not None and revenue > 0 else None
        mos_pct = safe_divide(mos, revenue) if mos is not None else None

        # --- Degree of Operating Leverage ---
        # DOL = Contribution Margin / Operating Income
        dol = None
        if contribution_margin is not None and oi is not None and abs(oi) > 1e-12:
            dol = contribution_margin / oi

        # --- Cost structure classification ---
        fixed_pct = safe_divide(fixed_costs, total_costs) if total_costs > 0 else None
        if fixed_pct is not None:
            if fixed_pct >= 0.60:
                structure = "High Fixed"
            elif fixed_pct >= 0.40:
                structure = "Balanced"
            else:
                structure = "High Variable"
        else:
            structure = "N/A"

        # --- Summary ---
        parts = []
        if dol is not None:
            parts.append(f"DOL={dol:.2f}x")
        if cm_ratio is not None:
            parts.append(f"CM Ratio={cm_ratio:.1%}")
        if break_even is not None:
            parts.append(f"Break-Even=${break_even:,.0f}")
        if mos_pct is not None:
            parts.append(f"Margin of Safety={mos_pct:.1%}")
        parts.append(f"Structure: {structure}")
        summary = "Operating Leverage: " + " | ".join(parts)

        return OperatingLeverageResult(
            degree_of_operating_leverage=dol,
            contribution_margin=contribution_margin,
            contribution_margin_ratio=cm_ratio,
            estimated_fixed_costs=fixed_costs,
            estimated_variable_costs=variable_costs,
            break_even_revenue=break_even,
            margin_of_safety=mos,
            margin_of_safety_pct=mos_pct,
            cost_structure=structure,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 17 – Cash Flow Quality & Free Cash Flow Yield
    # ------------------------------------------------------------------

    def cash_flow_quality(self, data: FinancialData) -> CashFlowQualityResult:
        """Analyze cash flow quality and free cash flow metrics.

        Computes FCF, FCF yield, FCF margin, OCF/NI quality ratio,
        capex intensity, and cash conversion efficiency.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        CashFlowQualityResult
        """
        ocf = data.operating_cash_flow
        ni = data.net_income
        capex = data.capex or 0
        revenue = data.revenue or 0
        ebitda = data.ebitda

        # --- Free Cash Flow ---
        fcf = (ocf - capex) if ocf is not None else None

        # --- FCF Yield (FCF / Enterprise Value proxy: equity + debt) ---
        ev_proxy = (data.total_equity or 0) + (data.total_debt or 0)
        fcf_yield = safe_divide(fcf, ev_proxy) if fcf is not None and ev_proxy > 0 else None

        # --- FCF Margin ---
        fcf_margin = safe_divide(fcf, revenue) if fcf is not None and revenue > 0 else None

        # --- OCF to Net Income ---
        ocf_to_ni = safe_divide(ocf, ni) if ocf is not None and ni is not None and abs(ni) > 1e-12 else None

        # --- Capex intensity ---
        capex_to_rev = safe_divide(capex, revenue) if revenue > 0 else None
        capex_to_ocf = safe_divide(capex, ocf) if ocf is not None and ocf > 0 else None

        # --- Cash Conversion Efficiency (OCF / EBITDA) ---
        cce = safe_divide(ocf, ebitda) if ocf is not None and ebitda is not None and abs(ebitda) > 1e-12 else None

        # --- Quality scoring ---
        score = 5.0  # Start at mid-point
        indicators = []

        # OCF > NI is positive (low accruals)
        if ocf_to_ni is not None:
            if ocf_to_ni >= 1.2:
                score += 2.0
                indicators.append("OCF exceeds NI significantly (strong)")
            elif ocf_to_ni >= 0.8:
                score += 1.0
                indicators.append("OCF roughly matches NI (adequate)")
            elif ocf_to_ni > 0:
                score -= 1.0
                indicators.append("OCF well below NI (weak cash conversion)")
            else:
                score -= 2.0
                indicators.append("Negative OCF/NI ratio (poor)")

        # Positive FCF
        if fcf is not None:
            if fcf > 0:
                score += 1.5
                indicators.append("Positive free cash flow")
            else:
                score -= 1.5
                indicators.append("Negative free cash flow")

        # FCF margin
        if fcf_margin is not None:
            if fcf_margin >= 0.15:
                score += 1.0
                indicators.append("High FCF margin (>=15%)")
            elif fcf_margin >= 0.05:
                score += 0.5
                indicators.append("Moderate FCF margin")
            elif fcf_margin < 0:
                score -= 1.0
                indicators.append("Negative FCF margin")

        # Cash conversion efficiency
        if cce is not None:
            if cce >= 0.80:
                score += 1.0
                indicators.append("Strong cash conversion (>=80% of EBITDA)")
            elif cce >= 0.50:
                score += 0.5
                indicators.append("Adequate cash conversion")
            elif cce < 0.30:
                score -= 1.0
                indicators.append("Weak cash conversion (<30% of EBITDA)")

        # Capex intensity
        if capex_to_rev is not None:
            if capex_to_rev > 0.20:
                score -= 0.5
                indicators.append("High capex intensity (>20% of revenue)")
            elif capex_to_rev < 0.05:
                score += 0.5
                indicators.append("Low capex intensity (<5%)")

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Strong"
        elif score >= 6:
            grade = "Adequate"
        elif score >= 4:
            grade = "Weak"
        else:
            grade = "Poor"

        # Summary
        parts = [f"Cash Flow Quality: {grade} ({score:.1f}/10)"]
        if fcf is not None:
            parts.append(f"FCF=${fcf:,.0f}")
        if fcf_yield is not None:
            parts.append(f"FCF Yield={fcf_yield:.1%}")
        if cce is not None:
            parts.append(f"Cash Conversion={cce:.0%}")
        summary = " | ".join(parts)

        return CashFlowQualityResult(
            fcf=fcf,
            fcf_yield=fcf_yield,
            fcf_margin=fcf_margin,
            ocf_to_net_income=ocf_to_ni,
            capex_to_revenue=capex_to_rev,
            capex_to_ocf=capex_to_ocf,
            cash_conversion_efficiency=cce,
            quality_grade=grade,
            indicators=indicators,
            summary=summary,
        )

    def asset_efficiency_analysis(self, data: FinancialData) -> AssetEfficiencyResult:
        """Analyze asset efficiency and turnover ratios.

        Computes total asset turnover, fixed asset turnover, inventory turnover,
        receivables turnover, payables turnover, working capital turnover,
        equity turnover, and an efficiency score.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        AssetEfficiencyResult
        """
        revenue = data.revenue or 0
        cogs = data.cogs or 0
        ta = data.total_assets
        ca = data.current_assets
        cl = data.current_liabilities
        inv = data.inventory
        ar = data.accounts_receivable
        ap = data.accounts_payable
        equity = data.total_equity

        # Use average values if available, otherwise point-in-time
        avg_ta = data.avg_total_assets or ta
        avg_inv = data.avg_inventory or inv
        avg_ar = data.avg_receivables or ar
        avg_ap = data.avg_payables or ap

        # --- Turnover ratios ---
        total_at = safe_divide(revenue, avg_ta) if avg_ta is not None and avg_ta > 0 else None

        # Fixed assets = Total assets - Current assets
        fixed_assets = (ta - ca) if ta is not None and ca is not None and (ta - ca) > 0 else None
        fixed_at = safe_divide(revenue, fixed_assets) if fixed_assets is not None and revenue > 0 else None

        inv_turnover = safe_divide(cogs, avg_inv) if avg_inv is not None and avg_inv > 0 and cogs > 0 else None

        ar_turnover = safe_divide(revenue, avg_ar) if avg_ar is not None and avg_ar > 0 and revenue > 0 else None

        ap_turnover = safe_divide(cogs, avg_ap) if avg_ap is not None and avg_ap > 0 and cogs > 0 else None

        # Working capital turnover
        wc = (ca - cl) if ca is not None and cl is not None else None
        wc_turnover = safe_divide(revenue, wc) if wc is not None and wc > 0 and revenue > 0 else None

        # Equity turnover
        eq_turnover = safe_divide(revenue, equity) if equity is not None and equity > 0 and revenue > 0 else None

        # --- Efficiency scoring (0-10) ---
        score = 5.0
        count = 0

        if total_at is not None:
            count += 1
            if total_at >= 1.5:
                score += 1.5
            elif total_at >= 0.8:
                score += 0.5
            elif total_at < 0.3:
                score -= 1.5

        if inv_turnover is not None:
            count += 1
            if inv_turnover >= 8:
                score += 1.5
            elif inv_turnover >= 4:
                score += 0.5
            elif inv_turnover < 2:
                score -= 1.0

        if ar_turnover is not None:
            count += 1
            if ar_turnover >= 10:
                score += 1.0
            elif ar_turnover >= 6:
                score += 0.5
            elif ar_turnover < 3:
                score -= 1.0

        if fixed_at is not None:
            count += 1
            if fixed_at >= 3.0:
                score += 1.0
            elif fixed_at >= 1.5:
                score += 0.5
            elif fixed_at < 0.5:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Average"
        else:
            grade = "Below Average"

        # Summary
        parts = [f"Asset Efficiency: {grade} ({score:.1f}/10)"]
        if total_at is not None:
            parts.append(f"Asset Turnover={total_at:.2f}x")
        if inv_turnover is not None:
            parts.append(f"Inventory Turnover={inv_turnover:.1f}x")
        if ar_turnover is not None:
            parts.append(f"AR Turnover={ar_turnover:.1f}x")
        summary = " | ".join(parts)

        return AssetEfficiencyResult(
            total_asset_turnover=total_at,
            fixed_asset_turnover=fixed_at,
            inventory_turnover=inv_turnover,
            receivables_turnover=ar_turnover,
            payables_turnover=ap_turnover,
            working_capital_turnover=wc_turnover,
            equity_turnover=eq_turnover,
            efficiency_score=score,
            efficiency_grade=grade,
            summary=summary,
        )

    def profitability_decomposition(self, data: FinancialData) -> ProfitabilityDecompResult:
        """Decompose profitability into return components.

        Computes ROE, ROA, ROIC, NOPAT, invested capital, spread,
        economic profit, and related leverage/efficiency metrics.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ProfitabilityDecompResult
        """
        revenue = data.revenue or 0
        ni = data.net_income
        ta = data.total_assets
        equity = data.total_equity
        ebit = data.ebit or data.operating_income
        debt = data.total_debt or 0
        interest = data.interest_expense or 0
        cl = data.current_liabilities or 0

        # --- ROE = NI / Equity ---
        roe = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None

        # --- ROA = NI / Total Assets ---
        roa = safe_divide(ni, ta) if ni is not None and ta is not None and ta > 0 else None

        # --- Invested Capital = Total Debt + Equity (or TA - non-interest-bearing CL) ---
        invested_capital = None
        if equity is not None and equity > 0:
            invested_capital = equity + debt
        elif ta is not None and ta > 0:
            invested_capital = ta - cl

        # --- NOPAT = EBIT × (1 - tax_rate) ---
        nopat = None
        if ebit is not None:
            nopat = ebit * (1 - self._tax_rate)

        # --- ROIC = NOPAT / Invested Capital ---
        roic = safe_divide(nopat, invested_capital) if nopat is not None and invested_capital is not None and invested_capital > 0 else None

        # --- Cost of capital proxy (simplified WACC estimate) ---
        # Use interest rate on debt as debt cost, assume 10% equity cost
        cost_of_debt = safe_divide(interest, debt) if debt > 0 else 0.05
        equity_weight = safe_divide(equity, invested_capital) if invested_capital and invested_capital > 0 and equity else 0.5
        debt_weight = 1 - equity_weight
        wacc_proxy = (equity_weight * 0.10) + (debt_weight * (cost_of_debt or 0.05) * (1 - self._tax_rate))

        # --- Spread = ROIC - WACC proxy ---
        spread = (roic - wacc_proxy) if roic is not None else None

        # --- Economic profit = Spread × Invested Capital ---
        economic_profit = None
        if spread is not None and invested_capital is not None and invested_capital > 0:
            economic_profit = spread * invested_capital

        # --- Asset turnover ---
        asset_to = safe_divide(revenue, ta) if ta is not None and ta > 0 and revenue > 0 else None

        # --- Financial leverage = TA / Equity ---
        fin_leverage = safe_divide(ta, equity) if ta is not None and equity is not None and equity > 0 else None

        # --- Tax efficiency = NI / EBT ---
        ebt = data.ebt
        if ebt is None and ni is not None:
            ebt = safe_divide(ni, 1 - self._tax_rate)
        tax_eff = safe_divide(ni, ebt) if ni is not None and ebt is not None and ebt != 0 else None

        # --- Capital intensity = TA / Revenue ---
        cap_intensity = safe_divide(ta, revenue) if ta is not None and revenue > 0 else None

        # --- Scoring (0-10) ---
        score = 5.0

        if roe is not None:
            if roe >= 0.20:
                score += 2.0
            elif roe >= 0.12:
                score += 1.0
            elif roe < 0:
                score -= 2.0

        if roic is not None:
            if roic >= 0.15:
                score += 1.5
            elif roic >= 0.08:
                score += 0.5
            elif roic < 0:
                score -= 1.5

        if spread is not None:
            if spread > 0.05:
                score += 1.0
            elif spread > 0:
                score += 0.5
            elif spread < -0.05:
                score -= 1.0

        if roa is not None:
            if roa >= 0.10:
                score += 0.5
            elif roa < 0:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Elite"
        elif score >= 6:
            grade = "Strong"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Poor"

        # Summary
        parts = [f"Profitability: {grade} ({score:.1f}/10)"]
        if roe is not None:
            parts.append(f"ROE={roe:.1%}")
        if roic is not None:
            parts.append(f"ROIC={roic:.1%}")
        if spread is not None:
            parts.append(f"Spread={spread:.1%}")
        summary = " | ".join(parts)

        return ProfitabilityDecompResult(
            roe=roe,
            roa=roa,
            roic=roic,
            invested_capital=invested_capital,
            nopat=nopat,
            spread=spread,
            economic_profit=economic_profit,
            asset_turnover=asset_to,
            financial_leverage=fin_leverage,
            tax_efficiency=tax_eff,
            capital_intensity=cap_intensity,
            profitability_score=score,
            profitability_grade=grade,
            summary=summary,
        )

    def risk_adjusted_performance(self, data: FinancialData) -> RiskAdjustedResult:
        """Compute risk-adjusted performance metrics.

        Evaluates returns relative to financial and operating risk,
        including leverage-adjusted returns, margin of safety, and
        risk-return trade-off ratios.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        RiskAdjustedResult
        """
        ni = data.net_income
        equity = data.total_equity
        ta = data.total_assets
        debt = data.total_debt or 0
        ebit = data.ebit or data.operating_income
        interest = data.interest_expense or 0
        revenue = data.revenue or 0

        # --- ROE and ROA ---
        roe = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None
        roa = safe_divide(ni, ta) if ni is not None and ta is not None and ta > 0 else None

        # --- Leverage ratio ---
        leverage = safe_divide(ta, equity) if ta is not None and equity is not None and equity > 0 else None

        # --- Return on risk = ROE / leverage_ratio ---
        # Adjusts for the fact that high ROE from leverage isn't truly "better"
        return_on_risk = safe_divide(roe, leverage) if roe is not None and leverage is not None and leverage > 0 else None

        # --- Risk-adjusted ROE = ROE × (Equity / Total Assets) ---
        # Penalizes ROE that comes from high leverage
        risk_adj_roe = None
        if roe is not None and ta is not None and equity is not None and ta > 0:
            equity_ratio = equity / ta
            risk_adj_roe = roe * equity_ratio

        # --- Risk-adjusted ROA = ROA × (1 - Debt/Assets) ---
        risk_adj_roa = None
        if roa is not None and ta is not None and ta > 0:
            debt_ratio = debt / ta
            risk_adj_roa = roa * (1 - debt_ratio)

        # --- Debt-adjusted return = NI / (Equity + Debt) ---
        invested = (equity or 0) + debt
        debt_adj_return = safe_divide(ni, invested) if ni is not None and invested > 0 else None

        # --- Volatility proxy: abs(operating margin - net margin) ---
        op_margin = safe_divide(ebit, revenue) if ebit is not None and revenue > 0 else None
        net_margin = safe_divide(ni, revenue) if ni is not None and revenue > 0 else None
        vol_proxy = None
        if op_margin is not None and net_margin is not None:
            vol_proxy = abs(op_margin - net_margin)

        # --- Downside risk proxy: Debt / EBITDA (higher = more risk) ---
        ebitda = data.ebitda
        downside = safe_divide(debt, ebitda) if ebitda is not None and ebitda > 0 else None

        # --- Margin of safety = (EBIT - Interest) / Interest ---
        margin_safety = None
        if ebit is not None and interest > 0:
            margin_safety = (ebit - interest) / interest

        # --- Return per unit risk ---
        # Use ROE / (Debt/Equity ratio), higher = better risk-adjusted
        de_ratio = safe_divide(debt, equity) if equity is not None and equity > 0 else None
        return_per_risk = None
        if roe is not None and de_ratio is not None and de_ratio > 0:
            return_per_risk = roe / de_ratio

        # --- Scoring (0-10) ---
        score = 5.0

        if return_on_risk is not None:
            if return_on_risk >= 0.10:
                score += 1.5
            elif return_on_risk >= 0.05:
                score += 0.5
            elif return_on_risk < 0:
                score -= 1.5

        if margin_safety is not None:
            if margin_safety >= 5.0:
                score += 2.0
            elif margin_safety >= 2.0:
                score += 1.0
            elif margin_safety < 1.0:
                score -= 1.5

        if downside is not None:
            if downside <= 1.0:
                score += 1.0
            elif downside <= 3.0:
                score += 0.0
            elif downside > 5.0:
                score -= 1.5

        if risk_adj_roe is not None:
            if risk_adj_roe >= 0.12:
                score += 1.0
            elif risk_adj_roe >= 0.06:
                score += 0.5
            elif risk_adj_roe < 0:
                score -= 1.0

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Superior"
        elif score >= 6:
            grade = "Favorable"
        elif score >= 4:
            grade = "Neutral"
        else:
            grade = "Elevated"

        # Summary
        parts = [f"Risk-Adjusted: {grade} ({score:.1f}/10)"]
        if return_on_risk is not None:
            parts.append(f"Return/Risk={return_on_risk:.1%}")
        if margin_safety is not None:
            parts.append(f"Safety Margin={margin_safety:.1f}x")
        summary = " | ".join(parts)

        return RiskAdjustedResult(
            return_on_risk=return_on_risk,
            risk_adjusted_roe=risk_adj_roe,
            risk_adjusted_roa=risk_adj_roa,
            debt_adjusted_return=debt_adj_return,
            volatility_proxy=vol_proxy,
            downside_risk=downside,
            margin_of_safety=margin_safety,
            return_per_unit_risk=return_per_risk,
            risk_score=score,
            risk_grade=grade,
            summary=summary,
        )

    def capital_structure_analysis(self, data: FinancialData) -> CapitalStructureResult:
        """Analyze capital structure, leverage, and debt capacity.

        Evaluates the mix of debt and equity financing, coverage ratios,
        net debt position, and distance from optimal leverage.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        CapitalStructureResult
        """
        equity = data.total_equity
        ta = data.total_assets
        debt = data.total_debt or 0
        ebit = data.ebit or data.operating_income
        ebitda = data.ebitda
        interest = data.interest_expense or 0
        ocf = data.operating_cash_flow
        ni = data.net_income
        cash = data.cash or (data.current_assets - data.current_liabilities
                             if data.current_assets is not None and data.current_liabilities is not None
                             else None)

        # --- Debt / Equity ---
        de_ratio = safe_divide(debt, equity) if equity is not None and equity > 0 else None

        # --- Debt / Assets ---
        da_ratio = safe_divide(debt, ta) if ta is not None and ta > 0 else None

        # --- Debt / EBITDA ---
        debt_ebitda = safe_divide(debt, ebitda) if ebitda is not None and ebitda > 0 else None

        # --- Equity multiplier = TA / Equity ---
        eq_mult = safe_divide(ta, equity) if ta is not None and equity is not None and equity > 0 else None

        # --- Interest coverage = EBIT / Interest ---
        int_coverage = safe_divide(ebit, interest) if ebit is not None and interest > 0 else None

        # --- Debt service coverage = OCF / (Interest + assumed principal) ---
        # Use OCF / Interest as proxy when principal payment unknown
        dsc = safe_divide(ocf, interest) if ocf is not None and interest > 0 else None

        # --- Net debt = Debt - Cash ---
        net_debt_val = None
        if cash is not None:
            net_debt_val = debt - cash

        # --- Net debt / EBITDA ---
        nd_ebitda = safe_divide(net_debt_val, ebitda) if net_debt_val is not None and ebitda is not None and ebitda > 0 else None

        # --- Capitalization rate = Debt / (Debt + Equity) ---
        cap_rate = None
        if equity is not None:
            total_cap = debt + equity
            cap_rate = safe_divide(debt, total_cap) if total_cap > 0 else None

        # --- Equity ratio = Equity / TA ---
        eq_ratio = safe_divide(equity, ta) if equity is not None and ta is not None and ta > 0 else None

        # --- Simplified WACC ---
        wacc = None
        if equity is not None and ta is not None and ta > 0:
            e_weight = equity / ta
            d_weight = 1 - e_weight
            cost_debt = safe_divide(interest, debt) if debt > 0 else 0.05
            wacc = (e_weight * 0.10) + (d_weight * (cost_debt or 0.05) * (1 - self._tax_rate))

        # --- Distance from optimal D/E (target 0.5-1.0 for most firms) ---
        optimal_de = 0.75  # midpoint of healthy range
        opt_distance = abs(de_ratio - optimal_de) if de_ratio is not None else None

        # --- Scoring (0-10) ---
        score = 5.0

        # D/E ratio scoring
        if de_ratio is not None:
            if de_ratio <= 0.5:
                score += 1.5  # Conservative
            elif de_ratio <= 1.0:
                score += 1.0  # Balanced
            elif de_ratio <= 2.0:
                score -= 0.5  # Moderate leverage
            else:
                score -= 2.0  # High leverage

        # Interest coverage scoring
        if int_coverage is not None:
            if int_coverage >= 8.0:
                score += 2.0
            elif int_coverage >= 3.0:
                score += 1.0
            elif int_coverage >= 1.5:
                score += 0.0
            elif int_coverage < 1.0:
                score -= 2.0

        # Debt/EBITDA scoring
        if debt_ebitda is not None:
            if debt_ebitda <= 1.0:
                score += 1.0
            elif debt_ebitda <= 3.0:
                score += 0.0
            elif debt_ebitda > 5.0:
                score -= 1.5

        # Equity ratio scoring
        if eq_ratio is not None:
            if eq_ratio >= 0.60:
                score += 0.5
            elif eq_ratio < 0.20:
                score -= 1.0

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Conservative"
        elif score >= 6:
            grade = "Balanced"
        elif score >= 4:
            grade = "Aggressive"
        else:
            grade = "Distressed"

        # Summary
        parts = [f"Capital Structure: {grade} ({score:.1f}/10)"]
        if de_ratio is not None:
            parts.append(f"D/E={de_ratio:.2f}")
        if int_coverage is not None:
            parts.append(f"Interest Coverage={int_coverage:.1f}x")
        summary = " | ".join(parts)

        return CapitalStructureResult(
            debt_to_equity=de_ratio,
            debt_to_assets=da_ratio,
            debt_to_ebitda=debt_ebitda,
            equity_multiplier=eq_mult,
            interest_coverage=int_coverage,
            debt_service_coverage=dsc,
            net_debt=net_debt_val,
            net_debt_to_ebitda=nd_ebitda,
            capitalization_rate=cap_rate,
            equity_ratio=eq_ratio,
            wacc_estimate=wacc,
            optimal_leverage_distance=opt_distance,
            capital_score=score,
            capital_grade=grade,
            summary=summary,
        )

    def valuation_indicators(self, data: FinancialData) -> ValuationIndicatorsResult:
        """Compute valuation indicators and intrinsic value proxies.

        Since we don't have market price data, this uses fundamental-only
        proxies: earnings yield, EV/EBITDA, Graham Number, and a simplified
        DCF-lite intrinsic value estimate based on earnings power.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ValuationIndicatorsResult
        """
        ni = data.net_income
        equity = data.total_equity
        ta = data.total_assets
        debt = data.total_debt or 0
        ebit = data.ebit or data.operating_income
        ebitda = data.ebitda
        revenue = data.revenue or 0
        interest = data.interest_expense or 0
        cash = data.cash or (data.current_assets - data.current_liabilities
                             if data.current_assets is not None and data.current_liabilities is not None
                             else None)

        # --- Earnings yield proxy = NI / Equity (like E/P without market price) ---
        earnings_yield = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None

        # --- Book value proxy = Equity ---
        bv_proxy = equity if equity is not None and equity > 0 else None

        # --- Enterprise Value proxy = Equity + Debt - Cash ---
        ev_proxy = None
        if equity is not None:
            cash_val = cash if cash is not None else 0
            ev_proxy = equity + debt - cash_val
            if ev_proxy <= 0:
                ev_proxy = None

        # --- EV / EBITDA ---
        ev_ebitda = safe_divide(ev_proxy, ebitda) if ev_proxy is not None and ebitda is not None and ebitda > 0 else None

        # --- EV / Revenue ---
        ev_revenue = safe_divide(ev_proxy, revenue) if ev_proxy is not None and revenue > 0 else None

        # --- EV / EBIT ---
        ev_ebit = safe_divide(ev_proxy, ebit) if ev_proxy is not None and ebit is not None and ebit > 0 else None

        # --- Price-to-book proxy ---
        # Use (EV / Equity) as proxy since we lack market cap
        ptb_proxy = safe_divide(ev_proxy, equity) if ev_proxy is not None and equity is not None and equity > 0 else None

        # --- ROIC (from profitability decomp, repeated for convenience) ---
        invested = (equity or 0) + debt
        nopat = ebit * (1 - self._tax_rate) if ebit is not None else None
        roic = safe_divide(nopat, invested) if nopat is not None and invested > 0 else None

        # --- Graham Number proxy = sqrt(22.5 × EPS_proxy × BV_proxy) ---
        # Use NI as EPS proxy, Equity as BV proxy (no per-share data)
        graham = None
        if ni is not None and ni > 0 and bv_proxy is not None:
            import math
            graham = math.sqrt(22.5 * ni * bv_proxy)

        # --- DCF-lite intrinsic value proxy ---
        # Capitalized earnings: NI / discount_rate (assumes perpetuity)
        discount_rate = 0.10  # 10% required return
        intrinsic = None
        if ni is not None and ni > 0:
            intrinsic = ni / discount_rate

        # --- Margin of safety % = (intrinsic - book) / intrinsic ---
        mos_pct = None
        if intrinsic is not None and bv_proxy is not None and intrinsic > 0:
            mos_pct = (intrinsic - bv_proxy) / intrinsic

        # --- Scoring (0-10) ---
        score = 5.0

        # EV/EBITDA scoring (lower is cheaper)
        if ev_ebitda is not None:
            if ev_ebitda < 6.0:
                score += 1.5
            elif ev_ebitda < 10.0:
                score += 0.5
            elif ev_ebitda > 15.0:
                score -= 1.0

        # Earnings yield scoring (higher is cheaper)
        if earnings_yield is not None:
            if earnings_yield >= 0.15:
                score += 2.0
            elif earnings_yield >= 0.10:
                score += 1.0
            elif earnings_yield >= 0.05:
                score += 0.5
            elif earnings_yield < 0:
                score -= 1.5

        # ROIC scoring (quality of returns)
        if roic is not None:
            if roic >= 0.15:
                score += 1.0
            elif roic >= 0.08:
                score += 0.5
            elif roic < 0:
                score -= 1.0

        # Margin of safety scoring
        if mos_pct is not None:
            if mos_pct >= 0.30:
                score += 0.5
            elif mos_pct < 0:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Undervalued"
        elif score >= 6:
            grade = "Fair Value"
        elif score >= 4:
            grade = "Fully Valued"
        else:
            grade = "Overvalued"

        # Summary
        parts = [f"Valuation: {grade} ({score:.1f}/10)"]
        if ev_ebitda is not None:
            parts.append(f"EV/EBITDA={ev_ebitda:.1f}x")
        if earnings_yield is not None:
            parts.append(f"Earnings Yield={earnings_yield:.1%}")
        summary = " | ".join(parts)

        return ValuationIndicatorsResult(
            earnings_yield=earnings_yield,
            book_value_per_share_proxy=bv_proxy,
            ev_proxy=ev_proxy,
            ev_to_ebitda=ev_ebitda,
            ev_to_revenue=ev_revenue,
            ev_to_ebit=ev_ebit,
            price_to_book_proxy=ptb_proxy,
            return_on_invested_capital=roic,
            graham_number_proxy=graham,
            intrinsic_value_proxy=intrinsic,
            margin_of_safety_pct=mos_pct,
            valuation_score=score,
            valuation_grade=grade,
            summary=summary,
        )

    def sustainable_growth_analysis(self, data: FinancialData) -> SustainableGrowthResult:
        """Analyze sustainable and internal growth capacity.

        Computes the maximum growth rate achievable without external
        financing (internal growth) and with constant capital structure
        (sustainable growth), along with reinvestment and plowback metrics.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        SustainableGrowthResult
        """
        ni = data.net_income
        equity = data.total_equity
        ta = data.total_assets
        dividends = data.dividends_paid or 0
        capex = data.capex or 0
        depreciation = data.depreciation or 0
        retained_earnings = data.retained_earnings

        # --- ROE and ROA ---
        roe = safe_divide(ni, equity) if ni is not None and equity is not None and equity > 0 else None
        roa = safe_divide(ni, ta) if ni is not None and ta is not None and ta > 0 else None

        # --- Payout and retention ---
        payout = None
        retention = None
        if ni is not None and ni > 0:
            payout = dividends / ni if dividends > 0 else 0.0
            retention = 1 - payout

        # --- Sustainable growth rate = ROE × retention ---
        sgr = None
        if roe is not None and retention is not None:
            sgr = roe * retention

        # --- Internal growth rate = ROA × b / (1 - ROA × b) ---
        igr = None
        if roa is not None and retention is not None:
            roa_b = roa * retention
            if roa_b < 1.0:  # avoid division by zero / infinity
                igr = roa_b / (1 - roa_b)

        # --- Plowback capacity = retained earnings / NI ---
        plowback = None
        if retained_earnings is not None and ni is not None and ni > 0:
            plowback = retained_earnings / ni

        # --- Equity growth rate = retained earnings / equity ---
        eq_growth = safe_divide(retained_earnings, equity) if retained_earnings is not None and equity is not None and equity > 0 else None

        # --- Reinvestment rate = Capex / Depreciation ---
        reinvest = safe_divide(capex, depreciation) if depreciation > 0 else None

        # --- Growth-profitability balance = SGR / cost of equity proxy (10%) ---
        gpb = None
        cost_of_equity = 0.10
        if sgr is not None:
            gpb = sgr / cost_of_equity

        # --- Scoring (0-10) ---
        score = 5.0

        # SGR scoring
        if sgr is not None:
            if sgr >= 0.15:
                score += 2.0
            elif sgr >= 0.08:
                score += 1.0
            elif sgr >= 0.03:
                score += 0.0
            elif sgr < 0:
                score -= 2.0

        # ROE contribution
        if roe is not None:
            if roe >= 0.20:
                score += 1.0
            elif roe >= 0.10:
                score += 0.5
            elif roe < 0:
                score -= 1.5

        # Retention ratio (high retention = reinvesting for growth)
        if retention is not None:
            if retention >= 0.80:
                score += 1.0
            elif retention >= 0.50:
                score += 0.5

        # Reinvestment rate (maintaining/growing assets)
        if reinvest is not None:
            if reinvest >= 1.5:
                score += 0.5
            elif reinvest >= 1.0:
                score += 0.0
            elif reinvest < 0.5:
                score -= 0.5

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "High Growth"
        elif score >= 6:
            grade = "Sustainable"
        elif score >= 4:
            grade = "Moderate"
        else:
            grade = "Constrained"

        # Summary
        parts = [f"Growth Capacity: {grade} ({score:.1f}/10)"]
        if sgr is not None:
            parts.append(f"SGR={sgr:.1%}")
        if igr is not None:
            parts.append(f"IGR={igr:.1%}")
        summary = " | ".join(parts)

        return SustainableGrowthResult(
            sustainable_growth_rate=sgr,
            internal_growth_rate=igr,
            retention_ratio=retention,
            payout_ratio=payout,
            plowback_capacity=plowback,
            roe=roe,
            roa=roa,
            equity_growth_rate=eq_growth,
            reinvestment_rate=reinvest,
            growth_profitability_balance=gpb,
            growth_score=score,
            growth_grade=grade,
            summary=summary,
        )

    def concentration_risk_analysis(self, data: FinancialData) -> ConcentrationRiskResult:
        """Analyze concentration and structural risk across financial dimensions.

        Evaluates how diversified or concentrated the company's financial
        structure is across assets, liabilities, revenue, and earnings.

        Parameters
        ----------
        data : FinancialData

        Returns
        -------
        ConcentrationRiskResult
        """
        revenue = data.revenue
        ta = data.total_assets
        ca = data.current_assets
        cl = data.current_liabilities
        tl = data.total_liabilities
        equity = data.total_equity
        oi = data.operating_income
        ni = data.net_income
        ebit = data.ebit
        ebitda = data.ebitda
        interest = data.interest_expense
        ocf = data.operating_cash_flow
        capex = data.capex or 0
        debt = data.total_debt

        # --- Revenue-asset intensity (asset turnover) ---
        rev_asset = safe_divide(revenue, ta) if revenue is not None and ta is not None and ta > 0 else None

        # --- Operating dependency (operating margin) ---
        op_dep = safe_divide(oi, revenue) if oi is not None and revenue is not None and revenue > 0 else None

        # --- Asset composition ---
        asset_current = safe_divide(ca, ta) if ca is not None and ta is not None and ta > 0 else None
        asset_fixed = (1.0 - asset_current) if asset_current is not None else None

        # --- Liability structure ---
        liab_current = safe_divide(cl, tl) if cl is not None and tl is not None and tl > 0 else None

        # --- Earnings retention (NI / EBITDA) ---
        earn_retention = safe_divide(ni, ebitda) if ni is not None and ebitda is not None and ebitda > 0 else None

        # --- Working capital intensity (NWC / Revenue) ---
        nwc = None
        wc_intensity = None
        if ca is not None and cl is not None:
            nwc = ca - cl
            if revenue is not None and revenue > 0:
                wc_intensity = nwc / revenue

        # --- Capex intensity ---
        capex_int = safe_divide(capex, revenue) if revenue is not None and revenue > 0 and capex > 0 else None

        # --- Interest burden (Interest / EBIT) ---
        int_burden = safe_divide(interest, ebit) if interest is not None and ebit is not None and ebit > 0 else None

        # --- Cash conversion efficiency (OCF / NI) ---
        cash_conv = safe_divide(ocf, ni) if ocf is not None and ni is not None and ni > 0 else None

        # --- Fixed asset ratio (fixed assets / equity) ---
        fixed_assets = (ta - ca) if ta is not None and ca is not None else None
        fa_ratio = safe_divide(fixed_assets, equity) if fixed_assets is not None and equity is not None and equity > 0 else None

        # --- Debt concentration (debt / TL) ---
        debt_conc = safe_divide(debt, tl) if debt is not None and tl is not None and tl > 0 else None

        # --- Scoring (0-10) ---
        score = 5.0

        # Balanced asset composition: not too heavy on either side
        if asset_current is not None:
            if 0.30 <= asset_current <= 0.60:
                score += 1.0  # well balanced
            elif asset_current > 0.80 or asset_current < 0.10:
                score -= 1.0  # too concentrated

        # Operating dependency - moderate margins are healthier than extremes
        if op_dep is not None:
            if 0.10 <= op_dep <= 0.30:
                score += 1.0  # healthy operating margin
            elif op_dep >= 0.05:
                score += 0.5
            elif op_dep < 0:
                score -= 1.5  # operating losses

        # Cash conversion - OCF/NI near or above 1.0 is positive
        if cash_conv is not None:
            if cash_conv >= 1.2:
                score += 1.5
            elif cash_conv >= 0.8:
                score += 0.5
            elif cash_conv < 0.5:
                score -= 1.0

        # Interest burden - low is good
        if int_burden is not None:
            if int_burden <= 0.15:
                score += 0.5
            elif int_burden >= 0.50:
                score -= 1.0

        # Working capital intensity - moderate is ideal
        if wc_intensity is not None:
            if 0.05 <= wc_intensity <= 0.25:
                score += 0.5
            elif wc_intensity < 0:
                score -= 0.5  # negative NWC can be risky

        score = max(0, min(10, score))

        # Grade
        if score >= 8:
            grade = "Well Diversified"
        elif score >= 6:
            grade = "Balanced"
        elif score >= 4:
            grade = "Concentrated"
        else:
            grade = "Highly Concentrated"

        # Summary
        parts = [f"Concentration Risk: {grade} ({score:.1f}/10)"]
        if rev_asset is not None:
            parts.append(f"Asset Turnover={rev_asset:.2f}x")
        if op_dep is not None:
            parts.append(f"Op Margin={op_dep:.1%}")
        summary = " | ".join(parts)

        return ConcentrationRiskResult(
            revenue_asset_intensity=rev_asset,
            operating_dependency=op_dep,
            asset_composition_current=asset_current,
            asset_composition_fixed=asset_fixed,
            liability_structure_current=liab_current,
            earnings_retention_ratio=earn_retention,
            working_capital_intensity=wc_intensity,
            capex_intensity=capex_int,
            interest_burden=int_burden,
            cash_conversion_efficiency=cash_conv,
            fixed_asset_ratio=fa_ratio,
            debt_concentration=debt_conc,
            concentration_score=score,
            concentration_grade=grade,
            summary=summary,
        )

    def margin_of_safety_analysis(self, data: FinancialData) -> MarginOfSafetyResult:
        """Analyze margin of safety using intrinsic value vs market price."""
        ni = data.net_income
        equity = data.total_equity
        shares = data.shares_outstanding
        price = data.share_price
        ca = data.current_assets
        tl = data.total_liabilities
        ta = data.total_assets

        # --- Market Cap ---
        mcap = None
        if shares and shares > 0 and price and price > 0:
            mcap = shares * price

        # --- Earnings Yield ---
        earn_yield = None
        if ni is not None and mcap and mcap > 0:
            earn_yield = safe_divide(ni, mcap)

        # --- Book Value ---
        bv_per_share = None
        if equity is not None and shares and shares > 0:
            bv_per_share = safe_divide(equity, shares)

        # --- Price to Book ---
        ptb = None
        bv_discount = None
        if bv_per_share is not None and bv_per_share > 0 and price and price > 0:
            ptb = safe_divide(price, bv_per_share)
            bv_discount = 1.0 - ptb  # Positive = below book value

        # --- Intrinsic Value (capitalized earnings at 10%) ---
        iv = None
        iv_margin = None
        if ni is not None and ni > 0:
            iv = ni / 0.10  # Capitalize at 10%
            if mcap and mcap > 0:
                iv_margin = safe_divide(iv - mcap, iv)

        # --- Tangible Book Value (approximate: equity as proxy) ---
        tbv = equity  # No intangibles field, use equity as approximation

        # --- Liquidation Value (70% of tangible BV) ---
        liq_val = None
        if tbv is not None and tbv > 0:
            liq_val = tbv * 0.70

        # --- Net Current Asset Value (Graham NCAV) ---
        ncav = None
        ncav_ps = None
        if ca is not None and tl is not None:
            ncav = ca - tl
            if shares and shares > 0:
                ncav_ps = safe_divide(ncav, shares)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # Earnings yield
        if earn_yield is not None:
            if earn_yield >= 0.15:
                score += 2.0  # Very high yield = cheap
            elif earn_yield >= 0.10:
                score += 1.0
            elif earn_yield >= 0.05:
                score += 0.0
            elif earn_yield < 0:
                score -= 1.5

        # Intrinsic value margin
        if iv_margin is not None:
            if iv_margin >= 0.50:
                score += 1.5  # IV is 2x+ market cap
            elif iv_margin >= 0.25:
                score += 1.0
            elif iv_margin < 0:
                score -= 1.0  # Overvalued vs IV

        # Book value discount
        if bv_discount is not None:
            if bv_discount > 0:
                score += 0.5  # Trading below book
            elif bv_discount < -1.0:
                score -= 0.5  # Trading at 2x+ book

        # NCAV check
        if ncav_ps is not None and price and price > 0:
            if ncav_ps >= price:
                score += 1.0  # Graham's net-net

        score = max(0.0, min(10.0, score))

        # --- Grade ---
        if score >= 8.0:
            grade = "Wide Margin"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Thin"
        else:
            grade = "No Margin"

        # --- Summary ---
        parts = ["Margin of Safety:"]
        if earn_yield is not None:
            parts.append(f"Earnings yield {earn_yield:.1%}.")
        if iv_margin is not None:
            parts.append(f"Intrinsic value margin {iv_margin:.1%}.")
        if ptb is not None:
            parts.append(f"P/B ratio {ptb:.2f}.")
        parts.append(f"Grade: {grade} (score {score:.1f}/10).")
        summary = " ".join(parts)

        return MarginOfSafetyResult(
            earnings_yield=earn_yield,
            book_value_per_share=bv_per_share,
            price_to_book=ptb,
            book_value_discount=bv_discount,
            intrinsic_value_estimate=iv,
            market_cap=mcap,
            intrinsic_margin=iv_margin,
            tangible_bv=tbv,
            liquidation_value=liq_val,
            net_current_asset_value=ncav,
            ncav_per_share=ncav_ps,
            safety_score=score,
            safety_grade=grade,
            summary=summary,
        )

    def earnings_quality_analysis(self, data: FinancialData) -> EarningsQualityResult:
        """Analyze earnings quality and sustainability."""
        ni = data.net_income
        ocf = data.operating_cash_flow
        ta = data.total_assets
        rev = data.revenue
        ebitda = data.ebitda
        dep = data.depreciation
        capex = data.capex

        # --- Accruals Ratio: (NI - OCF) / TA ---
        accruals = None
        if ni is not None and ocf is not None and ta and ta > 0:
            accruals = safe_divide(ni - ocf, ta)

        # --- Cash to Income: OCF / NI ---
        cash_to_ni = None
        if ocf is not None and ni is not None and ni > 0:
            cash_to_ni = safe_divide(ocf, ni)

        # --- Earnings Persistence (net margin proxy): NI / Revenue ---
        persist = None
        if ni is not None and rev and rev > 0:
            persist = safe_divide(ni, rev)

        # --- Cash Return on Assets: OCF / TA ---
        cash_roa = None
        if ocf is not None and ta and ta > 0:
            cash_roa = safe_divide(ocf, ta)

        # --- Operating Cash Quality: OCF / EBITDA ---
        ocf_ebitda = None
        if ocf is not None and ebitda and ebitda > 0:
            ocf_ebitda = safe_divide(ocf, ebitda)

        # --- Revenue Cash Ratio: OCF / Revenue ---
        rev_cash = None
        if ocf is not None and rev and rev > 0:
            rev_cash = safe_divide(ocf, rev)

        # --- Depreciation to Capex ---
        dep_capex = None
        if dep is not None and capex and capex > 0:
            dep_capex = safe_divide(dep, capex)

        # --- Capex to Revenue ---
        capex_rev = None
        if capex is not None and rev and rev > 0:
            capex_rev = safe_divide(capex, rev)

        # --- Reinvestment Rate: Capex / Depreciation ---
        reinvest = None
        if capex is not None and dep and dep > 0:
            reinvest = safe_divide(capex, dep)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # Cash-to-income quality
        if cash_to_ni is not None:
            if cash_to_ni >= 1.2:
                score += 2.0  # Cash earnings exceed accrual earnings
            elif cash_to_ni >= 1.0:
                score += 1.0
            elif cash_to_ni >= 0.5:
                score += 0.0
            elif cash_to_ni < 0.5:
                score -= 1.5  # Cash generation much lower than reported NI

        # Accruals quality (lower is better - less accrual manipulation)
        if accruals is not None:
            if accruals <= -0.05:
                score += 1.0  # OCF exceeds NI (conservative accounting)
            elif accruals <= 0.05:
                score += 0.5  # Low accruals
            elif accruals > 0.15:
                score -= 1.5  # High accruals = potential manipulation
            elif accruals > 0.10:
                score -= 0.5

        # Operating cash quality
        if ocf_ebitda is not None:
            if ocf_ebitda >= 0.80:
                score += 0.5  # Strong cash conversion
            elif ocf_ebitda < 0.40:
                score -= 0.5  # Weak conversion

        score = max(0.0, min(10.0, score))

        # --- Grade ---
        if score >= 8.0:
            grade = "High"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Questionable"
        else:
            grade = "Poor"

        # --- Summary ---
        parts = ["Earnings Quality:"]
        if cash_to_ni is not None:
            parts.append(f"Cash/Income ratio {cash_to_ni:.2f}x.")
        if accruals is not None:
            parts.append(f"Accruals ratio {accruals:.2%}.")
        if ocf_ebitda is not None:
            parts.append(f"OCF/EBITDA {ocf_ebitda:.1%}.")
        parts.append(f"Grade: {grade} (score {score:.1f}/10).")
        summary = " ".join(parts)

        return EarningsQualityResult(
            accruals_ratio=accruals,
            cash_to_income=cash_to_ni,
            earnings_persistence=persist,
            cash_return_on_assets=cash_roa,
            accrual_return_on_assets=accruals,
            operating_cash_quality=ocf_ebitda,
            revenue_cash_ratio=rev_cash,
            depreciation_to_capex=dep_capex,
            capex_to_revenue=capex_rev,
            reinvestment_rate=reinvest,
            earnings_quality_score=score,
            earnings_quality_grade=grade,
            summary=summary,
        )

    def financial_flexibility_analysis(self, data: FinancialData) -> FinancialFlexibilityResult:
        """Analyze financial flexibility and adaptive capacity."""
        cash = data.cash
        ta = data.total_assets
        tl = data.total_liabilities
        debt = data.total_debt
        rev = data.revenue
        ocf = data.operating_cash_flow
        capex = data.capex
        ebitda = data.ebitda
        equity = data.total_equity
        re = data.retained_earnings
        ca = data.current_assets
        cl = data.current_liabilities

        # Cash fallback
        if cash is None and ca is not None and cl is not None:
            cash = ca - cl

        # FCF
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - capex

        # --- Cash / Assets ---
        cash_ta = safe_divide(cash, ta) if cash is not None and ta and ta > 0 else None

        # --- Cash / Debt ---
        cash_debt = safe_divide(cash, debt) if cash is not None and debt and debt > 0 else None

        # --- Cash / Revenue ---
        cash_rev = safe_divide(cash, rev) if cash is not None and rev and rev > 0 else None

        # --- FCF / Revenue ---
        fcf_rev = safe_divide(fcf, rev) if fcf is not None and rev and rev > 0 else None

        # --- Spare borrowing capacity: (TA * 0.50 - Debt) / TA ---
        spare_cap = None
        if ta and ta > 0 and debt is not None:
            spare_cap = safe_divide(ta * 0.50 - debt, ta)

        # --- Unencumbered assets: TA - TL ---
        unencumbered = None
        if ta is not None and tl is not None:
            unencumbered = ta - tl

        # --- Financial slack: (Cash + FCF) / Debt ---
        slack = None
        if cash is not None and fcf is not None and debt and debt > 0:
            slack = safe_divide(cash + fcf, debt)

        # --- Debt headroom: 3x EBITDA - current debt ---
        headroom = None
        if ebitda is not None and ebitda > 0 and debt is not None:
            headroom = ebitda * 3.0 - debt

        # --- Retained earnings ratio: RE / Equity ---
        re_ratio = None
        if re is not None and equity and equity > 0:
            re_ratio = safe_divide(re, equity)

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # Cash-to-assets
        if cash_ta is not None:
            if cash_ta >= 0.20:
                score += 1.5  # Substantial cash buffer
            elif cash_ta >= 0.10:
                score += 0.5
            elif cash_ta < 0.03:
                score -= 1.0  # Very thin cash

        # FCF margin
        if fcf_rev is not None:
            if fcf_rev >= 0.15:
                score += 1.5  # Strong free cash generation
            elif fcf_rev >= 0.05:
                score += 0.5
            elif fcf_rev < 0:
                score -= 1.0  # Negative FCF

        # Debt headroom
        if headroom is not None:
            if headroom > 0:
                score += 1.0  # Room to borrow more
            elif headroom < 0:
                score -= 1.0  # Over-leveraged vs 3x EBITDA

        # Financial slack
        if slack is not None:
            if slack >= 1.0:
                score += 0.5  # Can cover debt with cash+FCF
            elif slack < 0.2:
                score -= 0.5

        score = max(0.0, min(10.0, score))

        # --- Grade ---
        if score >= 8.0:
            grade = "Highly Flexible"
        elif score >= 6.0:
            grade = "Flexible"
        elif score >= 4.0:
            grade = "Constrained"
        else:
            grade = "Rigid"

        # --- Summary ---
        parts = ["Financial Flexibility:"]
        if cash_ta is not None:
            parts.append(f"Cash/Assets {cash_ta:.1%}.")
        if fcf_rev is not None:
            parts.append(f"FCF margin {fcf_rev:.1%}.")
        if headroom is not None:
            parts.append(f"Debt headroom ${headroom:,.0f}.")
        parts.append(f"Grade: {grade} (score {score:.1f}/10).")
        summary = " ".join(parts)

        return FinancialFlexibilityResult(
            cash_to_assets=cash_ta,
            cash_to_debt=cash_debt,
            cash_to_revenue=cash_rev,
            fcf_to_revenue=fcf_rev,
            fcf_margin=fcf_rev,
            spare_borrowing_capacity=spare_cap,
            unencumbered_assets=unencumbered,
            financial_slack=slack,
            debt_headroom=headroom,
            retained_earnings_ratio=re_ratio,
            flexibility_score=score,
            flexibility_grade=grade,
            summary=summary,
        )

    def altman_z_score_analysis(self, data: FinancialData) -> AltmanZScoreResult:
        """Altman Z-Score bankruptcy prediction model (original manufacturing formula)."""
        ca = data.current_assets
        cl = data.current_liabilities
        ta = data.total_assets
        re = data.retained_earnings
        ebit = data.ebit
        eq = data.total_equity
        tl = data.total_liabilities
        rev = data.revenue

        # Component ratios
        wc_ta = None
        if ca is not None and cl is not None and ta and ta > 0:
            wc_ta = (ca - cl) / ta

        re_ta = safe_divide(re, ta)
        ebit_ta = safe_divide(ebit, ta)
        eq_tl = safe_divide(eq, tl)
        rev_ta = safe_divide(rev, ta)

        # Weighted components
        x1 = 1.2 * wc_ta if wc_ta is not None else None
        x2 = 1.4 * re_ta if re_ta is not None else None
        x3 = 3.3 * ebit_ta if ebit_ta is not None else None
        x4 = 0.6 * eq_tl if eq_tl is not None else None
        x5 = 1.0 * rev_ta if rev_ta is not None else None

        # Z-Score (only if all components available)
        z = None
        zone = ""
        if all(v is not None for v in [x1, x2, x3, x4, x5]):
            z = x1 + x2 + x3 + x4 + x5
            if z > 2.99:
                zone = "Safe"
            elif z >= 1.81:
                zone = "Gray"
            else:
                zone = "Distress"

        # --- Scoring (base 5.0) ---
        score = 5.0

        if z is not None:
            if z > 3.0:
                score += 2.5
            elif z > 2.99:
                score += 2.0
            elif z >= 2.5:
                score += 1.0
            elif z >= 1.81:
                score += 0.0
            elif z >= 1.0:
                score -= 1.5
            else:
                score -= 2.5

        # Bonus/penalty for individual components
        if wc_ta is not None:
            if wc_ta > 0.20:
                score += 0.5
            elif wc_ta < 0:
                score -= 0.5

        if ebit_ta is not None:
            if ebit_ta > 0.15:
                score += 0.5
            elif ebit_ta < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            grade = "Strong"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Watch"
        else:
            grade = "Critical"

        parts = [f"Altman Z-Score — Grade: {grade} ({score:.1f}/10)."]
        if z is not None:
            parts.append(f"Z-Score: {z:.2f} ({zone})")
        summary = " | ".join(parts)

        return AltmanZScoreResult(
            working_capital_to_assets=wc_ta,
            retained_earnings_to_assets=re_ta,
            ebit_to_assets=ebit_ta,
            equity_to_liabilities=eq_tl,
            revenue_to_assets=rev_ta,
            x1_weighted=x1,
            x2_weighted=x2,
            x3_weighted=x3,
            x4_weighted=x4,
            x5_weighted=x5,
            z_score=z,
            z_zone=zone,
            altman_score=score,
            altman_grade=grade,
            summary=summary,
        )

    def piotroski_f_score_analysis(self, data: FinancialData) -> PiotroskiFScoreResult:
        """Piotroski F-Score (adapted for single-period: 7 signals instead of 9)."""
        ni = data.net_income
        ta = data.total_assets
        ocf = data.operating_cash_flow
        ca = data.current_assets
        cl = data.current_liabilities
        debt = data.total_debt
        rev = data.revenue
        gp = data.gross_profit

        # Profitability signals
        roa = safe_divide(ni, ta)
        roa_pos = roa > 0 if roa is not None else None
        ocf_pos = ocf > 0 if ocf is not None else None
        accruals_neg = None
        if ocf is not None and ni is not None:
            accruals_neg = ocf > ni  # Cash earnings exceed accrual earnings

        # Leverage / Liquidity signals
        cr = safe_divide(ca, cl)
        cr_above_1 = cr > 1.0 if cr is not None else None
        dta = safe_divide(debt, ta)
        low_lev = dta < 0.5 if dta is not None else None

        # Efficiency signals
        gm = safe_divide(gp, rev)
        gm_healthy = gm > 0.20 if gm is not None else None
        at = safe_divide(rev, ta)
        at_adequate = at > 0.5 if at is not None else None

        # Count score (each True signal = +1)
        signals = [roa_pos, ocf_pos, accruals_neg, cr_above_1, low_lev, gm_healthy, at_adequate]
        f_score = sum(1 for s in signals if s is True)
        testable = sum(1 for s in signals if s is not None)

        if f_score >= 6:
            grade = "Strong Value"
        elif f_score >= 4:
            grade = "Moderate Value"
        elif f_score >= 2:
            grade = "Weak"
        else:
            grade = "Avoid"

        parts = [f"Piotroski F-Score — {grade} ({f_score}/{testable} signals)."]
        if roa is not None:
            parts.append(f"ROA: {roa:.1%}")
        if cr is not None:
            parts.append(f"Current Ratio: {cr:.2f}x")
        summary = " | ".join(parts)

        return PiotroskiFScoreResult(
            roa_positive=roa_pos,
            ocf_positive=ocf_pos,
            accruals_negative=accruals_neg,
            current_ratio_above_1=cr_above_1,
            low_leverage=low_lev,
            gross_margin_healthy=gm_healthy,
            asset_turnover_adequate=at_adequate,
            roa=roa,
            current_ratio=cr,
            debt_to_assets=dta,
            gross_margin=gm,
            asset_turnover=at,
            f_score=f_score,
            f_score_max=testable if testable > 0 else 7,
            piotroski_grade=grade,
            summary=summary,
        )

    def interest_coverage_analysis(self, data: FinancialData) -> InterestCoverageResult:
        """Comprehensive interest coverage and debt capacity analysis."""
        ebit = data.ebit
        ebitda = data.ebitda
        interest = data.interest_expense
        debt = data.total_debt
        ocf = data.operating_cash_flow
        capex = data.capex
        rev = data.revenue
        eq = data.total_equity

        ebit_cov = safe_divide(ebit, interest)
        ebitda_cov = safe_divide(ebitda, interest)
        d_ebitda = safe_divide(debt, ebitda)
        ocf_debt = safe_divide(ocf, debt)

        fcf_debt = None
        if ocf is not None and capex is not None and debt and debt > 0:
            fcf_debt = (ocf - capex) / debt

        # Fixed charge coverage: approximate (no lease data, so FCCR ≈ EBIT coverage)
        fcc = ebit_cov  # Simplification when lease data unavailable

        max_cap = None
        spare_cap = None
        if ebitda is not None:
            max_cap = 3.0 * ebitda
            if debt is not None:
                spare_cap = max_cap - debt

        int_rev = safe_divide(interest, rev)
        d_eq = safe_divide(debt, eq)

        # --- Scoring (base 5.0) ---
        score = 5.0

        if ebit_cov is not None:
            if ebit_cov >= 5.0:
                score += 2.0
            elif ebit_cov >= 3.0:
                score += 1.0
            elif ebit_cov >= 1.5:
                score += 0.0
            elif ebit_cov >= 1.0:
                score -= 1.0
            else:
                score -= 2.0

        if d_ebitda is not None:
            if d_ebitda <= 1.5:
                score += 1.0
            elif d_ebitda <= 3.0:
                score += 0.5
            elif d_ebitda > 5.0:
                score -= 1.0
            elif d_ebitda > 4.0:
                score -= 0.5

        if ocf_debt is not None:
            if ocf_debt >= 0.30:
                score += 0.5
            elif ocf_debt < 0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Strained"
        else:
            grade = "Critical"

        parts = [f"Interest Coverage — Grade: {grade} ({score:.1f}/10)."]
        if ebit_cov is not None:
            parts.append(f"EBIT Coverage: {ebit_cov:.1f}x")
        if d_ebitda is not None:
            parts.append(f"Debt/EBITDA: {d_ebitda:.1f}x")
        summary = " | ".join(parts)

        return InterestCoverageResult(
            ebit_coverage=ebit_cov,
            ebitda_coverage=ebitda_cov,
            debt_to_ebitda=d_ebitda,
            ocf_to_debt=ocf_debt,
            fcf_to_debt=fcf_debt,
            fixed_charge_coverage=fcc,
            max_debt_capacity=max_cap,
            spare_debt_capacity=spare_cap,
            interest_to_revenue=int_rev,
            debt_to_equity=d_eq,
            coverage_score=score,
            coverage_grade=grade,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Phase 37: WACC & Cost of Capital
    # ------------------------------------------------------------------
    def wacc_analysis(self, data: FinancialData) -> WACCResult:
        """Estimate WACC from financial statement data.

        Cost of debt = Interest Expense / Total Debt.
        Cost of equity = implied from ROE (capped/floored for realism).
        WACC = Wd * Rd * (1-T) + We * Re.
        """
        result = WACCResult()

        debt = data.total_debt
        equity = data.total_equity

        # Capital structure weights
        if debt is not None and equity is not None and (debt + equity) > 0:
            total_cap = debt + equity
            result.total_capital = total_cap
            result.debt_weight = debt / total_cap
            result.equity_weight = equity / total_cap
            result.debt_to_total_capital = debt / total_cap
            result.equity_to_total_capital = equity / total_cap
        else:
            # Without both debt and equity, cannot compute WACC
            result.wacc_score = 5.0
            result.wacc_grade = "Fair"
            result.summary = "WACC: Insufficient data to estimate cost of capital."
            return result

        # Cost of debt
        cost_of_debt = safe_divide(data.interest_expense, debt)
        result.cost_of_debt = cost_of_debt

        # Effective tax rate (proxy: 1 - NI/EBT, where EBT = EBIT - Interest)
        eff_tax = None
        ebit = data.ebit
        ie = data.interest_expense
        ni = data.net_income
        if ebit is not None and ie is not None and ni is not None:
            ebt = ebit - ie
            if ebt > 0:
                eff_tax = 1.0 - (ni / ebt)
                eff_tax = max(0.0, min(eff_tax, 0.60))  # Clamp 0-60%
        if eff_tax is None:
            eff_tax = 0.25  # Default assumption
        result.effective_tax_rate = eff_tax

        # After-tax cost of debt
        if cost_of_debt is not None:
            result.after_tax_cost_of_debt = cost_of_debt * (1 - eff_tax)

        # Implied cost of equity from ROE (floored at 8%, capped at 30%)
        roe = safe_divide(ni, equity)
        if roe is not None:
            implied_re = max(0.08, min(abs(roe) + 0.03, 0.30))
        else:
            implied_re = 0.12  # Default
        result.implied_cost_of_equity = implied_re

        # WACC calculation
        wd = result.debt_weight
        we = result.equity_weight
        if result.after_tax_cost_of_debt is not None:
            wacc = wd * result.after_tax_cost_of_debt + we * implied_re
        elif cost_of_debt is not None:
            wacc = wd * cost_of_debt * (1 - eff_tax) + we * implied_re
        else:
            # No cost of debt info; use equity cost only weighted by equity
            wacc = we * implied_re
        result.wacc = wacc

        # Scoring (base 5.0)
        score = 5.0

        # Lower WACC is better
        if wacc is not None:
            if wacc < 0.08:
                score += 2.0
            elif wacc < 0.10:
                score += 1.5
            elif wacc < 0.12:
                score += 0.5
            elif wacc > 0.20:
                score -= 2.0
            elif wacc > 0.15:
                score -= 1.0

        # Capital structure balance
        if wd is not None:
            if 0.20 <= wd <= 0.50:
                score += 0.5  # Balanced mix
            elif wd > 0.70:
                score -= 1.0  # Over-levered
            elif wd < 0.05 and debt > 0:
                score -= 0.5  # Under-utilizing tax shield

        # Cost of debt efficiency
        if cost_of_debt is not None:
            if cost_of_debt < 0.04:
                score += 0.5
            elif cost_of_debt > 0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.wacc_score = score

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Fair"
        else:
            grade = "Expensive"
        result.wacc_grade = grade

        wacc_pct = f"{wacc * 100:.1f}%" if wacc is not None else "N/A"
        result.summary = (
            f"WACC Analysis: Estimated WACC of {wacc_pct}. "
            f"Capital structure: {wd * 100:.0f}% debt / {we * 100:.0f}% equity. "
            f"Grade: {grade}."
        )

        return result

    # ------------------------------------------------------------------
    # Phase 38: Economic Value Added (EVA)
    # ------------------------------------------------------------------
    def eva_analysis(self, data: FinancialData) -> EVAResult:
        """Compute Economic Value Added = NOPAT - (Invested Capital * WACC).

        NOPAT = EBIT * (1 - Tax Rate).
        Invested Capital = Total Assets - Current Liabilities.
        Uses WACC from wacc_analysis().
        """
        result = EVAResult()

        ebit = data.ebit
        ta = data.total_assets
        cl = data.current_liabilities

        # Invested Capital = Total Assets - Current Liabilities
        if ta is not None and cl is not None and ta > 0:
            invested_cap = ta - cl
        elif data.total_debt is not None and data.total_equity is not None:
            invested_cap = data.total_debt + data.total_equity
        else:
            result.eva_score = 5.0
            result.eva_grade = "Marginal"
            result.summary = "EVA: Insufficient data to compute invested capital."
            return result

        result.invested_capital = invested_cap

        # Effective tax rate (same logic as WACC)
        eff_tax = None
        ie = data.interest_expense
        ni = data.net_income
        if ebit is not None and ie is not None and ni is not None:
            ebt = ebit - ie
            if ebt > 0:
                eff_tax = 1.0 - (ni / ebt)
                eff_tax = max(0.0, min(eff_tax, 0.60))
        if eff_tax is None:
            eff_tax = 0.25

        # NOPAT
        if ebit is not None:
            nopat = ebit * (1 - eff_tax)
            result.nopat = nopat
        else:
            result.eva_score = 5.0
            result.eva_grade = "Marginal"
            result.summary = "EVA: Insufficient data (no EBIT) to compute NOPAT."
            return result

        # Get WACC
        wacc_result = self.wacc_analysis(data)
        wacc = wacc_result.wacc
        if wacc is None:
            wacc = 0.10  # Default fallback
        result.wacc_used = wacc

        # Capital charge & EVA
        capital_charge = invested_cap * wacc
        result.capital_charge = capital_charge
        eva = nopat - capital_charge
        result.eva = eva

        # EVA margin
        if data.revenue and data.revenue > 0:
            result.eva_margin = eva / data.revenue

        # ROIC
        roic = safe_divide(nopat, invested_cap)
        result.roic = roic

        # ROIC-WACC spread
        if roic is not None:
            result.roic_wacc_spread = roic - wacc

        # Scoring (base 5.0)
        score = 5.0

        # EVA positive/negative
        if eva > 0:
            score += 2.0
        elif eva < 0:
            score -= 2.0

        # ROIC-WACC spread
        spread = result.roic_wacc_spread
        if spread is not None:
            if spread > 0.10:
                score += 1.0
            elif spread > 0.05:
                score += 0.5
            elif spread < -0.05:
                score -= 1.0
            elif spread < 0:
                score -= 0.5

        # EVA margin
        if result.eva_margin is not None:
            if result.eva_margin > 0.10:
                score += 0.5
            elif result.eva_margin < -0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.eva_score = score

        if score >= 8.0:
            grade = "Value Creator"
        elif score >= 6.0:
            grade = "Adequate"
        elif score >= 4.0:
            grade = "Marginal"
        else:
            grade = "Value Destroyer"
        result.eva_grade = grade

        eva_str = f"${eva:,.0f}" if eva is not None else "N/A"
        result.summary = (
            f"EVA Analysis: Economic Value Added of {eva_str}. "
            f"ROIC: {roic * 100:.1f}% vs WACC: {wacc * 100:.1f}%. "
            f"Grade: {grade}."
        )

        return result

    # ------------------------------------------------------------------
    # Phase 39: Free Cash Flow Yield
    # ------------------------------------------------------------------
    def fcf_yield_analysis(self, data: FinancialData) -> FCFYieldResult:
        """Compute Free Cash Flow yield and quality metrics.

        FCF = Operating Cash Flow - CapEx.
        """
        result = FCFYieldResult()

        ocf = data.operating_cash_flow
        capex = data.capex or 0

        if ocf is None:
            result.fcf_score = 5.0
            result.fcf_grade = "Weak"
            result.summary = "FCF Yield: No operating cash flow data available."
            return result

        fcf = ocf - capex
        result.fcf = fcf

        # FCF margin
        result.fcf_margin = safe_divide(fcf, data.revenue)

        # FCF conversion (vs net income)
        result.fcf_to_net_income = safe_divide(fcf, data.net_income)

        # FCF yield on invested capital
        ta = data.total_assets
        cl = data.current_liabilities
        if ta is not None and cl is not None and (ta - cl) > 0:
            result.fcf_yield_on_capital = fcf / (ta - cl)
        elif data.total_debt is not None and data.total_equity is not None:
            total_cap = data.total_debt + data.total_equity
            if total_cap > 0:
                result.fcf_yield_on_capital = fcf / total_cap

        # FCF yield on equity
        result.fcf_yield_on_equity = safe_divide(fcf, data.total_equity)

        # FCF to debt
        result.fcf_to_debt = safe_divide(fcf, data.total_debt)

        # CapEx ratios
        if ocf != 0:
            result.capex_to_ocf = capex / ocf
        result.capex_to_revenue = safe_divide(capex, data.revenue)

        # Scoring (base 5.0)
        score = 5.0

        # FCF positive/negative
        if fcf > 0:
            score += 1.5
        elif fcf < 0:
            score -= 2.0

        # FCF margin
        fm = result.fcf_margin
        if fm is not None:
            if fm > 0.15:
                score += 1.0
            elif fm > 0.08:
                score += 0.5
            elif fm < 0:
                score -= 1.0

        # FCF conversion quality
        conv = result.fcf_to_net_income
        if conv is not None and data.net_income is not None and data.net_income > 0:
            if conv > 1.0:
                score += 0.5  # FCF exceeds NI (high quality)
            elif conv < 0.5:
                score -= 0.5  # Poor conversion

        # Capital intensity
        capex_ratio = result.capex_to_ocf
        if capex_ratio is not None and capex_ratio >= 0:
            if capex_ratio < 0.30:
                score += 0.5  # Low reinvestment needs
            elif capex_ratio > 0.70:
                score -= 0.5  # Heavy capex burden

        score = max(0.0, min(10.0, score))
        result.fcf_score = score

        if score >= 8.0:
            grade = "Strong"
        elif score >= 6.0:
            grade = "Healthy"
        elif score >= 4.0:
            grade = "Weak"
        else:
            grade = "Negative"
        result.fcf_grade = grade

        fcf_str = f"${fcf:,.0f}" if fcf is not None else "N/A"
        margin_str = f"{fm * 100:.1f}%" if fm is not None else "N/A"
        result.summary = (
            f"FCF Yield: Free cash flow of {fcf_str} ({margin_str} margin). "
            f"Grade: {grade}."
        )

        return result

    def cash_conversion_analysis(self, data: FinancialData) -> CashConversionResult:
        """Phase 41: Cash Conversion Efficiency.

        Computes DSO, DIO, DPO, Cash Conversion Cycle (CCC),
        cash/revenue ratio, OCF/revenue, and OCF/EBITDA.
        """
        result = CashConversionResult()

        revenue = data.revenue
        cogs = data.cogs

        # --- DSO: AR / Revenue * 365 ---
        ar = data.accounts_receivable
        dso = None
        if ar is not None and revenue and revenue > 0:
            dso = (ar / revenue) * 365
            result.dso = dso

        # --- DIO: Inventory / COGS * 365 ---
        inv = data.inventory
        dio = None
        if inv is not None and cogs and cogs > 0:
            dio = (inv / cogs) * 365
            result.dio = dio

        # --- DPO: AP / COGS * 365 ---
        ap = data.accounts_payable
        dpo = None
        if ap is not None and cogs and cogs > 0:
            dpo = (ap / cogs) * 365
            result.dpo = dpo

        # --- CCC: DSO + DIO - DPO ---
        if dso is not None and dio is not None and dpo is not None:
            ccc = dso + dio - dpo
            result.ccc = ccc
        elif dso is not None and dio is not None:
            ccc = dso + dio
            result.ccc = ccc
        else:
            ccc = None

        # --- Cash / Revenue ---
        cash = data.cash
        if cash is not None and revenue and revenue > 0:
            result.cash_to_revenue = cash / revenue

        # --- OCF / Revenue ---
        ocf = data.operating_cash_flow
        if ocf is not None and revenue and revenue > 0:
            result.ocf_to_revenue = ocf / revenue

        # --- OCF / EBITDA ---
        ebitda = data.ebitda
        if ocf is not None and ebitda and ebitda > 0:
            result.ocf_to_ebitda = ocf / ebitda

        # --- Scoring (start at 5.0) ---
        score = 5.0

        # CCC scoring: lower is better
        if ccc is not None:
            if ccc < 30:
                score += 2.0
            elif ccc < 60:
                score += 1.0
            elif ccc < 90:
                score += 0.0
            elif ccc < 120:
                score -= 1.0
            else:
                score -= 2.0

        # OCF/Revenue
        ocf_rev = result.ocf_to_revenue
        if ocf_rev is not None:
            if ocf_rev >= 0.20:
                score += 1.0
            elif ocf_rev >= 0.10:
                score += 0.5
            elif ocf_rev < 0:
                score -= 1.0

        # OCF/EBITDA quality
        ocf_ebitda = result.ocf_to_ebitda
        if ocf_ebitda is not None:
            if ocf_ebitda >= 0.80:
                score += 0.5
            elif ocf_ebitda < 0.50:
                score -= 0.5

        # DSO efficiency
        if dso is not None:
            if dso < 30:
                score += 0.5
            elif dso > 90:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cash_conversion_score = score

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Fair"
        else:
            grade = "Poor"
        result.cash_conversion_grade = grade

        ccc_str = f"{ccc:.0f} days" if ccc is not None else "N/A"
        result.summary = (
            f"Cash Conversion: CCC of {ccc_str}. "
            f"Grade: {grade}."
        )

        return result

    def beneish_m_score_analysis(self, data: FinancialData) -> BeneishMScoreResult:
        """Phase 43: Beneish M-Score (Earnings Manipulation Detection).

        Single-period simplified M-Score using available data.
        Without prior-period data, indices default to 1.0 (neutral).
        TATA = (NI - OCF) / TA is the most reliable single-period indicator.

        M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

        M > -1.78: likely manipulator
        M <= -1.78: unlikely manipulator
        """
        result = BeneishMScoreResult()

        revenue = data.revenue
        ta = data.total_assets
        ni = data.net_income
        ocf = data.operating_cash_flow
        gp = data.gross_profit
        dep = data.depreciation
        tl = data.total_liabilities

        if not revenue or revenue <= 0 or not ta or ta <= 0:
            result.summary = "Insufficient data for Beneish M-Score."
            return result

        # --- DSRI: Days Sales in Receivables Index ---
        # Single-period proxy: AR/Revenue ratio vs benchmark (0.15)
        ar = data.accounts_receivable
        if ar is not None:
            ar_ratio = ar / revenue
            dsri = ar_ratio / 0.15 if ar_ratio > 0 else 1.0
        else:
            dsri = 1.0
        result.dsri = dsri

        # --- GMI: Gross Margin Index ---
        # Single-period proxy: (1 - GM) where GM = GP/Revenue
        if gp is not None:
            gm = gp / revenue
            gmi = (1.0 - gm) / 0.60 if gm < 1.0 else 1.0  # benchmark 40% GM
        else:
            gmi = 1.0
        result.gmi = gmi

        # --- AQI: Asset Quality Index ---
        # Proxy: (1 - (CA + PP&E)/TA); without PP&E use CA/TA
        ca = data.current_assets
        if ca is not None:
            hard_assets_ratio = ca / ta
            aqi = 1.0 + (1.0 - hard_assets_ratio) * 0.5
        else:
            aqi = 1.0
        result.aqi = aqi

        # --- SGI: Sales Growth Index ---
        # Without prior revenue, default to 1.0
        sgi = 1.0
        result.sgi = sgi

        # --- DEPI: Depreciation Index ---
        if dep is not None and ta > 0:
            dep_rate = dep / ta
            depi = 0.05 / dep_rate if dep_rate > 0 else 1.0
        else:
            depi = 1.0
        result.depi = depi

        # --- SGAI: SGA Expense Index ---
        opex = data.operating_expenses
        if opex is not None and revenue > 0:
            sga_ratio = opex / revenue
            sgai = sga_ratio / 0.20 if sga_ratio > 0 else 1.0
        else:
            sgai = 1.0
        result.sgai = sgai

        # --- LVGI: Leverage Index ---
        if tl is not None and ta > 0:
            leverage = tl / ta
            lvgi = leverage / 0.40 if leverage > 0 else 1.0
        else:
            lvgi = 1.0
        result.lvgi = lvgi

        # --- TATA: Total Accruals to Total Assets ---
        if ni is not None and ocf is not None and ta > 0:
            tata = (ni - ocf) / ta
        else:
            tata = 0.0
        result.tata = tata

        # --- Compute M-Score ---
        m = (-4.84
             + 0.920 * dsri
             + 0.528 * gmi
             + 0.404 * aqi
             + 0.892 * sgi
             + 0.115 * depi
             - 0.172 * sgai
             + 4.679 * tata
             - 0.327 * lvgi)
        result.m_score = m

        # --- Scoring (invert: lower M = better, score 0-10) ---
        # M < -2.22: very unlikely => 9-10
        # M < -1.78: unlikely => 7-8
        # M >= -1.78: likely => 2-4
        # M >= -1.0: highly likely => 0-2
        if m < -2.50:
            score = 10.0
        elif m < -2.22:
            score = 9.0
        elif m < -1.78:
            score = 7.0
        elif m < -1.50:
            score = 5.0
        elif m < -1.0:
            score = 3.0
        else:
            score = 1.0

        # Adjust based on TATA
        if tata <= -0.05:
            score += 0.5  # Negative accruals (good)
        elif tata > 0.05:
            score -= 0.5  # High accruals (bad)

        score = max(0.0, min(10.0, score))
        result.manipulation_score = score

        if score >= 8.0:
            grade = "Unlikely"
        elif score >= 6.0:
            grade = "Possible"
        elif score >= 4.0:
            grade = "Likely"
        else:
            grade = "Highly Likely"
        result.manipulation_grade = grade

        result.summary = (
            f"Beneish M-Score: {m:.2f} "
            f"(threshold -1.78). "
            f"Manipulation: {grade}."
        )

        return result

    def profit_retention_power_analysis(self, data: FinancialData) -> ProfitRetentionPowerResult:
        """Phase 356: Profit Retention Power Analysis.

        Measures how effectively the company retains profits.
        Primary metric: RE / Total Assets.
        """
        result = ProfitRetentionPowerResult()

        re = data.retained_earnings
        ta = data.total_assets
        te = data.total_equity
        rev = data.revenue
        ni = data.net_income
        div = data.dividends_paid

        # RE / Total Assets
        prp_ratio = safe_divide(re, ta)
        result.prp_ratio = prp_ratio

        # RE / Total Equity
        result.re_to_equity = safe_divide(re, te)

        # RE / Revenue
        result.re_to_revenue = safe_divide(re, rev)

        # Retention rate = (NI - Div) / NI
        if ni is not None and ni > 0:
            div_val = div if div is not None else 0.0
            ret_rate = (ni - div_val) / ni
            result.retention_rate = ret_rate
            # RE growth capacity = (NI - Div) / RE
            result.re_growth_capacity = safe_divide(ni - div_val, re)
        else:
            ret_rate = None

        if prp_ratio is None:
            result.prp_score = 0.0
            result.prp_grade = ""
            result.summary = "Profit Retention Power: Insufficient data."
            return result

        # Scoring
        if prp_ratio >= 0.40:
            base = 10.0
        elif prp_ratio >= 0.30:
            base = 8.5
        elif prp_ratio >= 0.20:
            base = 7.0
        elif prp_ratio >= 0.15:
            base = 5.5
        elif prp_ratio >= 0.10:
            base = 4.0
        elif prp_ratio >= 0.05:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if ret_rate is not None and ret_rate >= 0.60:
            adj += 0.5
        if re is not None and re > 0 and ta is not None and ta > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.prp_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.prp_grade = grade

        # Spread vs benchmark (0.20)
        result.prp_spread = prp_ratio - 0.20

        prp_str = f"{prp_ratio:.4f}" if prp_ratio is not None else "N/A"
        rr_str = f"{ret_rate:.4f}" if ret_rate is not None else "N/A"
        rer_str = f"{result.re_to_equity:.4f}" if result.re_to_equity is not None else "N/A"
        result.summary = (
            f"Profit Retention Power: RE/Assets={prp_str}, "
            f"Retention rate={rr_str}, "
            f"RE/Equity={rer_str}. "
            f"{'Above' if prp_ratio >= 0.20 else 'Below'} benchmark (0.20). "
            f"Status: {grade}."
        )

        return result

    def earnings_to_debt_analysis(self, data: FinancialData) -> EarningsToDebtResult:
        """Phase 353: Earnings To Debt Analysis.

        Measures how well net income can service total debt.
        Primary metric: NI / Total Debt.
        """
        result = EarningsToDebtResult()

        ni = data.net_income
        td = data.total_debt
        ie = data.interest_expense
        tl = data.total_liabilities

        # NI / Total Debt
        etd_ratio = safe_divide(ni, td)
        result.etd_ratio = etd_ratio
        result.earnings_yield_on_debt = etd_ratio

        # NI / Interest Expense
        ni_to_int = safe_divide(ni, ie)
        result.ni_to_interest = ni_to_int

        # NI / Total Liabilities
        result.ni_to_liabilities = safe_divide(ni, tl)

        # Debt payback from earnings
        result.debt_years_from_earnings = safe_divide(td, ni)

        if etd_ratio is None:
            result.etd_score = 0.0
            result.etd_grade = ""
            result.summary = "Earnings To Debt: Insufficient data."
            return result

        # Scoring
        if etd_ratio >= 0.40:
            base = 10.0
        elif etd_ratio >= 0.30:
            base = 8.5
        elif etd_ratio >= 0.20:
            base = 7.0
        elif etd_ratio >= 0.15:
            base = 5.5
        elif etd_ratio >= 0.10:
            base = 4.0
        elif etd_ratio >= 0.05:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if ni_to_int is not None and ni_to_int >= 3.0:
            adj += 0.5
        if ni is not None and ni > 0 and td is not None and td > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.etd_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.etd_grade = grade

        # Spread vs benchmark (0.20)
        result.etd_spread = etd_ratio - 0.20

        etd_str = f"{etd_ratio:.4f}" if etd_ratio is not None else "N/A"
        nti_str = f"{ni_to_int:.2f}" if ni_to_int is not None else "N/A"
        dye_str = f"{result.debt_years_from_earnings:.2f}" if result.debt_years_from_earnings is not None else "N/A"
        result.summary = (
            f"Earnings To Debt: NI/TD={etd_str}, "
            f"NI/Interest={nti_str}x, "
            f"Payback={dye_str} yrs. "
            f"{'Above' if etd_ratio >= 0.20 else 'Below'} benchmark (0.20). "
            f"Status: {grade}."
        )

        return result

    def revenue_growth_analysis(self, data: FinancialData) -> RevenueGrowthResult:
        """Phase 350: Revenue Growth Capacity Analysis.

        Measures sustainable growth capacity via ROE * plowback.
        Primary metric: Sustainable Growth Rate = ROE * (1 - Dividend Payout).
        Supporting: ROE, plowback rate, revenue/assets.

        Scoring:
            SGR >= 0.15 => base 10
            SGR >= 0.12 => base 8.5
            SGR >= 0.10 => base 7.0
            SGR >= 0.07 => base 5.5
            SGR >= 0.05 => base 4.0
            SGR >= 0.02 => base 2.5
            SGR < 0.02  => base 1.0
        Adjustments:
            ROE >= 0.12 => +0.5
            Both NI and TE > 0 => +0.5
        """
        result = RevenueGrowthResult()

        roe = safe_divide(data.net_income, data.total_equity)
        rev_asset = safe_divide(data.revenue, data.total_assets)

        # Plowback = (NI - Dividends) / NI
        plowback = None
        if data.net_income is not None and data.net_income != 0:
            div = data.dividends_paid or 0
            plowback = (data.net_income - div) / data.net_income

        result.roe = roe
        result.plowback = plowback
        result.revenue_per_asset = rev_asset

        # Sustainable growth rate = ROE * plowback
        sgr = None
        if roe is not None and plowback is not None:
            sgr = roe * plowback
        result.sustainable_growth = sgr
        result.rg_capacity = sgr

        if sgr is not None and roe is not None:
            result.rg_spread = sgr - roe

        if sgr is None:
            result.rg_score = 0.0
            result.summary = "Revenue Growth: Insufficient data for analysis."
            return result

        if sgr >= 0.15:
            base = 10.0
        elif sgr >= 0.12:
            base = 8.5
        elif sgr >= 0.10:
            base = 7.0
        elif sgr >= 0.07:
            base = 5.5
        elif sgr >= 0.05:
            base = 4.0
        elif sgr >= 0.02:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if roe is not None and roe >= 0.12:
            adj += 0.5
        if data.net_income is not None and data.net_income > 0 and data.total_equity is not None and data.total_equity > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.rg_score = round(score, 1)

        if score >= 8:
            result.rg_grade = "Excellent"
        elif score >= 6:
            result.rg_grade = "Good"
        elif score >= 4:
            result.rg_grade = "Adequate"
        else:
            result.rg_grade = "Weak"

        pb_str = f"{plowback:.4f}" if plowback is not None else "N/A"
        ra_str = f"{rev_asset:.4f}" if rev_asset is not None else "N/A"
        result.summary = (
            f"Revenue Growth: SGR={sgr:.4f}, ROE={roe:.4f}, Plowback={pb_str}, "
            f"Rev/Assets={ra_str}. "
            f"Score={result.rg_score}/10 ({result.rg_grade})."
        )

        return result

    def operating_margin_analysis(self, data: FinancialData) -> OperatingMarginResult:
        """Phase 349: Operating Margin Analysis.

        Measures core operating profitability as OI/Revenue.
        Primary metric: Operating Income / Revenue.
        Supporting: EBIT margin, EBITDA margin, margin spread.

        Scoring:
            OI/Rev >= 0.25 => base 10
            OI/Rev >= 0.20 => base 8.5
            OI/Rev >= 0.15 => base 7.0
            OI/Rev >= 0.10 => base 5.5
            OI/Rev >= 0.05 => base 4.0
            OI/Rev >= 0.02 => base 2.5
            OI/Rev < 0.02  => base 1.0
        Adjustments:
            EBITDA margin >= 0.20 => +0.5
            Both OI and Revenue > 0 => +0.5
        """
        result = OperatingMarginResult()

        oi_rev = safe_divide(data.operating_income, data.revenue)
        ebit_m = safe_divide(data.ebit, data.revenue)
        ebitda_m = safe_divide(data.ebitda, data.revenue)

        result.oi_to_revenue = oi_rev
        result.operating_margin = oi_rev
        result.ebit_margin = ebit_m
        result.ebitda_margin = ebitda_m

        if oi_rev is not None and ebitda_m is not None:
            result.opm_spread = ebitda_m - oi_rev

        if oi_rev is None:
            result.opm_score = 0.0
            result.summary = "Operating Margin: Insufficient data for analysis."
            return result

        if oi_rev >= 0.25:
            base = 10.0
        elif oi_rev >= 0.20:
            base = 8.5
        elif oi_rev >= 0.15:
            base = 7.0
        elif oi_rev >= 0.10:
            base = 5.5
        elif oi_rev >= 0.05:
            base = 4.0
        elif oi_rev >= 0.02:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if ebitda_m is not None and ebitda_m >= 0.20:
            adj += 0.5
        if data.operating_income is not None and data.operating_income > 0 and data.revenue is not None and data.revenue > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.opm_score = round(score, 1)

        if score >= 8:
            result.opm_grade = "Excellent"
        elif score >= 6:
            result.opm_grade = "Good"
        elif score >= 4:
            result.opm_grade = "Adequate"
        else:
            result.opm_grade = "Weak"

        ebit_str = f"{ebit_m:.4f}" if ebit_m is not None else "N/A"
        ebitda_str = f"{ebitda_m:.4f}" if ebitda_m is not None else "N/A"
        result.summary = (
            f"Operating Margin: OI/Revenue={oi_rev:.4f}, EBIT Margin={ebit_str}, "
            f"EBITDA Margin={ebitda_str}. "
            f"Score={result.opm_score}/10 ({result.opm_grade})."
        )

        return result

    def debt_to_equity_analysis(self, data: FinancialData) -> DebtToEquityResult:
        """Phase 348: Debt To Equity Analysis.

        Measures financial leverage via total debt relative to equity.
        Primary metric: TD/TE — lower means less reliance on debt.
        Supporting: LT debt/equity, debt/assets, equity multiplier.

        Scoring (lower is better):
            TD/TE <= 0.30 => base 10
            TD/TE <= 0.50 => base 8.5
            TD/TE <= 0.80 => base 7.0
            TD/TE <= 1.00 => base 5.5
            TD/TE <= 1.50 => base 4.0
            TD/TE <= 2.00 => base 2.5
            TD/TE > 2.00  => base 1.0
        Adjustments:
            Debt/Assets <= 0.50 => +0.5
            Both TD and TE > 0 => +0.5
        """
        result = DebtToEquityResult()

        td_te = safe_divide(data.total_debt, data.total_equity)
        d_a = safe_divide(data.total_debt, data.total_assets)

        # Equity multiplier = TA / TE
        eq_mult = safe_divide(data.total_assets, data.total_equity)

        result.td_to_te = td_te
        result.dte_ratio = td_te
        result.debt_to_assets = d_a
        result.equity_multiplier = eq_mult

        # LT debt approximation: total_debt (proxy when LT not separate)
        if data.total_debt is not None and data.total_equity is not None and data.total_equity != 0:
            result.lt_debt_to_equity = data.total_debt / data.total_equity

        if td_te is not None and d_a is not None:
            result.dte_spread = td_te - d_a

        if td_te is None:
            result.dte_score = 0.0
            result.summary = "Debt to Equity: Insufficient data for analysis."
            return result

        # Lower leverage is better
        if td_te <= 0.30:
            base = 10.0
        elif td_te <= 0.50:
            base = 8.5
        elif td_te <= 0.80:
            base = 7.0
        elif td_te <= 1.00:
            base = 5.5
        elif td_te <= 1.50:
            base = 4.0
        elif td_te <= 2.00:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if d_a is not None and d_a <= 0.50:
            adj += 0.5
        if data.total_debt is not None and data.total_debt > 0 and data.total_equity is not None and data.total_equity > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.dte_score = round(score, 1)

        if score >= 8:
            result.dte_grade = "Excellent"
        elif score >= 6:
            result.dte_grade = "Good"
        elif score >= 4:
            result.dte_grade = "Adequate"
        else:
            result.dte_grade = "Weak"

        em_str = f"{eq_mult:.2f}" if eq_mult is not None else "N/A"
        da_str = f"{d_a:.4f}" if d_a is not None else "N/A"
        result.summary = (
            f"Debt to Equity: TD/TE={td_te:.4f}, Debt/Assets={da_str}, "
            f"Equity Multiplier={em_str}. "
            f"Score={result.dte_score}/10 ({result.dte_grade})."
        )

        return result

    def cash_flow_to_debt_analysis(self, data: FinancialData) -> CashFlowToDebtResult:
        """Phase 347: Cash Flow To Debt Analysis.

        Measures ability to service debt from operating cash flows.
        Primary metric: OCF/Total Debt — higher means faster potential paydown.
        Supporting: FCF/TD, debt payback years, OCF/Interest.

        Scoring:
            OCF/TD >= 0.50 => base 10
            OCF/TD >= 0.40 => base 8.5
            OCF/TD >= 0.30 => base 7.0
            OCF/TD >= 0.20 => base 5.5
            OCF/TD >= 0.10 => base 4.0
            OCF/TD >= 0.05 => base 2.5
            OCF/TD < 0.05  => base 1.0
        Adjustments:
            OCF/Interest >= 3.0 => +0.5
            Both OCF and TD > 0 => +0.5
        """
        result = CashFlowToDebtResult()

        ocf_td = safe_divide(data.operating_cash_flow, data.total_debt)
        ocf_int = safe_divide(data.operating_cash_flow, data.interest_expense)

        # FCF = OCF - CapEx
        fcf_td = None
        if data.operating_cash_flow is not None and data.capex is not None and data.total_debt is not None and data.total_debt != 0:
            fcf = data.operating_cash_flow - (data.capex or 0)
            fcf_td = fcf / data.total_debt

        result.ocf_to_td = ocf_td
        result.fcf_to_td = fcf_td
        result.ocf_to_interest = ocf_int
        result.cf_to_debt = ocf_td

        # Debt payback years = TD / OCF
        if data.total_debt is not None and data.operating_cash_flow is not None and data.operating_cash_flow > 0:
            result.debt_payback_years = data.total_debt / data.operating_cash_flow

        if ocf_td is not None and fcf_td is not None:
            result.cf_debt_spread = ocf_td - fcf_td

        if ocf_td is None:
            result.cfd_score = 0.0
            result.summary = "Cash Flow to Debt: Insufficient data for analysis."
            return result

        if ocf_td >= 0.50:
            base = 10.0
        elif ocf_td >= 0.40:
            base = 8.5
        elif ocf_td >= 0.30:
            base = 7.0
        elif ocf_td >= 0.20:
            base = 5.5
        elif ocf_td >= 0.10:
            base = 4.0
        elif ocf_td >= 0.05:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if ocf_int is not None and ocf_int >= 3.0:
            adj += 0.5
        if data.operating_cash_flow is not None and data.operating_cash_flow > 0 and data.total_debt is not None and data.total_debt > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.cfd_score = round(score, 1)

        if result.cfd_score >= 8:
            result.cfd_grade = "Excellent"
        elif result.cfd_score >= 6:
            result.cfd_grade = "Good"
        elif result.cfd_score >= 4:
            result.cfd_grade = "Adequate"
        else:
            result.cfd_grade = "Weak"

        dpb_str = f"{result.debt_payback_years:.1f}" if result.debt_payback_years is not None else "N/A"
        result.summary = (
            f"Cash Flow to Debt: OCF/TD={ocf_td:.4f}, "
            f"Debt Payback={dpb_str} years. "
            f"Score={result.cfd_score:.1f}/10 ({result.cfd_grade})."
        )

        return result

    def net_worth_growth_analysis(self, data: FinancialData) -> NetWorthGrowthResult:
        """Phase 346: Net Worth Growth Analysis.

        Measures how well the company grows equity from internal earnings.
        Primary metric: RE/TE — higher means more self-funded growth.
        Supporting: Equity/Assets, NI/Equity (ROE proxy), plowback rate.

        Scoring:
            RE/TE >= 0.70 => base 10
            RE/TE >= 0.60 => base 8.5
            RE/TE >= 0.50 => base 7.0
            RE/TE >= 0.40 => base 5.5
            RE/TE >= 0.25 => base 4.0
            RE/TE >= 0.10 => base 2.5
            RE/TE < 0.10  => base 1.0
        Adjustments:
            Equity/Assets >= 0.40 => +0.5
            Both RE and TE > 0 => +0.5
        """
        result = NetWorthGrowthResult()

        re_te = safe_divide(data.retained_earnings, data.total_equity)
        ea = safe_divide(data.total_equity, data.total_assets)
        ni_eq = safe_divide(data.net_income, data.total_equity)

        # Plowback = (NI - Dividends) / NI
        plowback = None
        if data.net_income is not None and data.net_income != 0:
            div = data.dividends_paid or 0
            plowback = (data.net_income - div) / data.net_income

        result.re_to_equity = re_te
        result.equity_to_assets = ea
        result.ni_to_equity = ni_eq
        result.plowback_rate = plowback
        result.nw_growth_ratio = re_te

        if re_te is not None and ea is not None:
            result.nw_spread = re_te - ea

        if re_te is None:
            result.nwg_score = 0.0
            result.summary = "Net Worth Growth: Insufficient data for analysis."
            return result

        if re_te >= 0.70:
            base = 10.0
        elif re_te >= 0.60:
            base = 8.5
        elif re_te >= 0.50:
            base = 7.0
        elif re_te >= 0.40:
            base = 5.5
        elif re_te >= 0.25:
            base = 4.0
        elif re_te >= 0.10:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if ea is not None and ea >= 0.40:
            adj += 0.5
        if data.retained_earnings is not None and data.retained_earnings > 0 and data.total_equity is not None and data.total_equity > 0:
            adj += 0.5

        score = min(10.0, max(0.0, base + adj))
        result.nwg_score = round(score, 1)

        if result.nwg_score >= 8:
            result.nwg_grade = "Excellent"
        elif result.nwg_score >= 6:
            result.nwg_grade = "Good"
        elif result.nwg_score >= 4:
            result.nwg_grade = "Adequate"
        else:
            result.nwg_grade = "Weak"

        pb_str = f"{plowback:.4f}" if plowback is not None else "N/A"
        result.summary = (
            f"Net Worth Growth: RE/Equity={re_te:.4f}, Plowback Rate={pb_str}. "
            f"Score={result.nwg_score:.1f}/10 ({result.nwg_grade})."
        )

        return result

    def asset_lightness_analysis(self, data: FinancialData) -> AssetLightnessResult:
        """Phase 341: Asset Lightness Analysis.

        Lightness Ratio = CA / TA. Higher means more current/liquid assets
        vs fixed assets, indicating an asset-light business model.
        Complemented by Revenue/TA (asset turnover) for efficiency.
        """
        result = AssetLightnessResult()

        ca = data.current_assets
        ta = data.total_assets

        # Primary: CA / TA
        lightness = safe_divide(ca, ta)
        result.lightness_ratio = lightness
        result.ca_to_ta = lightness

        # Revenue to assets (asset turnover)
        result.revenue_to_assets = safe_divide(data.revenue, ta)

        # Fixed asset ratio = (TA - CA) / TA = 1 - lightness
        if lightness is not None:
            result.fixed_asset_ratio = 1.0 - lightness
        else:
            result.fixed_asset_ratio = None

        # Intangible intensity proxy: (TA - CA - Inventory) / TA
        if ca is not None and ta is not None and ta > 0:
            inv = data.inventory or 0
            tangible_current = ca - inv
            result.intangible_intensity = safe_divide(ta - tangible_current, ta)
        else:
            result.intangible_intensity = None

        # Lightness spread: lightness - 0.50 benchmark
        if lightness is not None:
            result.lightness_spread = lightness - 0.50

        # Scoring
        if lightness is None:
            result.alt_score = 0.0
            result.alt_grade = ""
            result.summary = "Asset Lightness: Insufficient data."
            return result

        # CA/TA ranges: higher = more asset-light
        if lightness >= 0.70:
            score = 10.0
        elif lightness >= 0.60:
            score = 8.5
        elif lightness >= 0.50:
            score = 7.0
        elif lightness >= 0.40:
            score = 5.5
        elif lightness >= 0.30:
            score = 4.0
        elif lightness >= 0.15:
            score = 2.5
        else:
            score = 1.0

        # Adj: revenue/TA >= 0.50 (+0.5) — efficient asset use
        rat = result.revenue_to_assets
        if rat is not None and rat >= 0.50:
            score += 0.5

        # Adj: both > 0 (+0.5)
        if lightness > 0 and ta is not None and ta > 0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.alt_score = score

        if score >= 8:
            result.alt_grade = "Excellent"
        elif score >= 6:
            result.alt_grade = "Good"
        elif score >= 4:
            result.alt_grade = "Adequate"
        else:
            result.alt_grade = "Weak"

        rat_str = f"{result.revenue_to_assets:.4f}" if result.revenue_to_assets is not None else "N/A"
        result.summary = (
            f"Asset Lightness: CA/TA={lightness:.4f}, "
            f"Revenue/TA={rat_str}, "
            f"Score={score:.1f}/10 ({result.alt_grade})."
        )
        return result

    def internal_growth_rate_analysis(self, data: FinancialData) -> InternalGrowthRateResult:
        """Phase 337: Internal Growth Rate Analysis.

        Primary: IGR = (ROA * b) / (1 - ROA * b), where b = retention ratio.
        Higher IGR means more self-funded growth capacity.
        """
        result = InternalGrowthRateResult()
        ni = getattr(data, 'net_income', None) or 0.0
        ta = getattr(data, 'total_assets', None) or 0.0
        div = getattr(data, 'dividends_paid', None) or 0.0
        te = getattr(data, 'total_equity', None) or 0.0

        if not ni and not ta:
            result.summary = "Internal Growth Rate: Insufficient data."
            return result

        roa = safe_divide(ni, ta)
        result.roa = roa
        b = safe_divide(ni - div, ni) if ni else None
        result.retention_ratio = b

        if roa is not None and b is not None:
            roa_b = roa * b
            result.roa_times_b = roa_b
            denom = 1.0 - roa_b
            if abs(denom) > 1e-9:
                result.igr = roa_b / denom
            else:
                result.igr = None
        else:
            result.roa_times_b = None
            result.igr = None

        # Sustainable growth = ROE * b
        roe = safe_divide(ni, te) if te else None
        result.sustainable_growth = (roe * b) if roe is not None and b is not None else None
        result.growth_capacity = result.igr

        igr = result.igr
        if igr is None:
            result.summary = "Internal Growth Rate: Cannot compute."
            return result

        # Scoring: higher IGR is better (as percentage)
        igr_pct = igr * 100.0  # convert to percentage
        if igr_pct >= 15.0:
            base = 10.0
        elif igr_pct >= 12.0:
            base = 8.5
        elif igr_pct >= 8.0:
            base = 7.0
        elif igr_pct >= 5.0:
            base = 5.5
        elif igr_pct >= 3.0:
            base = 4.0
        elif igr_pct >= 0.0:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if b is not None and b >= 0.70:
            adj += 0.5
        if ni > 0 and ta > 0:
            adj += 0.5

        score = min(base + adj, 10.0)
        result.igr_score = score
        if score >= 8:
            result.igr_grade = "Excellent"
        elif score >= 6:
            result.igr_grade = "Good"
        elif score >= 4:
            result.igr_grade = "Adequate"
        else:
            result.igr_grade = "Weak"

        result.summary = (
            f"Internal Growth Rate: IGR={f'{igr_pct:.2f}%' if igr is not None else 'N/A'}, "
            f"ROA={f'{roa:.4f}' if roa is not None else 'N/A'}, "
            f"Retention={f'{b:.4f}' if b is not None else 'N/A'}, "
            f"Score={result.igr_score:.1f}/10 ({result.igr_grade})."
        )
        return result

    def operating_expense_ratio_analysis(self, data: FinancialData) -> OperatingExpenseRatioResult:
        """Phase 330: Operating Expense Ratio Analysis.

        Primary: OpEx / Revenue.
        Lower is better — efficient overhead management. <0.15 excellent, >0.40 weak.
        """
        result = OperatingExpenseRatioResult()
        opex = getattr(data, 'operating_expenses', None) or 0.0
        revenue = getattr(data, 'revenue', None) or 0.0
        gp = getattr(data, 'gross_profit', None) or 0.0
        ebitda = getattr(data, 'ebitda', None) or 0.0

        if not revenue and not opex:
            result.summary = "Operating Expense Ratio: Insufficient data."
            return result

        oer = safe_divide(opex, revenue)
        result.opex_ratio = oer
        result.opex_per_revenue = oer
        result.opex_to_gross_profit = safe_divide(opex, gp) if gp else None
        result.opex_to_ebitda = safe_divide(opex, ebitda) if ebitda else None
        result.opex_coverage = safe_divide(revenue, opex) if opex else None
        result.efficiency_gap = (oer - 0.20) if oer is not None else None

        if oer is None:
            result.summary = "Operating Expense Ratio: Cannot compute."
            return result

        # Scoring: lower OpEx/Revenue = more efficient
        if oer <= 0.10:
            base = 10.0
        elif oer <= 0.15:
            base = 8.5
        elif oer <= 0.20:
            base = 7.0
        elif oer <= 0.25:
            base = 5.5
        elif oer <= 0.30:
            base = 4.0
        elif oer <= 0.40:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        opgp = result.opex_to_gross_profit
        if opgp is not None and opgp < 1.0:
            adj += 0.5
        if opex > 0 and revenue > 0:
            adj += 0.5

        score = min(base + adj, 10.0)
        result.oer_score = score
        if score >= 8:
            result.oer_grade = "Excellent"
        elif score >= 6:
            result.oer_grade = "Good"
        elif score >= 4:
            result.oer_grade = "Adequate"
        else:
            result.oer_grade = "Weak"

        result.summary = (
            f"Operating Expense Ratio: OpEx/Rev={f'{oer:.4f}' if oer is not None else 'N/A'}, "
            f"OpEx/GP={f'{opgp:.4f}' if opgp is not None else 'N/A'}, "
            f"Score={result.oer_score:.1f}/10 ({result.oer_grade})."
        )
        return result

    def noncurrent_asset_ratio_analysis(self, data: FinancialData) -> NoncurrentAssetRatioResult:
        """Phase 327: Noncurrent Asset Ratio Analysis.

        Primary: (Total Assets - Current Assets) / Total Assets.
        Measures long-term asset concentration. Moderate 0.40-0.65 is balanced.
        Very high (>0.80) means illiquid, very low (<0.20) means mostly current assets.
        """
        result = NoncurrentAssetRatioResult()
        ta = getattr(data, 'total_assets', None) or 0.0
        ca = getattr(data, 'current_assets', None) or 0.0
        te = getattr(data, 'total_equity', None) or 0.0
        td = getattr(data, 'total_debt', None) or 0.0

        if not ta:
            result.summary = "Noncurrent Asset Ratio: Insufficient data (need total_assets)."
            return result

        nca = ta - ca
        ratio = safe_divide(nca, ta)
        result.nca_ratio = ratio

        result.current_asset_ratio = safe_divide(ca, ta)
        result.nca_to_equity = safe_divide(nca, te) if te else None
        result.nca_to_debt = safe_divide(nca, td) if td else None
        result.asset_structure_spread = (ratio - 0.50) if ratio is not None else None
        result.liquidity_complement = safe_divide(ca, ta)

        if ratio is None:
            result.summary = "Noncurrent Asset Ratio: Cannot compute."
            return result

        # Scoring: moderate ratio is best (inverted U around 0.40-0.65)
        if 0.40 <= ratio <= 0.65:
            base = 10.0
        elif (0.30 <= ratio < 0.40) or (0.65 < ratio <= 0.75):
            base = 8.5
        elif (0.20 <= ratio < 0.30) or (0.75 < ratio <= 0.85):
            base = 7.0
        elif (0.10 <= ratio < 0.20) or (0.85 < ratio <= 0.90):
            base = 5.5
        elif ratio < 0.10 or ratio > 0.90:
            base = 4.0
        else:
            base = 2.5

        adj = 0.0
        nca_eq = result.nca_to_equity
        if nca_eq is not None and nca_eq <= 1.0:
            adj += 0.5
        if ta > 0 and ca > 0:
            adj += 0.5

        score = min(base + adj, 10.0)
        result.nar_score = score
        if score >= 8:
            result.nar_grade = "Excellent"
        elif score >= 6:
            result.nar_grade = "Good"
        elif score >= 4:
            result.nar_grade = "Adequate"
        else:
            result.nar_grade = "Weak"

        result.summary = (
            f"Noncurrent Asset Ratio: NCA Ratio={f'{ratio:.4f}' if ratio is not None else 'N/A'}, "
            f"NCA/Equity={f'{nca_eq:.4f}' if nca_eq is not None else 'N/A'}, "
            f"Score={result.nar_score:.1f}/10 ({result.nar_grade})."
        )
        return result

    def payout_resilience_analysis(self, data: FinancialData) -> PayoutResilienceResult:
        """Phase 317: Payout Resilience Analysis.

        Measures sustainability of dividend payments from earnings and cash flow.
        Primary metric: Div/NI payout ratio (moderate 0.20-0.50 is ideal).
        """
        result = PayoutResilienceResult()

        div = data.dividends_paid
        ni = data.net_income
        ocf = data.operating_cash_flow
        revenue = data.revenue
        ebitda = data.ebitda

        if not div or div <= 0:
            return result
        if not ni or ni <= 0:
            return result

        # Ratios
        result.div_to_ni = safe_divide(div, ni)
        result.div_to_ocf = safe_divide(div, ocf)
        result.div_to_revenue = safe_divide(div, revenue)
        result.div_to_ebitda = safe_divide(div, ebitda)

        # Primary: Div/NI
        primary = result.div_to_ni
        if primary is None:
            return result

        result.payout_ratio = primary

        # Resilience buffer = 1 - payout ratio (how much earnings retained)
        result.resilience_buffer = 1.0 - primary if primary <= 1.0 else 0.0

        # Scoring: moderate payout [0.20, 0.50] is ideal
        if 0.20 <= primary <= 0.50:
            score = 10.0
        elif (0.10 <= primary < 0.20) or (0.50 < primary <= 0.60):
            score = 8.5
        elif (0.05 <= primary < 0.10) or (0.60 < primary <= 0.70):
            score = 7.0
        elif 0.70 < primary <= 0.80:
            score = 5.5
        elif 0.80 < primary <= 0.90:
            score = 4.0
        elif 0.90 < primary <= 1.0:
            score = 2.5
        elif primary < 0.05:
            score = 5.5
        else:
            score = 1.0

        # Adjustments
        if result.div_to_ocf is not None and result.div_to_ocf <= 0.40:
            score += 0.5
        if div > 0 and ni > 0:
            score += 0.5

        result.prs_score = min(score, 10.0)

        # Grade
        if result.prs_score >= 8:
            result.prs_grade = "Excellent"
        elif result.prs_score >= 6:
            result.prs_grade = "Good"
        elif result.prs_score >= 4:
            result.prs_grade = "Adequate"
        else:
            result.prs_grade = "Weak"

        primary_str = f"{primary:.4f}" if primary is not None else "N/A"
        ocf_str = f"{result.div_to_ocf:.4f}" if result.div_to_ocf is not None else "N/A"
        result.summary = (
            f"Payout Resilience Analysis: Div/NI={primary_str}, "
            f"Div/OCF={ocf_str}, "
            f"Score={result.prs_score:.1f}/10 ({result.prs_grade})"
        )

        return result

    def debt_burden_index_analysis(self, data: FinancialData) -> DebtBurdenIndexResult:
        """Phase 314: Debt Burden Index Analysis.

        Measures total debt burden relative to earnings and assets.
        Primary metric: Debt/EBITDA (lower = less leveraged = better).
        """
        result = DebtBurdenIndexResult()

        debt = data.total_debt
        ebitda = data.ebitda
        assets = data.total_assets
        equity = data.total_equity
        revenue = data.revenue

        if not debt or debt <= 0:
            return result

        # Ratios
        result.debt_to_ebitda = safe_divide(debt, ebitda)
        result.debt_to_assets = safe_divide(debt, assets)
        result.debt_to_equity = safe_divide(debt, equity)
        result.debt_to_revenue = safe_divide(debt, revenue)

        # Primary: Debt/EBITDA
        primary = result.debt_to_ebitda
        if primary is None:
            # Fallback to Debt/Revenue
            primary = result.debt_to_revenue
            if primary is None:
                return result

        result.debt_ratio = primary

        # Burden intensity = Debt/Assets
        result.burden_intensity = result.debt_to_assets

        # Scoring: lower Debt/EBITDA is better
        if primary <= 1.0:
            score = 10.0
        elif primary <= 2.0:
            score = 8.5
        elif primary <= 3.0:
            score = 7.0
        elif primary <= 4.0:
            score = 5.5
        elif primary <= 5.0:
            score = 4.0
        elif primary <= 6.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.debt_to_assets is not None and result.debt_to_assets <= 0.40:
            score += 0.5
        if debt > 0 and ebitda is not None and ebitda > 0:
            score += 0.5

        result.dbi_score = min(score, 10.0)

        # Grade
        if result.dbi_score >= 8:
            result.dbi_grade = "Excellent"
        elif result.dbi_score >= 6:
            result.dbi_grade = "Good"
        elif result.dbi_score >= 4:
            result.dbi_grade = "Adequate"
        else:
            result.dbi_grade = "Weak"

        primary_str = f"{primary:.4f}" if primary is not None else "N/A"
        da_str = f"{result.debt_to_assets:.4f}" if result.debt_to_assets is not None else "N/A"
        result.summary = (
            f"Debt Burden Index Analysis: Debt/EBITDA={primary_str}, "
            f"Debt/Assets={da_str}, "
            f"Score={result.dbi_score:.1f}/10 ({result.dbi_grade})"
        )

        return result

    def inventory_coverage_analysis(self, data: FinancialData) -> InventoryCoverageResult:
        """Phase 309: Inventory Coverage Analysis.

        Evaluates how well inventory levels support revenue operations.
        Primary metric: Inventory/COGS (lower = faster turnover, better coverage).
        """
        result = InventoryCoverageResult()

        inv = data.inventory
        cogs = data.cogs
        rev = data.revenue
        ta = data.total_assets
        ca = data.current_assets

        if inv is None or (cogs is None and rev is None):
            return result

        if inv == 0 and cogs in (None, 0) and rev in (None, 0):
            return result

        # Ratios
        result.inventory_to_cogs = safe_divide(inv, cogs)
        result.inventory_to_revenue = safe_divide(inv, rev)
        result.inventory_to_assets = safe_divide(inv, ta)
        result.inventory_to_current_assets = safe_divide(inv, ca)

        # Inventory days = Inv / (COGS / 365)
        if cogs and cogs > 0:
            result.inventory_days = (inv / cogs) * 365
        # Buffer = Inv / (Rev / 365)
        if rev and rev > 0:
            result.inventory_buffer = (inv / rev) * 365

        # Scoring: based on inventory_to_cogs (lower is generally better — efficient)
        primary = result.inventory_to_cogs
        if primary is None:
            primary = result.inventory_to_revenue

        if primary is None:
            return result

        # Scoring — moderate inventory relative to COGS is ideal
        # Very low (<0.05) means possible stockouts; moderate (0.08-0.20) is ideal; high (>0.40) is excess
        if 0.08 <= primary <= 0.20:
            base = 10.0
        elif 0.05 <= primary < 0.08 or 0.20 < primary <= 0.25:
            base = 8.5
        elif 0.03 <= primary < 0.05 or 0.25 < primary <= 0.30:
            base = 7.0
        elif 0.02 <= primary < 0.03 or 0.30 < primary <= 0.40:
            base = 5.5
        elif primary < 0.02 or 0.40 < primary <= 0.50:
            base = 4.0
        elif 0.50 < primary <= 0.60:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Bonus: inventory days 30-90 (healthy range)
        if result.inventory_days is not None and 30 <= result.inventory_days <= 90:
            adj += 0.5
        # Bonus: inventory > 0 and cogs > 0
        if inv and inv > 0 and cogs and cogs > 0:
            adj += 0.5

        result.icv_score = min(10.0, max(0.0, base + adj))

        if result.icv_score >= 8:
            result.icv_grade = "Excellent"
        elif result.icv_score >= 6:
            result.icv_grade = "Good"
        elif result.icv_score >= 4:
            result.icv_grade = "Adequate"
        else:
            result.icv_grade = "Weak"

        primary_str = f"{primary:.4f}" if primary is not None else "N/A"
        days_str = f"{result.inventory_days:.1f}" if result.inventory_days is not None else "N/A"
        result.summary = (
            f"Inventory Coverage Analysis: Inv/COGS={primary_str}, "
            f"Days={days_str}, "
            f"Score={result.icv_score:.1f}/10 ({result.icv_grade})"
        )

        return result

    def capex_to_revenue_analysis(self, data: FinancialData) -> CapexToRevenueResult:
        """Phase 307: CapEx to Revenue Analysis.

        Measures capital expenditure relative to revenue — investment intensity.
        Moderate CapEx/Revenue indicates balanced investment without over-spending.
        """
        result = CapexToRevenueResult()

        capex = data.capex
        rev = data.revenue
        ocf = data.operating_cash_flow
        ebitda = data.ebitda
        ta = data.total_assets

        if not capex and not rev:
            return result

        # Primary: CapEx/Revenue
        cap_rev = safe_divide(capex, rev)
        result.capex_to_revenue = cap_rev

        # Secondary metrics
        result.capex_to_ocf = safe_divide(capex, ocf)
        result.capex_to_ebitda = safe_divide(capex, ebitda)
        result.capex_to_assets = safe_divide(capex, ta)

        # Investment intensity
        result.investment_intensity = cap_rev

        # CapEx yield: Revenue / CapEx
        result.capex_yield = safe_divide(rev, capex)

        # Scoring: moderate CapEx is best (5-15% of revenue)
        # Too low = underinvesting, too high = over-spending
        if cap_rev is not None:
            if 0.05 <= cap_rev <= 0.15:
                base = 10.0
            elif 0.03 <= cap_rev < 0.05 or 0.15 < cap_rev <= 0.20:
                base = 8.5
            elif 0.02 <= cap_rev < 0.03 or 0.20 < cap_rev <= 0.25:
                base = 7.0
            elif 0.01 <= cap_rev < 0.02 or 0.25 < cap_rev <= 0.30:
                base = 5.5
            elif cap_rev < 0.01 or 0.30 < cap_rev <= 0.40:
                base = 4.0
            elif 0.40 < cap_rev <= 0.50:
                base = 2.5
            else:
                base = 1.0

            adj = 0.0
            # Bonus: OCF covers CapEx (OCF > CapEx)
            if ocf is not None and capex is not None and (ocf or 0) > (capex or 0):
                adj += 0.5
            # Bonus: CapEx > 0 and Rev > 0
            if (capex or 0) > 0 and (rev or 0) > 0:
                adj += 0.5

            result.ctr_score = min(10.0, base + adj)
        else:
            result.ctr_score = 0.0

        # Grade
        if result.ctr_score >= 8:
            result.ctr_grade = "Excellent"
        elif result.ctr_score >= 6:
            result.ctr_grade = "Good"
        elif result.ctr_score >= 4:
            result.ctr_grade = "Adequate"
        else:
            result.ctr_grade = "Weak"

        result.summary = (
            f"CapEx to Revenue: CapEx/Rev={cap_rev:.2f}, "
            f"Score={result.ctr_score:.1f}/10 ({result.ctr_grade})"
            if cap_rev is not None
            else "CapEx to Revenue: Insufficient data"
        )

        return result

    def inventory_holding_cost_analysis(self, data: FinancialData) -> InventoryHoldingCostResult:
        """Phase 294: Inventory Holding Cost Analysis.

        Measures how much capital is tied up in inventory relative to
        revenue, indicating inventory management efficiency and holding costs.
        """
        result = InventoryHoldingCostResult()

        revenue = data.revenue
        inventory = data.inventory
        ca = data.current_assets
        ta = data.total_assets
        cogs = data.cogs

        if not revenue or revenue <= 0:
            return result
        if inventory is None:
            return result

        # Primary: Inventory / Revenue (lower = better)
        inv_to_rev = safe_divide(inventory, revenue)
        result.inventory_to_revenue = inv_to_rev

        # Secondary metrics
        result.inventory_to_current_assets = safe_divide(inventory, ca)
        result.inventory_to_total_assets = safe_divide(inventory, ta)
        result.inventory_days = safe_divide(inventory, cogs) * 365 if cogs and cogs > 0 else None
        result.inventory_carrying_cost = inv_to_rev
        result.inventory_intensity = safe_divide(inventory, revenue)

        if inv_to_rev is None:
            return result

        # Scoring: lower inventory/revenue = better efficiency
        if inv_to_rev <= 0.05:
            score = 10.0
        elif inv_to_rev <= 0.08:
            score = 8.5
        elif inv_to_rev <= 0.12:
            score = 7.0
        elif inv_to_rev <= 0.18:
            score = 5.5
        elif inv_to_rev <= 0.25:
            score = 4.0
        elif inv_to_rev <= 0.35:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: Inventory/CA <= 0.30 (+0.5)
        inv_ca = result.inventory_to_current_assets
        if inv_ca is not None and inv_ca <= 0.30:
            score += 0.5

        # Adjustment: Inventory > 0 and Revenue > 0 (+0.5)
        if inventory > 0 and revenue > 0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.ihc_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ihc_grade = grade

        result.summary = (
            f"Inventory Holding Cost: Inventory/Revenue={inv_to_rev:.2f}. "
            f"Score={score:.1f}/10. "
            f"Status: {grade}."
        )

        return result

    def funding_mix_balance_analysis(self, data: FinancialData) -> FundingMixBalanceResult:
        """Phase 293: Funding Mix Balance Analysis.

        Evaluates the balance between debt and equity in the company's
        capital structure, indicating financial stability and risk level.
        """
        result = FundingMixBalanceResult()

        equity = data.total_equity
        debt = data.total_debt

        if equity is None or equity <= 0:
            return result

        total_capital = equity + (debt or 0)
        if total_capital <= 0:
            return result

        # Primary: Equity / Total Capital (higher = more equity-funded)
        eq_to_cap = safe_divide(equity, total_capital)
        result.equity_to_total_capital = eq_to_cap

        # Secondary metrics
        result.debt_to_equity = safe_divide(debt, equity)
        result.debt_to_total_capital = safe_divide(debt, total_capital)
        result.equity_multiplier = safe_divide(total_capital, equity)
        result.leverage_headroom = 1.0 - (result.debt_to_total_capital or 0)
        result.funding_stability = eq_to_cap

        if eq_to_cap is None:
            return result

        # Scoring: higher equity proportion = better balance
        if eq_to_cap >= 0.80:
            score = 10.0
        elif eq_to_cap >= 0.70:
            score = 8.5
        elif eq_to_cap >= 0.60:
            score = 7.0
        elif eq_to_cap >= 0.50:
            score = 5.5
        elif eq_to_cap >= 0.40:
            score = 4.0
        elif eq_to_cap >= 0.30:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: D/E <= 0.50 (+0.5)
        de_ratio = result.debt_to_equity
        if de_ratio is not None and de_ratio <= 0.50:
            score += 0.5

        # Adjustment: Equity > 0 and Debt >= 0 (+0.5)
        if equity > 0 and debt is not None and debt >= 0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.fmb_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.fmb_grade = grade

        result.summary = (
            f"Funding Mix Balance: Equity/TotalCapital={eq_to_cap:.2f}. "
            f"Score={score:.1f}/10. "
            f"Status: {grade}."
        )

        return result

    def expense_ratio_discipline_analysis(self, data: FinancialData) -> ExpenseRatioDisciplineResult:
        """Phase 292: Expense Ratio Discipline Analysis.

        Measures how well the company controls operating expenses relative
        to revenue, indicating cost management discipline and efficiency.
        """
        result = ExpenseRatioDisciplineResult()

        revenue = data.revenue
        opex = data.operating_expenses
        cogs = data.cogs
        oi = data.operating_income

        if not revenue or revenue <= 0:
            return result

        # Primary: OpEx / Revenue (lower = better)
        opex_to_rev = safe_divide(opex, revenue)
        result.opex_to_revenue = opex_to_rev

        # Secondary metrics
        result.cogs_to_revenue = safe_divide(cogs, revenue)
        result.total_expense_ratio = safe_divide(
            (opex or 0) + (cogs or 0), revenue
        )
        result.operating_margin = safe_divide(oi, revenue)
        result.expense_efficiency = safe_divide(revenue, (opex or 0) + (cogs or 0)) if ((opex or 0) + (cogs or 0)) > 0 else None

        if opex_to_rev is None:
            return result

        # Scoring: lower OpEx/Revenue = better discipline
        if opex_to_rev <= 0.30:
            score = 10.0
        elif opex_to_rev <= 0.40:
            score = 8.5
        elif opex_to_rev <= 0.50:
            score = 7.0
        elif opex_to_rev <= 0.60:
            score = 5.5
        elif opex_to_rev <= 0.70:
            score = 4.0
        elif opex_to_rev <= 0.80:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: COGS/Revenue <= 0.60 (+0.5)
        cogs_ratio = result.cogs_to_revenue
        if cogs_ratio is not None and cogs_ratio <= 0.60:
            score += 0.5

        # Adjustment: Positive operating income (+0.5)
        if oi is not None and oi > 0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.erd_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.erd_grade = grade

        result.summary = (
            f"Expense Ratio Discipline: OpEx/Revenue={opex_to_rev:.2f}. "
            f"Score={score:.1f}/10. "
            f"Status: {grade}."
        )

        return result

    def revenue_cash_realization_analysis(self, data: FinancialData) -> RevenueCashRealizationResult:
        """Phase 291: Revenue Cash Realization Analysis.

        Measures how effectively reported revenue converts to actual
        cash collected, indicating revenue quality and collection efficiency.
        """
        result = RevenueCashRealizationResult()

        revenue = getattr(data, 'revenue', None) or 0
        ocf = getattr(data, 'operating_cash_flow', None) or 0
        ar = getattr(data, 'accounts_receivable', None) or 0

        if not revenue or revenue <= 0:
            return result

        # Core metrics
        result.ocf_to_revenue = safe_divide(ocf, revenue)
        cash_collected = revenue - ar  # simplified approximation
        result.cash_to_revenue = safe_divide(cash_collected, revenue) if cash_collected else None
        result.collection_rate = safe_divide(revenue - ar, revenue) if ar is not None else None
        result.revenue_cash_gap = revenue - ocf
        result.cash_conversion_speed = safe_divide(ocf, revenue)
        result.revenue_quality_ratio = safe_divide(ocf, revenue)

        # Scoring: OCF/Revenue primary — higher = better cash realization
        ocf_rev = result.ocf_to_revenue or 0
        if ocf_rev >= 0.30:
            base = 10.0
        elif ocf_rev >= 0.22:
            base = 8.5
        elif ocf_rev >= 0.15:
            base = 7.0
        elif ocf_rev >= 0.10:
            base = 5.5
        elif ocf_rev >= 0.05:
            base = 4.0
        elif ocf_rev >= 0.02:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Bonus: collection rate >= 85%
        if result.collection_rate is not None and result.collection_rate >= 0.85:
            adj += 0.5
        # Bonus: OCF > 0 and revenue > 0
        if ocf > 0 and revenue > 0:
            adj += 0.5

        result.rcr_score = min(10.0, max(0.0, base + adj))

        if result.rcr_score >= 8:
            result.rcr_grade = "Excellent"
        elif result.rcr_score >= 6:
            result.rcr_grade = "Good"
        elif result.rcr_score >= 4:
            result.rcr_grade = "Adequate"
        else:
            result.rcr_grade = "Weak"

        result.summary = (
            f"Revenue Cash Realization: {result.rcr_grade} "
            f"(Score: {result.rcr_score:.1f}/10). "
            f"OCF/Revenue={ocf_rev:.1%}, "
            f"Collection Rate={result.collection_rate:.1%}."
            if result.collection_rate is not None
            else f"Revenue Cash Realization: {result.rcr_grade} "
            f"(Score: {result.rcr_score:.1f}/10). "
            f"OCF/Revenue={ocf_rev:.1%}."
        )

        return result

    def net_debt_position_analysis(self, data: FinancialData) -> NetDebtPositionResult:
        """Phase 286: Net Debt Position Analysis.

        Measures the company's net indebtedness after accounting for cash,
        indicating how leveraged the balance sheet truly is.
        """
        result = NetDebtPositionResult()

        total_debt = getattr(data, 'total_debt', None) or 0
        cash = getattr(data, 'cash', None) or 0
        ebitda = getattr(data, 'ebitda', None) or 0
        total_equity = getattr(data, 'total_equity', None) or 0
        total_assets = getattr(data, 'total_assets', None) or 0
        ocf = getattr(data, 'operating_cash_flow', None) or 0

        if not total_debt or total_debt <= 0:
            return result

        # Core metrics
        net_debt = total_debt - cash
        result.net_debt = net_debt
        result.net_debt_to_ebitda = safe_divide(net_debt, ebitda)
        result.net_debt_to_equity = safe_divide(net_debt, total_equity)
        result.net_debt_to_assets = safe_divide(net_debt, total_assets)
        result.cash_to_debt = safe_divide(cash, total_debt)
        result.net_debt_to_ocf = safe_divide(net_debt, ocf)

        # Scoring: Net Debt / EBITDA — lower = better (negative = net cash position)
        nd_ebitda = result.net_debt_to_ebitda
        if nd_ebitda is None:
            # No EBITDA — use net_debt sign
            if net_debt <= 0:
                base = 10.0
            else:
                base = 2.0
        elif nd_ebitda <= 0:
            base = 10.0  # Net cash position
        elif nd_ebitda <= 1.0:
            base = 8.5
        elif nd_ebitda <= 2.0:
            base = 7.0
        elif nd_ebitda <= 3.0:
            base = 5.5
        elif nd_ebitda <= 4.0:
            base = 4.0
        elif nd_ebitda <= 6.0:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Bonus: Cash/Debt >= 0.30
        if result.cash_to_debt is not None and result.cash_to_debt >= 0.30:
            adj += 0.5
        # Bonus: total_debt > 0
        if total_debt > 0:
            adj += 0.5

        result.ndp_score = min(10.0, max(0.0, base + adj))

        if result.ndp_score >= 8:
            result.ndp_grade = "Excellent"
        elif result.ndp_score >= 6:
            result.ndp_grade = "Good"
        elif result.ndp_score >= 4:
            result.ndp_grade = "Adequate"
        else:
            result.ndp_grade = "Weak"

        result.summary = (
            f"Net Debt Position: {result.ndp_grade} "
            f"(Score: {result.ndp_score:.1f}/10). "
            f"Net Debt=${net_debt:,.0f}, "
            f"Net Debt/EBITDA={nd_ebitda:.2f}x."
            if nd_ebitda is not None
            else f"Net Debt Position: {result.ndp_grade} "
            f"(Score: {result.ndp_score:.1f}/10). "
            f"Net Debt=${net_debt:,.0f}."
        )

        return result

    def liability_coverage_strength_analysis(self, data: FinancialData) -> LiabilityCoverageStrengthResult:
        """Phase 281: Liability Coverage Strength Analysis.

        Measures ability to cover total liabilities from operations.
        Primary metric: OCF / Total Liabilities (higher = stronger).

        Scoring (base from ocf_to_liabilities):
            >= 0.50 => 10  (can pay off liabilities in ~2 years)
            >= 0.35 => 8.5
            >= 0.25 => 7.0
            >= 0.15 => 5.5
            >= 0.10 => 4.0
            >= 0.05 => 2.5
            <  0.05 => 1.0

        Adjustments:
            assets_to_liabilities >= 2.0 (asset-backed)   => +0.5
            OCF > 0 and TL > 0 (data present)             => +0.5
        """
        result = LiabilityCoverageStrengthResult()

        ocf = getattr(data, 'operating_cash_flow', None) or 0
        tl = getattr(data, 'total_liabilities', None) or 0
        ta = getattr(data, 'total_assets', None) or 0
        te = getattr(data, 'total_equity', None) or 0
        ebitda = getattr(data, 'ebitda', None) or 0
        revenue = getattr(data, 'revenue', None) or 0

        if not ocf or ocf <= 0 or not tl or tl <= 0:
            result.summary = "Liability Coverage Strength: Insufficient data (need positive OCF and total liabilities)."
            return result

        # Core ratios
        result.ocf_to_liabilities = safe_divide(ocf, tl)
        result.ebitda_to_liabilities = safe_divide(ebitda, tl) if ebitda else None
        result.assets_to_liabilities = safe_divide(ta, tl) if ta and ta > 0 else None
        result.equity_to_liabilities = safe_divide(te, tl) if te else None
        result.liability_to_revenue = safe_divide(tl, revenue) if revenue and revenue > 0 else None
        result.liability_burden = safe_divide(tl, ta) if ta and ta > 0 else None

        # Scoring
        otl = result.ocf_to_liabilities
        if otl is None:
            result.summary = "Liability Coverage Strength: Could not compute ratio."
            return result

        if otl >= 0.50:
            base = 10.0
        elif otl >= 0.35:
            base = 8.5
        elif otl >= 0.25:
            base = 7.0
        elif otl >= 0.15:
            base = 5.5
        elif otl >= 0.10:
            base = 4.0
        elif otl >= 0.05:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if result.assets_to_liabilities is not None and result.assets_to_liabilities >= 2.0:
            adj += 0.5
        if ocf > 0 and tl > 0:
            adj += 0.5

        result.lcs_score = min(10.0, max(0.0, base + adj))

        if result.lcs_score >= 8:
            result.lcs_grade = "Excellent"
        elif result.lcs_score >= 6:
            result.lcs_grade = "Good"
        elif result.lcs_score >= 4:
            result.lcs_grade = "Adequate"
        else:
            result.lcs_grade = "Weak"

        result.summary = (
            f"Liability Coverage Strength: OCF/TL={otl:.2%}, "
            f"Score={result.lcs_score:.1f}/10 ({result.lcs_grade})."
        )

        return result

    def capital_adequacy_analysis(self, data: FinancialData) -> CapitalAdequacyResult:
        """Phase 279: Capital Adequacy Analysis.

        Measures whether equity base is sufficient relative to risk profile.
        Primary metric: Equity / Total Assets (equity ratio, higher = more adequate).

        Scoring (base from equity_ratio):
            >= 0.60 => 10  (fortress balance sheet)
            >= 0.50 => 8.5
            >= 0.40 => 7.0
            >= 0.30 => 5.5
            >= 0.20 => 4.0
            >= 0.10 => 2.5
            <  0.10 => 1.0

        Adjustments:
            retained_to_equity >= 0.50 (quality equity)     => +0.5
            equity > 0 and total_assets > 0 (data present)  => +0.5
        """
        result = CapitalAdequacyResult()

        equity = getattr(data, 'total_equity', None) or 0
        ta = getattr(data, 'total_assets', None) or 0
        tl = getattr(data, 'total_liabilities', None) or 0
        td = getattr(data, 'total_debt', None) or 0
        re = getattr(data, 'retained_earnings', None) or 0

        if not equity or equity <= 0 or not ta or ta <= 0:
            result.summary = "Capital Adequacy: Insufficient data (need positive equity and total assets)."
            return result

        # Core ratios
        result.equity_ratio = safe_divide(equity, ta)
        result.equity_to_debt = safe_divide(equity, td) if td and td > 0 else None
        result.retained_to_equity = safe_divide(re, equity) if re else None
        result.equity_to_liabilities = safe_divide(equity, tl) if tl and tl > 0 else None
        result.tangible_equity_ratio = safe_divide(equity, ta)  # Simplified (no intangibles tracked)
        result.capital_buffer = safe_divide(equity - tl, ta) if tl else None

        # Scoring
        er = result.equity_ratio
        if er is None:
            result.summary = "Capital Adequacy: Could not compute equity ratio."
            return result

        if er >= 0.60:
            base = 10.0
        elif er >= 0.50:
            base = 8.5
        elif er >= 0.40:
            base = 7.0
        elif er >= 0.30:
            base = 5.5
        elif er >= 0.20:
            base = 4.0
        elif er >= 0.10:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        rte = result.retained_to_equity
        if rte is not None and rte >= 0.50:
            adj += 0.5
        if equity > 0 and ta > 0:
            adj += 0.5

        result.caq_score = min(10.0, max(0.0, base + adj))

        if result.caq_score >= 8:
            result.caq_grade = "Excellent"
        elif result.caq_score >= 6:
            result.caq_grade = "Good"
        elif result.caq_score >= 4:
            result.caq_grade = "Adequate"
        else:
            result.caq_grade = "Weak"

        result.summary = (
            f"Capital Adequacy: Equity Ratio={er:.2%}, "
            f"Score={result.caq_score:.1f}/10 ({result.caq_grade})."
        )

        return result

    def operating_income_quality_analysis(self, data: FinancialData) -> OperatingIncomeQualityResult:
        """Phase 275: Operating Income Quality Analysis.

        Measures the quality and sustainability of operating income.
        Primary metric: OI / Revenue (operating margin, higher = better).

        Scoring (base from oi_to_revenue):
            >= 0.30 => 10  (strong operating margin)
            >= 0.20 => 8.5
            >= 0.15 => 7.0
            >= 0.10 => 5.5
            >= 0.05 => 4.0
            >= 0.02 => 2.5
            <  0.02 => 1.0

        Adjustments:
            OCF > OI (cash-backed)   => +0.5
            OI > 0 and Revenue > 0   => +0.5
        """
        result = OperatingIncomeQualityResult()

        oi = getattr(data, 'operating_income', None) or 0
        revenue = getattr(data, 'revenue', None) or 0
        ebitda = getattr(data, 'ebitda', None) or 0
        ocf = getattr(data, 'operating_cash_flow', None) or 0
        total_assets = getattr(data, 'total_assets', None) or 0

        if not oi or oi <= 0 or not revenue or revenue <= 0:
            result.summary = "Operating Income Quality: Insufficient data (need positive OI and revenue)."
            return result

        # Core ratios
        result.oi_to_revenue = safe_divide(oi, revenue)
        result.oi_to_ebitda = safe_divide(oi, ebitda) if ebitda and ebitda > 0 else None
        result.oi_to_ocf = safe_divide(oi, ocf) if ocf and ocf > 0 else None
        result.oi_to_total_assets = safe_divide(oi, total_assets) if total_assets and total_assets > 0 else None
        result.operating_spread = safe_divide(oi, revenue)  # same as margin
        result.oi_cash_backing = safe_divide(ocf, oi) if ocf else None

        # Scoring
        oi_margin = result.oi_to_revenue
        if oi_margin is None:
            result.summary = "Operating Income Quality: Could not compute margin."
            return result

        if oi_margin >= 0.30:
            base = 10.0
        elif oi_margin >= 0.20:
            base = 8.5
        elif oi_margin >= 0.15:
            base = 7.0
        elif oi_margin >= 0.10:
            base = 5.5
        elif oi_margin >= 0.05:
            base = 4.0
        elif oi_margin >= 0.02:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if ocf > 0 and ocf > oi:
            adj += 0.5
        if oi > 0 and revenue > 0:
            adj += 0.5

        result.oiq_score = min(10.0, max(0.0, base + adj))

        if result.oiq_score >= 8:
            result.oiq_grade = "Excellent"
        elif result.oiq_score >= 6:
            result.oiq_grade = "Good"
        elif result.oiq_score >= 4:
            result.oiq_grade = "Adequate"
        else:
            result.oiq_grade = "Weak"

        result.summary = (
            f"Operating Income Quality: OI Margin={oi_margin:.2%}, "
            f"Score={result.oiq_score:.1f}/10, Grade={result.oiq_grade}. "
            f"{'Strong' if oi_margin >= 0.20 else 'Moderate' if oi_margin >= 0.10 else 'Thin'} "
            f"operating profitability from core business."
        )

        return result

    def ebitda_to_debt_coverage_analysis(self, data: FinancialData) -> EbitdaToDebtCoverageResult:
        """Phase 274: EBITDA-to-Debt Coverage Analysis.

        Measures how well EBITDA covers total debt obligations.
        Primary metric: EBITDA / Total Debt (higher = better coverage).

        Scoring (base from ebitda_to_debt):
            >= 1.0  => 10  (can repay all debt from 1 year EBITDA)
            >= 0.60 => 8.5
            >= 0.40 => 7.0
            >= 0.25 => 5.5
            >= 0.15 => 4.0
            >= 0.08 => 2.5
            <  0.08 => 1.0

        Adjustments:
            ebitda_to_interest >= 3.0 => +0.5
            NI > 0 and EBITDA > 0    => +0.5
        """
        result = EbitdaToDebtCoverageResult()

        ebitda = getattr(data, 'ebitda', None) or 0
        total_debt = getattr(data, 'total_debt', None) or 0
        interest_expense = getattr(data, 'interest_expense', None) or 0
        total_liabilities = getattr(data, 'total_liabilities', None) or 0
        net_income = getattr(data, 'net_income', None) or 0

        if not ebitda or ebitda <= 0 or not total_debt or total_debt <= 0:
            result.summary = "EBITDA-to-Debt Coverage: Insufficient data (need positive EBITDA and total debt)."
            return result

        # Core ratios
        result.ebitda_to_debt = safe_divide(ebitda, total_debt)
        result.ebitda_to_interest = safe_divide(ebitda, interest_expense) if interest_expense and interest_expense > 0 else None
        result.debt_to_ebitda = safe_divide(total_debt, ebitda)
        result.ebitda_to_total_liabilities = safe_divide(ebitda, total_liabilities) if total_liabilities and total_liabilities > 0 else None
        result.debt_service_buffer = safe_divide(ebitda - interest_expense, total_debt) if interest_expense else result.ebitda_to_debt
        result.leverage_headroom = safe_divide(ebitda, total_debt) if total_debt else None

        # Scoring
        etd = result.ebitda_to_debt
        if etd is None:
            result.summary = "EBITDA-to-Debt Coverage: Could not compute ratio."
            return result

        if etd >= 1.0:
            base = 10.0
        elif etd >= 0.60:
            base = 8.5
        elif etd >= 0.40:
            base = 7.0
        elif etd >= 0.25:
            base = 5.5
        elif etd >= 0.15:
            base = 4.0
        elif etd >= 0.08:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        if result.ebitda_to_interest is not None and result.ebitda_to_interest >= 3.0:
            adj += 0.5
        if net_income > 0 and ebitda > 0:
            adj += 0.5

        result.etdc_score = min(10.0, max(0.0, base + adj))

        if result.etdc_score >= 8:
            result.etdc_grade = "Excellent"
        elif result.etdc_score >= 6:
            result.etdc_grade = "Good"
        elif result.etdc_score >= 4:
            result.etdc_grade = "Adequate"
        else:
            result.etdc_grade = "Weak"

        result.summary = (
            f"EBITDA-to-Debt Coverage: EBITDA/Debt={etd:.2f}, "
            f"Score={result.etdc_score:.1f}/10, Grade={result.etdc_grade}. "
            f"{'Strong' if etd >= 0.60 else 'Moderate' if etd >= 0.25 else 'Limited'} "
            f"debt repayment capacity from operating earnings."
        )

        return result

    def debt_quality_analysis(self, data: FinancialData) -> DebtQualityResult:
        """Phase 267: Debt Quality Assessment.

        Evaluates the quality and sustainability of a company's debt structure.
        Debt-to-Equity is the primary metric. Lower = less leveraged = higher quality.

        Scoring (D/E, lower = better):
            <=0.20 → 10 (minimal debt)
            <=0.50 → 8.5
            <=1.00 → 7.0
            <=1.50 → 5.5
            <=2.00 → 4.0
            <=3.00 → 2.5
            >3.00  → 1.0 (highly leveraged)
        Adjustments:
            +0.5 if interest_coverage >= 5.0 (strong coverage)
            +0.5 if debt_to_ebitda <= 3.0 (manageable debt load)
        """
        result = DebtQualityResult()

        td = getattr(data, 'total_debt', None)
        te = getattr(data, 'total_equity', None)
        ta = getattr(data, 'total_assets', None)

        if td is None or te is None or te <= 0:
            result.summary = "Debt Quality: Insufficient data."
            return result

        result.debt_to_equity = safe_divide(td, te)

        if ta and ta > 0:
            result.debt_to_assets = safe_divide(td, ta)

        tl = getattr(data, 'total_liabilities', None)
        if tl and tl > 0 and td > 0:
            result.long_term_debt_ratio = safe_divide(td, tl)

        ebitda = getattr(data, 'ebitda', None)
        if ebitda and ebitda > 0:
            result.debt_to_ebitda = safe_divide(td, ebitda)

        ie = getattr(data, 'interest_expense', None)
        ebit = getattr(data, 'ebit', None)
        if ie and ie > 0 and ebit is not None:
            result.interest_coverage = safe_divide(ebit, ie)

        if ie and ie > 0 and td > 0:
            result.debt_cost = safe_divide(ie, td)

        # Scoring
        de = result.debt_to_equity
        if de is None:
            result.summary = "Debt Quality: Could not compute ratio."
            return result

        if de <= 0.20:
            score = 10.0
        elif de <= 0.50:
            score = 8.5
        elif de <= 1.00:
            score = 7.0
        elif de <= 1.50:
            score = 5.5
        elif de <= 2.00:
            score = 4.0
        elif de <= 3.00:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.interest_coverage is not None and result.interest_coverage >= 5.0:
            score += 0.5
        if result.debt_to_ebitda is not None and result.debt_to_ebitda <= 3.0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.dq_score = score
        if score >= 8:
            result.dq_grade = "Excellent"
        elif score >= 6:
            result.dq_grade = "Good"
        elif score >= 4:
            result.dq_grade = "Adequate"
        else:
            result.dq_grade = "Weak"

        result.summary = (
            f"Debt Quality: D/E={de:.2f}, "
            f"score={score:.1f}/10, grade={result.dq_grade}."
        )

        return result

    def fixed_asset_productivity_analysis(self, data: FinancialData) -> FixedAssetProductivityResult:
        """Phase 263: Fixed Asset Productivity Analysis.

        Evaluates how effectively the company uses its fixed (non-current) assets
        to generate revenue. Fixed Asset Turnover = Revenue / Fixed Assets
        where Fixed Assets = Total Assets - Current Assets.

        Scoring (FAT-based, higher = better):
            >=8.0  → 10
            >=5.0  → 8.5
            >=3.0  → 7.0
            >=2.0  → 5.5
            >=1.0  → 4.0
            >=0.5  → 2.5
            <0.5   → 1.0
        Adjustments:
            +0.5 if fixed_to_total_assets <= 0.40 (asset-light)
            +0.5 if depreciation_to_fixed <= 0.10 (newer assets)
        """
        result = FixedAssetProductivityResult()

        revenue = getattr(data, 'revenue', None)
        ta = getattr(data, 'total_assets', None)
        ca = getattr(data, 'current_assets', None)

        if not revenue or not ta or revenue <= 0 or ta <= 0:
            result.summary = "Fixed Asset Productivity: Insufficient data."
            return result

        fixed_assets = ta - (ca if ca else 0)
        if fixed_assets <= 0:
            result.summary = "Fixed Asset Productivity: No fixed assets."
            return result

        # Core metrics
        result.fixed_asset_turnover = safe_divide(revenue, fixed_assets)
        result.revenue_per_fixed_asset = result.fixed_asset_turnover
        result.fixed_to_total_assets = safe_divide(fixed_assets, ta)
        result.net_fixed_asset_intensity = safe_divide(fixed_assets, revenue)

        capex = getattr(data, 'capex', None)
        if capex and fixed_assets > 0:
            result.capex_to_fixed_assets = safe_divide(capex, fixed_assets)

        dep = getattr(data, 'depreciation', None)
        if dep and fixed_assets > 0:
            result.depreciation_to_fixed = safe_divide(dep, fixed_assets)

        # Scoring
        fat = result.fixed_asset_turnover
        if fat is None:
            result.summary = "Fixed Asset Productivity: Could not compute turnover."
            return result

        if fat >= 8.0:
            score = 10.0
        elif fat >= 5.0:
            score = 8.5
        elif fat >= 3.0:
            score = 7.0
        elif fat >= 2.0:
            score = 5.5
        elif fat >= 1.0:
            score = 4.0
        elif fat >= 0.5:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.fixed_to_total_assets is not None and result.fixed_to_total_assets <= 0.40:
            score += 0.5
        if result.depreciation_to_fixed is not None and result.depreciation_to_fixed <= 0.10:
            score += 0.5

        score = max(0.0, min(10.0, score))
        result.fap_score = score
        if score >= 8:
            result.fap_grade = "Excellent"
        elif score >= 6:
            result.fap_grade = "Good"
        elif score >= 4:
            result.fap_grade = "Adequate"
        else:
            result.fap_grade = "Weak"

        result.summary = f"Fixed Asset Productivity: Turnover {fat:.1f}x — {result.fap_grade} ({score:.1f}/10)."
        return result

    def depreciation_burden_analysis(self, data: FinancialData) -> DepreciationBurdenResult:
        """Phase 259: Depreciation Burden Analysis.

        Measures how heavily depreciation weighs on earnings.
        Lower dep_to_revenue = lighter burden = more asset-light.
        """
        result = DepreciationBurdenResult()

        rev = getattr(data, 'revenue', None)
        dep = getattr(data, 'depreciation', None)
        ta = getattr(data, 'total_assets', None)
        ebitda = getattr(data, 'ebitda', None)
        ebit = getattr(data, 'ebit', None)
        gp = getattr(data, 'gross_profit', None)

        if not rev or not dep or rev <= 0 or dep <= 0:
            return result

        result.dep_to_revenue = safe_divide(dep, rev)
        if ta and ta > 0:
            result.dep_to_assets = safe_divide(dep, ta)
        if ebitda and ebitda > 0:
            result.dep_to_ebitda = safe_divide(dep, ebitda)
        if gp and gp > 0:
            result.dep_to_gross_profit = safe_divide(dep, gp)

        # EBITDA-to-EBIT spread (how much D&A adds to EBITDA vs EBIT)
        if ebitda and ebit and ebit > 0:
            result.ebitda_to_ebit_spread = safe_divide(ebitda, ebit)

        # Asset age proxy: D&A / Total Assets (higher = older assets)
        if ta and ta > 0:
            result.asset_age_proxy = safe_divide(dep, ta)

        # Scoring based on dep_to_revenue (lower = lighter burden)
        dtr = result.dep_to_revenue
        if dtr is not None:
            if dtr <= 0.03:
                score = 10.0
            elif dtr <= 0.05:
                score = 8.5
            elif dtr <= 0.08:
                score = 7.0
            elif dtr <= 0.12:
                score = 5.5
            elif dtr <= 0.18:
                score = 4.0
            elif dtr <= 0.25:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: dep/EBITDA <= 0.20 (low D&A relative to cash earnings)
            if result.dep_to_ebitda is not None and result.dep_to_ebitda <= 0.20:
                score += 0.5

            # Adjustment: dep/assets <= 0.03 (newer/lighter assets)
            if result.dep_to_assets is not None and result.dep_to_assets <= 0.03:
                score += 0.5

            result.db_score = min(10.0, max(0.0, score))

            if result.db_score >= 8:
                result.db_grade = "Excellent"
            elif result.db_score >= 6:
                result.db_grade = "Good"
            elif result.db_score >= 4:
                result.db_grade = "Adequate"
            else:
                result.db_grade = "Weak"

            result.summary = (
                f"Depreciation Burden: D&A/Rev={dtr:.1%}, "
                f"Score={result.db_score:.1f}/10 ({result.db_grade})."
            )

        return result

    def debt_to_capital_analysis(self, data: FinancialData) -> DebtToCapitalResult:
        """Phase 258: Debt-to-Capital Analysis.

        Measures the proportion of debt in the capital structure.
        Debt-to-Capital = Total Debt / (Total Debt + Total Equity).
        Lower ratio = less financial risk.
        """
        result = DebtToCapitalResult()

        td = getattr(data, 'total_debt', None)
        te = getattr(data, 'total_equity', None)
        cash = getattr(data, 'cash', None)

        if not td or not te or te <= 0:
            return result

        total_capital = td + te
        if total_capital <= 0:
            return result

        result.debt_to_capital = safe_divide(td, total_capital)
        result.debt_to_equity = safe_divide(td, te)
        result.equity_ratio = safe_divide(te, total_capital)

        # Long-term debt proxy (use total_debt as proxy)
        result.long_term_debt_to_capital = result.debt_to_capital

        # Net debt to capital
        if cash is not None:
            net_debt = td - cash
            result.net_debt_to_capital = safe_divide(max(0, net_debt), total_capital)

        # Financial risk index: D/E * (1 - equity_ratio)
        if result.debt_to_equity is not None and result.equity_ratio is not None:
            result.financial_risk_index = result.debt_to_equity * (1 - result.equity_ratio)

        # Scoring based on debt_to_capital (lower = better)
        dtc = result.debt_to_capital
        if dtc is not None:
            if dtc <= 0.20:
                score = 10.0
            elif dtc <= 0.30:
                score = 8.5
            elif dtc <= 0.40:
                score = 7.0
            elif dtc <= 0.50:
                score = 5.5
            elif dtc <= 0.60:
                score = 4.0
            elif dtc <= 0.75:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: net debt to capital < debt to capital (cash cushion)
            if result.net_debt_to_capital is not None and result.net_debt_to_capital < dtc - 0.05:
                score += 0.5

            # Adjustment: equity ratio >= 0.60
            if result.equity_ratio is not None and result.equity_ratio >= 0.60:
                score += 0.5

            result.dtc_score = min(10.0, max(0.0, score))

            if result.dtc_score >= 8:
                result.dtc_grade = "Excellent"
            elif result.dtc_score >= 6:
                result.dtc_grade = "Good"
            elif result.dtc_score >= 4:
                result.dtc_grade = "Adequate"
            else:
                result.dtc_grade = "Weak"

            result.summary = (
                f"Debt-to-Capital: D/C={dtc:.1%}, D/E={result.debt_to_equity:.2f}, "
                f"Score={result.dtc_score:.1f}/10 ({result.dtc_grade})."
            )

        return result

    def dividend_payout_analysis(self, data: FinancialData) -> DividendPayoutResult:
        """Phase 251: Dividend Payout Ratio Analysis.

        Measures proportion of earnings distributed as dividends.
        Primary metric: Dividends / Net Income (moderate 20-60% is ideal)."""
        result = DividendPayoutResult()

        div = data.dividends_paid
        ni = data.net_income
        ocf = data.operating_cash_flow
        capex = data.capex
        rev = data.revenue

        if div is None or ni is None or ni <= 0:
            result.summary = "Dividend Payout: Insufficient data (need Dividends and Net Income > 0)."
            return result

        # Dividends_paid is often stored as positive; ensure ratio is positive
        div_abs = abs(div) if div else 0

        result.div_to_ni = safe_divide(div_abs, ni)
        result.retention_ratio = 1.0 - result.div_to_ni if result.div_to_ni is not None else None
        result.div_to_ocf = safe_divide(div_abs, ocf) if ocf and ocf > 0 else None
        fcf = (ocf - capex) if ocf is not None and capex is not None else None
        result.div_to_fcf = safe_divide(div_abs, fcf) if fcf and fcf > 0 else None
        result.div_to_revenue = safe_divide(div_abs, rev)
        result.div_coverage = safe_divide(ni, div_abs) if div_abs > 0 else None

        rat = result.div_to_ni
        if rat is None:
            result.summary = "Dividend Payout: Cannot compute Div/NI."
            return result

        # Scoring: Moderate payout (20-60%) is ideal
        if 0.20 <= rat <= 0.60:
            score = 10.0
        elif 0.10 <= rat < 0.20 or 0.60 < rat <= 0.75:
            score = 8.0
        elif 0.05 <= rat < 0.10 or 0.75 < rat <= 0.90:
            score = 6.0
        elif rat < 0.05:
            score = 5.0  # Very low — may not attract income investors
        else:
            score = 3.0  # >90% — unsustainable

        # Adjustments
        if result.div_coverage is not None:
            if result.div_coverage >= 2.0:
                score += 0.5  # Well-covered dividends
            elif result.div_coverage < 1.0:
                score -= 0.5  # Dividend exceeds earnings — unsustainable

        if result.div_to_ocf is not None and result.div_to_ocf <= 0.50:
            score += 0.5  # Dividends well within cash flow

        score = max(0.0, min(10.0, score))
        result.dpr_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.dpr_grade = grade

        result.summary = (
            f"Dividend Payout: Div/NI={rat:.1%}, Retention={result.retention_ratio:.1%}. "
            f"Score={result.dpr_score:.1f}/10 ({grade})."
        )

        return result

    def operating_cash_flow_ratio_analysis(self, data: FinancialData) -> OperatingCashFlowRatioResult:
        """Phase 249: Operating Cash Flow Ratio Analysis.

        Measures ability to cover obligations from operations.
        Primary metric: OCF / Current Liabilities (higher = better)."""
        result = OperatingCashFlowRatioResult()

        ocf = data.operating_cash_flow
        cl = data.current_liabilities
        tl = data.total_liabilities
        rev = data.revenue
        ni = data.net_income
        td = data.total_debt

        if ocf is None or cl is None or cl <= 0:
            result.summary = "Operating Cash Flow Ratio: Insufficient data (need OCF and CL > 0)."
            return result

        result.ocf_to_cl = safe_divide(ocf, cl)
        result.ocf_to_tl = safe_divide(ocf, tl)
        result.ocf_to_revenue = safe_divide(ocf, rev)
        result.ocf_to_ni = safe_divide(ocf, ni) if ni and ni != 0 else None
        result.ocf_to_debt = safe_divide(ocf, td)
        result.ocf_margin = safe_divide(ocf, rev)

        rat = result.ocf_to_cl
        if rat is None:
            result.summary = "Operating Cash Flow Ratio: Cannot compute OCF/CL."
            return result

        # Scoring: Higher OCF/CL = better short-term coverage
        if rat >= 2.0:
            score = 10.0
        elif rat >= 1.5:
            score = 8.5
        elif rat >= 1.0:
            score = 7.0
        elif rat >= 0.75:
            score = 5.5
        elif rat >= 0.50:
            score = 4.0
        elif rat >= 0.25:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.ocf_to_revenue is not None:
            if result.ocf_to_revenue >= 0.20:
                score += 0.5  # Strong OCF margin
            elif result.ocf_to_revenue < 0.05:
                score -= 0.5  # Weak OCF margin

        if result.ocf_to_ni is not None and result.ocf_to_ni >= 1.0:
            score += 0.5  # OCF exceeds net income — quality earnings

        score = max(0.0, min(10.0, score))
        result.ocfr_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ocfr_grade = grade

        result.summary = (
            f"Operating Cash Flow Ratio: OCF/CL={rat:.2f}x. "
            f"Score={result.ocfr_score:.1f}/10 ({grade})."
        )

        return result

    def cash_conversion_cycle_analysis(self, data: FinancialData) -> CashConversionCycleResult:
        """Phase 248: Cash Conversion Cycle Analysis.

        Measures time to convert resource inputs into cash.
        CCC = DSO + DIO - DPO. Lower/negative is better."""
        result = CashConversionCycleResult()

        rev = data.revenue
        cogs = data.cogs
        ar = data.accounts_receivable
        inv = data.inventory
        ap = data.accounts_payable

        # Need at least revenue and COGS to compute any component
        if rev is None or rev <= 0 or cogs is None or cogs <= 0:
            result.summary = "Cash Conversion Cycle: Insufficient data (need Revenue and COGS > 0)."
            return result

        # Compute components (each may be None if numerator is None/0)
        result.dso = safe_divide(ar * 365, rev) if ar and ar > 0 else None
        result.dio = safe_divide(inv * 365, cogs) if inv and inv > 0 else None
        result.dpo = safe_divide(ap * 365, cogs) if ap and ap > 0 else None

        # CCC = DSO + DIO - DPO (need at least DSO or DIO to be meaningful)
        components = [result.dso, result.dio]
        if all(c is None for c in components):
            result.summary = "Cash Conversion Cycle: Insufficient data (need AR or Inventory > 0)."
            return result

        dso_val = result.dso if result.dso is not None else 0.0
        dio_val = result.dio if result.dio is not None else 0.0
        dpo_val = result.dpo if result.dpo is not None else 0.0

        result.ccc = dso_val + dio_val - dpo_val
        result.working_cap_days = dso_val + dio_val
        result.ccc_to_revenue = safe_divide(result.ccc, 365)

        ccc = result.ccc

        # Scoring: Lower CCC is better; negative is excellent
        if ccc <= 0:
            score = 10.0
        elif ccc <= 15:
            score = 9.0
        elif ccc <= 30:
            score = 8.0
        elif ccc <= 45:
            score = 7.0
        elif ccc <= 60:
            score = 6.0
        elif ccc <= 90:
            score = 5.0
        elif ccc <= 120:
            score = 3.5
        else:
            score = 1.5

        # Adjustments
        if result.dpo is not None and result.dpo >= 30:
            score += 0.5  # Good supplier terms
        if result.dso is not None and result.dso <= 30:
            score += 0.5  # Fast collection

        score = max(0.0, min(10.0, score))
        result.ccc_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ccc_grade = grade

        result.summary = (
            f"Cash Conversion Cycle: CCC={ccc:.1f} days "
            f"(DSO={dso_val:.0f} + DIO={dio_val:.0f} - DPO={dpo_val:.0f}). "
            f"Score={result.ccc_score:.1f}/10 ({grade})."
        )

        return result

    def inventory_turnover_analysis(self, data: FinancialData) -> InventoryTurnoverResult:
        """Phase 247: Inventory Turnover Analysis.

        Measures how efficiently inventory is managed.
        Primary metric: COGS / Inventory (higher = faster turns)."""
        result = InventoryTurnoverResult()

        cogs = data.cogs
        inv = data.inventory
        ca = data.current_assets
        ta = data.total_assets
        rev = data.revenue

        if inv is None or inv <= 0 or cogs is None or cogs <= 0:
            result.summary = "Inventory Turnover Analysis: Insufficient data (need COGS and Inventory > 0)."
            return result

        result.cogs_to_inv = safe_divide(cogs, inv)
        result.dio = safe_divide(inv * 365, cogs)
        result.inv_to_ca = safe_divide(inv, ca)
        result.inv_to_ta = safe_divide(inv, ta)
        result.inv_to_revenue = safe_divide(inv, rev)
        result.inv_velocity = safe_divide(cogs, inv * 12)  # Monthly turns

        rat = result.cogs_to_inv
        if rat is None:
            result.summary = "Inventory Turnover Analysis: Cannot compute COGS/Inventory."
            return result

        # Scoring: Higher COGS/Inv = faster turns = better
        if rat >= 12.0:
            score = 10.0
        elif rat >= 8.0:
            score = 8.5
        elif rat >= 6.0:
            score = 7.0
        elif rat >= 4.0:
            score = 5.5
        elif rat >= 2.0:
            score = 4.0
        elif rat >= 1.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.dio is not None:
            if result.dio <= 30:
                score += 0.5  # Very fast inventory movement
            elif result.dio > 120:
                score -= 0.5  # Very slow — obsolescence risk

        if result.inv_to_ca is not None:
            if result.inv_to_ca <= 0.15:
                score += 0.5  # Low inventory relative to CA
            elif result.inv_to_ca > 0.50:
                score -= 0.5  # Inventory dominates CA

        score = max(0.0, min(10.0, score))
        result.ito_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ito_grade = grade

        result.summary = (
            f"Inventory Turnover Analysis: COGS/Inv={rat:.2f}x, DIO={result.dio:.0f} days. "
            f"Score={result.ito_score:.1f}/10 ({grade})."
        )

        return result

    def payables_turnover_analysis(self, data: FinancialData) -> PayablesTurnoverResult:
        """Phase 246: Payables Turnover Analysis.

        Measures how efficiently AP is managed.
        Primary metric: COGS / AP (moderate is ideal — not too fast, not too slow).
        Optimal DPO: 30-60 days (balances cash management with supplier relations)."""
        result = PayablesTurnoverResult()

        cogs = data.cogs
        ap = data.accounts_payable
        cl = data.current_liabilities
        tl = data.total_liabilities

        if ap is None or ap <= 0 or cogs is None or cogs <= 0:
            result.summary = "Payables Turnover Analysis: Insufficient data (need COGS and AP > 0)."
            return result

        result.cogs_to_ap = safe_divide(cogs, ap)
        result.dpo = safe_divide(ap * 365, cogs)
        result.ap_to_cl = safe_divide(ap, cl)
        result.ap_to_tl = safe_divide(ap, tl)
        result.ap_to_cogs = safe_divide(ap, cogs)
        result.payment_velocity = safe_divide(cogs, ap * 12)  # Monthly turns

        rat = result.cogs_to_ap
        if rat is None:
            result.summary = "Payables Turnover Analysis: Cannot compute COGS/AP."
            return result

        dpo = result.dpo if result.dpo is not None else 0

        # Scoring: Moderate COGS/AP is ideal (DPO 30-60 days = COGS/AP 6-12)
        if 6.0 <= rat <= 12.0:
            score = 10.0  # Ideal range
        elif 4.0 <= rat < 6.0 or 12.0 < rat <= 15.0:
            score = 8.5
        elif 3.0 <= rat < 4.0 or 15.0 < rat <= 20.0:
            score = 7.0
        elif 2.0 <= rat < 3.0:
            score = 5.5  # Paying too slowly
        elif rat > 20.0:
            score = 5.5  # Paying too fast
        elif 1.0 <= rat < 2.0:
            score = 4.0
        else:
            score = 1.0

        # Adjustments based on DPO
        if 30 <= dpo <= 60:
            score += 0.5  # Ideal payment window
        elif dpo > 120:
            score -= 0.5  # Very slow — supplier risk

        if result.ap_to_cl is not None:
            if result.ap_to_cl <= 0.30:
                score += 0.5  # AP well within CL capacity
            elif result.ap_to_cl > 0.70:
                score -= 0.5  # AP dominates CL

        score = max(0.0, min(10.0, score))
        result.pto_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pto_grade = grade

        result.summary = (
            f"Payables Turnover Analysis: COGS/AP={rat:.2f}x, DPO={dpo:.0f} days. "
            f"Score={result.pto_score:.1f}/10 ({grade})."
        )

        return result

    def receivables_turnover_analysis(self, data: FinancialData) -> ReceivablesTurnoverResult:
        """Phase 245: Receivables Turnover Analysis.

        Measures how efficiently AR is collected.
        Primary metric: Revenue / AR (higher = faster collection)."""
        result = ReceivablesTurnoverResult()

        rev = data.revenue
        ar = data.accounts_receivable
        ca = data.current_assets
        ta = data.total_assets

        if ar is None or ar <= 0 or rev is None or rev <= 0:
            result.summary = "Receivables Turnover Analysis: Insufficient data (need Revenue and AR > 0)."
            return result

        result.rev_to_ar = safe_divide(rev, ar)
        result.dso = safe_divide(ar * 365, rev)
        result.ar_to_ca = safe_divide(ar, ca)
        result.ar_to_ta = safe_divide(ar, ta)
        result.ar_to_revenue = safe_divide(ar, rev)
        result.collection_efficiency = safe_divide(rev - ar, rev)

        rat = result.rev_to_ar
        if rat is None:
            result.summary = "Receivables Turnover Analysis: Cannot compute Rev/AR."
            return result

        # Scoring: Higher Rev/AR = better (faster collection)
        if rat >= 15.0:
            score = 10.0
        elif rat >= 10.0:
            score = 8.5
        elif rat >= 8.0:
            score = 7.0
        elif rat >= 6.0:
            score = 5.5
        elif rat >= 4.0:
            score = 4.0
        elif rat >= 2.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if result.dso is not None:
            if result.dso <= 30:
                score += 0.5  # Very fast collection
            elif result.dso > 90:
                score -= 0.5  # Slow collection

        if result.ar_to_revenue is not None:
            if result.ar_to_revenue <= 0.05:
                score += 0.5  # Low AR relative to revenue
            elif result.ar_to_revenue > 0.25:
                score -= 0.5  # High AR relative to revenue

        score = max(0.0, min(10.0, score))
        result.rto_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.rto_grade = grade

        result.summary = (
            f"Receivables Turnover Analysis: Rev/AR={rat:.2f}x, DSO={result.dso:.0f} days. "
            f"Score={result.rto_score:.1f}/10 ({grade})."
        )

        return result

    def cash_conversion_efficiency_analysis(self, data: FinancialData) -> CashConversionEfficiencyResult:
        """Phase 237: Cash Conversion Efficiency Analysis.

        Measures how effectively operating income converts to cash.
        Primary metric: OCF/OI (higher = better cash conversion quality).
        """
        result = CashConversionEfficiencyResult()

        ocf = data.operating_cash_flow
        oi = data.operating_income
        ni = data.net_income
        rev = data.revenue
        ebitda = data.ebitda
        capex = data.capex
        cash = data.cash

        if ocf is None or oi is None:
            result.summary = "Cash Conversion Efficiency: Insufficient data."
            return result

        ocf_to_oi = safe_divide(ocf, oi)
        result.ocf_to_oi = ocf_to_oi

        if ni is not None and ni != 0:
            result.ocf_to_ni = safe_divide(ocf, ni)
        if rev is not None and rev != 0:
            result.ocf_to_revenue = safe_divide(ocf, rev)
        if ebitda is not None and ebitda != 0:
            result.ocf_to_ebitda = safe_divide(ocf, ebitda)
        if capex is not None:
            fcf = ocf - capex
            result.fcf_to_oi = safe_divide(fcf, oi)
        if cash is not None:
            result.cash_to_oi = safe_divide(cash, oi)

        if ocf_to_oi is None:
            result.summary = "Cash Conversion Efficiency: OCF/OI not computable."
            return result

        # Scoring: OCF/OI
        if ocf_to_oi >= 1.20:
            score = 10.0
        elif ocf_to_oi >= 1.00:
            score = 8.5
        elif ocf_to_oi >= 0.80:
            score = 7.0
        elif ocf_to_oi >= 0.60:
            score = 5.5
        elif ocf_to_oi >= 0.40:
            score = 4.0
        elif ocf_to_oi >= 0.20:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OCF/NI
        if result.ocf_to_ni is not None:
            if result.ocf_to_ni >= 1.20:
                score += 0.5
            elif result.ocf_to_ni < 0.80:
                score -= 0.5

        # Adjustment: OCF/Revenue
        if result.ocf_to_revenue is not None:
            if result.ocf_to_revenue >= 0.20:
                score += 0.5
            elif result.ocf_to_revenue < 0.05:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cce_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.cce_grade = grade

        result.summary = (
            f"Cash Conversion Efficiency: OCF/OI={ocf_to_oi:.4f}. "
            f"Score={score:.1f}/10. Grade={grade}."
        )

        return result

    def fixed_cost_leverage_ratio_analysis(self, data: FinancialData) -> FixedCostLeverageRatioResult:
        """Phase 236: Fixed Cost Leverage Ratio Analysis.

        Measures degree of operating leverage (DOL) as GP/OI.
        Lower DOL means less sensitivity to revenue changes (more stable earnings).
        """
        result = FixedCostLeverageRatioResult()

        gp = data.gross_profit
        oi = data.operating_income
        rev = data.revenue
        cogs = data.cogs
        opex = data.operating_expenses

        if gp is None or oi is None or rev is None:
            result.summary = "Fixed Cost Leverage Ratio: Insufficient data."
            return result

        dol = safe_divide(gp, oi)
        result.dol = dol

        result.contribution_margin = safe_divide(gp, rev)
        result.oi_to_revenue = safe_divide(oi, rev)
        if cogs is not None:
            result.cogs_to_revenue = safe_divide(cogs, rev)
        if opex is not None:
            result.opex_to_revenue = safe_divide(opex, rev)

        # Breakeven proxy: OpEx / GP (how much of GP is consumed by fixed costs)
        if opex is not None:
            result.breakeven_proxy = safe_divide(opex, gp)

        if dol is None:
            result.summary = "Fixed Cost Leverage Ratio: DOL not computable (OI=0)."
            return result

        # Scoring: DOL (lower = more stable = better)
        if dol <= 1.5:
            score = 10.0
        elif dol <= 2.0:
            score = 8.5
        elif dol <= 2.5:
            score = 7.0
        elif dol <= 3.0:
            score = 5.5
        elif dol <= 4.0:
            score = 4.0
        elif dol <= 5.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OI/Revenue (operating margin)
        if result.oi_to_revenue is not None:
            if result.oi_to_revenue >= 0.15:
                score += 0.5
            elif result.oi_to_revenue < 0.05:
                score -= 0.5

        # Adjustment: COGS/Revenue (cost efficiency)
        if result.cogs_to_revenue is not None:
            if result.cogs_to_revenue <= 0.50:
                score += 0.5
            elif result.cogs_to_revenue > 0.80:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.fclr_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.fclr_grade = grade

        result.summary = (
            f"Fixed Cost Leverage Ratio: DOL={dol:.4f}. "
            f"Score={score:.1f}/10. Grade={grade}."
        )

        return result

    def revenue_quality_index_analysis(self, data: FinancialData) -> RevenueQualityIndexResult:
        """Phase 232: Revenue Quality Index Analysis.

        Primary metric: OCF/Revenue — how much operating cash flow each dollar
        of revenue generates. Higher = higher quality revenue.

        Scoring thresholds (ocf_to_revenue):
          >= 0.25 => 10
          >= 0.20 => 8.5
          >= 0.15 => 7.0
          >= 0.10 => 5.5
          >= 0.05 => 4.0
          >= 0.00 => 2.5
          < 0.00  => 1.0

        Adjustments:
          Gross margin >= 0.50     => +0.5
          Gross margin < 0.20     => -0.5
          AR/Revenue <= 0.10      => +0.5
          AR/Revenue > 0.30       => -0.5

        Score clamped to [0, 10].
        """
        result = RevenueQualityIndexResult()

        rev = data.revenue
        ocf = data.operating_cash_flow
        gp = data.gross_profit
        ni = data.net_income
        ebitda = data.ebitda
        ar = data.accounts_receivable
        cash = data.cash

        if rev is None or ocf is None:
            result.summary = "Revenue Quality Index: Insufficient data."
            return result

        ocf_rev = safe_divide(ocf, rev)
        result.ocf_to_revenue = ocf_rev

        result.gross_margin = safe_divide(gp, rev)
        result.ni_to_revenue = safe_divide(ni, rev)
        result.ebitda_to_revenue = safe_divide(ebitda, rev)
        result.ar_to_revenue = safe_divide(ar, rev)
        result.cash_to_revenue = safe_divide(cash, rev)

        if ocf_rev is None:
            result.summary = "Revenue Quality Index: Cannot compute OCF/Revenue."
            return result

        if ocf_rev >= 0.25:
            base = 10.0
        elif ocf_rev >= 0.20:
            base = 8.5
        elif ocf_rev >= 0.15:
            base = 7.0
        elif ocf_rev >= 0.10:
            base = 5.5
        elif ocf_rev >= 0.05:
            base = 4.0
        elif ocf_rev >= 0.00:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        gm = result.gross_margin
        if gm is not None:
            if gm >= 0.50:
                adj += 0.5
            elif gm < 0.20:
                adj -= 0.5

        ar_rev = result.ar_to_revenue
        if ar_rev is not None:
            if ar_rev <= 0.10:
                adj += 0.5
            elif ar_rev > 0.30:
                adj -= 0.5

        result.rqi_score = max(0.0, min(10.0, base + adj))

        if result.rqi_score >= 8:
            result.rqi_grade = "Excellent"
        elif result.rqi_score >= 6:
            result.rqi_grade = "Good"
        elif result.rqi_score >= 4:
            result.rqi_grade = "Adequate"
        else:
            result.rqi_grade = "Weak"

        ocf_s = f"{result.ocf_to_revenue:.4f}" if result.ocf_to_revenue is not None else "N/A"
        gm_s = f"{result.gross_margin:.4f}" if result.gross_margin is not None else "N/A"
        ar_s = f"{result.ar_to_revenue:.4f}" if result.ar_to_revenue is not None else "N/A"
        result.summary = (
            f"Revenue Quality Index: OCF/Rev={ocf_s}, "
            f"GM={gm_s}, AR/Rev={ar_s}. "
            f"Score={result.rqi_score:.1f}/10 ({result.rqi_grade})."
        )
        return result

    def fixed_asset_utilization_analysis(self, data: FinancialData) -> FixedAssetUtilizationResult:
        """Phase 221: Fixed Asset Utilization Analysis.

        Measures how effectively fixed assets generate revenue.
        Primary metric: Revenue / Fixed Assets (higher is better).
        Fixed Assets proxy: Total Assets - Current Assets.
        """
        result = FixedAssetUtilizationResult()

        # Fixed assets proxy
        fixed_assets = None
        if data.total_assets is not None and data.current_assets is not None:
            fixed_assets = data.total_assets - data.current_assets
            if fixed_assets <= 0:
                fixed_assets = None

        fa_turnover = safe_divide(data.revenue, fixed_assets) if fixed_assets is not None else None
        result.fixed_asset_turnover = fa_turnover
        result.depreciation_to_revenue = safe_divide(data.depreciation, data.revenue)
        result.capex_to_depreciation = safe_divide(data.capex, data.depreciation)
        result.capex_to_total_assets = safe_divide(data.capex, data.total_assets)

        if fixed_assets is not None and data.total_assets is not None:
            result.fixed_to_total_assets = safe_divide(fixed_assets, data.total_assets)

        result.depreciation_to_total_assets = safe_divide(data.depreciation, data.total_assets)

        if fa_turnover is None:
            result.fau_score = 0.0
            result.fau_grade = "Weak"
            result.summary = "Fixed Asset Utilization: Insufficient data."
            return result

        # Scoring: FA Turnover higher is better
        if fa_turnover >= 3.0:
            score = 10.0
        elif fa_turnover >= 2.5:
            score = 8.5
        elif fa_turnover >= 2.0:
            score = 7.0
        elif fa_turnover >= 1.5:
            score = 5.5
        elif fa_turnover >= 1.0:
            score = 4.0
        elif fa_turnover >= 0.5:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: CapEx/D&A — reinvestment ratio
        capex_da = result.capex_to_depreciation
        if capex_da is not None:
            if capex_da >= 1.5:
                score += 0.5
            elif capex_da < 0.5:
                score -= 0.5

        # Adjustment: D&A/Rev — low means assets are productive
        da_rev = result.depreciation_to_revenue
        if da_rev is not None:
            if da_rev <= 0.03:
                score += 0.5
            elif da_rev > 0.15:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.fau_score = score

        if score >= 8.0:
            result.fau_grade = "Excellent"
        elif score >= 6.0:
            result.fau_grade = "Good"
        elif score >= 4.0:
            result.fau_grade = "Adequate"
        else:
            result.fau_grade = "Weak"

        result.summary = (
            f"Fixed Asset Utilization: FA Turnover={fa_turnover:.4f}. "
            f"Score={score:.1f}/10 ({result.fau_grade})."
        )

        return result

    def cost_control_analysis(self, data: FinancialData) -> CostControlResult:
        """Phase 215: Cost Control Analysis.

        Primary metric: OpEx/Revenue — lower is better.
        Measures how well the company controls operating costs.
        """
        result = CostControlResult()

        # OpEx/Revenue (primary)
        opex_ratio = safe_divide(data.operating_expenses, data.revenue)
        result.opex_to_revenue = opex_ratio

        # COGS/Revenue
        cogs_ratio = safe_divide(data.cogs, data.revenue)
        result.cogs_to_revenue = cogs_ratio

        # SGA proxy: (OpEx - COGS) / Revenue — only if both exist
        if data.operating_expenses is not None and data.cogs is not None and data.revenue is not None and data.revenue != 0:
            sga = data.operating_expenses - data.cogs
            if sga >= 0:
                result.sga_to_revenue = sga / data.revenue
            else:
                result.sga_to_revenue = None
        else:
            result.sga_to_revenue = None

        # Operating Margin: OI/Revenue
        oi_ratio = safe_divide(data.operating_income, data.revenue)
        result.operating_margin = oi_ratio

        # OpEx/Gross Profit
        result.opex_to_gross_profit = safe_divide(data.operating_expenses, data.gross_profit)

        # EBITDA Margin
        result.ebitda_margin = safe_divide(data.ebitda, data.revenue)

        # Scoring: based on OpEx/Revenue (lower is better)
        if opex_ratio is None:
            result.cc_score = 0.0
            result.cc_grade = "Weak"
            result.summary = "Cost Control: Insufficient data for analysis."
            return result

        if opex_ratio <= 0.15:
            score = 10.0
        elif opex_ratio <= 0.20:
            score = 8.5
        elif opex_ratio <= 0.25:
            score = 7.0
        elif opex_ratio <= 0.30:
            score = 5.5
        elif opex_ratio <= 0.40:
            score = 4.0
        elif opex_ratio <= 0.55:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: Operating Margin
        if oi_ratio is not None:
            if oi_ratio >= 0.25:
                score += 0.5
            elif oi_ratio < 0.05:
                score -= 0.5

        # Adjustment: COGS/Revenue
        if cogs_ratio is not None:
            if cogs_ratio <= 0.40:
                score += 0.5
            elif cogs_ratio > 0.75:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cc_score = score

        if score >= 8.0:
            result.cc_grade = "Excellent"
        elif score >= 6.0:
            result.cc_grade = "Good"
        elif score >= 4.0:
            result.cc_grade = "Adequate"
        else:
            result.cc_grade = "Weak"

        or_str = f"{opex_ratio:.4f}" if opex_ratio is not None else "N/A"
        oi_str = f"{oi_ratio:.4f}" if oi_ratio is not None else "N/A"
        cogs_str = f"{cogs_ratio:.4f}" if cogs_ratio is not None else "N/A"
        result.summary = (
            f"Cost Control: OpEx/Rev={or_str}, "
            f"OI/Rev={oi_str}, COGS/Rev={cogs_str}. "
            f"Score={score:.1f}/10 ({result.cc_grade})."
        )

        return result

    def valuation_signal_analysis(self, data: FinancialData) -> ValuationSignalResult:
        """Phase 212: Valuation Signal Analysis.

        Evaluates valuation attractiveness using proxy multiples.
        Primary metric: EV/EBITDA proxy (lower is better — more undervalued).
        EV proxy = Total Assets + Total Debt - Cash.
        """
        result = ValuationSignalResult()

        # Enterprise Value proxy
        ev = None
        if data.total_assets is not None:
            ev = data.total_assets
            if data.total_debt is not None:
                ev += data.total_debt
            if data.cash is not None:
                ev -= data.cash

        # EV/EBITDA (primary — lower is better)
        ev_ebitda = safe_divide(ev, data.ebitda)
        result.ev_to_ebitda = ev_ebitda

        # P/E proxy (TA/NI)
        result.price_to_earnings = safe_divide(data.total_assets, data.net_income)

        # P/B proxy (TA/TE)
        result.price_to_book = safe_divide(data.total_assets, data.total_equity)

        # EV/Revenue
        result.ev_to_revenue = safe_divide(ev, data.revenue)

        # Earnings Yield (NI/TA)
        result.earnings_yield = safe_divide(data.net_income, data.total_assets)

        # FCF Yield (FCF/TA)
        fcf = None
        if data.operating_cash_flow is not None and data.capex is not None:
            fcf = data.operating_cash_flow - data.capex
        result.fcf_yield = safe_divide(fcf, data.total_assets)

        # --- Scoring: EV/EBITDA-based (lower is better) ---
        if ev_ebitda is None:
            result.vsg_score = 0.0
            result.vsg_grade = "Weak"
            result.summary = "Valuation Signal: Insufficient data for analysis."
            return result

        if ev_ebitda <= 5.0:
            score = 10.0
        elif ev_ebitda <= 8.0:
            score = 8.5
        elif ev_ebitda <= 12.0:
            score = 7.0
        elif ev_ebitda <= 16.0:
            score = 5.5
        elif ev_ebitda <= 20.0:
            score = 4.0
        elif ev_ebitda <= 30.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: Earnings Yield (higher is better)
        ey = result.earnings_yield
        if ey is not None:
            if ey >= 0.10:
                score += 0.5
            elif ey < 0.03:
                score -= 0.5

        # Adjustment: P/B (lower is better)
        pb = result.price_to_book
        if pb is not None:
            if pb <= 1.0:
                score += 0.5
            elif pb > 5.0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.vsg_score = score

        if score >= 8:
            result.vsg_grade = "Excellent"
        elif score >= 6:
            result.vsg_grade = "Good"
        elif score >= 4:
            result.vsg_grade = "Adequate"
        else:
            result.vsg_grade = "Weak"

        eve_s = f"{ev_ebitda:.2f}" if ev_ebitda is not None else "N/A"
        ey_s = f"{ey:.4f}" if ey is not None else "N/A"
        pb_s = f"{pb:.2f}" if pb is not None else "N/A"
        result.summary = (
            f"Valuation Signal: EV/EBITDA={eve_s}, Earnings Yield={ey_s}, "
            f"P/B={pb_s}. Score={score:.1f}/10 ({result.vsg_grade})."
        )

        return result

    def capital_discipline_analysis(self, data: FinancialData) -> CapitalDisciplineResult:
        """Phase 211: Capital Discipline Analysis.

        Evaluates how disciplined a company is with capital allocation.
        Primary metric: Retained Earnings / Total Equity (higher is better).
        """
        result = CapitalDisciplineResult()

        # RE/TE (primary)
        rete = safe_divide(data.retained_earnings, data.total_equity)
        result.retained_to_equity = rete

        # RE/TA
        result.retained_to_assets = safe_divide(data.retained_earnings, data.total_assets)

        # Dividend Payout (Div/NI)
        result.dividend_payout = safe_divide(data.dividends_paid, data.net_income)

        # CapEx/OCF (reinvestment ratio)
        result.capex_to_ocf = safe_divide(data.capex, data.operating_cash_flow)

        # Debt/Equity
        result.debt_to_equity = safe_divide(data.total_debt, data.total_equity)

        # OCF/Debt
        result.ocf_to_debt = safe_divide(data.operating_cash_flow, data.total_debt)

        # --- Scoring: RE/TE-based ---
        if rete is None:
            result.cd_score = 0.0
            result.cd_grade = "Weak"
            result.summary = "Capital Discipline: Insufficient data for analysis."
            return result

        if rete >= 0.70:
            score = 10.0
        elif rete >= 0.60:
            score = 8.5
        elif rete >= 0.50:
            score = 7.0
        elif rete >= 0.40:
            score = 5.5
        elif rete >= 0.30:
            score = 4.0
        elif rete >= 0.15:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OCF/TD (ability to repay)
        ocftd = result.ocf_to_debt
        if ocftd is not None:
            if ocftd >= 0.50:
                score += 0.5
            elif ocftd < 0.15:
                score -= 0.5

        # Adjustment: CapEx/OCF (capital intensity — lower is better)
        cxo = result.capex_to_ocf
        if cxo is not None:
            if cxo <= 0.30:
                score += 0.5
            elif cxo > 0.80:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cd_score = score

        if score >= 8:
            result.cd_grade = "Excellent"
        elif score >= 6:
            result.cd_grade = "Good"
        elif score >= 4:
            result.cd_grade = "Adequate"
        else:
            result.cd_grade = "Weak"

        rete_s = f"{rete:.4f}" if rete is not None else "N/A"
        ocftd_s = f"{ocftd:.4f}" if ocftd is not None else "N/A"
        cxo_s = f"{cxo:.4f}" if cxo is not None else "N/A"
        result.summary = (
            f"Capital Discipline: RE/TE={rete_s}, OCF/Debt={ocftd_s}, "
            f"CapEx/OCF={cxo_s}. Score={score:.1f}/10 ({result.cd_grade})."
        )

        return result

    def resource_optimization_analysis(self, data: FinancialData) -> ResourceOptimizationResult:
        """Phase 210: Resource Optimization Analysis.

        Evaluates how efficiently a company converts revenue into free cash flow.
        Primary metric: FCF/Revenue (higher is better).
        """
        result = ResourceOptimizationResult()

        # Free Cash Flow = OCF - CapEx
        fcf = None
        if data.operating_cash_flow is not None and data.capex is not None:
            fcf = data.operating_cash_flow - data.capex

        # FCF/Revenue (primary)
        fcf_rev = safe_divide(fcf, data.revenue)
        result.fcf_to_revenue = fcf_rev

        # OCF/Revenue
        result.ocf_to_revenue = safe_divide(data.operating_cash_flow, data.revenue)

        # CapEx/Revenue
        result.capex_to_revenue = safe_divide(data.capex, data.revenue)

        # OCF/Total Assets
        result.ocf_to_assets = safe_divide(data.operating_cash_flow, data.total_assets)

        # FCF/Total Assets
        result.fcf_to_assets = safe_divide(fcf, data.total_assets)

        # Dividend Payout Ratio (Div/NI)
        result.dividend_payout_ratio = safe_divide(data.dividends_paid, data.net_income)

        # --- Scoring: FCF/Revenue-based ---
        if fcf_rev is None:
            result.ro_score = 0.0
            result.ro_grade = "Weak"
            result.summary = "Resource Optimization: Insufficient data for analysis."
            return result

        if fcf_rev >= 0.20:
            score = 10.0
        elif fcf_rev >= 0.15:
            score = 8.5
        elif fcf_rev >= 0.12:
            score = 7.0
        elif fcf_rev >= 0.08:
            score = 5.5
        elif fcf_rev >= 0.05:
            score = 4.0
        elif fcf_rev >= 0.02:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OCF/Revenue (cash generation efficiency)
        ocfr = result.ocf_to_revenue
        if ocfr is not None:
            if ocfr >= 0.20:
                score += 0.5
            elif ocfr < 0.08:
                score -= 0.5

        # Adjustment: CapEx/Revenue (capital intensity — lower is better)
        cxr = result.capex_to_revenue
        if cxr is not None:
            if cxr <= 0.05:
                score += 0.5
            elif cxr > 0.20:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.ro_score = score

        if score >= 8:
            result.ro_grade = "Excellent"
        elif score >= 6:
            result.ro_grade = "Good"
        elif score >= 4:
            result.ro_grade = "Adequate"
        else:
            result.ro_grade = "Weak"

        fr_s = f"{fcf_rev:.4f}" if fcf_rev is not None else "N/A"
        ocfr_s = f"{ocfr:.4f}" if ocfr is not None else "N/A"
        cxr_s = f"{cxr:.4f}" if cxr is not None else "N/A"
        result.summary = (
            f"Resource Optimization: FCF/Rev={fr_s}, OCF/Rev={ocfr_s}, "
            f"CapEx/Rev={cxr_s}. Score={score:.1f}/10 ({result.ro_grade})."
        )

        return result

    def financial_productivity_analysis(self, data: FinancialData) -> FinancialProductivityResult:
        """Phase 205: Financial Productivity Analysis.

        Measures how productively a company converts financial resources into output.
        Primary metric: Revenue per Asset (Rev/TA — higher is better).
        """
        result = FinancialProductivityResult()

        # Revenue per Asset (asset turnover)
        rpa = safe_divide(data.revenue, data.total_assets)
        result.revenue_per_asset = rpa

        # Revenue per Equity
        result.revenue_per_equity = safe_divide(data.revenue, data.total_equity)

        # EBITDA per Employee proxy: EBITDA / OpEx (higher = more productive per cost unit)
        result.ebitda_per_employee_proxy = safe_divide(data.ebitda, data.operating_expenses)

        # Operating Income per Asset
        result.operating_income_per_asset = safe_divide(data.operating_income, data.total_assets)

        # Net Income per Revenue (net margin as productivity)
        result.net_income_per_revenue = safe_divide(data.net_income, data.revenue)

        # Cash Flow per Asset
        result.cash_flow_per_asset = safe_divide(data.operating_cash_flow, data.total_assets)

        # --- Scoring: RPA-based ---
        if rpa is None:
            result.fp_score = 0.0
            result.fp_grade = "Weak"
            result.summary = "Financial Productivity: Insufficient data for analysis."
            return result

        if rpa >= 2.0:
            score = 10.0
        elif rpa >= 1.5:
            score = 8.5
        elif rpa >= 1.0:
            score = 7.0
        elif rpa >= 0.70:
            score = 5.5
        elif rpa >= 0.40:
            score = 4.0
        elif rpa >= 0.20:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: EBITDA/OpEx (operational productivity)
        epo = result.ebitda_per_employee_proxy
        if epo is not None:
            if epo >= 1.5:
                score += 0.5
            elif epo < 0.50:
                score -= 0.5

        # Adjustment: OCF/TA (cash productivity)
        cfpa = result.cash_flow_per_asset
        if cfpa is not None:
            if cfpa >= 0.12:
                score += 0.5
            elif cfpa < 0.05:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.fp_score = score

        if score >= 8:
            result.fp_grade = "Excellent"
        elif score >= 6:
            result.fp_grade = "Good"
        elif score >= 4:
            result.fp_grade = "Adequate"
        else:
            result.fp_grade = "Weak"

        rpa_s = f"{rpa:.2f}" if rpa is not None else "N/A"
        epo_s = f"{epo:.2f}" if epo is not None else "N/A"
        cfpa_s = f"{cfpa:.2f}" if cfpa is not None else "N/A"
        result.summary = (
            f"Financial Productivity: Rev/Assets={rpa_s}, EBITDA/OpEx={epo_s}, "
            f"OCF/Assets={cfpa_s}. Score={score:.1f}/10 ({result.fp_grade})."
        )

        return result

    def equity_preservation_analysis(self, data: FinancialData) -> EquityPreservationResult:
        """Phase 198: Equity Preservation Analysis.

        Metrics:
            equity_to_assets: TE / TA (primary — higher is better)
            retained_to_equity: RE / TE
            equity_growth_capacity: NI / TE
            equity_to_liabilities: TE / TL
            tangible_equity_ratio: TE / TA (proxy)
            equity_per_revenue: TE / Rev
        """
        result = EquityPreservationResult()
        te = data.total_equity
        ta = data.total_assets
        re = data.retained_earnings
        ni = data.net_income
        tl = data.total_liabilities
        rev = data.revenue

        # Primary: equity-to-assets
        eta = safe_divide(te, ta)
        result.equity_to_assets = eta

        # Secondary metrics
        result.retained_to_equity = safe_divide(re, te)
        result.equity_growth_capacity = safe_divide(ni, te)
        result.equity_to_liabilities = safe_divide(te, tl)
        result.tangible_equity_ratio = safe_divide(te, ta)
        result.equity_per_revenue = safe_divide(te, rev)

        # Scoring on equity-to-assets (higher is better)
        if eta is None:
            result.ep_score = 0.0
            result.ep_grade = "Weak"
            result.summary = "Equity Preservation: Insufficient data."
            return result

        if eta >= 0.60:
            base = 10.0
        elif eta >= 0.50:
            base = 8.5
        elif eta >= 0.40:
            base = 7.0
        elif eta >= 0.30:
            base = 5.5
        elif eta >= 0.20:
            base = 4.0
        elif eta >= 0.10:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Retained-to-equity adjustment
        rte = result.retained_to_equity
        if rte is not None:
            if rte >= 0.60:
                adj += 0.5
            elif rte < 0.20:
                adj -= 0.5

        # Equity growth capacity adjustment
        egc = result.equity_growth_capacity
        if egc is not None:
            if egc >= 0.15:
                adj += 0.5
            elif egc < 0.05:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.ep_score = score

        if score >= 8:
            result.ep_grade = "Excellent"
        elif score >= 6:
            result.ep_grade = "Good"
        elif score >= 4:
            result.ep_grade = "Adequate"
        else:
            result.ep_grade = "Weak"

        eta_str = f"{eta:.2f}" if eta is not None else "N/A"
        result.summary = (
            f"Equity Preservation: Equity-to-assets {eta_str}, "
            f"score {score:.1f}/10 ({result.ep_grade})."
        )

        return result

    def debt_management_analysis(self, data: FinancialData) -> DebtManagementResult:
        """Phase 197: Debt Management Analysis.

        Metrics:
            debt_to_operating_income: TD / OI (primary — lower is better)
            debt_to_ocf: TD / OCF
            interest_to_revenue: IE / Rev
            debt_to_gross_profit: TD / GP
            net_debt_ratio: (TD - Cash) / TA
            debt_coverage_ratio: EBITDA / (IE + TD * 0.1)
        """
        result = DebtManagementResult()
        td = data.total_debt
        oi = data.operating_income
        ocf = data.operating_cash_flow
        ie = data.interest_expense
        rev = data.revenue
        gp = data.gross_profit
        cash = data.cash
        ta = data.total_assets
        ebitda = data.ebitda

        # Primary: debt-to-operating-income
        dtoi = safe_divide(td, oi)
        result.debt_to_operating_income = dtoi

        # Secondary metrics
        result.debt_to_ocf = safe_divide(td, ocf)
        result.interest_to_revenue = safe_divide(ie, rev)
        result.debt_to_gross_profit = safe_divide(td, gp)

        # Net debt ratio: (TD - Cash) / TA
        if td is not None and cash is not None and ta is not None and ta > 0:
            result.net_debt_ratio = (td - cash) / ta
        else:
            result.net_debt_ratio = None

        # Debt coverage ratio: EBITDA / (IE + TD * 0.1)
        if ebitda is not None and ie is not None and td is not None:
            denom = ie + td * 0.1
            if denom > 0:
                result.debt_coverage_ratio = ebitda / denom
            else:
                result.debt_coverage_ratio = None
        else:
            result.debt_coverage_ratio = None

        # Scoring on debt-to-OI (lower is better)
        if dtoi is None:
            result.dm_score = 0.0
            result.dm_grade = "Weak"
            result.summary = "Debt Management: Insufficient data."
            return result

        if dtoi <= 1.0:
            base = 10.0
        elif dtoi <= 2.0:
            base = 8.5
        elif dtoi <= 3.0:
            base = 7.0
        elif dtoi <= 4.0:
            base = 5.5
        elif dtoi <= 5.0:
            base = 4.0
        elif dtoi <= 7.0:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Interest-to-revenue adjustment
        itr = result.interest_to_revenue
        if itr is not None:
            if itr <= 0.03:
                adj += 0.5
            elif itr > 0.10:
                adj -= 0.5

        # Debt coverage ratio adjustment
        dcr = result.debt_coverage_ratio
        if dcr is not None:
            if dcr >= 3.0:
                adj += 0.5
            elif dcr < 1.5:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.dm_score = score

        if score >= 8:
            result.dm_grade = "Excellent"
        elif score >= 6:
            result.dm_grade = "Good"
        elif score >= 4:
            result.dm_grade = "Adequate"
        else:
            result.dm_grade = "Weak"

        dtoi_str = f"{dtoi:.2f}" if dtoi is not None else "N/A"
        result.summary = (
            f"Debt Management: Debt-to-OI ratio {dtoi_str}, "
            f"score {score:.1f}/10 ({result.dm_grade})."
        )

        return result

    def income_retention_analysis(self, data: FinancialData) -> IncomeRetentionResult:
        """Phase 196: Income Retention Analysis.

        Metrics:
            net_to_gross_ratio: NI / GP (primary — higher is better)
            net_to_operating_ratio: NI / OI
            net_to_ebitda_ratio: NI / EBITDA
            retention_rate: RE / NI
            income_to_asset_generation: NI / TA
            after_tax_margin: NI / Rev
        """
        result = IncomeRetentionResult()
        ni = data.net_income
        gp = data.gross_profit
        oi = data.operating_income
        ebitda = data.ebitda
        re = data.retained_earnings
        ta = data.total_assets
        rev = data.revenue

        # Primary: net-to-gross ratio
        ntgr = safe_divide(ni, gp)
        result.net_to_gross_ratio = ntgr

        # Secondary metrics
        result.net_to_operating_ratio = safe_divide(ni, oi)
        result.net_to_ebitda_ratio = safe_divide(ni, ebitda)
        result.retention_rate = safe_divide(re, ni)
        result.income_to_asset_generation = safe_divide(ni, ta)
        result.after_tax_margin = safe_divide(ni, rev)

        # Scoring on net-to-gross ratio (higher is better)
        if ntgr is None:
            result.ir_score = 0.0
            result.ir_grade = "Weak"
            result.summary = "Income Retention: Insufficient data."
            return result

        if ntgr >= 0.45:
            base = 10.0
        elif ntgr >= 0.35:
            base = 8.5
        elif ntgr >= 0.25:
            base = 7.0
        elif ntgr >= 0.18:
            base = 5.5
        elif ntgr >= 0.10:
            base = 4.0
        elif ntgr >= 0.05:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Net-to-operating ratio adjustment
        ntor = result.net_to_operating_ratio
        if ntor is not None:
            if ntor >= 0.80:
                adj += 0.5
            elif ntor < 0.50:
                adj -= 0.5

        # After-tax margin adjustment
        atm = result.after_tax_margin
        if atm is not None:
            if atm >= 0.15:
                adj += 0.5
            elif atm < 0.05:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.ir_score = score

        if score >= 8:
            result.ir_grade = "Excellent"
        elif score >= 6:
            result.ir_grade = "Good"
        elif score >= 4:
            result.ir_grade = "Adequate"
        else:
            result.ir_grade = "Weak"

        ntgr_str = f"{ntgr:.2f}" if ntgr is not None else "N/A"
        result.summary = (
            f"Income Retention: Net-to-gross ratio {ntgr_str}, "
            f"score {score:.1f}/10 ({result.ir_grade})."
        )

        return result

    def operational_efficiency_analysis(self, data: FinancialData) -> OperationalEfficiencyResult:
        """Phase 195: Operational Efficiency Analysis.

        Metrics:
            oi_margin: OI / Rev (primary — higher is better)
            revenue_to_assets: Rev / TA
            gross_profit_per_asset: GP / TA
            opex_efficiency: Rev / OpEx
            asset_utilization: Rev / CA
            income_per_liability: OI / TL
        """
        result = OperationalEfficiencyResult()
        oi = data.operating_income
        rev = data.revenue
        ta = data.total_assets
        gp = data.gross_profit
        opex = data.operating_expenses
        ca = data.current_assets
        tl = data.total_liabilities

        # Primary: OI margin
        oim = safe_divide(oi, rev)
        result.oi_margin = oim

        # Secondary metrics
        result.revenue_to_assets = safe_divide(rev, ta)
        result.gross_profit_per_asset = safe_divide(gp, ta)
        result.opex_efficiency = safe_divide(rev, opex)
        result.asset_utilization = safe_divide(rev, ca)
        result.income_per_liability = safe_divide(oi, tl)

        # Scoring on OI margin (higher is better)
        if oim is None:
            result.oe_score = 0.0
            result.oe_grade = "Weak"
            result.summary = "Operational Efficiency: Insufficient data."
            return result

        if oim >= 0.25:
            base = 10.0
        elif oim >= 0.20:
            base = 8.5
        elif oim >= 0.15:
            base = 7.0
        elif oim >= 0.10:
            base = 5.5
        elif oim >= 0.05:
            base = 4.0
        elif oim >= 0.02:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Revenue-to-assets adjustment
        rta = result.revenue_to_assets
        if rta is not None:
            if rta >= 0.80:
                adj += 0.5
            elif rta < 0.30:
                adj -= 0.5

        # Opex efficiency adjustment
        oe = result.opex_efficiency
        if oe is not None:
            if oe >= 5.0:
                adj += 0.5
            elif oe < 2.0:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.oe_score = score

        if score >= 8:
            result.oe_grade = "Excellent"
        elif score >= 6:
            result.oe_grade = "Good"
        elif score >= 4:
            result.oe_grade = "Adequate"
        else:
            result.oe_grade = "Weak"

        oim_str = f"{oim:.2f}" if oim is not None else "N/A"
        result.summary = (
            f"Operational Efficiency: OI margin {oim_str}, "
            f"score {score:.1f}/10 ({result.oe_grade})."
        )
        return result

    def operating_momentum_analysis(self, data: FinancialData) -> OperatingMomentumResult:
        """Phase 191: Operating Momentum Analysis.

        Metrics:
            ebitda_margin: EBITDA / Rev (primary — higher is better)
            ebit_margin: EBIT / Rev
            ocf_margin: OCF / Rev
            gross_to_operating_conversion: OI / GP
            operating_cash_conversion: OCF / OI
            overhead_absorption: OI / OpEx
        """
        result = OperatingMomentumResult()
        rev = data.revenue
        ebitda = data.ebitda
        ebit = data.ebit
        ocf = data.operating_cash_flow
        gp = data.gross_profit
        oi = data.operating_income
        opex = data.operating_expenses

        # Primary: EBITDA margin
        em = safe_divide(ebitda, rev)
        result.ebitda_margin = em

        # Secondary metrics
        result.ebit_margin = safe_divide(ebit, rev)
        result.ocf_margin = safe_divide(ocf, rev)
        result.gross_to_operating_conversion = safe_divide(oi, gp)
        result.operating_cash_conversion = safe_divide(ocf, oi)
        result.overhead_absorption = safe_divide(oi, opex)

        # Scoring on EBITDA margin (higher is better)
        if em is None:
            result.om_score = 0.0
            result.om_grade = "Weak"
            result.summary = "Operating Momentum: Insufficient data."
            return result

        if em >= 0.30:
            base = 10.0
        elif em >= 0.25:
            base = 8.5
        elif em >= 0.20:
            base = 7.0
        elif em >= 0.15:
            base = 5.5
        elif em >= 0.10:
            base = 4.0
        elif em >= 0.05:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        # Gross-to-operating conversion adjustment
        gtoc = result.gross_to_operating_conversion
        if gtoc is not None:
            if gtoc >= 0.50:
                adj += 0.5
            elif gtoc < 0.25:
                adj -= 0.5

        # Operating cash conversion adjustment
        occ = result.operating_cash_conversion
        if occ is not None:
            if occ >= 1.0:
                adj += 0.5
            elif occ < 0.60:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.om_score = score

        if score >= 8:
            result.om_grade = "Excellent"
        elif score >= 6:
            result.om_grade = "Good"
        elif score >= 4:
            result.om_grade = "Adequate"
        else:
            result.om_grade = "Weak"

        em_str = f"{em:.2f}" if em is not None else "N/A"
        result.summary = (
            f"Operating Momentum: EBITDA margin {em_str}, "
            f"score {score:.1f}/10 ({result.om_grade})."
        )
        return result

    def payout_discipline_analysis(self, data: FinancialData) -> PayoutDisciplineResult:
        """Phase 185: Payout Discipline Analysis.

        Measures the sustainability and prudence of a company's dividend
        and capital return policies relative to cash generation.
        """
        result = PayoutDisciplineResult()

        ocf = data.operating_cash_flow
        div = data.dividends_paid
        ni = data.net_income
        capex = data.capex
        rev = data.revenue

        # --- Primary: Cash Dividend Coverage = OCF / Div ---
        result.cash_dividend_coverage = safe_divide(ocf, div)

        # --- Payout Ratio = Div / NI ---
        result.payout_ratio = safe_divide(div, ni)

        # --- Retention Ratio = (NI - Div) / NI ---
        if ni is not None and div is not None and ni > 0:
            result.retention_ratio = (ni - div) / ni
        else:
            result.retention_ratio = None

        # --- Dividend-to-OCF = Div / OCF ---
        result.dividend_to_ocf = safe_divide(div, ocf)

        # --- CapEx Priority = CapEx / (CapEx + Div) ---
        if capex is not None and div is not None and (capex + div) > 0:
            result.capex_priority = capex / (capex + div)
        else:
            result.capex_priority = None

        # --- Free Cash After Dividends = (OCF - CapEx - Div) / Rev ---
        if ocf is not None and capex is not None and div is not None and rev is not None and rev > 0:
            result.free_cash_after_dividends = (ocf - capex - div) / rev
        else:
            result.free_cash_after_dividends = None

        # --- Scoring (primary = Cash Dividend Coverage) ---
        cdc = result.cash_dividend_coverage
        if cdc is None:
            result.pd_score = 0.0
            result.pd_grade = "Weak"
            result.summary = "Payout Discipline: Insufficient data."
            return result

        if cdc >= 5.0:
            base = 10.0
        elif cdc >= 4.0:
            base = 8.5
        elif cdc >= 3.0:
            base = 7.0
        elif cdc >= 2.0:
            base = 5.5
        elif cdc >= 1.5:
            base = 4.0
        elif cdc >= 1.0:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        cp = result.capex_priority
        if cp is not None:
            if cp >= 0.60:
                adj += 0.5
            elif cp < 0.30:
                adj -= 0.5

        rr = result.retention_ratio
        if rr is not None:
            if rr >= 0.60:
                adj += 0.5
            elif rr < 0.30:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.pd_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pd_grade = grade

        cp_str = f"{cp:.2f}" if cp is not None else "N/A"
        rr_str = f"{rr:.2f}" if rr is not None else "N/A"
        result.summary = (
            f"Payout Discipline: Cash Div Coverage {cdc:.2f}x, "
            f"CapEx Priority {cp_str}, "
            f"Retention {rr_str}. "
            f"Score {score:.1f}/10. "
            f"Status: {grade}."
        )

        return result

    def income_resilience_analysis(self, data: FinancialData) -> IncomeResilienceResult:
        """Phase 184: Income Resilience Analysis.

        Measures how well a company's income stream withstands cost pressures,
        interest burden, and tax drag from operations to bottom line.
        """
        result = IncomeResilienceResult()

        oi = data.operating_income
        rev = data.revenue
        ebit = data.ebit
        ie = data.interest_expense
        ni = data.net_income
        da = data.depreciation
        ebitda = data.ebitda

        # --- Primary: Operating Income Stability = OI / Rev ---
        result.operating_income_stability = safe_divide(oi, rev)

        # --- EBIT Coverage = EBIT / IE ---
        result.ebit_coverage = safe_divide(ebit, ie)

        # --- Net Margin Resilience = NI / OI ---
        result.net_margin_resilience = safe_divide(ni, oi)

        # --- Depreciation Buffer = D&A / OI ---
        result.depreciation_buffer = safe_divide(da, oi)

        # --- Tax & Interest Drag = (OI - NI) / OI ---
        if oi is not None and ni is not None and oi > 0:
            result.tax_interest_drag = (oi - ni) / oi
        else:
            result.tax_interest_drag = None

        # --- EBITDA Cushion = EBITDA / IE ---
        result.ebitda_cushion = safe_divide(ebitda, ie)

        # --- Scoring (primary = Operating Income Stability) ---
        ois = result.operating_income_stability
        if ois is None:
            result.ir_score = 0.0
            result.ir_grade = "Weak"
            result.summary = "Income Resilience: Insufficient data."
            return result

        if ois >= 0.25:
            base = 10.0
        elif ois >= 0.20:
            base = 8.5
        elif ois >= 0.15:
            base = 7.0
        elif ois >= 0.10:
            base = 5.5
        elif ois >= 0.06:
            base = 4.0
        elif ois >= 0.03:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        ec = result.ebit_coverage
        if ec is not None:
            if ec >= 5.0:
                adj += 0.5
            elif ec < 2.0:
                adj -= 0.5

        nmr = result.net_margin_resilience
        if nmr is not None:
            if nmr >= 0.70:
                adj += 0.5
            elif nmr < 0.40:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.ir_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ir_grade = grade

        ec_str = f"{ec:.2f}x" if ec is not None else "N/A"
        nmr_str = f"{nmr:.2f}" if nmr is not None else "N/A"
        result.summary = (
            f"Income Resilience: OI Stability {ois:.2f}, "
            f"EBIT Coverage {ec_str}, "
            f"NM Resilience {nmr_str}. "
            f"Score {score:.1f}/10. "
            f"Status: {grade}."
        )

        return result

    def structural_strength_analysis(self, data: FinancialData) -> StructuralStrengthResult:
        """Phase 182: Structural Strength Analysis.

        Measures the structural soundness of a company's balance sheet
        through leverage, equity cushion, and liability composition.
        """
        result = StructuralStrengthResult()

        ta = data.total_assets
        te = data.total_equity
        td = data.total_debt
        tl = data.total_liabilities
        cl = data.current_liabilities
        ca = data.current_assets

        # --- Primary: Equity Multiplier = TA / TE (lower = stronger) ---
        result.equity_multiplier = safe_divide(ta, te)

        # --- Debt-to-Equity = TD / TE ---
        result.debt_to_equity = safe_divide(td, te)

        # --- Liability Composition = CL / TL ---
        result.liability_composition = safe_divide(cl, tl)

        # --- Equity Cushion = (TE - TD) / TA ---
        if te is not None and td is not None and ta is not None and ta > 0:
            result.equity_cushion = (te - td) / ta
        else:
            result.equity_cushion = None

        # --- Fixed Asset Coverage = TE / (TA - CA) ---
        if te is not None and ta is not None and ca is not None:
            fixed_assets = ta - ca
            result.fixed_asset_coverage = safe_divide(te, fixed_assets) if fixed_assets > 0 else None
        else:
            result.fixed_asset_coverage = None

        # --- Financial Leverage Ratio = TL / TE ---
        result.financial_leverage_ratio = safe_divide(tl, te)

        # --- Scoring (primary = Equity Multiplier, inverted) ---
        em = result.equity_multiplier
        if em is None:
            result.ss_score = 0.0
            result.ss_grade = "Weak"
            result.summary = "Structural Strength: Insufficient data."
            return result

        if em <= 1.25:
            base = 10.0
        elif em <= 1.50:
            base = 8.5
        elif em <= 1.75:
            base = 7.0
        elif em <= 2.00:
            base = 5.5
        elif em <= 2.50:
            base = 4.0
        elif em <= 3.50:
            base = 2.5
        else:
            base = 1.0

        adj = 0.0
        dte = result.debt_to_equity
        if dte is not None:
            if dte <= 0.50:
                adj += 0.5
            elif dte >= 2.00:
                adj -= 0.5

        ec = result.equity_cushion
        if ec is not None:
            if ec >= 0.30:
                adj += 0.5
            elif ec < 0.10:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.ss_score = score

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ss_grade = grade

        dte_str = f"{dte:.2f}x" if dte is not None else "N/A"
        ec_str = f"{ec:.2f}" if ec is not None else "N/A"
        result.summary = (
            f"Structural Strength: Equity Multiplier {em:.2f}x, "
            f"Debt-to-Equity {dte_str}, "
            f"Equity Cushion {ec_str}. "
            f"Score {score:.1f}/10. "
            f"Status: {grade}."
        )

        return result

    def profit_conversion_analysis(self, data: FinancialData) -> ProfitConversionResult:
        """Phase 179: Profit Conversion Analysis.

        Measures how effectively a company converts revenue into various
        levels of profit and cash, indicating operational discipline.
        """
        result = ProfitConversionResult()

        revenue = data.revenue
        gross_profit = data.gross_profit
        operating_income = data.operating_income
        net_income = data.net_income
        ebitda = data.ebitda
        operating_cash_flow = data.operating_cash_flow

        # Gross Conversion: GP / Revenue — primary
        result.gross_conversion = safe_divide(gross_profit, revenue)

        # Operating Conversion: OI / Revenue
        result.operating_conversion = safe_divide(operating_income, revenue)

        # Net Conversion: NI / Revenue
        result.net_conversion = safe_divide(net_income, revenue)

        # EBITDA Conversion: EBITDA / Revenue
        result.ebitda_conversion = safe_divide(ebitda, revenue)

        # Cash Conversion: OCF / Revenue
        result.cash_conversion = safe_divide(operating_cash_flow, revenue)

        # Profit-to-Cash Ratio: OCF / NI
        result.profit_to_cash_ratio = safe_divide(operating_cash_flow, net_income)

        # --- Scoring on Gross Conversion ---
        gc = result.gross_conversion
        if gc is not None:
            if gc >= 0.60:
                base = 10.0
            elif gc >= 0.50:
                base = 8.5
            elif gc >= 0.40:
                base = 7.0
            elif gc >= 0.30:
                base = 5.5
            elif gc >= 0.20:
                base = 4.0
            elif gc >= 0.10:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Operating Conversion
            oc = result.operating_conversion
            if oc is not None:
                if oc >= 0.20:
                    base += 0.5
                elif oc < 0.05:
                    base -= 0.5

            # Adjustment: Cash Conversion
            cc = result.cash_conversion
            if cc is not None:
                if cc >= 0.20:
                    base += 0.5
                elif cc < 0.05:
                    base -= 0.5

            result.pc_score = max(0.0, min(10.0, base))
        else:
            result.pc_score = 0.0

        # Grade
        if result.pc_score >= 8:
            result.pc_grade = "Excellent"
        elif result.pc_score >= 6:
            result.pc_grade = "Good"
        elif result.pc_score >= 4:
            result.pc_grade = "Adequate"
        else:
            result.pc_grade = "Weak"

        # Summary
        gc_str = f"{gc:.2%}" if gc is not None else "N/A"
        grade = result.pc_grade
        result.summary = (
            f"Profit Conversion score {result.pc_score:.1f}/10 ({grade}). "
            f"Gross Conversion {gc_str}."
        )

        return result

    def asset_deployment_efficiency_analysis(self, data: FinancialData) -> AssetDeploymentEfficiencyResult:
        """Phase 172: Asset Deployment Efficiency Analysis.

        Measures how efficiently assets are deployed to generate revenue,
        income, and cash flow returns.
        """
        result = AssetDeploymentEfficiencyResult()

        revenue = data.revenue
        total_assets = data.total_assets
        current_assets = data.current_assets
        operating_income = data.operating_income
        operating_cash_flow = data.operating_cash_flow
        inventory = data.inventory
        accounts_receivable = data.accounts_receivable

        # Asset Turnover: Rev / TA — primary
        result.asset_turnover = safe_divide(revenue, total_assets)

        # Fixed Asset Leverage: Rev / (TA - CA)
        if revenue is not None and total_assets is not None and current_assets is not None:
            fixed_assets = total_assets - current_assets
            if fixed_assets > 0:
                result.fixed_asset_leverage = revenue / fixed_assets

        # Asset Income Yield: OI / TA
        result.asset_income_yield = safe_divide(operating_income, total_assets)

        # Asset Cash Yield: OCF / TA
        result.asset_cash_yield = safe_divide(operating_cash_flow, total_assets)

        # Inventory Velocity: Rev / Inv
        result.inventory_velocity = safe_divide(revenue, inventory)

        # Receivables Velocity: Rev / AR
        result.receivables_velocity = safe_divide(revenue, accounts_receivable)

        # --- Scoring on Asset Turnover ---
        at = result.asset_turnover
        if at is not None:
            if at >= 1.50:
                base = 10.0
            elif at >= 1.20:
                base = 8.5
            elif at >= 0.90:
                base = 7.0
            elif at >= 0.60:
                base = 5.5
            elif at >= 0.40:
                base = 4.0
            elif at >= 0.20:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Asset Income Yield
            aiy = result.asset_income_yield
            if aiy is not None:
                if aiy >= 0.10:
                    base += 0.5
                elif aiy < 0.02:
                    base -= 0.5

            # Adjustment: Asset Cash Yield
            acy = result.asset_cash_yield
            if acy is not None:
                if acy >= 0.10:
                    base += 0.5
                elif acy < 0.02:
                    base -= 0.5

            result.ade_score = max(0.0, min(10.0, base))
        else:
            result.ade_score = 0.0

        # Grade
        if result.ade_score >= 8:
            result.ade_grade = "Excellent"
        elif result.ade_score >= 6:
            result.ade_grade = "Good"
        elif result.ade_score >= 4:
            result.ade_grade = "Adequate"
        else:
            result.ade_grade = "Weak"

        # Summary
        at_str = f"{at:.2f}" if at is not None else "N/A"
        result.summary = (
            f"Asset Deployment Efficiency score {result.ade_score:.1f}/10 ({result.ade_grade}). "
            f"Asset Turnover {at_str}."
        )

        return result

    def profit_sustainability_analysis(self, data: FinancialData) -> ProfitSustainabilityResult:
        """Phase 171: Profit Sustainability Analysis.

        Measures whether profits are sustainable, cash-backed, and efficiently
        retained for future growth.
        """
        result = ProfitSustainabilityResult()

        operating_cash_flow = data.operating_cash_flow
        net_income = data.net_income
        revenue = data.revenue
        dividends_paid = data.dividends_paid
        total_assets = data.total_assets
        operating_income = data.operating_income
        ebitda = data.ebitda

        # Profit Cash Backing: OCF / NI — primary
        result.profit_cash_backing = safe_divide(operating_cash_flow, net_income)

        # Profit Margin Depth: NI / Rev
        result.profit_margin_depth = safe_divide(net_income, revenue)

        # Profit Reinvestment: (NI - Div) / NI
        if net_income is not None and dividends_paid is not None and net_income != 0:
            result.profit_reinvestment = (net_income - dividends_paid) / net_income

        # Profit-to-Asset: NI / TA
        result.profit_to_asset = safe_divide(net_income, total_assets)

        # Profit Stability Proxy: OI / EBITDA
        result.profit_stability_proxy = safe_divide(operating_income, ebitda)

        # Profit Leverage: NI / OI
        result.profit_leverage = safe_divide(net_income, operating_income)

        # --- Scoring on Profit Cash Backing ---
        pcb = result.profit_cash_backing
        if pcb is not None:
            if pcb >= 1.50:
                base = 10.0
            elif pcb >= 1.20:
                base = 8.5
            elif pcb >= 1.00:
                base = 7.0
            elif pcb >= 0.80:
                base = 5.5
            elif pcb >= 0.50:
                base = 4.0
            elif pcb >= 0.20:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Profit Margin Depth
            pmd = result.profit_margin_depth
            if pmd is not None:
                if pmd >= 0.15:
                    base += 0.5
                elif pmd < 0.05:
                    base -= 0.5

            # Adjustment: Profit Reinvestment
            pri = result.profit_reinvestment
            if pri is not None:
                if pri >= 0.70:
                    base += 0.5
                elif pri < 0.30:
                    base -= 0.5

            result.ps_score = max(0.0, min(10.0, base))
        else:
            result.ps_score = 0.0

        # Grade
        if result.ps_score >= 8:
            result.ps_grade = "Excellent"
        elif result.ps_score >= 6:
            result.ps_grade = "Good"
        elif result.ps_score >= 4:
            result.ps_grade = "Adequate"
        else:
            result.ps_grade = "Weak"

        # Summary
        pcb_str = f"{pcb:.2f}" if pcb is not None else "N/A"
        result.summary = (
            f"Profit Sustainability score {result.ps_score:.1f}/10 ({result.ps_grade}). "
            f"Profit Cash Backing {pcb_str}."
        )

        return result

    def debt_discipline_analysis(self, data: FinancialData) -> DebtDisciplineResult:
        """Phase 170: Debt Discipline Analysis.

        Measures how responsibly a company manages its debt obligations,
        servicing capacity, and leverage prudence.
        """
        result = DebtDisciplineResult()

        total_debt = data.total_debt
        total_assets = data.total_assets
        ebitda = data.ebitda
        interest_expense = data.interest_expense
        operating_cash_flow = data.operating_cash_flow
        total_equity = data.total_equity
        revenue = data.revenue
        capex = data.capex

        # Debt Prudence Ratio: TD / TA — primary (lower is better)
        result.debt_prudence_ratio = safe_divide(total_debt, total_assets)

        # Debt Servicing Power: EBITDA / (IE + TD/5)
        if ebitda is not None and interest_expense is not None and total_debt is not None:
            denom = interest_expense + total_debt / 5.0
            if denom > 0:
                result.debt_servicing_power = ebitda / denom

        # Debt Coverage Spread: OCF / TD
        result.debt_coverage_spread = safe_divide(operating_cash_flow, total_debt)

        # Debt-to-Equity Leverage: TD / TE
        result.debt_to_equity_leverage = safe_divide(total_debt, total_equity)

        # Interest Absorption: IE / Rev (lower is better)
        result.interest_absorption = safe_divide(interest_expense, revenue)

        # Debt Repayment Capacity: (OCF - CapEx) / TD
        if operating_cash_flow is not None and capex is not None and total_debt is not None and total_debt != 0:
            result.debt_repayment_capacity = (operating_cash_flow - capex) / total_debt

        # --- Scoring on Debt Prudence Ratio (INVERTED: lower is better) ---
        dpr = result.debt_prudence_ratio
        if dpr is not None:
            if dpr <= 0.10:
                base = 10.0
            elif dpr <= 0.20:
                base = 8.5
            elif dpr <= 0.30:
                base = 7.0
            elif dpr <= 0.40:
                base = 5.5
            elif dpr <= 0.50:
                base = 4.0
            elif dpr <= 0.70:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Debt Coverage Spread
            dcs = result.debt_coverage_spread
            if dcs is not None:
                if dcs >= 0.50:
                    base += 0.5
                elif dcs < 0.10:
                    base -= 0.5

            # Adjustment: Debt-to-Equity Leverage
            del_ = result.debt_to_equity_leverage
            if del_ is not None:
                if del_ <= 0.30:
                    base += 0.5
                elif del_ >= 1.50:
                    base -= 0.5

            result.dd_score = max(0.0, min(10.0, base))
        else:
            result.dd_score = 0.0

        # Grade
        if result.dd_score >= 8:
            result.dd_grade = "Excellent"
        elif result.dd_score >= 6:
            result.dd_grade = "Good"
        elif result.dd_score >= 4:
            result.dd_grade = "Adequate"
        else:
            result.dd_grade = "Weak"

        # Summary
        dpr_str = f"{dpr:.1%}" if dpr is not None else "N/A"
        result.summary = (
            f"Debt Discipline score {result.dd_score:.1f}/10 ({result.dd_grade}). "
            f"Debt Prudence Ratio {dpr_str}."
        )

        return result

    def capital_preservation_analysis(self, data: FinancialData) -> CapitalPreservationResult:
        """Phase 168: Capital Preservation Analysis.

        Measures how well a company preserves and protects its capital base
        through retained earnings power, equity stability, and capital buffers.
        """
        result = CapitalPreservationResult()

        retained_earnings = data.retained_earnings
        total_assets = data.total_assets
        total_liabilities = data.total_liabilities
        total_equity = data.total_equity
        cash = data.cash
        operating_cash_flow = data.operating_cash_flow
        total_debt = data.total_debt
        net_income = data.net_income
        current_assets = data.current_assets
        current_liabilities = data.current_liabilities

        # Retained Earnings Power: RE / TA — primary
        result.retained_earnings_power = safe_divide(retained_earnings, total_assets)

        # Capital Erosion Rate: (TL - Cash) / TE
        if total_liabilities is not None and cash is not None and total_equity is not None and total_equity != 0:
            result.capital_erosion_rate = (total_liabilities - cash) / total_equity

        # Asset Integrity Ratio: (TA - TL) / TA
        if total_assets is not None and total_liabilities is not None and total_assets != 0:
            result.asset_integrity_ratio = (total_assets - total_liabilities) / total_assets

        # Operating Capital Ratio: OCF / TD
        result.operating_capital_ratio = safe_divide(operating_cash_flow, total_debt)

        # Net Worth Growth Proxy: NI / TE
        result.net_worth_growth_proxy = safe_divide(net_income, total_equity)

        # Capital Buffer: (CA - CL) / TA
        if current_assets is not None and current_liabilities is not None and total_assets is not None and total_assets != 0:
            result.capital_buffer = (current_assets - current_liabilities) / total_assets

        # --- Scoring on Retained Earnings Power ---
        rep = result.retained_earnings_power
        if rep is not None:
            if rep >= 0.40:
                base = 10.0
            elif rep >= 0.35:
                base = 8.5
            elif rep >= 0.30:
                base = 7.0
            elif rep >= 0.25:
                base = 5.5
            elif rep >= 0.20:
                base = 4.0
            elif rep >= 0.10:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Capital Erosion Rate
            cer = result.capital_erosion_rate
            if cer is not None:
                if cer <= 0.50:
                    base += 0.5
                elif cer >= 1.50:
                    base -= 0.5

            # Adjustment: Operating Capital Ratio
            ocr = result.operating_capital_ratio
            if ocr is not None:
                if ocr >= 0.50:
                    base += 0.5
                elif ocr < 0.10:
                    base -= 0.5

            result.cp_score = max(0.0, min(10.0, base))
        else:
            result.cp_score = 0.0

        # Grade
        if result.cp_score >= 8:
            result.cp_grade = "Excellent"
        elif result.cp_score >= 6:
            result.cp_grade = "Good"
        elif result.cp_score >= 4:
            result.cp_grade = "Adequate"
        else:
            result.cp_grade = "Weak"

        # Summary
        rep_str = f"{rep:.1%}" if rep is not None else "N/A"
        result.summary = (
            f"Capital Preservation score {result.cp_score:.1f}/10 ({result.cp_grade}). "
            f"Retained Earnings Power {rep_str}."
        )

        return result

    def obligation_coverage_analysis(self, data: FinancialData) -> ObligationCoverageResult:
        """Phase 160: Obligation Coverage Analysis.

        Evaluates the company's ability to meet its debt and fixed-charge
        obligations from operating performance.
        """
        result = ObligationCoverageResult()

        ebitda = data.ebitda
        interest_expense = data.interest_expense
        operating_cash_flow = data.operating_cash_flow
        capex = data.capex
        total_debt = data.total_debt
        revenue = data.revenue

        # EBITDA Interest Coverage: EBITDA / IE
        result.ebitda_interest_coverage = safe_divide(ebitda, interest_expense)

        # Cash Interest Coverage: OCF / IE
        result.cash_interest_coverage = safe_divide(operating_cash_flow, interest_expense)

        # Debt Amortization Capacity: (OCF - CapEx) / TD
        if operating_cash_flow is not None and capex is not None:
            fcf = operating_cash_flow - capex
            result.debt_amortization_capacity = safe_divide(fcf, total_debt)

        # Fixed Charge Coverage: EBITDA / (IE + CapEx)
        if ebitda is not None:
            ie_val = interest_expense if interest_expense is not None else 0.0
            cap_val = capex if capex is not None else 0.0
            denom = ie_val + cap_val
            if denom > 0:
                result.fixed_charge_coverage = ebitda / denom

        # Debt Burden Ratio: TD / EBITDA
        result.debt_burden_ratio = safe_divide(total_debt, ebitda)

        # Interest to Revenue: IE / Rev
        result.interest_to_revenue = safe_divide(interest_expense, revenue)

        # --- Scoring on EBITDA Interest Coverage ---
        eic = result.ebitda_interest_coverage
        if eic is not None:
            if eic >= 10.0:
                base = 10.0
            elif eic >= 7.0:
                base = 8.5
            elif eic >= 5.0:
                base = 7.0
            elif eic >= 3.0:
                base = 5.5
            elif eic >= 2.0:
                base = 4.0
            elif eic >= 1.0:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Debt Burden Ratio (lower is better)
            dbr = result.debt_burden_ratio
            if dbr is not None:
                if dbr <= 2.0:
                    base += 0.5
                elif dbr >= 5.0:
                    base -= 0.5

            # Adjustment: Cash Interest Coverage
            cic = result.cash_interest_coverage
            if cic is not None:
                if cic >= 8.0:
                    base += 0.5
                elif cic < 1.5:
                    base -= 0.5

            result.oc_score = max(0.0, min(10.0, base))
        else:
            result.oc_score = 0.0

        # Grade
        if result.oc_score >= 8:
            result.oc_grade = "Excellent"
        elif result.oc_score >= 6:
            result.oc_grade = "Good"
        elif result.oc_score >= 4:
            result.oc_grade = "Adequate"
        else:
            result.oc_grade = "Weak"

        result.summary = (
            f"Obligation Coverage: EBITDA Interest Coverage={eic:.2f}x "
            if eic is not None else "Obligation Coverage: EBITDA Interest Coverage=N/A "
        ) + f"Grade={result.oc_grade}."

        return result

    def internal_growth_capacity_analysis(self, data: FinancialData) -> InternalGrowthCapacityResult:
        """Phase 159: Internal Growth Capacity Analysis.

        Evaluates the company's ability to fund growth internally
        through retained earnings and operating cash flows.
        """
        result = InternalGrowthCapacityResult()

        net_income = data.net_income
        total_equity = data.total_equity
        total_assets = data.total_assets
        dividends_paid = data.dividends_paid
        operating_cash_flow = data.operating_cash_flow
        capex = data.capex
        depreciation = data.depreciation
        retained_earnings = data.retained_earnings

        # Plowback Ratio: (NI - Div) / NI
        if net_income is not None and net_income != 0:
            div = dividends_paid if dividends_paid is not None else 0.0
            result.plowback_ratio = (net_income - div) / net_income

        # Sustainable Growth Rate: ROE * Plowback
        roe = safe_divide(net_income, total_equity)
        if roe is not None and result.plowback_ratio is not None:
            result.sustainable_growth_rate = roe * result.plowback_ratio

        # Internal Growth Rate: ROA * b / (1 - ROA * b)
        roa = safe_divide(net_income, total_assets)
        if roa is not None and result.plowback_ratio is not None:
            roa_b = roa * result.plowback_ratio
            if roa_b < 1.0:
                result.internal_growth_rate = roa_b / (1.0 - roa_b)

        # Reinvestment Rate: CapEx / D&A
        result.reinvestment_rate = safe_divide(capex, depreciation)

        # Growth Financing Ratio: OCF / (CapEx + Div)
        if operating_cash_flow is not None:
            div_val = dividends_paid if dividends_paid is not None else 0.0
            cap_val = capex if capex is not None else 0.0
            denom = cap_val + div_val
            if denom > 0:
                result.growth_financing_ratio = operating_cash_flow / denom

        # Equity Growth Rate: RE / TE
        result.equity_growth_rate = safe_divide(retained_earnings, total_equity)

        # --- Scoring on Growth Financing Ratio ---
        gfr = result.growth_financing_ratio
        if gfr is not None:
            if gfr >= 2.5:
                base = 10.0
            elif gfr >= 2.0:
                base = 8.5
            elif gfr >= 1.5:
                base = 7.0
            elif gfr >= 1.2:
                base = 5.5
            elif gfr >= 1.0:
                base = 4.0
            elif gfr >= 0.5:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Plowback Ratio
            pb = result.plowback_ratio
            if pb is not None:
                if pb >= 0.60:
                    base += 0.5
                elif pb < 0.20:
                    base -= 0.5

            # Adjustment: Reinvestment Rate
            rr = result.reinvestment_rate
            if rr is not None:
                if 1.0 <= rr <= 2.5:
                    base += 0.5
                elif rr > 4.0:
                    base -= 0.5

            result.igc_score = max(0.0, min(10.0, base))
        else:
            result.igc_score = 0.0

        # Grade
        if result.igc_score >= 8:
            result.igc_grade = "Excellent"
        elif result.igc_score >= 6:
            result.igc_grade = "Good"
        elif result.igc_score >= 4:
            result.igc_grade = "Adequate"
        else:
            result.igc_grade = "Weak"

        result.summary = (
            f"Internal Growth Capacity: Growth Financing={gfr:.2f}x "
            if gfr is not None else "Internal Growth Capacity: Growth Financing=N/A "
        ) + f"Grade={result.igc_grade}."

        return result

    def liability_management_analysis(self, data: FinancialData) -> LiabilityManagementResult:
        """Phase 146: Liability Management Analysis."""
        result = LiabilityManagementResult()
        tl = data.total_liabilities
        ta = data.total_assets
        te = data.total_equity
        cl = data.current_liabilities
        ebitda = data.ebitda
        rev = data.revenue
        cash = data.cash

        result.liability_to_assets = safe_divide(tl, ta)
        result.liability_to_equity = safe_divide(tl, te)
        result.current_liability_ratio = safe_divide(cl, tl)
        result.liability_coverage = safe_divide(ebitda, tl)
        result.liability_to_revenue = safe_divide(tl, rev)

        # Net Liability = (TL - Cash) / TA
        if tl is not None and ta is not None and ta > 0:
            c = cash if cash is not None else 0.0
            result.net_liability = (tl - c) / ta

        # Scoring on Liability to Assets (lower = better)
        la = result.liability_to_assets
        if la is not None:
            if la <= 0.30:
                score = 10.0
            elif la <= 0.35:
                score = 8.5
            elif la <= 0.40:
                score = 7.0
            elif la <= 0.50:
                score = 5.5
            elif la <= 0.60:
                score = 4.0
            elif la <= 0.70:
                score = 2.5
            else:
                score = 1.0

            lc = result.liability_coverage
            if lc is not None:
                if lc >= 0.30:
                    score += 0.5
                elif lc < 0.10:
                    score -= 0.5

            clr = result.current_liability_ratio
            if clr is not None:
                if clr <= 0.40:
                    score += 0.5
                elif clr > 0.70:
                    score -= 0.5

            result.lm_score = max(0.0, min(10.0, score))
        else:
            result.lm_score = 0.0

        s = result.lm_score
        grade = "Excellent" if s >= 8 else "Good" if s >= 6 else "Adequate" if s >= 4 else "Weak"
        result.lm_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Liability Management — Liab/Assets: {_pct(result.liability_to_assets)}, "
            f"Liab/Equity: {_r2(result.liability_to_equity)}, "
            f"Coverage: {_r2(result.liability_coverage)}, "
            f"Net Liability: {_pct(result.net_liability)}. "
            f"Score: {result.lm_score}/10. Grade: {grade}."
        )

        return result

    def revenue_predictability_analysis(self, data: FinancialData) -> RevenuePredictabilityResult:
        """Phase 142: Revenue Predictability Analysis."""
        result = RevenuePredictabilityResult()

        rev = data.revenue
        ta = data.total_assets
        te = data.total_equity
        td = data.total_debt
        gp = data.gross_profit
        oi = data.operating_income
        ni = data.net_income

        result.revenue_to_assets = safe_divide(rev, ta)
        result.revenue_to_equity = safe_divide(rev, te)
        result.revenue_to_debt = safe_divide(rev, td)
        result.gross_margin = safe_divide(gp, rev)
        result.operating_margin = safe_divide(oi, rev)
        result.net_margin = safe_divide(ni, rev)

        # Scoring on Operating Margin
        om = result.operating_margin
        if om is not None:
            if om >= 0.30:
                score = 10.0
            elif om >= 0.25:
                score = 8.5
            elif om >= 0.20:
                score = 7.0
            elif om >= 0.15:
                score = 5.5
            elif om >= 0.10:
                score = 4.0
            elif om >= 0.05:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: Gross Margin
            gm = result.gross_margin
            if gm is not None:
                if gm >= 0.50:
                    score += 0.5
                elif gm < 0.20:
                    score -= 0.5

            # Adjustment: Net Margin
            nm = result.net_margin
            if nm is not None:
                if nm >= 0.15:
                    score += 0.5
                elif nm < 0.03:
                    score -= 0.5

            result.rp_score = max(0.0, min(10.0, round(score, 2)))
        else:
            result.rp_score = 0.0

        if result.rp_score >= 8:
            result.rp_grade = "Excellent"
        elif result.rp_score >= 6:
            result.rp_grade = "Good"
        elif result.rp_score >= 4:
            result.rp_grade = "Adequate"
        else:
            result.rp_grade = "Weak"

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Revenue Predictability — Grade: {result.rp_grade} ({result.rp_score}/10). "
            f"Op Margin={_pct(result.operating_margin)}, "
            f"Gross Margin={_pct(result.gross_margin)}, "
            f"Net Margin={_pct(result.net_margin)}, "
            f"Rev/Assets={_r2(result.revenue_to_assets)}."
        )

        return result

    def equity_reinvestment_analysis(self, data: FinancialData) -> EquityReinvestmentResult:
        """Phase 139: Equity Reinvestment Analysis."""
        result = EquityReinvestmentResult()

        ni = data.net_income
        div = data.dividends_paid
        capex = data.capex
        re = data.retained_earnings
        te = data.total_equity
        ta = data.total_assets

        # Retention ratio = (NI - Div) / NI
        if ni is not None and div is not None:
            retained = ni - div
            result.retention_ratio = safe_divide(retained, ni)
            result.plowback_to_assets = safe_divide(retained, ta)
        else:
            result.retention_ratio = safe_divide(ni, ni) if ni is not None and div is None else None

        result.reinvestment_rate = safe_divide(capex, ni)
        result.equity_growth_proxy = safe_divide(re, te)
        result.dividend_coverage = safe_divide(ni, div)

        # Internal growth rate = (RE/TE) * (NI/TE)
        egp = result.equity_growth_proxy
        roe = safe_divide(ni, te)
        if egp is not None and roe is not None:
            result.internal_growth_rate = egp * roe
        else:
            result.internal_growth_rate = None

        # Scoring on Retention Ratio
        rr = result.retention_ratio
        if rr is not None:
            if rr >= 0.90:
                score = 10.0
            elif rr >= 0.80:
                score = 8.5
            elif rr >= 0.70:
                score = 7.0
            elif rr >= 0.60:
                score = 5.5
            elif rr >= 0.50:
                score = 4.0
            elif rr >= 0.30:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: Dividend Coverage
            dc = result.dividend_coverage
            if dc is not None:
                if dc >= 3.0:
                    score += 0.5
                elif dc < 1.0:
                    score -= 0.5

            # Adjustment: Equity Growth Proxy
            if egp is not None:
                if egp >= 0.50:
                    score += 0.5
                elif egp < 0.20:
                    score -= 0.5

            result.er_score = max(0.0, min(10.0, round(score, 2)))
        else:
            result.er_score = 0.0

        if result.er_score >= 8:
            result.er_grade = "Excellent"
        elif result.er_score >= 6:
            result.er_grade = "Good"
        elif result.er_score >= 4:
            result.er_grade = "Adequate"
        else:
            result.er_grade = "Weak"

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Equity Reinvestment — Grade: {result.er_grade} ({result.er_score}/10). "
            f"Retention Ratio={_pct(result.retention_ratio)}, "
            f"Reinvestment Rate={_pct(result.reinvestment_rate)}, "
            f"Equity Growth Proxy={_pct(result.equity_growth_proxy)}, "
            f"Div Coverage={_r2(result.dividend_coverage)}."
        )

        return result

    def fixed_asset_efficiency_analysis(self, data: FinancialData) -> FixedAssetEfficiencyResult:
        """Phase 138: Fixed Asset Efficiency Analysis."""
        result = FixedAssetEfficiencyResult()

        ta = data.total_assets
        ca = data.current_assets
        rev = data.revenue
        te = data.total_equity
        dep = data.depreciation
        capex = data.capex

        # Fixed assets = total assets - current assets
        if ta is not None and ca is not None:
            fa = ta - ca
        else:
            fa = None

        result.fixed_asset_ratio = safe_divide(fa, ta)
        result.fixed_asset_turnover = safe_divide(rev, fa)
        result.fixed_to_equity = safe_divide(fa, te)
        result.fixed_asset_coverage = safe_divide(te, fa)
        result.depreciation_to_fixed = safe_divide(dep, fa)
        result.capex_to_fixed = safe_divide(capex, fa)

        # Scoring on Fixed Asset Turnover
        fat = result.fixed_asset_turnover
        if fat is not None:
            if fat >= 5.0:
                score = 10.0
            elif fat >= 3.0:
                score = 8.5
            elif fat >= 2.0:
                score = 7.0
            elif fat >= 1.5:
                score = 5.5
            elif fat >= 1.0:
                score = 4.0
            elif fat >= 0.5:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: CapEx to Fixed (reinvestment)
            ctf = result.capex_to_fixed
            if ctf is not None:
                if ctf >= 0.10:
                    score += 0.5
                elif ctf < 0.02:
                    score -= 0.5

            # Adjustment: Fixed Asset Coverage
            fac = result.fixed_asset_coverage
            if fac is not None:
                if fac >= 1.0:
                    score += 0.5
                elif fac < 0.3:
                    score -= 0.5

            result.fae_score = max(0.0, min(10.0, round(score, 2)))
        else:
            result.fae_score = 0.0

        if result.fae_score >= 8:
            result.fae_grade = "Excellent"
        elif result.fae_score >= 6:
            result.fae_grade = "Good"
        elif result.fae_score >= 4:
            result.fae_grade = "Adequate"
        else:
            result.fae_grade = "Weak"

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Fixed Asset Efficiency — Grade: {result.fae_grade} ({result.fae_score}/10). "
            f"FA Ratio={_pct(result.fixed_asset_ratio)}, "
            f"FA Turnover={_r2(result.fixed_asset_turnover)}, "
            f"FA Coverage={_r2(result.fixed_asset_coverage)}, "
            f"CapEx/FA={_pct(result.capex_to_fixed)}."
        )

        return result

    def income_stability_analysis(self, data: FinancialData) -> IncomeStabilityResult:
        """Phase 134: Income Stability Analysis.

        Metrics:
            - Net Income Margin = NI / Revenue
            - Retained Earnings Ratio = RE / TA
            - Operating Income Cushion = OI / IE
            - Net-to-Gross Ratio = NI / GP
            - EBITDA Margin = EBITDA / Revenue
            - Income Resilience = OCF / NI
        Scoring on Operating Income Cushion (OI / IE).
        """
        result = IncomeStabilityResult()

        ni_margin = safe_divide(data.net_income, data.revenue)
        result.net_income_margin = ni_margin

        re_ratio = safe_divide(data.retained_earnings, data.total_assets)
        result.retained_earnings_ratio = re_ratio

        oi_cushion = safe_divide(data.operating_income, data.interest_expense)
        result.operating_income_cushion = oi_cushion

        ntg = safe_divide(data.net_income, data.gross_profit)
        result.net_to_gross_ratio = ntg

        ebitda_m = safe_divide(data.ebitda, data.revenue)
        result.ebitda_margin = ebitda_m

        resilience = safe_divide(data.operating_cash_flow, data.net_income)
        result.income_resilience = resilience

        # --- Scoring on Operating Income Cushion ---
        if oi_cushion is None:
            result.is_score = 0.0
            result.is_grade = "Weak"
            result.summary = "Income Stability: Insufficient data."
            return result

        if oi_cushion >= 10.0:
            score = 10.0
        elif oi_cushion >= 7.0:
            score = 8.5
        elif oi_cushion >= 5.0:
            score = 7.0
        elif oi_cushion >= 3.0:
            score = 5.5
        elif oi_cushion >= 2.0:
            score = 4.0
        elif oi_cushion >= 1.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustments
        if ni_margin is not None:
            if ni_margin >= 0.15:
                score += 0.5
            elif ni_margin < 0.02:
                score -= 0.5

        if resilience is not None:
            if resilience >= 1.0:
                score += 0.5
            elif resilience < 0.5:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.is_score = score

        if score >= 8.0:
            result.is_grade = "Excellent"
        elif score >= 6.0:
            result.is_grade = "Good"
        elif score >= 4.0:
            result.is_grade = "Adequate"
        else:
            result.is_grade = "Weak"

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Income Stability: OI Cushion {_r2(oi_cushion)}x, "
            f"NI Margin {_pct(ni_margin)}, "
            f"Resilience {_r2(resilience)}x — "
            f"Score {score:.1f}/10 ({result.is_grade})"
        )

        return result

    def defensive_posture_analysis(self, data: FinancialData) -> DefensivePostureResult:
        """Phase 133: Defensive Posture Analysis.

        Metrics:
          - Defensive Interval = CA / (OpEx / 365)
          - Cash Ratio = Cash / CL
          - Quick Ratio = (CA - Inv) / CL
          - Cash Flow Coverage = OCF / TA
          - Equity Buffer = TE / TA
          - Debt Shield = EBITDA / TD
        """
        result = DefensivePostureResult()

        # Defensive Interval (days of coverage)
        if data.operating_expenses is not None and data.operating_expenses > 0:
            daily_opex = data.operating_expenses / 365
            result.defensive_interval = safe_divide(data.current_assets, daily_opex)

        # Cash Ratio
        result.cash_ratio = safe_divide(data.cash, data.current_liabilities)

        # Quick Ratio
        ca = data.current_assets
        inv = data.inventory or 0
        if ca is not None and data.current_liabilities is not None and data.current_liabilities > 0:
            result.quick_ratio = (ca - inv) / data.current_liabilities

        # Cash Flow Coverage
        result.cash_flow_coverage = safe_divide(data.operating_cash_flow, data.total_assets)

        # Equity Buffer
        result.equity_buffer = safe_divide(data.total_equity, data.total_assets)

        # Debt Shield
        result.debt_shield = safe_divide(data.ebitda, data.total_debt)

        # --- Scoring on Defensive Interval ---
        di = result.defensive_interval
        if di is not None:
            if di >= 365:
                base = 10.0
            elif di >= 270:
                base = 8.5
            elif di >= 180:
                base = 7.0
            elif di >= 120:
                base = 5.5
            elif di >= 90:
                base = 4.0
            elif di >= 30:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Cash Ratio
            cr = result.cash_ratio
            if cr is not None:
                if cr >= 0.50:
                    base += 0.5
                elif cr < 0.10:
                    base -= 0.5

            # Adjustment: Debt Shield
            ds = result.debt_shield
            if ds is not None:
                if ds >= 2.0:
                    base += 0.5
                elif ds < 0.5:
                    base -= 0.5

            result.dp_score = max(0.0, min(10.0, base))
        else:
            result.dp_score = 0.0

        # Grade
        s = result.dp_score
        if s >= 8:
            result.dp_grade = "Excellent"
        elif s >= 6:
            result.dp_grade = "Good"
        elif s >= 4:
            result.dp_grade = "Adequate"
        else:
            result.dp_grade = "Weak"

        # Summary
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        _r0 = lambda v: f"{v:.0f}" if v is not None else "N/A"
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        result.summary = (
            f"Defensive Posture — Score: {result.dp_score:.1f}/10 ({result.dp_grade}). "
            f"Defensive Interval: {_r0(result.defensive_interval)} days. "
            f"Cash Ratio: {_r2(result.cash_ratio)}. "
            f"Quick Ratio: {_r2(result.quick_ratio)}. "
            f"CF Coverage: {_pct(result.cash_flow_coverage)}. "
            f"Equity Buffer: {_pct(result.equity_buffer)}. "
            f"Debt Shield: {_r2(result.debt_shield)}x."
        )

        return result

    def funding_efficiency_analysis(self, data: FinancialData) -> FundingEfficiencyResult:
        """Phase 131: Funding Efficiency Analysis.

        Metrics:
          - Debt-to-Capitalization = TD / (TD + TE)
          - Equity Multiplier = TA / TE
          - Interest Coverage (EBITDA) = EBITDA / IE
          - Cost of Debt = IE / TD
          - Weighted Funding Cost = CoD * (TD/(TD+TE))
          - Funding Spread = ROA - Weighted Funding Cost
        """
        result = FundingEfficiencyResult()

        # Debt-to-Capitalization
        td = data.total_debt
        te = data.total_equity
        if td is not None and te is not None and (td + te) > 0:
            result.debt_to_capitalization = td / (td + te)

        # Equity Multiplier
        result.equity_multiplier = safe_divide(data.total_assets, data.total_equity)

        # Interest Coverage (EBITDA-based)
        result.interest_coverage_ebitda = safe_divide(data.ebitda, data.interest_expense)

        # Cost of Debt
        cod = safe_divide(data.interest_expense, data.total_debt)
        result.cost_of_debt = cod

        # Weighted Funding Cost
        if cod is not None and result.debt_to_capitalization is not None:
            result.weighted_funding_cost = cod * result.debt_to_capitalization

        # Funding Spread = ROA - Weighted Funding Cost
        roa = safe_divide(data.net_income, data.total_assets)
        if roa is not None and result.weighted_funding_cost is not None:
            result.funding_spread = roa - result.weighted_funding_cost

        # --- Scoring on Interest Coverage (EBITDA) ---
        ic = result.interest_coverage_ebitda
        if ic is not None:
            if ic >= 10.0:
                base = 10.0
            elif ic >= 7.0:
                base = 8.5
            elif ic >= 5.0:
                base = 7.0
            elif ic >= 3.0:
                base = 5.5
            elif ic >= 2.0:
                base = 4.0
            elif ic >= 1.0:
                base = 2.5
            else:
                base = 1.0

            # Adjustment: Debt-to-Capitalization
            dc = result.debt_to_capitalization
            if dc is not None:
                if dc <= 0.30:
                    base += 0.5
                elif dc > 0.60:
                    base -= 0.5

            # Adjustment: Funding Spread
            fs = result.funding_spread
            if fs is not None:
                if fs >= 0.05:
                    base += 0.5
                elif fs < 0.0:
                    base -= 0.5

            result.fe_score = max(0.0, min(10.0, base))
        else:
            result.fe_score = 0.0

        # Grade
        s = result.fe_score
        if s >= 8:
            result.fe_grade = "Excellent"
        elif s >= 6:
            result.fe_grade = "Good"
        elif s >= 4:
            result.fe_grade = "Adequate"
        else:
            result.fe_grade = "Weak"

        # Summary
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Funding Efficiency — Score: {result.fe_score:.1f}/10 ({result.fe_grade}). "
            f"Debt/Cap: {_pct(result.debt_to_capitalization)}. "
            f"Equity Multiplier: {_r2(result.equity_multiplier)}x. "
            f"EBITDA Coverage: {_r2(result.interest_coverage_ebitda)}x. "
            f"Cost of Debt: {_pct(result.cost_of_debt)}. "
            f"Weighted Cost: {_pct(result.weighted_funding_cost)}. "
            f"Funding Spread: {_pct(result.funding_spread)}."
        )

        return result

    def cash_flow_stability_analysis(self, data: FinancialData) -> CashFlowStabilityResult:
        """Phase 125: Cash Flow Stability Analysis.

        Metrics:
        - OCF Margin = OCF / Revenue
        - OCF/EBITDA = Operating Cash Flow / EBITDA
        - OCF/Debt Service = OCF / (Interest Expense + Debt Repayment)
        - CapEx/OCF = CapEx / OCF
        - Dividend Coverage = OCF / Dividends
        - Cash Flow Sufficiency = OCF / (CapEx + Dividends + Interest)
        """
        result = CashFlowStabilityResult()

        ocf = getattr(data, "operating_cash_flow", None)
        revenue = getattr(data, "revenue", None)
        ebitda = getattr(data, "ebitda", None)
        interest_expense = getattr(data, "interest_expense", None)
        capex = getattr(data, "capex", None)
        dividends = getattr(data, "dividends_paid", None)

        has_ocf = ocf is not None and ocf > 0
        has_rev = revenue is not None and revenue > 0

        # OCF Margin = OCF / Revenue
        if ocf is not None and has_rev:
            result.ocf_margin = safe_divide(ocf, revenue)

        # OCF / EBITDA
        if ocf is not None and ebitda is not None and ebitda > 0:
            result.ocf_to_ebitda = safe_divide(ocf, ebitda)

        # OCF / Debt Service (Interest)
        if has_ocf and interest_expense is not None and interest_expense > 0:
            result.ocf_to_debt_service = safe_divide(ocf, interest_expense)

        # CapEx / OCF
        if capex is not None and has_ocf:
            result.capex_to_ocf = safe_divide(capex, ocf)

        # Dividend Coverage = OCF / Dividends
        if has_ocf and dividends is not None and dividends > 0:
            result.dividend_coverage = safe_divide(ocf, dividends)

        # Cash Flow Sufficiency = OCF / (CapEx + Dividends + Interest)
        if has_ocf:
            cap = capex if capex is not None else 0.0
            div = dividends if dividends is not None else 0.0
            ie = interest_expense if interest_expense is not None else 0.0
            total_needs = cap + div + ie
            if total_needs > 0:
                result.cash_flow_sufficiency = safe_divide(ocf, total_needs)

        # --- Scoring based on OCF Margin ---
        om = result.ocf_margin
        if om is not None:
            if om >= 0.25:
                score = 10.0
            elif om >= 0.20:
                score = 8.5
            elif om >= 0.15:
                score = 7.0
            elif om >= 0.10:
                score = 5.5
            elif om >= 0.05:
                score = 4.0
            elif om >= 0.0:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: OCF/EBITDA
            oe = result.ocf_to_ebitda
            if oe is not None:
                if oe >= 0.90:
                    score += 0.5
                elif oe < 0.50:
                    score -= 0.5

            # Adjustment: Cash Flow Sufficiency
            cfs = result.cash_flow_sufficiency
            if cfs is not None:
                if cfs >= 1.5:
                    score += 0.5
                elif cfs < 1.0:
                    score -= 0.5

            score = max(0.0, min(10.0, score))
            result.cfs_score = round(score, 1)
        else:
            result.cfs_score = 0.0

        # Grade
        s = result.cfs_score
        if s >= 8.0:
            result.cfs_grade = "Excellent"
        elif s >= 6.0:
            result.cfs_grade = "Good"
        elif s >= 4.0:
            result.cfs_grade = "Adequate"
        else:
            result.cfs_grade = "Weak"

        # Summary
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Cash Flow Stability: OCF Margin={_pct(result.ocf_margin)}, "
            f"OCF/EBITDA={_r2(result.ocf_to_ebitda)}, "
            f"Sufficiency={_r2(result.cash_flow_sufficiency)} | "
            f"Score: {result.cfs_score}/10 ({result.cfs_grade})"
        )

        return result

    def income_quality_analysis(self, data: FinancialData) -> IncomeQualityResult:
        """Phase 124: Income Quality Analysis.

        Metrics:
        - OCF/Net Income = Operating Cash Flow / Net Income
        - Accruals Ratio = (Net Income - OCF) / Total Assets
        - Cash Earnings Ratio = OCF / EBITDA
        - Non-Cash Ratio = Depreciation / Net Income
        - Earnings Persistence = Operating Income / Revenue
        - Operating Income Ratio = Operating Income / Net Income
        """
        result = IncomeQualityResult()

        ocf = getattr(data, "operating_cash_flow", None)
        net_income = getattr(data, "net_income", None)
        total_assets = getattr(data, "total_assets", None)
        ebitda = getattr(data, "ebitda", None)
        depreciation = getattr(data, "depreciation", None)
        operating_income = getattr(data, "operating_income", None)
        revenue = getattr(data, "revenue", None)

        has_ni = net_income is not None and net_income > 0
        has_rev = revenue is not None and revenue > 0

        # OCF / Net Income
        if ocf is not None and has_ni:
            result.ocf_to_net_income = safe_divide(ocf, net_income)

        # Accruals Ratio = (NI - OCF) / TA
        if has_ni and ocf is not None and total_assets is not None and total_assets > 0:
            result.accruals_ratio = safe_divide(net_income - ocf, total_assets)

        # Cash Earnings Ratio = OCF / EBITDA
        if ocf is not None and ebitda is not None and ebitda > 0:
            result.cash_earnings_ratio = safe_divide(ocf, ebitda)

        # Non-Cash Ratio = Depreciation / NI
        if depreciation is not None and has_ni:
            result.non_cash_ratio = safe_divide(depreciation, net_income)

        # Earnings Persistence = OI / Revenue
        if operating_income is not None and has_rev:
            result.earnings_persistence = safe_divide(operating_income, revenue)

        # Operating Income Ratio = OI / NI
        if operating_income is not None and has_ni:
            result.operating_income_ratio = safe_divide(operating_income, net_income)

        # --- Scoring based on OCF/NI ---
        ocf_ni = result.ocf_to_net_income
        if ocf_ni is not None:
            if ocf_ni >= 1.5:
                score = 10.0
            elif ocf_ni >= 1.2:
                score = 8.5
            elif ocf_ni >= 1.0:
                score = 7.0
            elif ocf_ni >= 0.8:
                score = 5.5
            elif ocf_ni >= 0.5:
                score = 4.0
            elif ocf_ni >= 0.0:
                score = 2.5
            else:
                score = 1.0

            # Adjustment: Accruals Ratio
            ar = result.accruals_ratio
            if ar is not None:
                if ar <= -0.05:
                    score += 0.5
                elif ar > 0.10:
                    score -= 0.5

            # Adjustment: Cash Earnings Ratio
            cer = result.cash_earnings_ratio
            if cer is not None:
                if cer >= 0.90:
                    score += 0.5
                elif cer < 0.50:
                    score -= 0.5

            score = max(0.0, min(10.0, score))
            result.iq_score = round(score, 1)
        else:
            result.iq_score = 0.0

        # Grade
        s = result.iq_score
        if s >= 8.0:
            result.iq_grade = "Excellent"
        elif s >= 6.0:
            result.iq_grade = "Good"
        elif s >= 4.0:
            result.iq_grade = "Adequate"
        else:
            result.iq_grade = "Weak"

        # Summary
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        result.summary = (
            f"Income Quality: OCF/NI={_r2(result.ocf_to_net_income)}, "
            f"Accruals={_pct(result.accruals_ratio)}, "
            f"Cash Earnings={_r2(result.cash_earnings_ratio)} | "
            f"Score: {result.iq_score}/10 ({result.iq_grade})"
        )

        return result

    def receivables_management_analysis(self, data: FinancialData) -> ReceivablesManagementResult:
        """Phase 114: Receivables Management Analysis."""
        result = ReceivablesManagementResult()

        rev = data.revenue
        ar = data.accounts_receivable
        ca = data.current_assets
        ni = data.net_income or 0

        if not rev or not ar:
            return result

        # DSO = AR / Revenue * 365
        result.dso = safe_divide(ar * 365, rev)

        # AR / Revenue
        result.ar_to_revenue = safe_divide(ar, rev)

        # AR / Current Assets
        result.ar_to_current_assets = safe_divide(ar, ca) if ca else None

        # Receivables Turnover = Revenue / AR
        result.receivables_turnover = safe_divide(rev, ar)

        # Collection Effectiveness = (Rev - AR ending) / Rev
        result.collection_effectiveness = safe_divide(rev - ar, rev)

        # AR Concentration = AR / (AR + Cash)
        cash = data.cash or 0
        denom_arc = ar + cash
        result.ar_concentration = safe_divide(ar, denom_arc) if denom_arc else None

        # Scoring on DSO
        dso = result.dso
        if dso is not None:
            if dso <= 30:
                score = 10.0
            elif dso <= 45:
                score = 8.5
            elif dso <= 60:
                score = 7.0
            elif dso <= 75:
                score = 5.5
            elif dso <= 90:
                score = 4.0
            elif dso <= 120:
                score = 2.5
            else:
                score = 1.0
        else:
            return result

        # Adjustments
        rt = result.receivables_turnover
        if rt is not None:
            if rt >= 12:
                score += 0.5
            elif rt < 4:
                score -= 0.5

        ce = result.collection_effectiveness
        if ce is not None:
            if ce >= 0.90:
                score += 0.5
            elif ce < 0.70:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.rm_score = score

        if score >= 8:
            result.rm_grade = "Excellent"
        elif score >= 6:
            result.rm_grade = "Good"
        elif score >= 4:
            result.rm_grade = "Adequate"
        else:
            result.rm_grade = "Weak"

        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        _d = lambda v: f"{v:.1f}" if v is not None else "N/A"
        result.summary = (
            f"Receivables Management: DSO={_d(result.dso)} days, "
            f"AR/Rev={_r2(result.ar_to_revenue)}, "
            f"Turnover={_r2(result.receivables_turnover)}. "
            f"Grade: {result.rm_grade}."
        )

        return result

    def solvency_depth_analysis(self, data: FinancialData) -> SolvencyDepthResult:
        """Phase 109: Solvency Depth Analysis."""
        result = SolvencyDepthResult()
        td = data.total_debt or 0
        te = data.total_equity or 0
        ta = data.total_assets or 0
        ebit = data.ebit or 0
        ie = data.interest_expense or 0
        ebitda = data.ebitda or 0

        if ta == 0 and te == 0:
            return result

        # Debt-to-Equity = TD / TE
        result.debt_to_equity = safe_divide(td, te) if te > 0 else None

        # Debt-to-Assets = TD / TA
        result.debt_to_assets = safe_divide(td, ta) if ta > 0 else None

        # Equity-to-Assets = TE / TA
        result.equity_to_assets = safe_divide(te, ta) if ta > 0 else None

        # Interest Coverage = EBIT / IE
        result.interest_coverage_ratio = safe_divide(ebit, ie) if ie > 0 else None

        # Debt-to-EBITDA = TD / EBITDA
        result.debt_to_ebitda = safe_divide(td, ebitda) if ebitda > 0 else None

        # Financial Leverage = TA / TE
        result.financial_leverage = safe_divide(ta, te) if te > 0 else None

        # Scoring based on Debt-to-EBITDA
        de = result.debt_to_ebitda
        if de is not None:
            if de <= 1.0:
                score = 10.0
            elif de <= 2.0:
                score = 8.5
            elif de <= 3.0:
                score = 7.0
            elif de <= 4.0:
                score = 5.5
            elif de <= 5.0:
                score = 4.0
            elif de <= 6.0:
                score = 2.5
            else:
                score = 1.0
        else:
            # No EBITDA, fall back to D/A
            da = result.debt_to_assets
            if da is not None:
                if da <= 0.30:
                    score = 8.0
                elif da <= 0.50:
                    score = 5.0
                else:
                    score = 2.0
            else:
                score = 0.0

        # Adjustments
        dte = result.debt_to_equity
        if dte is not None:
            if dte <= 1.0:
                score += 0.5
            elif dte > 3.0:
                score -= 0.5

        ic = result.interest_coverage_ratio
        if ic is not None:
            if ic >= 5.0:
                score += 0.5
            elif ic < 2.0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.sd_score = round(score, 2)

        if score >= 8:
            result.sd_grade = "Excellent"
        elif score >= 6:
            result.sd_grade = "Good"
        elif score >= 4:
            result.sd_grade = "Adequate"
        else:
            result.sd_grade = "Weak"

        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Solvency Depth: {result.sd_grade} ({result.sd_score}/10). "
            f"D/EBITDA={_r2(result.debt_to_ebitda)}, "
            f"D/E={_r2(result.debt_to_equity)}, "
            f"IC={_r2(result.interest_coverage_ratio)}."
        )

        return result

    def operational_leverage_depth_analysis(self, data: FinancialData) -> OperationalLeverageDepthResult:
        """Phase 105: Operational Leverage Depth Analysis."""
        result = OperationalLeverageDepthResult()
        rev = data.revenue or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        oi = data.operating_income or 0
        gp = data.gross_profit or 0
        da = data.depreciation or 0

        if rev <= 0:
            return result

        total_costs = cogs + opex
        if total_costs <= 0:
            return result

        # Fixed Cost Ratio = (OpEx + D&A) / Total Costs (proxy: opex as fixed)
        fixed_proxy = opex + da
        result.fixed_cost_ratio = safe_divide(fixed_proxy, total_costs)

        # Variable Cost Ratio = COGS / Total Costs
        result.variable_cost_ratio = safe_divide(cogs, total_costs)

        # Contribution Margin = (Rev - COGS) / Rev = GP / Rev
        result.contribution_margin = safe_divide(gp, rev) if gp else safe_divide(rev - cogs, rev)

        # DOL Proxy = GP / OI (if both available)
        result.dol_proxy = safe_divide(gp, oi) if oi and oi != 0 else None

        # Breakeven Coverage = Rev / (Rev - OI) = Rev / Total Costs
        result.breakeven_coverage = safe_divide(rev, total_costs) if total_costs > 0 else None

        # Cost Flexibility = Variable Costs / Total Costs (higher = more flexible)
        result.cost_flexibility = safe_divide(cogs, total_costs)

        # Scoring based on contribution margin
        cm = result.contribution_margin
        if cm is not None:
            if cm >= 0.60:
                score = 10.0
            elif cm >= 0.50:
                score = 8.5
            elif cm >= 0.40:
                score = 7.0
            elif cm >= 0.30:
                score = 5.5
            elif cm >= 0.20:
                score = 4.0
            elif cm >= 0.10:
                score = 2.5
            else:
                score = 1.0
        else:
            score = 3.0

        # Adjustments
        bc = result.breakeven_coverage
        if bc is not None:
            if bc >= 1.5:
                score += 0.5
            elif bc < 1.0:
                score -= 0.5

        cf = result.cost_flexibility
        if cf is not None:
            if cf >= 0.60:
                score += 0.5
            elif cf < 0.30:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.old_score = round(score, 2)

        if score >= 8:
            result.old_grade = "Excellent"
        elif score >= 6:
            result.old_grade = "Good"
        elif score >= 4:
            result.old_grade = "Adequate"
        else:
            result.old_grade = "Weak"

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        _r2 = lambda v: f"{v:.2f}" if v is not None else "N/A"
        result.summary = (
            f"Operational Leverage Depth: {result.old_grade} ({result.old_score}/10). "
            f"Contribution margin={_pct(result.contribution_margin)}, "
            f"Breakeven coverage={_r2(result.breakeven_coverage)}, "
            f"Cost flexibility={_pct(result.cost_flexibility)}."
        )

        return result

    def profitability_depth_analysis(self, data: FinancialData) -> ProfitabilityDepthResult:
        """Phase 103: Profitability Depth Analysis."""
        result = ProfitabilityDepthResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        gp = data.gross_profit or 0
        cogs = data.cogs or 0
        oi = data.operating_income or 0
        ebitda = data.ebitda or 0
        ni = data.net_income or 0

        # 1. Gross Margin
        gp_calc = gp if gp > 0 else (rev - cogs) if cogs > 0 else None
        if gp_calc is not None:
            result.gross_margin = round(safe_divide(gp_calc, rev), 4)

        # 2. Operating Margin
        if oi != 0 or (data.operating_income is not None):
            result.operating_margin = round(safe_divide(oi, rev), 4)

        # 3. EBITDA Margin
        if ebitda != 0 or (data.ebitda is not None):
            result.ebitda_margin = round(safe_divide(ebitda, rev), 4)

        # 4. Net Margin
        result.net_margin = round(safe_divide(ni, rev), 4)

        # 5. Margin Spread = Gross Margin - Net Margin (cost/tax leakage)
        if result.gross_margin is not None and result.net_margin is not None:
            result.margin_spread = round(result.gross_margin - result.net_margin, 4)

        # 6. Profit Retention Ratio = NI / GP (what fraction of gross profit survives)
        if gp_calc is not None and gp_calc > 0:
            result.profit_retention_ratio = round(safe_divide(ni, gp_calc), 4)

        # Scoring: based on Operating Margin (core profitability)
        om = result.operating_margin
        if om is not None:
            if om >= 0.25:
                score = 10.0
            elif om >= 0.20:
                score = 8.5
            elif om >= 0.15:
                score = 7.0
            elif om >= 0.10:
                score = 5.5
            elif om >= 0.05:
                score = 4.0
            elif om >= 0.0:
                score = 2.5
            else:
                score = 1.0
        elif result.net_margin is not None:
            nm = result.net_margin
            if nm >= 0.15:
                score = 8.0
            elif nm >= 0.05:
                score = 5.0
            else:
                score = 2.0
        else:
            return result

        # Adjustment: Gross Margin
        gm = result.gross_margin
        if gm is not None:
            if gm >= 0.50:
                score += 0.5
            elif gm < 0.20:
                score -= 0.5

        # Adjustment: Profit Retention
        prr = result.profit_retention_ratio
        if prr is not None:
            if prr >= 0.40:
                score += 0.5
            elif prr < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.pd_score = round(score, 2)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pd_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        result.summary = (
            f"Profitability Depth: OM={_pct(result.operating_margin)}, "
            f"NM={_pct(result.net_margin)}, GM={_pct(result.gross_margin)}. "
            f"Grade: {grade} ({result.pd_score}/10)."
        )

        return result

    def revenue_efficiency_analysis(self, data: FinancialData) -> RevenueEfficiencyResult:
        """Phase 102: Revenue Efficiency Analysis."""
        result = RevenueEfficiencyResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        ta = data.total_assets or 0
        te = data.total_equity or 0
        gp = data.gross_profit or 0
        oi = data.operating_income or 0
        ni = data.net_income or 0
        ocf = data.operating_cash_flow or 0
        cogs = data.cogs or 0

        # 1. Revenue per Asset = Rev / TA
        if ta > 0:
            result.revenue_per_asset = round(safe_divide(rev, ta), 4)

        # 2. Cash Conversion Efficiency = OCF / Rev
        result.cash_conversion_efficiency = round(safe_divide(ocf, rev), 4)

        # 3. Gross Margin Efficiency = GP / Rev
        if gp > 0 or cogs > 0:
            gp_calc = gp if gp > 0 else (rev - cogs)
            result.gross_margin_efficiency = round(safe_divide(gp_calc, rev), 4)

        # 4. Operating Leverage Ratio = OI / GP
        if gp > 0:
            result.operating_leverage_ratio = round(safe_divide(oi, gp), 4)
        elif cogs > 0:
            gp_calc = rev - cogs
            if gp_calc > 0:
                result.operating_leverage_ratio = round(safe_divide(oi, gp_calc), 4)

        # 5. Revenue to Equity = Rev / TE
        if te > 0:
            result.revenue_to_equity = round(safe_divide(rev, te), 4)

        # 6. Net Revenue Retention = NI / Rev (net margin as efficiency proxy)
        result.net_revenue_retention = round(safe_divide(ni, rev), 4)

        # Scoring: based on Cash Conversion Efficiency (higher = better)
        cce = result.cash_conversion_efficiency
        if cce is not None:
            if cce >= 0.25:
                score = 10.0
            elif cce >= 0.20:
                score = 8.5
            elif cce >= 0.15:
                score = 7.0
            elif cce >= 0.10:
                score = 5.5
            elif cce >= 0.05:
                score = 4.0
            elif cce >= 0.0:
                score = 2.5
            else:
                score = 1.0
        else:
            return result

        # Adjustment: Gross Margin Efficiency
        gme = result.gross_margin_efficiency
        if gme is not None:
            if gme >= 0.50:
                score += 0.5
            elif gme < 0.20:
                score -= 0.5

        # Adjustment: Operating Leverage
        olr = result.operating_leverage_ratio
        if olr is not None:
            if olr >= 0.50:
                score += 0.5
            elif olr < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.rev_eff_score = round(score, 2)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.rev_eff_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        result.summary = (
            f"Revenue Efficiency: CCE={_pct(result.cash_conversion_efficiency)}, "
            f"GM={_pct(result.gross_margin_efficiency)}. "
            f"Grade: {grade} ({result.rev_eff_score}/10)."
        )

        return result

    def debt_composition_analysis(self, data: FinancialData) -> DebtCompositionResult:
        """Phase 101: Debt Composition Analysis."""
        result = DebtCompositionResult()
        td = data.total_debt or 0
        te = data.total_equity or 0
        ta = data.total_assets or 0
        ie = abs(data.interest_expense or 0)
        ebit = data.ebit or 0
        ni = data.net_income or 0
        ocf = data.operating_cash_flow or 0
        tl = data.total_liabilities or 0

        if td <= 0 and tl <= 0:
            # No debt => perfect score
            result.dco_score = 10.0
            result.dco_grade = "Excellent"
            result.summary = "Debt Composition: No debt — Excellent (10.0/10)."
            return result

        # 1. Debt-to-Equity = TD / TE
        if te > 0:
            result.debt_to_equity = round(safe_divide(td, te), 4)

        # 2. Debt-to-Assets = TD / TA
        if ta > 0:
            result.debt_to_assets = round(safe_divide(td, ta), 4)

        # 3. Long-term Debt Ratio = (TD - CL_portion) / TD approximated as (TL - CL) / TL
        cl = data.current_liabilities or 0
        if tl > 0:
            long_term = max(0, tl - cl)
            result.long_term_debt_ratio = round(safe_divide(long_term, tl), 4)

        # 4. Interest Burden = IE / EBIT
        if ebit > 0 and ie > 0:
            result.interest_burden = round(safe_divide(ie, ebit), 4)

        # 5. Debt Cost Ratio = IE / TD
        if td > 0 and ie > 0:
            result.debt_cost_ratio = round(safe_divide(ie, td), 4)

        # 6. Debt Coverage Margin = (OCF - IE) / TD
        if td > 0:
            result.debt_coverage_margin = round(safe_divide(ocf - ie, td), 4)

        # Scoring: based on Debt-to-Equity (lower = better)
        de = result.debt_to_equity
        if de is not None:
            if de <= 0.30:
                score = 10.0
            elif de <= 0.50:
                score = 8.5
            elif de <= 1.00:
                score = 7.0
            elif de <= 1.50:
                score = 5.5
            elif de <= 2.00:
                score = 4.0
            elif de <= 3.00:
                score = 2.5
            else:
                score = 1.0
        elif result.debt_to_assets is not None:
            da = result.debt_to_assets
            if da <= 0.20:
                score = 10.0
            elif da <= 0.40:
                score = 7.0
            elif da <= 0.60:
                score = 4.0
            else:
                score = 1.0
        else:
            return result

        # Adjustment: Interest Burden
        ib = result.interest_burden
        if ib is not None:
            if ib <= 0.10:
                score += 0.5
            elif ib > 0.50:
                score -= 0.5

        # Adjustment: Debt Coverage Margin
        dcm = result.debt_coverage_margin
        if dcm is not None:
            if dcm >= 0.30:
                score += 0.5
            elif dcm < 0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.dco_score = round(score, 2)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.dco_grade = grade

        _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
        de_str = f"{result.debt_to_equity:.2f}x" if result.debt_to_equity is not None else "N/A"
        result.summary = (
            f"Debt Composition: D/E={de_str}, "
            f"D/A={_pct(result.debt_to_assets)}. "
            f"Grade: {grade} ({result.dco_score}/10)."
        )

        return result

    def operational_risk_analysis(self, data: FinancialData) -> OperationalRiskResult:
        """Phase 92: Operational Risk Analysis."""
        result = OperationalRiskResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        gp = data.gross_profit or 0
        oi = data.operating_income or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        ocf = data.operating_cash_flow or 0
        ni = data.net_income or 0
        ta = data.total_assets or 0

        # Operating leverage: GP / OI (how much GP translates to OI)
        result.operating_leverage = safe_divide(gp, oi) if oi != 0 else None

        # Cost rigidity: fixed costs (opex) as proportion of total costs
        total_costs = cogs + opex
        result.cost_rigidity = safe_divide(opex, total_costs) if total_costs > 0 else None

        # Breakeven ratio: total costs / revenue
        result.breakeven_ratio = safe_divide(total_costs, rev)

        # Margin of safety: (Rev - breakeven) / Rev ≈ 1 - breakeven_ratio
        be = result.breakeven_ratio
        if be is not None:
            result.margin_of_safety = 1.0 - be

        # Cash burn ratio: OCF / total costs (how well cash covers costs)
        result.cash_burn_ratio = safe_divide(ocf, total_costs) if total_costs > 0 else None

        # Risk buffer: (OCF + cash) / opex
        cash = data.cash or 0
        result.risk_buffer = safe_divide(ocf + cash, opex) if opex > 0 else None

        # Score on margin_of_safety (higher = less risk)
        mos = result.margin_of_safety
        if mos is not None:
            if mos >= 0.30:
                score = 10.0
            elif mos >= 0.25:
                score = 8.5
            elif mos >= 0.20:
                score = 7.0
            elif mos >= 0.15:
                score = 5.5
            elif mos >= 0.10:
                score = 4.0
            elif mos >= 0.05:
                score = 2.5
            else:
                score = 1.0

            rb = result.risk_buffer
            if rb is not None:
                if rb >= 2.0:
                    score += 0.5
                elif rb < 0.5:
                    score -= 0.5

            cr = result.cost_rigidity
            if cr is not None:
                if cr <= 0.20:
                    score += 0.5
                elif cr > 0.50:
                    score -= 0.5

            score = max(0.0, min(10.0, score))
            result.or_score = round(score, 2)

            if score >= 8:
                result.or_grade = "Excellent"
            elif score >= 6:
                result.or_grade = "Good"
            elif score >= 4:
                result.or_grade = "Adequate"
            else:
                result.or_grade = "Weak"

            def _pct(v):
                return f"{v:.1%}" if v is not None else "N/A"

            result.summary = (
                f"Operational Risk: Margin of safety {_pct(mos)}, "
                f"risk buffer {result.risk_buffer:.2f}x — {result.or_grade} ({result.or_score}/10)."
                if result.risk_buffer is not None else
                f"Operational Risk: Margin of safety {_pct(mos)} — {result.or_grade} ({result.or_score}/10)."
            )

        return result

    def financial_health_score_analysis(self, data: FinancialData) -> FinancialHealthScoreResult:
        """Phase 90: Financial Health Score Analysis — composite of 5 pillars."""
        result = FinancialHealthScoreResult()
        rev = data.revenue or 0
        ta = data.total_assets or 0
        if rev <= 0 and ta <= 0:
            return result

        # Profitability: net_margin
        ni = data.net_income or 0
        if rev > 0:
            nm = ni / rev
            if nm >= 0.15:
                result.profitability_score = 10.0
            elif nm >= 0.10:
                result.profitability_score = 8.0
            elif nm >= 0.05:
                result.profitability_score = 6.0
            elif nm >= 0.0:
                result.profitability_score = 4.0
            else:
                result.profitability_score = 2.0

        # Liquidity: current_ratio
        ca = data.current_assets or 0
        cl = data.current_liabilities or 0
        if cl > 0:
            cr = ca / cl
            if cr >= 2.0:
                result.liquidity_score = 10.0
            elif cr >= 1.5:
                result.liquidity_score = 8.0
            elif cr >= 1.0:
                result.liquidity_score = 6.0
            elif cr >= 0.5:
                result.liquidity_score = 4.0
            else:
                result.liquidity_score = 2.0

        # Solvency: debt_to_equity
        te = data.total_equity or 0
        td = data.total_debt or 0
        if te > 0:
            de = td / te
            if de <= 0.3:
                result.solvency_score = 10.0
            elif de <= 0.6:
                result.solvency_score = 8.0
            elif de <= 1.0:
                result.solvency_score = 6.0
            elif de <= 2.0:
                result.solvency_score = 4.0
            else:
                result.solvency_score = 2.0

        # Efficiency: asset_turnover
        if ta > 0 and rev > 0:
            at = rev / ta
            if at >= 1.5:
                result.efficiency_score = 10.0
            elif at >= 1.0:
                result.efficiency_score = 8.0
            elif at >= 0.5:
                result.efficiency_score = 6.0
            elif at >= 0.2:
                result.efficiency_score = 4.0
            else:
                result.efficiency_score = 2.0

        # Coverage: interest_coverage
        ebit = data.ebit or 0
        ie = data.interest_expense or 0
        if ie > 0:
            ic = ebit / ie
            if ic >= 8.0:
                result.coverage_score = 10.0
            elif ic >= 5.0:
                result.coverage_score = 8.0
            elif ic >= 3.0:
                result.coverage_score = 6.0
            elif ic >= 1.5:
                result.coverage_score = 4.0
            else:
                result.coverage_score = 2.0

        # Composite: weighted average of available pillars
        pillars = [
            (result.profitability_score, 0.25),
            (result.liquidity_score, 0.20),
            (result.solvency_score, 0.20),
            (result.efficiency_score, 0.15),
            (result.coverage_score, 0.20),
        ]
        available = [(s, w) for s, w in pillars if s is not None]
        if available:
            total_weight = sum(w for _, w in available)
            composite = sum(s * w for s, w in available) / total_weight
            result.composite_score = round(composite, 2)
            result.fh_score = result.composite_score

            if result.fh_score >= 8:
                result.fh_grade = "Excellent"
            elif result.fh_score >= 6:
                result.fh_grade = "Good"
            elif result.fh_score >= 4:
                result.fh_grade = "Adequate"
            else:
                result.fh_grade = "Weak"

            result.summary = (
                f"Financial Health: Composite {result.composite_score}/10 "
                f"across {len(available)} pillars — {result.fh_grade}."
            )

        return result

    def asset_quality_analysis(self, data: FinancialData) -> AssetQualityResult:
        """Phase 86: Asset Quality Analysis.

        Evaluates the composition and quality of the company's asset base,
        focusing on tangibility, liquidity of assets, and concentration risks.
        """
        result = AssetQualityResult()
        ta = data.total_assets or 0
        if ta <= 0:
            return result

        ca = data.current_assets or 0
        cash = data.cash or 0
        ar = data.accounts_receivable or 0
        inv = data.inventory or 0

        # Tangible asset ratio (assume no separate intangibles field; approx as 1.0)
        # Fixed assets approximated as TA - CA
        fixed_assets = max(0, ta - ca)

        result.tangible_asset_ratio = 1.0  # No intangibles data; assume all tangible
        result.fixed_asset_ratio = safe_divide(fixed_assets, ta)
        result.current_asset_ratio = safe_divide(ca, ta)
        result.cash_to_current_assets = safe_divide(cash, ca) if ca > 0 else None
        result.receivables_to_assets = safe_divide(ar, ta)
        result.inventory_to_assets = safe_divide(inv, ta)

        # Scoring: current_asset_ratio based (higher = more liquid asset base)
        car = result.current_asset_ratio
        if car is None:
            return result

        if car >= 0.60:
            score = 10.0
        elif car >= 0.50:
            score = 8.5
        elif car >= 0.40:
            score = 7.0
        elif car >= 0.30:
            score = 5.5
        elif car >= 0.20:
            score = 4.0
        elif car >= 0.10:
            score = 2.5
        else:
            score = 1.0

        # Adj: cash_to_current_assets
        cca = result.cash_to_current_assets
        if cca is not None:
            if cca >= 0.30:
                score += 0.5
            elif cca < 0.05:
                score -= 0.5

        # Adj: receivables concentration
        rta = result.receivables_to_assets
        if rta is not None:
            if rta > 0.40:
                score -= 0.5  # High AR concentration risk
            elif rta <= 0.10:
                score += 0.5  # Low collection risk

        score = max(0.0, min(10.0, score))
        result.aq_score = round(score, 1)

        if score >= 8:
            result.aq_grade = "Excellent"
        elif score >= 6:
            result.aq_grade = "Good"
        elif score >= 4:
            result.aq_grade = "Adequate"
        else:
            result.aq_grade = "Weak"

        result.summary = (
            f"Asset Quality: Current Asset Ratio={car:.1%}, "
            f"Cash/CA={cca:.1%}" if cca is not None else f"Asset Quality: Current Asset Ratio={car:.1%}, Cash/CA=N/A"
        )
        result.summary += (
            f", AR/TA={rta:.1%}" if rta is not None else ", AR/TA=N/A"
        )
        result.summary += f" — {result.aq_grade} ({result.aq_score}/10)"
        return result

    def financial_resilience_analysis(self, data: FinancialData) -> FinancialResilienceResult:
        """Phase 82: Financial Resilience Analysis.

        Measures a company's ability to withstand financial shocks through
        cash reserves, operating cash flow coverage, and buffer metrics.
        """
        result = FinancialResilienceResult()

        cash = data.cash or 0
        ta = data.total_assets or 0
        td = data.total_debt or 0
        tl = data.total_liabilities or 0
        ocf = data.operating_cash_flow or 0
        ie = data.interest_expense or 0
        rev = data.revenue or 0
        capex = data.capex or 0
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0

        if ta <= 0:
            return result

        # Core metrics
        result.cash_to_assets = safe_divide(cash, ta)

        result.cash_to_debt = safe_divide(cash, td) if td and td > 0 else None

        result.operating_cash_coverage = safe_divide(ocf, tl) if tl and tl > 0 else None

        result.interest_coverage_cash = safe_divide(ocf, ie) if ie and ie > 0 else None

        fcf = ocf - capex
        result.free_cash_margin = safe_divide(fcf, rev) if rev and rev > 0 else None

        # Resilience buffer = (Cash + OCF) / Annual OpEx
        annual_opex = cogs + opex
        if annual_opex > 0:
            result.resilience_buffer = safe_divide(cash + ocf, annual_opex)

        # Scoring based on resilience_buffer
        rb = result.resilience_buffer
        if rb is not None:
            if rb >= 1.0:
                base = 10.0
            elif rb >= 0.75:
                base = 8.5
            elif rb >= 0.50:
                base = 7.0
            elif rb >= 0.35:
                base = 5.5
            elif rb >= 0.20:
                base = 4.0
            elif rb >= 0.10:
                base = 2.5
            else:
                base = 1.0
        elif result.cash_to_assets is not None:
            # Fallback scoring on cash_to_assets
            ca = result.cash_to_assets
            if ca >= 0.30:
                base = 8.5
            elif ca >= 0.15:
                base = 5.5
            else:
                base = 2.5
        else:
            return result

        adj = 0.0

        # Cash-to-debt adjustment
        cd = result.cash_to_debt
        if cd is not None:
            if cd >= 1.0:
                adj += 0.5
            elif cd < 0.10:
                adj -= 0.5

        # Interest coverage cash adjustment
        icc = result.interest_coverage_cash
        if icc is not None:
            if icc >= 5.0:
                adj += 0.5
            elif icc < 1.5:
                adj -= 0.5

        score = max(0.0, min(10.0, base + adj))
        result.fr_score = score
        result.fr_grade = (
            "Excellent" if score >= 8.0
            else "Good" if score >= 6.0
            else "Adequate" if score >= 4.0
            else "Weak"
        )

        grade = result.fr_grade
        result.summary = (
            f"Financial Resilience Analysis: "
            f"Resilience Buffer={rb:.2f}x. " if rb is not None else "Financial Resilience Analysis: "
        ) + (
            f"Cash/Assets={result.cash_to_assets:.1%}. "
            f"Score: {result.fr_score:.1f}/10. Grade: {grade}."
        )

        return result

    def equity_multiplier_analysis(self, data: FinancialData) -> EquityMultiplierResult:
        """Phase 81: Equity Multiplier Analysis.

        Measures financial leverage through equity multiplier and related
        leverage decomposition metrics including DuPont ROE.
        """
        result = EquityMultiplierResult()

        ta = data.total_assets or 0
        te = data.total_equity or 0
        tl = data.total_liabilities or 0
        ni = data.net_income or 0
        rev = data.revenue or 0
        ie = data.interest_expense or 0
        td = data.total_debt or 0

        if te <= 0 or ta <= 0:
            return result

        # Core metrics
        em = safe_divide(ta, te)
        result.equity_multiplier = em

        result.debt_ratio = safe_divide(tl, ta)
        result.equity_ratio = safe_divide(te, ta)

        # Financial leverage index = ROE / ROA
        roe = safe_divide(ni, te)
        roa = safe_divide(ni, ta)
        result.financial_leverage_index = safe_divide(roe, roa) if roa and roa > 0 else None

        # DuPont ROE = Net Margin * Asset Turnover * Equity Multiplier
        if rev > 0 and em is not None:
            net_margin = safe_divide(ni, rev)
            asset_turnover = safe_divide(rev, ta)
            if net_margin is not None and asset_turnover is not None:
                result.dupont_roe = net_margin * asset_turnover * em

        # Leverage spread = ROA - Cost of Debt
        if roa is not None and td and td > 0 and ie is not None:
            cost_of_debt = safe_divide(ie, td)
            if cost_of_debt is not None:
                result.leverage_spread = roa - cost_of_debt

        # Scoring based on equity_multiplier
        # Lower EM = less leverage = safer
        if em is not None:
            if em <= 1.5:
                base = 10.0
            elif em <= 2.0:
                base = 8.5
            elif em <= 2.5:
                base = 7.0
            elif em <= 3.0:
                base = 5.5
            elif em <= 4.0:
                base = 4.0
            elif em <= 5.0:
                base = 2.5
            else:
                base = 1.0

            adj = 0.0

            # Debt ratio adjustment
            dr = result.debt_ratio
            if dr is not None:
                if dr <= 0.30:
                    adj += 0.5
                elif dr > 0.70:
                    adj -= 0.5

            # Leverage spread adjustment
            ls = result.leverage_spread
            if ls is not None:
                if ls > 0.05:
                    adj += 0.5
                elif ls < -0.05:
                    adj -= 0.5

            score = max(0.0, min(10.0, base + adj))
            result.em_score = score
            result.em_grade = (
                "Excellent" if score >= 8.0
                else "Good" if score >= 6.0
                else "Adequate" if score >= 4.0
                else "Weak"
            )

        grade = result.em_grade or "N/A"
        result.summary = (
            f"Equity Multiplier Analysis: EM={em if em is not None else 'N/A':.2f}. "
            f"Debt Ratio={result.debt_ratio if result.debt_ratio is not None else 'N/A'}. "
            f"Score: {result.em_score:.1f}/10. Grade: {grade}."
        ) if em is not None else "Equity Multiplier Analysis: Insufficient data."

        return result

    def defensive_interval_analysis(self, data: FinancialData) -> DefensiveIntervalResult:
        """Phase 80: Defensive Interval Analysis.

        Measures how long a company can operate from liquid assets
        without additional revenue.
        """
        result = DefensiveIntervalResult()
        cash = data.cash or 0
        ar = data.accounts_receivable or 0
        ta = data.total_assets or 0
        cl = data.current_liabilities or 0

        # Liquid assets = Cash + AR (conservative; no short-term investments field)
        liquid = cash + ar

        # Daily operating expenses
        # Use (COGS + OpEx) as annual operating expenses
        cogs = data.cogs or 0
        opex = data.operating_expenses or 0
        annual_opex = cogs + opex
        if annual_opex <= 0:
            return result

        daily_opex = annual_opex / 365.0

        result.defensive_interval_days = round(liquid / daily_opex, 1) if daily_opex > 0 else None
        result.cash_interval_days = round(cash / daily_opex, 1) if daily_opex > 0 else None
        result.liquid_assets_ratio = safe_divide(liquid, ta)
        result.days_cash_on_hand = round(cash / daily_opex, 1) if daily_opex > 0 else None
        result.liquid_reserve_adequacy = safe_divide(liquid, cl)
        result.operating_expense_coverage = safe_divide(liquid, annual_opex)

        # --- scoring based on defensive_interval_days ---
        did = result.defensive_interval_days
        if did is None:
            return result

        if did >= 180:
            score = 10.0
        elif did >= 120:
            score = 8.5
        elif did >= 90:
            score = 7.0
        elif did >= 60:
            score = 5.5
        elif did >= 30:
            score = 4.0
        elif did >= 15:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: liquid reserve adequacy
        lra = result.liquid_reserve_adequacy
        if lra is not None:
            if lra >= 2.0:
                score += 0.5
            elif lra < 0.5:
                score -= 0.5

        # Adjustment: liquid assets ratio
        lar = result.liquid_assets_ratio
        if lar is not None:
            if lar >= 0.30:
                score += 0.5
            elif lar < 0.05:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.di_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.di_grade = grade

        result.summary = (
            f"Defensive Interval: {did:.0f} days of operating coverage. "
            f"Score: {result.di_score}/10. Grade: {grade}."
        )

        return result

    def cash_burn_analysis(self, data: FinancialData) -> CashBurnResult:
        """Phase 79: Cash Burn Analysis.

        Evaluates cash generation vs consumption patterns and
        the company's cash self-sufficiency.
        """
        result = CashBurnResult()
        rev = data.revenue or 0
        if rev <= 0:
            return result

        ocf = data.operating_cash_flow or 0
        capex = data.capex or 0
        div = data.dividends_paid or 0
        cash = data.cash or 0
        td = data.total_debt or 0

        result.ocf_margin = safe_divide(ocf, rev)
        result.capex_intensity = safe_divide(capex, rev)
        fcf = ocf - capex
        result.fcf_margin = safe_divide(fcf, rev)
        total_outflows = capex + div
        result.cash_self_sufficiency = safe_divide(ocf, total_outflows) if total_outflows > 0 else None
        result.net_cash_position = cash - td

        # Runway: only meaningful if FCF is negative (burning cash)
        if fcf < 0 and cash > 0:
            monthly_burn = abs(fcf) / 12.0
            result.cash_runway_months = round(cash / monthly_burn, 1) if monthly_burn > 0 else None

        # --- scoring based on FCF margin ---
        fm = result.fcf_margin
        if fm is None:
            return result

        if fm >= 0.20:
            score = 10.0
        elif fm >= 0.15:
            score = 8.5
        elif fm >= 0.10:
            score = 7.0
        elif fm >= 0.05:
            score = 5.5
        elif fm >= 0.0:
            score = 4.0
        elif fm >= -0.10:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: cash self-sufficiency
        css = result.cash_self_sufficiency
        if css is not None:
            if css >= 2.0:
                score += 0.5
            elif css < 0.5:
                score -= 0.5

        # Adjustment: net cash position
        ncp = result.net_cash_position
        if ncp is not None:
            if ncp > 0:
                score += 0.5
            elif ncp < -rev * 0.5:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.cb_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.cb_grade = grade

        result.summary = (
            f"Cash Burn: FCF margin {fm:.1%}, OCF margin {result.ocf_margin:.1%}. "
            f"Score: {result.cb_score}/10. Grade: {grade}."
        )

        return result

    def profit_retention_analysis(self, data: FinancialData) -> ProfitRetentionResult:
        """Phase 78: Profit Retention Analysis.

        Measures how much profit a company retains versus distributes,
        and whether retained earnings drive sustainable growth.
        """
        result = ProfitRetentionResult()
        ni = data.net_income or 0
        if ni <= 0:
            return result

        div = data.dividends_paid or 0
        te = data.total_equity or 0
        ta = data.total_assets or 0
        re = data.retained_earnings or 0

        plowback = ni - div
        result.plowback_amount = plowback
        result.retention_ratio = safe_divide(plowback, ni)
        result.payout_ratio = safe_divide(div, ni)
        result.re_to_equity = safe_divide(re, te)

        # Sustainable growth rate = ROE * retention ratio
        roe = safe_divide(ni, te)
        rr = result.retention_ratio
        if roe is not None and rr is not None:
            result.sustainable_growth_rate = round(roe * rr, 6)

        # Internal growth rate = ROA * retention / (1 - ROA * retention)
        roa = safe_divide(ni, ta)
        if roa is not None and rr is not None:
            roa_rr = roa * rr
            if roa_rr < 1.0:
                result.internal_growth_rate = round(roa_rr / (1.0 - roa_rr), 6)

        # --- scoring based on retention_ratio ---
        if rr is None:
            return result

        if rr >= 0.80:
            score = 10.0
        elif rr >= 0.70:
            score = 8.5
        elif rr >= 0.60:
            score = 7.0
        elif rr >= 0.50:
            score = 5.5
        elif rr >= 0.30:
            score = 4.0
        elif rr >= 0.10:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: RE/Equity
        re_eq = result.re_to_equity
        if re_eq is not None:
            if re_eq >= 0.60:
                score += 0.5
            elif re_eq < 0.10:
                score -= 0.5

        # Adjustment: sustainable growth rate
        sgr = result.sustainable_growth_rate
        if sgr is not None:
            if sgr >= 0.15:
                score += 0.5
            elif sgr < 0.03:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.pr_score = round(score, 2)

        if score >= 8.0:
            grade = "Excellent"
        elif score >= 6.0:
            grade = "Good"
        elif score >= 4.0:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.pr_grade = grade

        result.summary = (
            f"Profit Retention: Retention ratio {rr:.1%}, "
            f"payout ratio {result.payout_ratio:.1%}. "
            f"Score: {result.pr_score}/10. Grade: {grade}."
        )

        return result

    def debt_service_coverage_analysis(self, data: FinancialData) -> DebtServiceCoverageResult:
        """Phase 76: Debt Service Coverage Analysis.

        Evaluates the company's ability to meet debt obligations
        from operating income and cash flow.
        """
        result = DebtServiceCoverageResult()

        ie = data.interest_expense or 0
        if ie <= 0:
            return result

        ebitda = data.ebitda or 0
        ocf = data.operating_cash_flow or 0
        rev = data.revenue or 0
        capex = data.capex or 0
        fcf = ocf - capex

        # Total debt service = interest expense (no principal data available)
        tds = ie

        result.dscr = safe_divide(ebitda, tds)
        result.ocf_to_debt_service = safe_divide(ocf, tds)
        result.ebitda_to_interest = safe_divide(ebitda, ie)
        result.fcf_to_debt_service = safe_divide(fcf, tds) if fcf != 0 else safe_divide(fcf, tds)
        result.debt_service_to_revenue = safe_divide(tds, rev)

        dscr = result.dscr
        if dscr is not None:
            result.coverage_cushion = dscr - 1.0

        # Scoring based on DSCR
        if dscr is None:
            return result

        if dscr >= 4.0:
            score = 10.0
        elif dscr >= 3.0:
            score = 8.5
        elif dscr >= 2.5:
            score = 7.0
        elif dscr >= 2.0:
            score = 5.5
        elif dscr >= 1.5:
            score = 4.0
        elif dscr >= 1.0:
            score = 2.5
        else:
            score = 1.0

        # Adjustment: OCF coverage bonus/penalty
        ocd = result.ocf_to_debt_service
        if ocd is not None:
            if ocd >= 5.0:
                score += 0.5
            elif ocd < 1.0:
                score -= 0.5

        # Adjustment: debt service burden on revenue
        dsr = result.debt_service_to_revenue
        if dsr is not None:
            if dsr <= 0.02:
                score += 0.5
            elif dsr >= 0.10:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.dsc_score = round(score, 1)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.dsc_grade = grade

        dscr_str = f"{dscr:.2f}x" if dscr is not None else "N/A"
        result.summary = (
            f"Debt Service Coverage — DSCR: {dscr_str}, "
            f"Score: {score}/10. Grade: {grade}."
        )

        return result

    def capital_allocation_analysis(self, data: FinancialData) -> CapitalAllocationResult:
        """Phase 73: Capital Allocation Analysis.

        Evaluates how effectively the company allocates capital
        across reinvestment, shareholder returns, and growth.
        """
        result = CapitalAllocationResult()

        capex = data.capex or 0
        rev = data.revenue or 0
        ocf = data.operating_cash_flow or 0
        ni = data.net_income or 0
        dep = data.depreciation or 0
        div = data.dividends_paid or 0
        bb = data.share_buybacks or 0

        # Need some capital allocation data
        if capex <= 0 and div <= 0 and bb <= 0:
            return result

        fcf = ocf - capex

        # Compute ratios
        result.capex_to_revenue = safe_divide(capex, rev)
        result.capex_to_ocf = safe_divide(capex, ocf)

        total_payout = div + bb
        result.shareholder_return_ratio = safe_divide(total_payout, ni) if total_payout > 0 else None
        result.reinvestment_rate = safe_divide(capex, dep) if dep > 0 else None
        result.fcf_yield = safe_divide(fcf, rev) if rev > 0 else None
        result.total_payout_to_fcf = safe_divide(total_payout, fcf) if fcf > 0 and total_payout > 0 else None

        # Scoring based on capex_to_ocf (reinvestment efficiency)
        cto = result.capex_to_ocf
        if cto is None:
            # Fall back to capex_to_revenue
            ctr = result.capex_to_revenue
            if ctr is None:
                return result
            if ctr <= 0.05:
                score = 10.0
            elif ctr <= 0.10:
                score = 8.5
            elif ctr <= 0.15:
                score = 7.0
            elif ctr <= 0.20:
                score = 5.5
            elif ctr <= 0.30:
                score = 4.0
            elif ctr <= 0.40:
                score = 2.5
            else:
                score = 1.0
        else:
            # Balanced reinvestment: 20-50% of OCF is healthy
            if cto <= 0.25:
                score = 10.0
            elif cto <= 0.35:
                score = 8.5
            elif cto <= 0.50:
                score = 7.0
            elif cto <= 0.65:
                score = 5.5
            elif cto <= 0.80:
                score = 4.0
            elif cto <= 0.95:
                score = 2.5
            else:
                score = 1.0

        # Adjustment: FCF yield bonus/penalty
        fy = result.fcf_yield
        if fy is not None:
            if fy >= 0.10:
                score += 0.5
            elif fy < 0.0:
                score -= 0.5

        # Adjustment: total payout sustainability
        tpf = result.total_payout_to_fcf
        if tpf is not None:
            if tpf <= 0.50:
                score += 0.5
            elif tpf >= 1.0:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.ca_score = round(score, 1)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ca_grade = grade

        cto_str = f"{cto:.1%}" if cto is not None else "N/A"
        result.summary = (
            f"Capital Allocation — CapEx/OCF: {cto_str}, "
            f"Score: {score}/10. Grade: {grade}."
        )

        return result

    def tax_efficiency_analysis(self, data: FinancialData) -> TaxEfficiencyResult:
        """Phase 70: Tax Efficiency Analysis.

        Measures how effectively a company manages its tax obligations.
        Examines effective tax rate, tax burden ratios, and after-tax margins.
        """
        result = TaxEfficiencyResult()

        # Need tax_expense or ability to derive it
        te = data.tax_expense or 0
        ebt_val = data.ebt or 0
        rev = data.revenue or 0
        ni = data.net_income or 0
        ebitda = data.ebitda or 0
        ebit_val = data.ebit or 0
        ie = data.interest_expense or 0

        # If no tax expense, try to derive: EBT - NI = tax
        if not data.tax_expense and data.ebt and data.net_income:
            te = data.ebt - data.net_income

        # If no EBT, try EBIT - IE
        if not ebt_val and ebit_val and data.interest_expense is not None:
            ebt_val = ebit_val - ie

        # Need at least some tax data to proceed
        if te <= 0 and ni <= 0 and rev <= 0:
            return result

        # Compute ratios
        result.effective_tax_rate = safe_divide(te, ebt_val) if ebt_val > 0 else None
        result.tax_to_revenue = safe_divide(te, rev)
        result.tax_to_ebitda = safe_divide(te, ebitda)
        result.after_tax_margin = safe_divide(ni, rev)
        # Tax shield: interest * ETR / NI (benefit of debt tax shield relative to NI)
        if result.effective_tax_rate is not None and ni > 0:
            result.tax_shield_ratio = safe_divide(ie * result.effective_tax_rate, ni)
        # PreTax/EBIT ratio (impact of non-operating items)
        result.pretax_to_ebit = safe_divide(ebt_val, ebit_val)

        # Scoring based on effective tax rate (lower = more efficient)
        etr = result.effective_tax_rate
        if etr is None:
            # Fall back to tax_to_revenue for scoring
            etr_proxy = result.tax_to_revenue
            if etr_proxy is None or etr_proxy <= 0:
                return result
            # Use after_tax_margin as primary scoring instead
            atm = result.after_tax_margin or 0
            if atm >= 0.20:
                score = 10.0
            elif atm >= 0.15:
                score = 8.5
            elif atm >= 0.10:
                score = 7.0
            elif atm >= 0.05:
                score = 5.5
            elif atm >= 0.02:
                score = 4.0
            elif atm >= 0.0:
                score = 2.5
            else:
                score = 1.0
        else:
            # Primary scoring on ETR (lower is better for tax efficiency)
            if etr <= 0.15:
                score = 10.0
            elif etr <= 0.20:
                score = 8.5
            elif etr <= 0.25:
                score = 7.0
            elif etr <= 0.30:
                score = 5.5
            elif etr <= 0.35:
                score = 4.0
            elif etr <= 0.40:
                score = 2.5
            else:
                score = 1.0

        # Adjustment: after-tax margin bonus/penalty
        atm = result.after_tax_margin
        if atm is not None:
            if atm >= 0.15:
                score += 0.5
            elif atm < 0.02:
                score -= 0.5

        # Adjustment: tax-to-revenue ratio bonus/penalty
        ttr = result.tax_to_revenue
        if ttr is not None:
            if ttr <= 0.03:
                score += 0.5
            elif ttr >= 0.15:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.te_score = round(score, 1)

        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.te_grade = grade

        etr_str = f"{etr:.1%}" if etr is not None else "N/A"
        result.summary = (
            f"Tax Efficiency — ETR: {etr_str}, "
            f"Score: {score}/10. Grade: {grade}."
        )

        return result

    def roic_analysis(self, data: FinancialData) -> ROICResult:
        """Phase 56: Return on Invested Capital Analysis.

        ROIC = NOPAT / Invested Capital.
        Invested Capital = Total Equity + Total Debt - Cash.
        NOPAT = EBIT * (1 - tax_rate), estimated tax_rate from NI/EBT.
        """
        result = ROICResult()
        ebit = data.ebit
        ni = data.net_income
        ie = data.interest_expense or 0
        te = data.total_equity
        td = data.total_debt or 0
        cash = data.cash or 0

        if ebit is None or te is None:
            return result

        # Estimate tax rate from NI/EBT
        ebt = ebit - ie
        if ebt > 0 and ni is not None:
            tax_rate = 1 - safe_divide(ni, ebt, default=0.75)
        else:
            tax_rate = 0.25  # default assumption

        nopat = ebit * (1 - tax_rate)
        result.nopat = round(nopat, 2)

        # Invested capital
        ic = te + td - cash
        if ic <= 0:
            return result
        result.invested_capital = round(ic, 2)

        # ROIC
        roic_val = safe_divide(nopat, ic)
        if roic_val is None:
            return result
        result.roic_pct = round(roic_val * 100, 2)

        # ROA for spread
        ta = data.total_assets
        if ta and ta > 0 and ni is not None:
            roa = safe_divide(ni, ta) * 100
            result.roic_roa_spread = round(result.roic_pct - roa, 2) if roa is not None else None

        # Capital efficiency = Revenue / IC
        rev = data.revenue
        if rev and rev > 0:
            result.capital_efficiency = safe_divide(rev, ic)
            if result.capital_efficiency is not None:
                result.capital_efficiency = round(result.capital_efficiency, 2)

        # Scoring
        roic = result.roic_pct
        if roic >= 20:
            score = 10.0
        elif roic >= 15:
            score = 8.5
        elif roic >= 10:
            score = 7.0
        elif roic >= 6:
            score = 5.5
        elif roic >= 0:
            score = 3.5
        elif roic >= -5:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.capital_efficiency is not None:
            if result.capital_efficiency > 2.0:
                score += 0.5
            elif result.capital_efficiency < 0.5:
                score -= 0.5
        if result.roic_roa_spread is not None:
            if result.roic_roa_spread > 5:
                score += 0.5

        score = max(0.0, min(10.0, score))
        result.roic_score = round(score, 2)

        if score >= 8:
            result.roic_grade = "Excellent"
        elif score >= 6:
            result.roic_grade = "Good"
        elif score >= 4:
            result.roic_grade = "Adequate"
        else:
            result.roic_grade = "Weak"

        result.summary = (
            f"ROIC: {roic:.1f}%. NOPAT: ${nopat:,.0f}. "
            f"Invested Capital: ${ic:,.0f}. "
            f"Grade: {result.roic_grade}."
        )

        return result

    def roa_quality_analysis(self, data: FinancialData) -> ROAQualityResult:
        """Phase 55: Return on Assets Quality Analysis.

        Examines asset productivity: ROA, operating ROA, cash ROA,
        asset turnover, fixed-asset turnover, and capital intensity.
        """
        result = ROAQualityResult()
        rev = data.revenue
        ta = data.total_assets

        if not ta or ta == 0:
            return result

        # Core ROA
        ni = data.net_income
        if ni is not None:
            result.roa_pct = safe_divide(ni, ta) * 100 if safe_divide(ni, ta) is not None else None

        # Operating ROA
        oi = data.operating_income or data.ebit
        if oi is not None:
            result.operating_roa_pct = safe_divide(oi, ta) * 100 if safe_divide(oi, ta) is not None else None

        # Cash ROA
        ocf = data.operating_cash_flow
        if ocf is not None:
            result.cash_roa_pct = safe_divide(ocf, ta) * 100 if safe_divide(ocf, ta) is not None else None

        # Asset turnover
        if rev and rev > 0:
            result.asset_turnover = safe_divide(rev, ta)

        # Fixed asset turnover
        ca = data.current_assets
        if rev and rev > 0 and ca is not None:
            fixed_assets = ta - ca
            if fixed_assets > 0:
                result.fixed_asset_turnover = safe_divide(rev, fixed_assets)

        # Capital intensity = TA / Revenue
        if rev and rev > 0:
            result.capital_intensity = safe_divide(ta, rev)

        # Scoring based on ROA
        if result.roa_pct is None:
            return result

        roa = result.roa_pct
        if roa >= 15:
            score = 10.0
        elif roa >= 10:
            score = 8.5
        elif roa >= 7:
            score = 7.0
        elif roa >= 4:
            score = 5.5
        elif roa >= 0:
            score = 3.5
        elif roa >= -5:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.asset_turnover is not None:
            if result.asset_turnover > 1.5:
                score += 0.5
            elif result.asset_turnover < 0.3:
                score -= 0.5
        if result.cash_roa_pct is not None and result.roa_pct is not None:
            if result.cash_roa_pct > result.roa_pct:
                score += 0.5

        score = max(0.0, min(10.0, score))
        result.roa_score = round(score, 2)

        if score >= 8:
            result.roa_grade = "Excellent"
        elif score >= 6:
            result.roa_grade = "Good"
        elif score >= 4:
            result.roa_grade = "Adequate"
        else:
            result.roa_grade = "Weak"

        result.summary = (
            f"Return on Assets: {roa:.1f}%. "
        )
        if result.operating_roa_pct is not None:
            result.summary += f"Operating ROA: {result.operating_roa_pct:.1f}%. "
        if result.cash_roa_pct is not None:
            result.summary += f"Cash ROA: {result.cash_roa_pct:.1f}%. "
        result.summary += f"Grade: {result.roa_grade}."

        return result

    def roe_analysis(self, data: FinancialData) -> ROEAnalysisResult:
        """Phase 54: Return on Equity Analysis.

        Decomposes ROE via DuPont: Net Margin × Asset Turnover × Equity Multiplier.
        Also computes ROA, retention ratio, and sustainable growth rate.
        """
        result = ROEAnalysisResult()
        rev = data.revenue
        ni = data.net_income
        ta = data.total_assets
        te = data.total_equity

        if not rev or not te or te == 0:
            return result

        # Core ROE
        if ni is not None:
            result.roe_pct = safe_divide(ni, te) * 100 if safe_divide(ni, te) is not None else None

        # DuPont components
        if ni is not None and rev > 0:
            result.net_margin_pct = safe_divide(ni, rev) * 100 if safe_divide(ni, rev) is not None else None
        if ta and ta > 0:
            result.asset_turnover = safe_divide(rev, ta)
            result.equity_multiplier = safe_divide(ta, te)
        if ni is not None and ta and ta > 0:
            result.roa_pct = safe_divide(ni, ta) * 100 if safe_divide(ni, ta) is not None else None

        # Retention & sustainable growth
        re = data.retained_earnings
        if ni is not None and ni > 0 and re is not None:
            dividends_paid = ni - re if re < ni else 0
            result.retention_ratio = safe_divide(ni - dividends_paid, ni)
            if result.roe_pct is not None and result.retention_ratio is not None:
                result.sustainable_growth_rate = (result.roe_pct / 100) * result.retention_ratio * 100

        # Scoring
        if result.roe_pct is None:
            return result

        roe = result.roe_pct
        if roe >= 25:
            score = 10.0
        elif roe >= 18:
            score = 8.5
        elif roe >= 12:
            score = 7.0
        elif roe >= 8:
            score = 5.5
        elif roe >= 0:
            score = 3.5
        elif roe >= -10:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.asset_turnover is not None:
            if result.asset_turnover > 1.5:
                score += 0.5
            elif result.asset_turnover < 0.3:
                score -= 0.5
        if result.equity_multiplier is not None:
            if result.equity_multiplier > 5.0:
                score -= 0.5
            elif result.equity_multiplier < 2.0:
                score += 0.5

        score = max(0.0, min(10.0, score))
        result.roe_score = round(score, 2)

        if score >= 8:
            result.roe_grade = "Excellent"
        elif score >= 6:
            result.roe_grade = "Good"
        elif score >= 4:
            result.roe_grade = "Adequate"
        else:
            result.roe_grade = "Weak"

        result.summary = (
            f"Return on Equity: {roe:.1f}%. "
            f"DuPont decomposition — Net Margin: "
            f"{result.net_margin_pct:.1f}%" if result.net_margin_pct is not None else "N/A"
        )
        if result.asset_turnover is not None:
            result.summary += f", Asset Turnover: {result.asset_turnover:.2f}x"
        if result.equity_multiplier is not None:
            result.summary += f", Equity Multiplier: {result.equity_multiplier:.2f}x"
        result.summary += f". Grade: {result.roe_grade}."

        return result

    def net_profit_margin_analysis(self, data: FinancialData) -> NetProfitMarginResult:
        """Phase 53: Net Profit Margin Analysis.

        Evaluates bottom-line profitability, tax/interest burden,
        and the path from EBITDA to net income.
        """
        result = NetProfitMarginResult()

        revenue = data.revenue
        ni = data.net_income
        ebitda = data.ebitda
        ebit = data.ebit
        ie = data.interest_expense

        if not revenue or revenue <= 0:
            result.summary = "Insufficient data for Net Profit Margin."
            return result

        # --- Compute ratios ---
        if ni is not None:
            result.net_margin_pct = safe_divide(ni, revenue, 0.0) * 100

        if ebitda is not None:
            result.ebitda_margin_pct = safe_divide(ebitda, revenue, 0.0) * 100

        if ebit is not None:
            result.ebit_margin_pct = safe_divide(ebit, revenue, 0.0) * 100

        # Tax burden = NI / EBT where EBT = EBIT - IE
        if ni is not None and ebit is not None:
            ebt = ebit - (ie or 0)
            if ebt and ebt > 0:
                result.tax_burden = safe_divide(ni, ebt)

        # Interest burden = EBT / EBIT
        if ebit is not None and ebit > 0:
            ebt = ebit - (ie or 0)
            result.interest_burden = safe_divide(ebt, ebit)

        # Net to EBITDA
        if ni is not None and ebitda and ebitda > 0:
            result.net_to_ebitda = safe_divide(ni, ebitda)

        # --- Scoring ---
        nm = result.net_margin_pct
        if nm is None:
            result.summary = "Net income not available."
            return result

        if nm >= 25:
            score = 10.0
        elif nm >= 15:
            score = 8.5
        elif nm >= 10:
            score = 7.0
        elif nm >= 5:
            score = 5.5
        elif nm >= 0:
            score = 3.5
        elif nm >= -10:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.tax_burden is not None:
            if result.tax_burden > 0.80:
                score += 0.5  # Low effective tax rate
            elif result.tax_burden < 0.50:
                score -= 0.5  # High tax burden

        if result.interest_burden is not None:
            if result.interest_burden > 0.90:
                score += 0.5  # Low interest burden
            elif result.interest_burden < 0.60:
                score -= 0.5  # Heavy interest cost

        score = max(0.0, min(10.0, score))
        result.npm_score = score

        # Grading
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.npm_grade = grade

        result.summary = (
            f"Net Profit Margin: {nm:.1f}% margin, "
            f"score {score:.1f}/10. Status: {grade}."
        )

        return result

    def ebitda_margin_quality_analysis(self, data: FinancialData) -> EbitdaMarginQualityResult:
        """Phase 52: EBITDA Margin Quality Analysis.

        Evaluates EBITDA margin level, D&A intensity, and spread between
        EBITDA and operating income as an earnings quality indicator.
        """
        result = EbitdaMarginQualityResult()

        revenue = data.revenue
        ebitda = data.ebitda
        oi = data.operating_income
        dep = data.depreciation
        gp = data.gross_profit

        if not revenue or revenue <= 0:
            result.summary = "Insufficient data for EBITDA Margin Quality."
            return result

        # --- Compute ratios ---
        if ebitda is not None:
            result.ebitda_margin_pct = safe_divide(ebitda, revenue, 0.0) * 100

        if oi is not None:
            result.operating_margin_pct = safe_divide(oi, revenue, 0.0) * 100

        if dep is not None:
            result.da_intensity = safe_divide(dep, revenue, 0.0) * 100

        if result.ebitda_margin_pct is not None and result.operating_margin_pct is not None:
            result.ebitda_oi_spread = result.ebitda_margin_pct - result.operating_margin_pct

        if ebitda is not None and gp and gp > 0:
            result.ebitda_to_gp = safe_divide(ebitda, gp)

        # --- Scoring ---
        em = result.ebitda_margin_pct
        if em is None:
            result.summary = "EBITDA not available."
            return result

        if em >= 40:
            score = 10.0
        elif em >= 30:
            score = 8.5
        elif em >= 20:
            score = 7.0
        elif em >= 15:
            score = 5.5
        elif em >= 10:
            score = 4.0
        elif em >= 0:
            score = 2.5
        else:
            score = 0.5

        # Adjustments
        if result.da_intensity is not None:
            if result.da_intensity > 15:
                score -= 0.5  # Capital-heavy business
            elif result.da_intensity < 3:
                score += 0.5  # Asset-light

        if result.ebitda_to_gp is not None:
            if result.ebitda_to_gp > 0.70:
                score += 0.5  # Strong conversion from GP to EBITDA
            elif result.ebitda_to_gp < 0.30:
                score -= 0.5  # OpEx eating too much

        score = max(0.0, min(10.0, score))
        result.ebitda_score = score

        # Grading
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.ebitda_grade = grade

        result.summary = (
            f"EBITDA Margin Quality: {em:.1f}% margin, "
            f"score {score:.1f}/10. Status: {grade}."
        )

        return result

    def gross_margin_stability_analysis(self, data: FinancialData) -> GrossMarginStabilityResult:
        """Phase 51: Gross Margin Stability Analysis.

        Assesses gross margin quality, cost structure efficiency, and
        margin sustainability using single-period data.
        """
        result = GrossMarginStabilityResult()

        revenue = data.revenue
        gp = data.gross_profit
        cogs = data.cogs
        oi = data.operating_income
        opex = data.operating_expenses

        if not revenue or revenue <= 0:
            result.summary = "Insufficient data for Gross Margin Stability."
            return result

        # --- Compute ratios ---
        gm_pct = safe_divide(gp, revenue, 0.0) * 100 if gp is not None else None
        cogs_r = safe_divide(cogs, revenue, None) * 100 if cogs is not None else None
        om_pct = safe_divide(oi, revenue, 0.0) * 100 if oi is not None else None

        result.gross_margin_pct = gm_pct
        result.cogs_ratio = cogs_r

        if gm_pct is not None and om_pct is not None:
            result.operating_margin_pct = om_pct
            result.margin_spread = gm_pct - om_pct

        if gp is not None and opex and opex > 0:
            result.opex_coverage = safe_divide(gp, opex)

        if gm_pct is not None:
            result.margin_buffer = gm_pct  # distance above 0% margin

        # --- Scoring ---
        if gm_pct is None:
            result.summary = "Gross profit not available."
            return result

        if gm_pct >= 60:
            score = 10.0
        elif gm_pct >= 45:
            score = 8.5
        elif gm_pct >= 30:
            score = 7.0
        elif gm_pct >= 20:
            score = 5.5
        elif gm_pct >= 10:
            score = 3.5
        elif gm_pct >= 0:
            score = 2.0
        else:
            score = 0.5

        # Adjustments
        if result.opex_coverage is not None:
            if result.opex_coverage > 2.0:
                score += 0.5
            elif result.opex_coverage < 1.0:
                score -= 0.5

        if result.margin_spread is not None:
            if result.margin_spread > 20:
                score += 0.5
            elif result.margin_spread < 5:
                score -= 0.5

        score = max(0.0, min(10.0, score))
        result.gm_stability_score = score

        # Grading
        if score >= 8:
            grade = "Excellent"
        elif score >= 6:
            grade = "Good"
        elif score >= 4:
            grade = "Adequate"
        else:
            grade = "Weak"
        result.gm_stability_grade = grade

        result.summary = (
            f"Gross Margin Stability: {gm_pct:.1f}% gross margin, "
            f"score {score:.1f}/10. Status: {grade}."
        )

        return result

    def analyze(self, data: Union[FinancialData, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run comprehensive analysis on financial data.
        """
        if isinstance(data, pd.DataFrame):
            financial_data = self._dataframe_to_financial_data(data)
        else:
            financial_data = data

        results = {
            'liquidity_ratios': self.calculate_liquidity_ratios(financial_data),
            'profitability_ratios': self.calculate_profitability_ratios(financial_data),
            'leverage_ratios': self.calculate_leverage_ratios(financial_data),
            'efficiency_ratios': self.calculate_efficiency_ratios(financial_data),
            'cash_flow': self.analyze_cash_flow(financial_data),
            'working_capital': self.analyze_working_capital(financial_data),
            'dupont': self.dupont_analysis(financial_data),
            'altman_z_score': self.altman_z_score(financial_data),
            'piotroski_f_score': self.piotroski_f_score(financial_data),
            'composite_health': self.composite_health_score(financial_data),
        }

        # Generate insights
        results['insights'] = self.generate_insights(results)

        return results

    def _dataframe_to_financial_data(self, df: pd.DataFrame) -> FinancialData:
        """
        Convert a DataFrame to FinancialData by detecting column mappings.
        """
        data = FinancialData()

        # Column name mapping (lowercase patterns to attribute names)
        mappings = {
            'revenue': ['revenue', 'sales', 'net sales', 'total revenue'],
            'cogs': ['cogs', 'cost of goods sold', 'cost of sales', 'cost of revenue'],
            'gross_profit': ['gross profit', 'gross margin'],
            'operating_income': ['operating income', 'operating profit', 'ebit'],
            'net_income': ['net income', 'net profit', 'net earnings', 'profit after tax'],
            'total_assets': ['total assets', 'assets'],
            'current_assets': ['current assets'],
            'cash': ['cash', 'cash and equivalents', 'cash & equivalents'],
            'inventory': ['inventory', 'inventories'],
            'accounts_receivable': ['accounts receivable', 'receivables', 'trade receivables'],
            'total_liabilities': ['total liabilities', 'liabilities'],
            'current_liabilities': ['current liabilities'],
            'accounts_payable': ['accounts payable', 'payables', 'trade payables'],
            'total_debt': ['total debt', 'long term debt', 'debt'],
            'total_equity': ['total equity', 'shareholders equity', 'stockholders equity'],
            'interest_expense': ['interest expense', 'interest'],
            'ebt': ['earnings before tax', 'ebt', 'income before tax', 'profit before tax', 'pre-tax income'],
            'retained_earnings': ['retained earnings', 'accumulated earnings'],
            'depreciation': ['depreciation', 'depreciation and amortization', 'd&a'],
            'operating_cash_flow': ['operating cash flow', 'cash from operations'],
            'capex': ['capex', 'capital expenditure', 'capital expenditures'],
        }

        # Build reverse lookup (prefer longest pattern match to avoid greedy short patterns)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            best_attr = None
            best_len = 0
            for attr, patterns in mappings.items():
                for pattern in patterns:
                    if pattern in col_lower and len(pattern) > best_len:
                        best_attr = attr
                        best_len = len(pattern)
            if best_attr is not None:
                column_map[col] = best_attr

        # Extract values (use first row if single row, or sum/latest)
        for col, attr in column_map.items():
            try:
                if len(df) == 1:
                    value = df[col].iloc[0]
                else:
                    # Use the most recent non-null value
                    value = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None

                if value is not None and not pd.isna(value):
                    setattr(data, attr, float(value))
            except (ValueError, TypeError):
                pass

        return data


# Convenience function for quick analysis
def quick_analyze(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick financial analysis of a DataFrame.
    """
    analyzer = CharlieAnalyzer()
    return analyzer.analyze(df)
