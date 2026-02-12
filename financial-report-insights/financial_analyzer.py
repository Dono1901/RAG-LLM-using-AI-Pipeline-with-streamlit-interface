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
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    net_income: Optional[float] = None
    ebt: Optional[float] = None  # Earnings Before Tax
    retained_earnings: Optional[float] = None
    depreciation: Optional[float] = None

    # Cash Flow items
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    capex: Optional[float] = None

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
