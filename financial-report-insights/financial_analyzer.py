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

    def __init__(self):
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
        nopat = data.operating_income * 0.75 if data.operating_income is not None else None
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
        """Calculate moving average."""
        if len(values) < window:
            return values

        result = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(np.mean(values[:i + 1]))
            else:
                result.append(np.mean(values[i - window + 1:i + 1]))
        return result

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
        except:
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

        return insights

    def detect_anomalies(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> List[Anomaly]:
        """
        Flag unusual patterns.
        """
        anomalies = []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            series = data[col].dropna()
            if len(series) < 3:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            for idx, value in series.items():
                z_score = (value - mean) / std
                if abs(z_score) > 2:
                    anomalies.append(Anomaly(
                        metric_name=col,
                        value=value,
                        expected_range=(mean - 2*std, mean + 2*std),
                        z_score=z_score,
                        description=f"Unusual value in {col}: {value:.2f} is {abs(z_score):.1f} standard deviations from mean"
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
            'operating_cash_flow': ['operating cash flow', 'cash from operations'],
            'capex': ['capex', 'capital expenditure', 'capital expenditures'],
        }

        # Build reverse lookup
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            for attr, patterns in mappings.items():
                if any(pattern in col_lower for pattern in patterns):
                    column_map[col] = attr
                    break

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
