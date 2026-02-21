"""Tests for portfolio_analyzer.py -- portfolio & multi-company analysis."""

import math
import pytest

from financial_analyzer import FinancialData
from portfolio_analyzer import (
    CompanySnapshot,
    CorrelationMatrix,
    DiversificationScore,
    PortfolioAnalyzer,
    PortfolioReport,
    PortfolioRiskSummary,
    _hhi,
    _hhi_label,
    _score_to_grade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strong_company() -> FinancialData:
    """Financially strong company."""
    return FinancialData(
        revenue=10_000_000,
        net_income=1_500_000,
        gross_profit=6_000_000,
        operating_income=2_500_000,
        ebit=2_500_000,
        ebitda=3_000_000,
        interest_expense=200_000,
        total_assets=20_000_000,
        current_assets=8_000_000,
        current_liabilities=3_000_000,
        total_debt=4_000_000,
        total_equity=14_000_000,
        total_liabilities=6_000_000,
        operating_cash_flow=2_000_000,
        cash=3_000_000,
    )


@pytest.fixture
def weak_company() -> FinancialData:
    """Financially weak company."""
    return FinancialData(
        revenue=2_000_000,
        net_income=-200_000,
        gross_profit=400_000,
        operating_income=-100_000,
        ebit=-100_000,
        ebitda=50_000,
        interest_expense=300_000,
        total_assets=5_000_000,
        current_assets=1_000_000,
        current_liabilities=2_000_000,
        total_debt=4_500_000,
        total_equity=100_000,
        total_liabilities=4_900_000,
        operating_cash_flow=100_000,
        cash=200_000,
    )


@pytest.fixture
def medium_company() -> FinancialData:
    """Average financial health company."""
    return FinancialData(
        revenue=5_000_000,
        net_income=250_000,
        gross_profit=2_000_000,
        operating_income=500_000,
        ebit=500_000,
        ebitda=800_000,
        interest_expense=100_000,
        total_assets=10_000_000,
        current_assets=4_000_000,
        current_liabilities=2_500_000,
        total_debt=3_000_000,
        total_equity=6_000_000,
        total_liabilities=4_000_000,
        operating_cash_flow=600_000,
        cash=1_000_000,
    )


@pytest.fixture
def three_company_portfolio(strong_company, weak_company, medium_company):
    """Dict of three companies for portfolio analysis."""
    return {
        "StrongCorp": strong_company,
        "WeakCo": weak_company,
        "MediumInc": medium_company,
    }


@pytest.fixture
def analyzer():
    return PortfolioAnalyzer()


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_hhi_single_company(self):
        assert _hhi([100.0]) == 1.0

    def test_hhi_equal_split(self):
        # 4 companies with equal revenue -> HHI = 0.25
        result = _hhi([100, 100, 100, 100])
        assert abs(result - 0.25) < 0.01

    def test_hhi_concentrated(self):
        # One company dominates
        result = _hhi([900, 10, 10, 10])
        assert result > 0.75

    def test_hhi_empty(self):
        assert _hhi([]) == 1.0

    def test_hhi_all_zeros(self):
        assert _hhi([0, 0, 0]) == 1.0

    def test_hhi_label_low(self):
        assert _hhi_label(0.10) == "low concentration"

    def test_hhi_label_moderate(self):
        assert _hhi_label(0.20) == "moderate concentration"

    def test_hhi_label_high(self):
        assert _hhi_label(0.50) == "high concentration"

    def test_score_to_grade(self):
        assert _score_to_grade(85) == "A"
        assert _score_to_grade(70) == "B"
        assert _score_to_grade(55) == "C"
        assert _score_to_grade(40) == "D"
        assert _score_to_grade(20) == "F"


# ---------------------------------------------------------------------------
# Company Snapshot
# ---------------------------------------------------------------------------


class TestCompanySnapshot:
    def test_strong_snapshot(self, analyzer, strong_company):
        snap = analyzer.company_snapshot("StrongCorp", strong_company)
        assert isinstance(snap, CompanySnapshot)
        assert snap.name == "StrongCorp"
        assert snap.health_score > 0
        assert snap.health_grade in ("A", "B", "C", "D", "F")
        assert "net_margin" in snap.key_ratios
        assert "roa" in snap.key_ratios

    def test_weak_snapshot(self, analyzer, weak_company):
        snap = analyzer.company_snapshot("WeakCo", weak_company)
        assert snap.name == "WeakCo"
        # Weak company should have low health score
        assert snap.health_score <= 60

    def test_snapshot_ratios_populated(self, analyzer, strong_company):
        snap = analyzer.company_snapshot("Test", strong_company)
        # net_margin = 1.5M / 10M = 0.15
        assert snap.key_ratios["net_margin"] is not None
        assert abs(snap.key_ratios["net_margin"] - 0.15) < 0.01


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_two_companies(self, analyzer, strong_company, medium_company):
        companies = {"A": strong_company, "B": medium_company}
        corr = analyzer.correlation_matrix(companies)
        assert isinstance(corr, CorrelationMatrix)
        assert len(corr.company_names) == 2
        assert len(corr.matrix) == 2
        assert len(corr.matrix[0]) == 2
        # Diagonal should be 1.0
        assert abs(corr.matrix[0][0] - 1.0) < 0.01
        assert abs(corr.matrix[1][1] - 1.0) < 0.01

    def test_single_company(self, analyzer, strong_company):
        companies = {"A": strong_company}
        corr = analyzer.correlation_matrix(companies)
        assert len(corr.matrix) == 1
        assert corr.avg_correlation == 1.0
        assert "at least 2" in corr.interpretation.lower()

    def test_three_companies(self, analyzer, three_company_portfolio):
        corr = analyzer.correlation_matrix(three_company_portfolio)
        assert len(corr.company_names) == 3
        assert len(corr.matrix) == 3
        # Avg correlation should be a float between -1 and 1
        assert -1.0 <= corr.avg_correlation <= 1.0

    def test_empty_portfolio(self, analyzer):
        corr = analyzer.correlation_matrix({})
        assert len(corr.matrix) == 0
        assert corr.avg_correlation == 0.0

    def test_interpretation_populated(self, analyzer, three_company_portfolio):
        corr = analyzer.correlation_matrix(three_company_portfolio)
        assert len(corr.interpretation) > 10


# ---------------------------------------------------------------------------
# Diversification Score
# ---------------------------------------------------------------------------


class TestDiversificationScore:
    def test_diverse_portfolio(self, analyzer, three_company_portfolio):
        div = analyzer.diversification_score(three_company_portfolio)
        assert isinstance(div, DiversificationScore)
        assert 0 <= div.overall_score <= 100
        assert div.grade in ("A", "B", "C", "D", "F")
        assert 0.0 <= div.hhi_revenue <= 1.0
        assert 0.0 <= div.hhi_assets <= 1.0

    def test_single_company_low_diversification(self, analyzer, strong_company):
        companies = {"Only": strong_company}
        div = analyzer.diversification_score(companies)
        # Single company = fully concentrated
        assert div.hhi_revenue == 1.0
        assert div.hhi_assets == 1.0

    def test_equal_revenue_good_diversification(self, analyzer, strong_company, medium_company):
        # Give them similar revenue
        medium_company.revenue = 10_000_000
        medium_company.total_assets = 20_000_000
        companies = {"A": strong_company, "B": medium_company}
        div = analyzer.diversification_score(companies)
        # Equal revenue -> HHI = 0.5
        assert div.hhi_revenue < 0.55

    def test_interpretation_populated(self, analyzer, three_company_portfolio):
        div = analyzer.diversification_score(three_company_portfolio)
        assert "HHI" in div.interpretation


# ---------------------------------------------------------------------------
# Portfolio Risk Summary
# ---------------------------------------------------------------------------


class TestPortfolioRiskSummary:
    def test_risk_summary_basic(self, analyzer, three_company_portfolio):
        snapshots = [
            analyzer.company_snapshot(n, d)
            for n, d in three_company_portfolio.items()
        ]
        risk = analyzer.portfolio_risk_summary(snapshots, three_company_portfolio)
        assert isinstance(risk, PortfolioRiskSummary)
        assert risk.num_companies == 3
        assert risk.weakest_company != ""
        assert risk.strongest_company != ""
        assert risk.overall_risk_level in ("low", "moderate", "high", "critical")

    def test_weak_company_flagged(self, analyzer, weak_company):
        companies = {"WeakCo": weak_company}
        snapshots = [analyzer.company_snapshot("WeakCo", weak_company)]
        risk = analyzer.portfolio_risk_summary(snapshots, companies)
        # Should have risk flags for the weak company
        assert len(risk.risk_flags) > 0

    def test_empty_portfolio(self, analyzer):
        risk = analyzer.portfolio_risk_summary([], {})
        assert risk.overall_risk_level == "critical"
        assert risk.num_companies == 0

    def test_negative_equity_flagged(self, analyzer):
        bad = FinancialData(
            revenue=1_000_000,
            net_income=-500_000,
            total_assets=2_000_000,
            total_equity=-100_000,
            total_liabilities=2_100_000,
            current_assets=500_000,
            current_liabilities=1_500_000,
            total_debt=2_000_000,
            ebit=-400_000,
            interest_expense=300_000,
            operating_cash_flow=-200_000,
        )
        companies = {"BadCo": bad}
        snapshots = [analyzer.company_snapshot("BadCo", bad)]
        risk = analyzer.portfolio_risk_summary(snapshots, companies)
        neg_equity_flags = [f for f in risk.risk_flags if "negative equity" in f.lower()]
        assert len(neg_equity_flags) > 0


# ---------------------------------------------------------------------------
# Full Portfolio Analysis
# ---------------------------------------------------------------------------


class TestFullPortfolioAnalysis:
    def test_full_analysis(self, analyzer, three_company_portfolio):
        report = analyzer.full_portfolio_analysis(three_company_portfolio)
        assert isinstance(report, PortfolioReport)
        assert report.num_companies == 3
        assert len(report.snapshots) == 3
        assert report.correlation is not None
        assert report.diversification is not None
        assert report.risk_summary is not None
        assert len(report.summary) > 20

    def test_single_company(self, analyzer, strong_company):
        companies = {"Only": strong_company}
        report = analyzer.full_portfolio_analysis(companies)
        assert report.num_companies == 1
        assert len(report.snapshots) == 1

    def test_summary_mentions_strongest_weakest(self, analyzer, three_company_portfolio):
        report = analyzer.full_portfolio_analysis(three_company_portfolio)
        # Summary should mention strongest and weakest
        assert "Strongest" in report.summary or "strongest" in report.summary
        assert "Weakest" in report.summary or "weakest" in report.summary
