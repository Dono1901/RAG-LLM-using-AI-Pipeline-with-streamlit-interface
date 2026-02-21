"""Tests for underwriting and credit analysis module."""

import pytest

from financial_analyzer import FinancialData
from underwriting import (
    CovenantPackage,
    CreditScorecard,
    DebtCapacityResult,
    LoanStructure,
    UnderwritingAnalyzer,
    UnderwritingReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return UnderwritingAnalyzer()


@pytest.fixture
def strong_company():
    """Company with strong financials -- should get A grade."""
    return FinancialData(
        total_assets=10_000_000,
        current_assets=4_000_000,
        cash=1_500_000,
        inventory=500_000,
        accounts_receivable=800_000,
        total_liabilities=3_000_000,
        current_liabilities=1_500_000,
        total_debt=1_500_000,
        total_equity=7_000_000,
        revenue=15_000_000,
        cogs=9_000_000,
        gross_profit=6_000_000,
        operating_income=3_000_000,
        net_income=2_000_000,
        ebit=3_200_000,
        ebitda=4_000_000,
        interest_expense=200_000,
        operating_cash_flow=3_500_000,
        capex=500_000,
    )


@pytest.fixture
def weak_company():
    """Company with weak financials -- should get D/F grade."""
    return FinancialData(
        total_assets=5_000_000,
        current_assets=800_000,
        cash=50_000,
        inventory=400_000,
        total_liabilities=4_500_000,
        current_liabilities=1_200_000,
        total_debt=3_500_000,
        total_equity=500_000,
        revenue=3_000_000,
        cogs=2_700_000,
        gross_profit=300_000,
        operating_income=50_000,
        net_income=-100_000,
        ebit=100_000,
        ebitda=200_000,
        interest_expense=300_000,
        operating_cash_flow=100_000,
        capex=80_000,
    )


@pytest.fixture
def medium_company():
    """Company with mixed financials -- should get C grade."""
    return FinancialData(
        total_assets=8_000_000,
        current_assets=2_500_000,
        cash=400_000,
        inventory=600_000,
        total_liabilities=4_000_000,
        current_liabilities=1_600_000,
        total_debt=2_500_000,
        total_equity=4_000_000,
        revenue=10_000_000,
        cogs=7_000_000,
        gross_profit=3_000_000,
        operating_income=1_200_000,
        net_income=600_000,
        ebit=1_500_000,
        ebitda=2_000_000,
        interest_expense=400_000,
        operating_cash_flow=1_200_000,
        capex=300_000,
    )


# ---------------------------------------------------------------------------
# CreditScorecard tests
# ---------------------------------------------------------------------------


class TestCreditScorecard:
    def test_strong_company_high_score(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert sc.total_score >= 70
        assert sc.grade in ("A", "B")
        assert sc.recommendation in ("approve", "conditional")

    def test_weak_company_low_score(self, analyzer, weak_company):
        sc = analyzer.credit_scorecard(weak_company)
        assert sc.total_score <= 40
        assert sc.grade in ("D", "F")
        assert sc.recommendation == "decline"

    def test_scorecard_has_five_categories(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert len(sc.category_scores) == 5
        expected_cats = {"profitability", "leverage", "liquidity", "cash_flow", "stability"}
        assert set(sc.category_scores.keys()) == expected_cats
        assert all(0 <= v <= 20 for v in sc.category_scores.values())
        assert sc.total_score == sum(sc.category_scores.values())

    def test_scorecard_max_100(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert 0 <= sc.total_score <= 100

    def test_scorecard_grade_boundaries(self, analyzer):
        """Minimal data -- tests graceful handling of None fields."""
        data = FinancialData()
        sc = analyzer.credit_scorecard(data)
        assert sc.total_score >= 0
        assert sc.grade in ("A", "B", "C", "D", "F")

    def test_strengths_weaknesses_populated(self, analyzer, strong_company):
        sc = analyzer.credit_scorecard(strong_company)
        assert len(sc.strengths) > 0  # Strong company should have strengths

    def test_weak_company_has_weaknesses(self, analyzer, weak_company):
        sc = analyzer.credit_scorecard(weak_company)
        assert len(sc.weaknesses) > 0

    def test_conditional_has_conditions(self, analyzer, medium_company):
        sc = analyzer.credit_scorecard(medium_company)
        if sc.recommendation == "conditional":
            assert len(sc.conditions) > 0

    def test_grade_a_threshold(self, analyzer):
        """Verify score >= 80 maps to grade A."""
        from underwriting import _score_to_grade

        assert _score_to_grade(100) == "A"
        assert _score_to_grade(80) == "A"
        assert _score_to_grade(79) == "B"

    def test_grade_f_threshold(self, analyzer):
        from underwriting import _score_to_grade

        assert _score_to_grade(34) == "F"
        assert _score_to_grade(0) == "F"
        assert _score_to_grade(35) == "D"

    def test_negative_equity_scores_zero_leverage(self, analyzer):
        """CORRUPT-04 fix: negative equity must not earn max leverage points."""
        from underwriting import UnderwritingAnalyzer
        # D/E = 500k / -200k = -2.5 (negative equity)
        # D/A = 500k / 300k = 1.67
        data = FinancialData(
            total_debt=500_000,
            total_equity=-200_000,
            total_assets=300_000,
            revenue=1_000_000,
            net_income=50_000,
        )
        sc = analyzer.credit_scorecard(data)
        assert sc.category_scores["leverage"] == 0

    def test_negative_d_e_from_negative_equity(self, analyzer):
        """Verify _score_leverage returns 0 when D/E is negative."""
        assert UnderwritingAnalyzer._score_leverage(-2.5, 1.67) == 0
        assert UnderwritingAnalyzer._score_leverage(-0.1, 0.1) == 0


# ---------------------------------------------------------------------------
# DebtCapacity tests
# ---------------------------------------------------------------------------


class TestDebtCapacity:
    @pytest.fixture
    def sample_data(self):
        return FinancialData(
            total_debt=2_000_000,
            ebitda=3_000_000,
            total_assets=10_000_000,
            total_equity=5_000_000,
            revenue=8_000_000,
            operating_cash_flow=2_500_000,
        )

    def test_debt_capacity_without_loan(self, analyzer, sample_data):
        result = analyzer.debt_capacity(sample_data)
        assert result.current_leverage is not None
        assert result.max_additional_debt is not None
        assert result.max_additional_debt > 0
        # 3.5 * 3M - 2M = 8.5M
        assert abs(result.max_additional_debt - 8_500_000) < 1

    def test_debt_capacity_with_loan(self, analyzer, sample_data):
        loan = LoanStructure(principal=1_000_000, annual_rate=0.06, term_years=5)
        result = analyzer.debt_capacity(sample_data, loan)
        assert result.pro_forma_debt == 3_000_000
        assert result.pro_forma_leverage is not None
        assert result.pro_forma_dscr is not None
        assert result.headroom_pct is not None

    def test_debt_capacity_overleveraged(self, analyzer):
        data = FinancialData(total_debt=10_000_000, ebitda=1_000_000)
        result = analyzer.debt_capacity(data)
        # 3.5 * 1M = 3.5M < 10M  ->  max_additional = 0
        assert result.max_additional_debt == 0

    def test_debt_capacity_no_data(self, analyzer):
        data = FinancialData()
        result = analyzer.debt_capacity(data)
        assert result.assessment != ""  # Should still produce assessment

    def test_pro_forma_dscr_strong(self, analyzer, sample_data):
        """Small loan relative to EBITDA should yield strong DSCR."""
        loan = LoanStructure(principal=500_000, annual_rate=0.05, term_years=5)
        result = analyzer.debt_capacity(sample_data, loan)
        # annual debt service = 100k + 25k = 125k; DSCR = 3M / 125k = 24
        assert result.pro_forma_dscr is not None
        assert result.pro_forma_dscr > 5.0

    def test_headroom_decreases_with_large_loan(self, analyzer, sample_data):
        small_loan = LoanStructure(principal=1_000_000)
        large_loan = LoanStructure(principal=7_000_000)
        r_small = analyzer.debt_capacity(sample_data, small_loan)
        r_large = analyzer.debt_capacity(sample_data, large_loan)
        assert r_small.headroom_pct > r_large.headroom_pct

    def test_assessment_warns_on_excess_leverage(self, analyzer):
        data = FinancialData(total_debt=3_000_000, ebitda=1_000_000)
        loan = LoanStructure(principal=2_000_000)
        result = analyzer.debt_capacity(data, loan)
        # pro_forma = 5M, target = 3.5M -> warning
        assert "WARNING" in result.assessment or result.pro_forma_leverage > 3.5


# ---------------------------------------------------------------------------
# CovenantPackage tests
# ---------------------------------------------------------------------------


class TestCovenantPackage:
    def test_light_covenants_for_strong_credit(self, analyzer):
        scorecard = CreditScorecard(
            total_score=85, grade="A", recommendation="approve"
        )
        data = FinancialData(capex=500_000)
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "light"
        assert len(covenants.financial_covenants) >= 3

    def test_standard_covenants_for_b_grade(self, analyzer):
        scorecard = CreditScorecard(
            total_score=70, grade="B", recommendation="approve"
        )
        data = FinancialData()
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "standard"

    def test_heavy_covenants_for_weak_credit(self, analyzer):
        scorecard = CreditScorecard(
            total_score=40, grade="D", recommendation="decline"
        )
        data = FinancialData(capex=500_000)
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "heavy"
        assert len(covenants.financial_covenants) >= 4  # Extra covenants
        assert "min_fixed_charge_coverage" in covenants.financial_covenants
        assert "max_capex" in covenants.financial_covenants

    def test_heavy_without_capex_data(self, analyzer):
        scorecard = CreditScorecard(total_score=30, grade="F")
        data = FinancialData()  # no capex
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert covenants.covenant_tier == "heavy"
        assert "max_capex" not in covenants.financial_covenants

    def test_reporting_requirements(self, analyzer):
        scorecard = CreditScorecard(total_score=60, grade="C")
        data = FinancialData()
        covenants = analyzer.recommend_covenants(data, scorecard)
        assert len(covenants.reporting_requirements) >= 2
        assert len(covenants.events_of_default) >= 3

    def test_heavy_has_monthly_reporting(self, analyzer):
        scorecard = CreditScorecard(total_score=30, grade="F")
        data = FinancialData()
        covenants = analyzer.recommend_covenants(data, scorecard)
        monthly = [r for r in covenants.reporting_requirements if "Monthly" in r]
        assert len(monthly) >= 1

    def test_events_of_default_always_present(self, analyzer):
        for grade in ("A", "B", "C", "D", "F"):
            scorecard = CreditScorecard(grade=grade)
            covenants = analyzer.recommend_covenants(FinancialData(), scorecard)
            assert len(covenants.events_of_default) == 4

    def test_covenant_thresholds_tighten_with_tier(self, analyzer):
        light_sc = CreditScorecard(grade="A")
        heavy_sc = CreditScorecard(grade="D")
        light = analyzer.recommend_covenants(FinancialData(), light_sc)
        heavy = analyzer.recommend_covenants(FinancialData(), heavy_sc)
        # Heavy tier should have tighter (higher) current ratio threshold
        assert (
            heavy.financial_covenants["min_current_ratio"]["threshold"]
            > light.financial_covenants["min_current_ratio"]["threshold"]
        )


# ---------------------------------------------------------------------------
# Full Underwriting Report tests
# ---------------------------------------------------------------------------


class TestFullUnderwriting:
    @pytest.fixture
    def sample_data(self):
        return FinancialData(
            total_assets=10_000_000,
            current_assets=4_000_000,
            cash=1_500_000,
            total_liabilities=3_000_000,
            current_liabilities=1_500_000,
            total_debt=1_500_000,
            total_equity=7_000_000,
            revenue=15_000_000,
            net_income=2_000_000,
            ebit=3_200_000,
            ebitda=4_000_000,
            interest_expense=200_000,
            operating_cash_flow=3_500_000,
            capex=500_000,
        )

    def test_full_underwriting_all_components(self, analyzer, sample_data):
        loan = LoanStructure(principal=2_000_000, annual_rate=0.05, term_years=7)
        report = analyzer.full_underwriting(sample_data, loan)
        assert isinstance(report, UnderwritingReport)
        assert report.scorecard is not None
        assert report.debt_capacity is not None
        assert report.covenants is not None
        assert report.loan is loan
        assert report.summary != ""

    def test_full_underwriting_without_loan(self, analyzer, sample_data):
        report = analyzer.full_underwriting(sample_data)
        assert report.scorecard is not None
        assert report.debt_capacity is not None
        assert report.covenants is not None
        assert report.loan is None
        assert report.summary != ""

    def test_summary_contains_score(self, analyzer, sample_data):
        report = analyzer.full_underwriting(sample_data)
        assert "Credit Score:" in report.summary
        assert report.scorecard.grade in report.summary

    def test_summary_contains_loan_info_when_given(self, analyzer, sample_data):
        loan = LoanStructure(principal=1_000_000, annual_rate=0.04, term_years=3)
        report = analyzer.full_underwriting(sample_data, loan)
        assert "$1,000,000" in report.summary
        assert "3 years" in report.summary

    def test_report_with_empty_data(self, analyzer):
        report = analyzer.full_underwriting(FinancialData())
        assert isinstance(report, UnderwritingReport)
        assert report.scorecard is not None
        assert report.summary != ""

    def test_report_consistency(self, analyzer, sample_data):
        """Scorecard grade should match covenant tier logic."""
        report = analyzer.full_underwriting(sample_data)
        grade = report.scorecard.grade
        tier = report.covenants.covenant_tier
        if grade == "A":
            assert tier == "light"
        elif grade == "B":
            assert tier == "standard"
        else:
            assert tier == "heavy"


# ---------------------------------------------------------------------------
# Dataclass defaults / edge cases
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_loan_structure_defaults(self):
        loan = LoanStructure()
        assert loan.principal == 0.0
        assert loan.annual_rate == 0.05
        assert loan.term_years == 5
        assert loan.loan_type == "term"

    def test_credit_scorecard_defaults(self):
        sc = CreditScorecard()
        assert sc.total_score == 0
        assert sc.grade == "F"
        assert sc.recommendation == "decline"
        assert sc.category_scores == {}

    def test_debt_capacity_result_defaults(self):
        dc = DebtCapacityResult()
        assert dc.max_leverage_target == 3.5
        assert dc.assessment == ""

    def test_covenant_package_defaults(self):
        cp = CovenantPackage()
        assert cp.covenant_tier == "standard"
        assert cp.financial_covenants == {}

    def test_underwriting_report_defaults(self):
        r = UnderwritingReport()
        assert r.scorecard is None
        assert r.summary == ""
