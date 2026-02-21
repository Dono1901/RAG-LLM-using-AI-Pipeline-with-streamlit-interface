"""Tests for compliance_scorer.py -- regulatory & compliance scoring."""

import pytest

from financial_analyzer import FinancialData
from compliance_scorer import (
    AuditRiskAssessment,
    ComplianceReport,
    ComplianceScorer,
    RegulatoryRatioReport,
    RegulatoryThreshold,
    SECFilingQuality,
    SOXComplianceResult,
    _score_to_grade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compliant_company() -> FinancialData:
    """Company that passes most compliance checks."""
    return FinancialData(
        revenue=10_000_000,
        net_income=1_000_000,
        gross_profit=6_000_000,
        operating_income=2_000_000,
        ebit=2_000_000,
        ebitda=2_500_000,
        interest_expense=200_000,
        cogs=4_000_000,
        total_assets=20_000_000,
        current_assets=8_000_000,
        current_liabilities=3_000_000,
        total_debt=5_000_000,
        total_equity=13_000_000,
        total_liabilities=7_000_000,
        operating_cash_flow=1_500_000,
        cash=2_000_000,
        accounts_receivable=2_000_000,
        inventory=1_500_000,
        accounts_payable=1_000_000,
        depreciation=500_000,
        capex=800_000,
    )


@pytest.fixture
def noncompliant_company() -> FinancialData:
    """Company that fails many compliance checks."""
    return FinancialData(
        revenue=2_000_000,
        net_income=-500_000,
        gross_profit=200_000,
        operating_income=-300_000,
        ebit=-300_000,
        ebitda=-100_000,
        interest_expense=400_000,
        cogs=1_800_000,
        total_assets=3_000_000,
        current_assets=500_000,
        current_liabilities=2_000_000,
        total_debt=3_500_000,
        total_equity=-500_000,  # Negative equity
        total_liabilities=3_500_000,
        operating_cash_flow=-100_000,
        cash=100_000,
        accounts_receivable=1_000_000,  # AR/Rev = 0.50 (high)
    )


@pytest.fixture
def scorer():
    return ComplianceScorer()


# ---------------------------------------------------------------------------
# SOX Compliance
# ---------------------------------------------------------------------------


class TestSOXCompliance:
    def test_compliant_low_risk(self, scorer, compliant_company):
        sox = scorer.sox_compliance(compliant_company)
        assert isinstance(sox, SOXComplianceResult)
        assert sox.overall_risk == "low"
        assert sox.risk_score >= 80
        assert sox.checks_performed > 0
        assert len(sox.material_weakness_indicators) == 0

    def test_noncompliant_high_risk(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        assert sox.overall_risk in ("moderate", "high")
        assert sox.risk_score < 80
        assert len(sox.flags) > 0

    def test_negative_equity_flagged(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        mw = sox.material_weakness_indicators
        neg_equity = [m for m in mw if "negative" in m.lower()]
        assert len(neg_equity) > 0

    def test_high_ar_revenue_flagged(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        ar_flags = [f for f in sox.flags if "AR/Revenue" in f]
        assert len(ar_flags) > 0

    def test_operating_loss_flagged(self, scorer, noncompliant_company):
        sox = scorer.sox_compliance(noncompliant_company)
        loss_flags = [f for f in sox.flags if "Operating loss" in f]
        assert len(loss_flags) > 0

    def test_negative_ni_ocf_ratio_flagged(self, scorer):
        """BUG-2 fix: negative NI with negative OCF should still flag concern,
        even though OCF/NI ratio > 0.5 (both negative → positive ratio)."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=-200_000,
            operating_cash_flow=-150_000,
            total_assets=2_000_000,
            total_equity=500_000,
        )
        sox = scorer.sox_compliance(data)
        # Should flag negative NI as earnings quality concern
        ni_flags = [f for f in sox.flags if "Negative net income" in f]
        assert len(ni_flags) > 0

    def test_positive_ni_low_ocf_flagged(self, scorer):
        """Positive NI with low OCF/NI should still flag divergence."""
        data = FinancialData(
            revenue=1_000_000,
            net_income=200_000,
            operating_cash_flow=50_000,  # OCF/NI = 0.25
            total_assets=2_000_000,
            total_equity=500_000,
        )
        sox = scorer.sox_compliance(data)
        ocf_flags = [f for f in sox.flags if "OCF/NI" in f]
        assert len(ocf_flags) > 0


# ---------------------------------------------------------------------------
# SEC Filing Quality
# ---------------------------------------------------------------------------


class TestSECFilingQuality:
    def test_complete_data_high_score(self, scorer, compliant_company):
        sec = scorer.sec_filing_quality(compliant_company)
        assert isinstance(sec, SECFilingQuality)
        assert sec.disclosure_score >= 60
        assert sec.grade in ("A", "B", "C")
        assert sec.data_completeness_pct > 50

    def test_missing_critical_fields(self, scorer):
        sparse = FinancialData(revenue=1_000_000)
        sec = scorer.sec_filing_quality(sparse)
        assert sec.disclosure_score < 50
        assert len(sec.missing_critical_fields) > 0

    def test_balance_sheet_consistency(self, scorer, compliant_company):
        sec = scorer.sec_filing_quality(compliant_company)
        # compliant_company: A=20M, L=7M, E=13M -> L+E=20M -> passes
        assert sec.consistency_checks_passed >= 1

    def test_balance_sheet_inconsistency_flagged(self, scorer):
        # Assets != Liabilities + Equity
        bad = FinancialData(
            revenue=1_000_000,
            net_income=100_000,
            total_assets=10_000_000,
            total_liabilities=3_000_000,
            total_equity=3_000_000,  # 3M + 3M = 6M != 10M
            operating_cash_flow=200_000,
            current_assets=4_000_000,
            current_liabilities=1_000_000,
        )
        sec = scorer.sec_filing_quality(bad)
        bs_flags = [f for f in sec.red_flags if "balance" in f.lower()]
        assert len(bs_flags) > 0


# ---------------------------------------------------------------------------
# Regulatory Ratios
# ---------------------------------------------------------------------------


class TestRegulatoryRatios:
    def test_compliant_passes_most(self, scorer, compliant_company):
        reg = scorer.regulatory_ratios(compliant_company)
        assert isinstance(reg, RegulatoryRatioReport)
        assert reg.pass_count > 0
        assert reg.compliance_pct > 50

    def test_noncompliant_fails(self, scorer, noncompliant_company):
        reg = scorer.regulatory_ratios(noncompliant_company)
        assert reg.fail_count > 0
        assert len(reg.critical_failures) > 0

    def test_all_thresholds_checked(self, scorer, compliant_company):
        reg = scorer.regulatory_ratios(compliant_company)
        assert len(reg.thresholds_checked) == 6  # 6 rules defined

    def test_threshold_fields_populated(self, scorer, compliant_company):
        reg = scorer.regulatory_ratios(compliant_company)
        for t in reg.thresholds_checked:
            assert isinstance(t, RegulatoryThreshold)
            assert t.rule_name != ""
            assert t.framework != ""
            assert t.threshold_value is not None


# ---------------------------------------------------------------------------
# Audit Risk Assessment
# ---------------------------------------------------------------------------


class TestAuditRisk:
    def test_low_risk_compliant(self, scorer, compliant_company):
        audit = scorer.audit_risk_assessment(compliant_company)
        assert isinstance(audit, AuditRiskAssessment)
        assert audit.risk_level in ("low", "moderate")
        assert audit.score >= 50
        assert not audit.going_concern_risk

    def test_high_risk_noncompliant(self, scorer, noncompliant_company):
        audit = scorer.audit_risk_assessment(noncompliant_company)
        assert audit.risk_level in ("high", "critical")
        assert audit.going_concern_risk is True
        assert len(audit.restatement_risk_indicators) > 0

    def test_recommendations_populated(self, scorer, noncompliant_company):
        audit = scorer.audit_risk_assessment(noncompliant_company)
        assert len(audit.recommendations) > 0


# ---------------------------------------------------------------------------
# Full Compliance Report
# ---------------------------------------------------------------------------


class TestFullComplianceReport:
    def test_full_report_compliant(self, scorer, compliant_company):
        report = scorer.full_compliance_report(compliant_company)
        assert isinstance(report, ComplianceReport)
        assert report.sox is not None
        assert report.sec is not None
        assert report.regulatory is not None
        assert report.audit_risk is not None
        assert len(report.summary) > 20

    def test_full_report_noncompliant(self, scorer, noncompliant_company):
        report = scorer.full_compliance_report(noncompliant_company)
        assert report.audit_risk.going_concern_risk is True
        assert "GOING CONCERN" in report.summary

    def test_summary_contains_scores(self, scorer, compliant_company):
        report = scorer.full_compliance_report(compliant_company)
        assert "SOX" in report.summary
        assert "SEC" in report.summary
        assert "Regulatory" in report.summary
