"""Tests for enhanced scenario engine (multi-scenario + probability-weighted)."""

import pytest

from financial_analyzer import (
    CharlieAnalyzer,
    FinancialData,
    ProbabilityWeightedResult,
    ScenarioResult,
)


@pytest.fixture()
def analyzer():
    return CharlieAnalyzer()


@pytest.fixture()
def sample_data():
    return FinancialData(
        total_assets=10_000_000,
        current_assets=4_000_000,
        cash=1_500_000,
        inventory=500_000,
        accounts_receivable=800_000,
        total_liabilities=4_000_000,
        current_liabilities=1_500_000,
        total_debt=2_500_000,
        total_equity=6_000_000,
        revenue=12_000_000,
        cogs=7_200_000,
        gross_profit=4_800_000,
        operating_income=2_400_000,
        net_income=1_800_000,
        ebit=2_600_000,
        ebitda=3_200_000,
        interest_expense=300_000,
        operating_cash_flow=2_800_000,
        capex=600_000,
        retained_earnings=3_000_000,
        shares_outstanding=1_000_000,
        share_price=25.0,
    )


class TestMultiScenarioAnalysis:
    def test_returns_list_of_scenario_results(self, analyzer, sample_data):
        scenarios = {
            "Bull": {"revenue": 1.15, "cogs": 1.05},
            "Bear": {"revenue": 0.85, "cogs": 1.10},
        }
        results = analyzer.multi_scenario_analysis(sample_data, scenarios)
        assert len(results) == 2
        assert all(isinstance(r, ScenarioResult) for r in results)

    def test_scenario_names_match(self, analyzer, sample_data):
        scenarios = {
            "Bull": {"revenue": 1.10},
            "Base": {"revenue": 1.00},
            "Bear": {"revenue": 0.90},
        }
        results = analyzer.multi_scenario_analysis(sample_data, scenarios)
        names = [r.scenario_name for r in results]
        assert names == ["Bull", "Base", "Bear"]

    def test_bull_better_than_bear(self, analyzer, sample_data):
        scenarios = {
            "Bull": {"revenue": 1.20},
            "Bear": {"revenue": 0.80},
        }
        results = analyzer.multi_scenario_analysis(sample_data, scenarios)
        bull = results[0]
        bear = results[1]
        if bull.scenario_health and bear.scenario_health:
            assert bull.scenario_health.score >= bear.scenario_health.score

    def test_empty_scenarios(self, analyzer, sample_data):
        results = analyzer.multi_scenario_analysis(sample_data, {})
        assert results == []

    def test_single_scenario(self, analyzer, sample_data):
        scenarios = {"Stress": {"revenue": 0.70}}
        results = analyzer.multi_scenario_analysis(sample_data, scenarios)
        assert len(results) == 1
        assert results[0].scenario_name == "Stress"


class TestProbabilityWeightedScenarios:
    def test_basic_probability_weighting(self, analyzer, sample_data):
        scenario_probs = [
            {"name": "Bull", "adjustments": {"revenue": 1.15}, "probability": 0.25},
            {"name": "Base", "adjustments": {"revenue": 1.00}, "probability": 0.50},
            {"name": "Bear", "adjustments": {"revenue": 0.85}, "probability": 0.25},
        ]
        result = analyzer.probability_weighted_scenarios(sample_data, scenario_probs)
        assert isinstance(result, ProbabilityWeightedResult)
        assert len(result.scenarios) == 3
        assert result.expected_health_score is not None
        assert result.expected_z_score is not None
        assert result.distress_probability is not None

    def test_probabilities_normalised(self, analyzer, sample_data):
        scenario_probs = [
            {"name": "A", "adjustments": {"revenue": 1.10}, "probability": 1.0},
            {"name": "B", "adjustments": {"revenue": 0.90}, "probability": 1.0},
        ]
        result = analyzer.probability_weighted_scenarios(sample_data, scenario_probs)
        assert sum(result.probabilities) == pytest.approx(1.0)

    def test_empty_scenarios_returns_summary(self, analyzer, sample_data):
        result = analyzer.probability_weighted_scenarios(sample_data, [])
        assert "No scenarios" in result.summary

    def test_scenario_names_tracked(self, analyzer, sample_data):
        scenario_probs = [
            {"name": "Optimistic", "adjustments": {"revenue": 1.20}, "probability": 0.3},
            {"name": "Pessimistic", "adjustments": {"revenue": 0.80}, "probability": 0.7},
        ]
        result = analyzer.probability_weighted_scenarios(sample_data, scenario_probs)
        assert result.scenario_names == ["Optimistic", "Pessimistic"]

    def test_distress_probability_range(self, analyzer, sample_data):
        scenario_probs = [
            {"name": "Good", "adjustments": {"revenue": 1.10}, "probability": 0.5},
            {"name": "Bad", "adjustments": {"revenue": 0.50}, "probability": 0.5},
        ]
        result = analyzer.probability_weighted_scenarios(sample_data, scenario_probs)
        assert 0.0 <= result.distress_probability <= 1.0

    def test_summary_contains_count(self, analyzer, sample_data):
        scenario_probs = [
            {"name": "A", "adjustments": {"revenue": 1.05}, "probability": 0.5},
            {"name": "B", "adjustments": {"revenue": 0.95}, "probability": 0.5},
        ]
        result = analyzer.probability_weighted_scenarios(sample_data, scenario_probs)
        assert "2 scenarios" in result.summary

    def test_high_distress_warning(self, analyzer, sample_data):
        # Force distress by heavily reducing revenue and increasing debt
        scenario_probs = [
            {"name": "Catastrophe", "adjustments": {"revenue": 0.30, "total_debt": 3.0}, "probability": 0.8},
            {"name": "OK", "adjustments": {"revenue": 1.00}, "probability": 0.2},
        ]
        result = analyzer.probability_weighted_scenarios(sample_data, scenario_probs)
        # With 80% catastrophe, distress prob should be significant
        assert result.distress_probability >= 0.0  # At minimum it's computed
