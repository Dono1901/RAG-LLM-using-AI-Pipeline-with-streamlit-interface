"""Tests for pdf_parser module."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pdf_parser import (
    _detect_section_type,
    _extract_metadata,
    _split_into_sections,
    _detect_tables_in_section,
    ParsedSection,
    ParsedDocument,
    SECTION_PATTERNS,
)


class TestDetectSectionType:
    def test_income_statement(self):
        assert _detect_section_type("Consolidated Statements of Income\nRevenue...") == "income_statement"
        assert _detect_section_type("INCOME STATEMENT\nLine items...") == "income_statement"
        assert _detect_section_type("Profit and Loss Statement\n...") == "income_statement"

    def test_balance_sheet(self):
        assert _detect_section_type("Consolidated Balance Sheets\n...") == "balance_sheet"
        assert _detect_section_type("Statement of Financial Position\n...") == "balance_sheet"

    def test_cash_flow(self):
        assert _detect_section_type("Consolidated Statements of Cash Flows\n...") == "cash_flow_statement"
        assert _detect_section_type("Cash Flow Statement\n...") == "cash_flow_statement"

    def test_mda(self):
        assert _detect_section_type("Management's Discussion and Analysis\n...") == "mda"
        assert _detect_section_type("MD&A\nThe following...") == "mda"

    def test_risk_factors(self):
        assert _detect_section_type("Risk Factors\nThe company faces...") == "risk_factors"

    def test_dcf(self):
        assert _detect_section_type("Discounted Cash Flow Analysis\n...") == "dcf"
        assert _detect_section_type("DCF Valuation\n...") == "dcf"

    def test_lbo(self):
        assert _detect_section_type("Leveraged Buyout Analysis\n...") == "lbo"
        assert _detect_section_type("LBO Model\n...") == "lbo"

    def test_cannabis_280e(self):
        assert _detect_section_type("280E Tax Classification\n...") == "280e_tax"

    def test_unknown_section(self):
        assert _detect_section_type("Some random text without financial headers") is None

    def test_notes(self):
        assert _detect_section_type("Notes to Financial Statements\n...") == "notes_to_financial_statements"

    def test_auditor_report(self):
        assert _detect_section_type("Report of Independent Registered Public Accounting Firm\n...") == "auditor_report"

    def test_comparable_companies(self):
        assert _detect_section_type("Comparable Companies Analysis\n...") == "comparable_companies"
        assert _detect_section_type("Trading Comps\n...") == "comparable_companies"

    def test_debt_schedule(self):
        assert _detect_section_type("Debt Schedule Summary\n...") == "debt_schedule"
        assert _detect_section_type("Capital Structure Summary\n...") == "debt_schedule"

    def test_covenant(self):
        assert _detect_section_type("Covenant Compliance Summary\n...") == "covenant_analysis"

    def test_rent_roll(self):
        assert _detect_section_type("Rent Roll\nUnit details...") == "rent_roll"

    def test_construction_budget(self):
        assert _detect_section_type("Construction Budget\nCode...") == "construction_budget"

    def test_kpi(self):
        assert _detect_section_type("KPI Dashboard\nMetrics...") == "kpi_dashboard"
        assert _detect_section_type("Key Performance Indicators\n...") == "kpi_dashboard"

    def test_sensitivity(self):
        assert _detect_section_type("Sensitivity Analysis\n...") == "sensitivity_analysis"


class TestExtractMetadata:
    def test_company_name_detection(self):
        text = "\nApple Inc.\nConsolidated Balance Sheets\n"
        meta = _extract_metadata(text, "test.pdf")
        assert meta.get("company") is not None

    def test_period_detection(self):
        text = "For the fiscal year ended December 31, 2024\n"
        meta = _extract_metadata(text, "test.pdf")
        assert "period" in meta
        assert "2024" in meta["period"]

    def test_filing_type_10k(self):
        text = "ANNUAL REPORT PURSUANT TO 10-K\n"
        meta = _extract_metadata(text, "test.pdf")
        assert meta.get("filing_type") == "10-K"

    def test_filing_type_10q(self):
        text = "QUARTERLY REPORT 10-Q\n"
        meta = _extract_metadata(text, "test.pdf")
        assert meta.get("filing_type") == "10-Q"

    def test_empty_text(self):
        meta = _extract_metadata("", "test.pdf")
        assert isinstance(meta, dict)


class TestSplitIntoSections:
    def test_markdown_headings(self):
        text = "# Introduction\nSome text.\n\n## Financial Overview\nMore text.\n\n### Revenue\nDetails."
        sections = _split_into_sections(text)
        assert len(sections) >= 2

    def test_no_headings(self):
        text = "Just a plain paragraph of text without any headings."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0]["title"] == "Document"

    def test_preamble_captured(self):
        # Preamble needs >100 chars of real text before first heading
        preamble = "This is preamble text. " * 10  # ~230 chars
        text = f"{preamble}\n\n# First Section\nContent here with enough words."
        sections = _split_into_sections(text)
        assert any(s["title"] == "Preamble" for s in sections)


class TestDetectTables:
    def test_markdown_table(self):
        content = "Some text\n| Col1 | Col2 |\n|---|---|\n| val1 | val2 |\n| val3 | val4 |\nMore text"
        tables = _detect_tables_in_section(content)
        assert len(tables) == 1
        assert "val1" in tables[0]

    def test_no_table(self):
        content = "Just text without any table formatting."
        tables = _detect_tables_in_section(content)
        assert len(tables) == 0

    def test_multiple_tables(self):
        content = (
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            "Some text between tables\n\n"
            "| C | D |\n|---|---|\n| 3 | 4 |\n"
        )
        tables = _detect_tables_in_section(content)
        assert len(tables) == 2


class TestSectionPatterns:
    def test_all_categories_have_patterns(self):
        for section_type, patterns in SECTION_PATTERNS.items():
            assert len(patterns) > 0, f"Section {section_type} has no patterns"
            for p in patterns:
                assert hasattr(p, "search"), f"Pattern in {section_type} is not a regex"

    def test_pattern_count(self):
        total = sum(len(p) for p in SECTION_PATTERNS.values())
        assert total >= 50, f"Expected 50+ patterns, got {total}"
