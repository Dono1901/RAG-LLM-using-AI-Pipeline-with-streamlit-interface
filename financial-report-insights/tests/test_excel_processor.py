"""Tests for excel_processor.py core processing engine."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor(tmp_path):
    from excel_processor import ExcelProcessor

    return ExcelProcessor(str(tmp_path))


@pytest.fixture
def sample_xlsx(tmp_path):
    """Create a sample xlsx file with financial data."""
    df = pd.DataFrame({
        "Line Item": ["Revenue", "COGS", "Gross Profit", "Operating Expenses", "Net Income"],
        "Q1 2024": [1000000, 400000, 600000, 200000, 400000],
        "Q2 2024": [1100000, 440000, 660000, 210000, 450000],
    })
    path = tmp_path / "test_financials.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    df = pd.DataFrame({
        "Category": ["Revenue", "Expenses", "Net Income"],
        "Amount": [500000, 300000, 200000],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_tsv(tmp_path):
    """Create a sample TSV file."""
    df = pd.DataFrame({
        "Item": ["Cash", "Inventory", "Receivables"],
        "Value": [100000, 50000, 75000],
    })
    path = tmp_path / "test_data.tsv"
    df.to_csv(path, index=False, sep="\t")
    return path


@pytest.fixture
def empty_xlsx(tmp_path):
    """Create an empty xlsx file."""
    df = pd.DataFrame()
    path = tmp_path / "empty.xlsx"
    df.to_excel(path, index=False)
    return path


# ---------------------------------------------------------------------------
# scan_for_excel_files
# ---------------------------------------------------------------------------


class TestScanForExcelFiles:
    def test_finds_xlsx(self, processor, sample_xlsx):
        files = processor.scan_for_excel_files()
        assert any(f.name == "test_financials.xlsx" for f in files)

    def test_finds_csv(self, processor, sample_csv):
        files = processor.scan_for_excel_files()
        assert any(f.name == "test_data.csv" for f in files)

    def test_finds_tsv(self, processor, sample_tsv):
        files = processor.scan_for_excel_files()
        assert any(f.name == "test_data.tsv" for f in files)

    def test_empty_folder(self, tmp_path):
        from excel_processor import ExcelProcessor

        proc = ExcelProcessor(str(tmp_path / "no_docs"))
        (tmp_path / "no_docs").mkdir()
        files = proc.scan_for_excel_files()
        assert files == []

    def test_ignores_non_excel(self, processor, tmp_path):
        (tmp_path / "readme.txt").write_text("not excel")
        (tmp_path / "image.png").write_bytes(b"fake png")
        files = processor.scan_for_excel_files()
        names = {f.name for f in files}
        assert "readme.txt" not in names
        assert "image.png" not in names


# ---------------------------------------------------------------------------
# load_workbook
# ---------------------------------------------------------------------------


class TestLoadWorkbook:
    def test_load_xlsx(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        assert len(wb.sheets) >= 1
        assert wb.metadata["sheet_count"] >= 1
        assert wb.metadata["total_rows"] >= 5
        assert wb.file_path == sample_xlsx

    def test_load_csv(self, processor, sample_csv):
        wb = processor.load_workbook(sample_csv)
        assert len(wb.sheets) == 1
        assert wb.sheets[0].name == "test_data"
        assert len(wb.sheets[0].df) == 3

    def test_load_tsv(self, processor, sample_tsv):
        wb = processor.load_workbook(sample_tsv)
        assert len(wb.sheets) == 1
        assert len(wb.sheets[0].df) == 3

    def test_unsupported_format_raises(self, processor, tmp_path):
        bad_file = tmp_path / "data.json"
        bad_file.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            processor.load_workbook(bad_file)

    def test_workbook_cached(self, processor, sample_xlsx):
        processor.load_workbook(sample_xlsx)
        assert str(sample_xlsx) in processor.loaded_workbooks

    def test_sheet_has_columns_analyzed(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        sheet = wb.sheets[0]
        assert len(sheet.columns) > 0
        assert all(hasattr(c, "name") for c in sheet.columns)
        assert all(hasattr(c, "is_numeric") for c in sheet.columns)


# ---------------------------------------------------------------------------
# _is_time_period
# ---------------------------------------------------------------------------


class TestIsTimePeriod:
    def test_year(self, processor):
        assert processor._is_time_period("2024") is True

    def test_quarter(self, processor):
        assert processor._is_time_period("Q1 2024") is True
        assert processor._is_time_period("q3 2023") is True

    def test_fiscal_year(self, processor):
        assert processor._is_time_period("FY24") is True
        assert processor._is_time_period("fy2024") is True

    def test_month_name(self, processor):
        assert processor._is_time_period("January") is True
        assert processor._is_time_period("mar") is True

    def test_not_time_period(self, processor):
        assert processor._is_time_period("Revenue") is False
        assert processor._is_time_period("Amount") is False
        assert processor._is_time_period("Category") is False


# ---------------------------------------------------------------------------
# extract_financial_tables
# ---------------------------------------------------------------------------


class TestExtractFinancialTables:
    def test_extracts_from_sheet(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        tables = processor.extract_financial_tables(wb.sheets[0])
        assert len(tables) >= 1
        table = tables[0]
        assert table.source_sheet == wb.sheets[0].name
        assert len(table.line_items) > 0

    def test_empty_sheet_returns_no_tables(self, processor, empty_xlsx):
        wb = processor.load_workbook(empty_xlsx)
        if wb.sheets:
            tables = processor.extract_financial_tables(wb.sheets[0])
            assert tables == []

    def test_time_periods_detected(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        tables = processor.extract_financial_tables(wb.sheets[0])
        if tables:
            # Q1 2024 and Q2 2024 should be detected
            assert any("Q" in tp or "2024" in tp for tp in tables[0].time_periods)


# ---------------------------------------------------------------------------
# combine_sheets_intelligently
# ---------------------------------------------------------------------------


class TestCombineSheetsIntelligently:
    def test_single_sheet_workbook(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        combined = processor.combine_sheets_intelligently(wb)
        # Single-sheet workbook should have the sheet in separate_sheets
        assert len(combined.separate_sheets) >= 1

    def test_csv_combine(self, processor, sample_csv):
        wb = processor.load_workbook(sample_csv)
        combined = processor.combine_sheets_intelligently(wb)
        assert combined is not None

    def test_relationships_is_list(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        combined = processor.combine_sheets_intelligently(wb)
        assert isinstance(combined.relationships, list)


# ---------------------------------------------------------------------------
# to_rag_chunks
# ---------------------------------------------------------------------------


class TestToRagChunks:
    def test_creates_chunks(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        combined = processor.combine_sheets_intelligently(wb)
        chunks = processor.to_rag_chunks(combined, wb)
        assert len(chunks) >= 1
        # First chunk should be the summary
        assert chunks[0].text_content  # Not empty

    def test_chunks_have_sheet_name(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        combined = processor.combine_sheets_intelligently(wb)
        chunks = processor.to_rag_chunks(combined, wb)
        for chunk in chunks:
            assert chunk.sheet_name  # Not empty or None

    def test_csv_chunks(self, processor, sample_csv):
        wb = processor.load_workbook(sample_csv)
        combined = processor.combine_sheets_intelligently(wb)
        chunks = processor.to_rag_chunks(combined, wb)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# get_sheet_as_dataframe
# ---------------------------------------------------------------------------


class TestGetSheetAsDataframe:
    def test_returns_first_sheet_by_default(self, processor, sample_xlsx):
        df = processor.get_sheet_as_dataframe(sample_xlsx)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 5

    def test_missing_sheet_raises(self, processor, sample_xlsx):
        processor.load_workbook(sample_xlsx)
        with pytest.raises(ValueError, match="not found"):
            processor.get_sheet_as_dataframe(sample_xlsx, sheet_name="NonExistent")

    def test_lazy_loads_workbook(self, processor, sample_xlsx):
        # Don't call load_workbook first
        assert str(sample_xlsx) not in processor.loaded_workbooks
        df = processor.get_sheet_as_dataframe(sample_xlsx)
        assert str(sample_xlsx) in processor.loaded_workbooks
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_all_numeric_columns
# ---------------------------------------------------------------------------


class TestGetAllNumericColumns:
    def test_finds_numeric_columns(self, processor, sample_xlsx):
        wb = processor.load_workbook(sample_xlsx)
        numeric = processor.get_all_numeric_columns(wb)
        assert len(numeric) >= 1
        # Should find Q1 2024 and Q2 2024 as numeric
        for sheet_name, cols in numeric.items():
            assert len(cols) >= 1


# ---------------------------------------------------------------------------
# process_excel_for_rag (convenience function)
# ---------------------------------------------------------------------------


class TestProcessExcelForRag:
    def test_end_to_end(self, tmp_path, sample_xlsx):
        from excel_processor import process_excel_for_rag

        docs = process_excel_for_rag(sample_xlsx, str(tmp_path))
        assert isinstance(docs, list)
        assert len(docs) >= 1
        # Each doc should have content and source
        for doc in docs:
            assert "content" in doc
            assert "source" in doc


# ---------------------------------------------------------------------------
# Column detection patterns
# ---------------------------------------------------------------------------


class TestColumnPatterns:
    def test_revenue_patterns_match(self):
        from excel_processor import ExcelProcessor

        patterns = ExcelProcessor.REVENUE_PATTERNS
        assert any(p.search("Revenue") for p in patterns)
        assert any(p.search("Net Sales") for p in patterns)
        assert any(p.search("Total Revenue") for p in patterns)

    def test_expense_patterns_match(self):
        from excel_processor import ExcelProcessor

        patterns = ExcelProcessor.EXPENSE_PATTERNS
        assert any(p.search("Operating Expenses") for p in patterns)
        assert any(p.search("COGS") for p in patterns)
        assert any(p.search("SG&A") for p in patterns)

    def test_asset_patterns_match(self):
        from excel_processor import ExcelProcessor

        patterns = ExcelProcessor.ASSET_PATTERNS
        assert any(p.search("Total Assets") for p in patterns)
        assert any(p.search("Cash") for p in patterns)
        assert any(p.search("Inventory") for p in patterns)

    def test_liability_patterns_match(self):
        from excel_processor import ExcelProcessor

        patterns = ExcelProcessor.LIABILITY_PATTERNS
        assert any(p.search("Accounts Payable") for p in patterns)
        assert any(p.search("Total Debt") for p in patterns)
        assert any(p.search("Accrued Expenses") for p in patterns)

    def test_equity_patterns_match(self):
        from excel_processor import ExcelProcessor

        patterns = ExcelProcessor.EQUITY_PATTERNS
        assert any(p.search("Shareholders Equity") for p in patterns)
        assert any(p.search("Retained Earnings") for p in patterns)
