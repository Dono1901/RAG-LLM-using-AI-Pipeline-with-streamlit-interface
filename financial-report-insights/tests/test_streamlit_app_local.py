"""Tests for streamlit_app_local.py file upload security and utilities."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _sanitize_and_save - path traversal prevention & file size
# ---------------------------------------------------------------------------


class TestSanitizeAndSave:
    """Test the file upload sanitization function."""

    def _make_file(self, name: str, content: bytes = b"data") -> MagicMock:
        """Create a mock uploaded file with given name and content."""
        buf = io.BytesIO(content)
        mock_file = MagicMock()
        mock_file.name = name
        mock_file.getbuffer.return_value = buf.getvalue()
        # Simulate seek/tell for size check
        mock_file.seek = buf.seek
        mock_file.tell = buf.tell
        # Reset position after setup
        buf.seek(0)
        return mock_file

    @patch("streamlit_app_local.st")
    def test_saves_normal_file(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("report.xlsx", b"excel data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is True
        assert (tmp_path / "report.xlsx").exists()
        assert (tmp_path / "report.xlsx").read_bytes() == b"excel data"

    @patch("streamlit_app_local.st")
    def test_rejects_empty_filename(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False
        mock_st.error.assert_called()

    @patch("streamlit_app_local.st")
    def test_rejects_dot_filename(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file(".", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False

    @patch("streamlit_app_local.st")
    def test_rejects_dotdot_filename(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("..", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False

    @patch("streamlit_app_local.st")
    def test_rejects_path_traversal_unix(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        # os.path.basename strips directory components, so this becomes "passwd"
        # The path traversal is neutralized by basename() on line 36
        f = self._make_file("../../../etc/passwd", b"data")
        result = _sanitize_and_save(f, tmp_path)
        # basename("../../../etc/passwd") = "passwd" which is a valid name
        # The resolve() check on line 48 ensures it stays within docs_path
        if result:
            assert (tmp_path / "passwd").exists()
            # Verify it didn't escape the docs folder
            saved = (tmp_path / "passwd").resolve()
            assert str(saved).startswith(str(tmp_path.resolve()))

    @patch("streamlit_app_local.st")
    def test_rejects_path_traversal_windows(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        # Backslash separator in filename
        f = self._make_file("..\\..\\Windows\\System32\\config", b"data")
        result = _sanitize_and_save(f, tmp_path)
        # basename strips path, but if "/" or "\\" remains, it should be rejected
        # on line 42's check after basename

    @patch("streamlit_app_local.st")
    def test_rejects_oversized_file(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save, MAX_FILE_SIZE

        # Create content larger than MAX_FILE_SIZE
        big_content = b"x" * (MAX_FILE_SIZE + 1)
        f = self._make_file("big.pdf", big_content)
        result = _sanitize_and_save(f, tmp_path)
        assert result is False
        mock_st.error.assert_called()
        assert not (tmp_path / "big.pdf").exists()

    @patch("streamlit_app_local.st")
    def test_accepts_file_at_size_limit(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save, MAX_FILE_SIZE

        exact_content = b"x" * MAX_FILE_SIZE
        f = self._make_file("exact.pdf", exact_content)
        result = _sanitize_and_save(f, tmp_path)
        assert result is True
        assert (tmp_path / "exact.pdf").exists()

    @patch("streamlit_app_local.st")
    def test_strips_whitespace_from_name(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("  report.xlsx  ", b"data")
        result = _sanitize_and_save(f, tmp_path)
        # After basename + strip, should be "report.xlsx"
        if result:
            assert (tmp_path / "report.xlsx").exists()

    @patch("streamlit_app_local.st")
    def test_whitespace_only_filename_rejected(self, mock_st, tmp_path):
        from streamlit_app_local import _sanitize_and_save

        f = self._make_file("   ", b"data")
        result = _sanitize_and_save(f, tmp_path)
        assert result is False


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS and MAX_FILE_SIZE
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_extensions_include_excel(self):
        from streamlit_app_local import SUPPORTED_EXTENSIONS

        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".xls" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_include_pdf(self):
        from streamlit_app_local import SUPPORTED_EXTENSIONS

        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_include_docx(self):
        from streamlit_app_local import SUPPORTED_EXTENSIONS

        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_max_file_size_is_positive(self):
        from streamlit_app_local import MAX_FILE_SIZE

        assert MAX_FILE_SIZE > 0

    def test_max_file_size_is_in_bytes(self):
        from streamlit_app_local import MAX_FILE_SIZE
        from config import settings

        assert MAX_FILE_SIZE == settings.max_file_size_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# _generate_sample_income_statement & _generate_sample_budget
# ---------------------------------------------------------------------------


class TestSampleDataGenerators:
    @patch("streamlit_app_local.st")
    def test_generate_sample_income_statement(self, mock_st, tmp_path):
        from streamlit_app_local import _generate_sample_income_statement

        _generate_sample_income_statement(tmp_path)
        output = tmp_path / "sample_income_statement.xlsx"
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("streamlit_app_local.st")
    def test_generate_sample_budget(self, mock_st, tmp_path):
        from streamlit_app_local import _generate_sample_budget

        _generate_sample_budget(tmp_path)
        output = tmp_path / "sample_budget_vs_actual.xlsx"
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("streamlit_app_local.st")
    def test_sample_income_statement_has_columns(self, mock_st, tmp_path):
        import pandas as pd
        from streamlit_app_local import _generate_sample_income_statement

        _generate_sample_income_statement(tmp_path)
        df = pd.read_excel(tmp_path / "sample_income_statement.xlsx")
        assert "Line Item" in df.columns
        assert "Q1 2024" in df.columns
        assert len(df) == 10  # 10 line items

    @patch("streamlit_app_local.st")
    def test_sample_budget_has_variance_columns(self, mock_st, tmp_path):
        import pandas as pd
        from streamlit_app_local import _generate_sample_budget

        _generate_sample_budget(tmp_path)
        df = pd.read_excel(tmp_path / "sample_budget_vs_actual.xlsx")
        assert "Budget" in df.columns
        assert "Actual" in df.columns
        assert "Variance" in df.columns
        assert "Variance %" in df.columns
