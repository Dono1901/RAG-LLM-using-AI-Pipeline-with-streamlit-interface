"""Shared helpers for financial export modules (XLSX and PDF).

Contains key-name classification utilities and ratio category grouping
that are used by both export_xlsx.py and export_pdf.py.
"""

from typing import Dict


# ---------------------------------------------------------------------------
# Key-name classification helpers
# ---------------------------------------------------------------------------

_PERCENT_KEYWORDS = (
    "margin", "ratio", "return", "yield", "roe", "roa", "roic",
    "turnover", "coverage", "rate",
)
_DOLLAR_KEYWORDS = (
    "revenue", "income", "assets", "debt", "equity", "cash",
    "expense", "liabilities", "ebit", "ebitda", "capex",
    "profit", "payable", "receivable", "inventory",
)


def _is_percent_key(key: str) -> bool:
    """Return True if *key* looks like a percentage metric."""
    lower = key.lower()
    return any(kw in lower for kw in _PERCENT_KEYWORDS)


def _is_dollar_key(key: str) -> bool:
    """Return True if *key* looks like a dollar-denominated metric."""
    lower = key.lower()
    return any(kw in lower for kw in _DOLLAR_KEYWORDS)


# ---------------------------------------------------------------------------
# Ratio category grouping
# ---------------------------------------------------------------------------

_CATEGORY_MAP: Dict[str, str] = {
    # Liquidity
    "current_ratio": "Liquidity",
    "quick_ratio": "Liquidity",
    "cash_ratio": "Liquidity",
    "working_capital": "Liquidity",
    # Profitability
    "gross_margin": "Profitability",
    "operating_margin": "Profitability",
    "net_margin": "Profitability",
    "roe": "Profitability",
    "roa": "Profitability",
    "roic": "Profitability",
    # Leverage
    "debt_to_equity": "Leverage",
    "debt_to_assets": "Leverage",
    "debt_ratio": "Leverage",
    "equity_multiplier": "Leverage",
    "interest_coverage": "Leverage",
    # Efficiency
    "asset_turnover": "Efficiency",
    "inventory_turnover": "Efficiency",
    "receivables_turnover": "Efficiency",
    "payables_turnover": "Efficiency",
}


def _categorize(key: str) -> str:
    """Return the category for a ratio key, falling back to 'Other'."""
    return _CATEGORY_MAP.get(key, "Other")
