"""
Prompt engineering module for the financial-report-insights RAG system.

Provides versioned, query-type-specific prompt templates and formatters
that replace hardcoded prompt strings throughout app_local.py.
"""

from prompts.templates import (
    PROMPT_VERSION,
    PromptTemplate,
    RATIO_LOOKUP_PROMPT,
    TREND_ANALYSIS_PROMPT,
    COMPARISON_PROMPT,
    EXPLANATION_PROMPT,
    GENERAL_PROMPT,
    get_prompt_for_query_type,
)
from prompts.formatters import (
    format_context_with_citations,
    format_few_shot_examples,
    build_prompt,
    format_json_instruction,
)

__all__ = [
    "PROMPT_VERSION",
    "PromptTemplate",
    "RATIO_LOOKUP_PROMPT",
    "TREND_ANALYSIS_PROMPT",
    "COMPARISON_PROMPT",
    "EXPLANATION_PROMPT",
    "GENERAL_PROMPT",
    "get_prompt_for_query_type",
    "format_context_with_citations",
    "format_few_shot_examples",
    "build_prompt",
    "format_json_instruction",
]
