# Audit Report: Agent Framework & Evaluation
**Date:** 2026-03-16
**Scope:** Agent base, specialized agents, tools, workflows, evaluation harness, metrics
**Files Reviewed:**
- `financial-report-insights/agents/__init__.py`
- `financial-report-insights/agents/base.py`
- `financial-report-insights/agents/specialized.py`
- `financial-report-insights/agents/tools.py`
- `financial-report-insights/agents/workflows.py`
- `financial-report-insights/evaluation/__init__.py`
- `financial-report-insights/evaluation/answer_metrics.py`
- `financial-report-insights/evaluation/eval_harness.py`
- `financial-report-insights/evaluation/golden_qa.py`
- `financial-report-insights/evaluation/response_quality.py`
- `financial-report-insights/evaluation/retrieval_metrics.py`

---

## P0 — Critical (Fix Immediately)

### 1. Prompt injection via unescaped user input in agent prompts
- **File:** `financial-report-insights/agents/base.py:356–360` (`plan`) and `base.py:604` (`_format_step_prompt`)
- **Severity:** Critical
- **Detail:** The raw user `query` string is interpolated directly into LLM prompts with no sanitization or escaping: `f"Task: {query}"` and `f"Question: {query}\n"`. A query containing sequences like `TOOL: calculate_ratio\nARGS: ratio_name=roe` would be parsed by `_select_tool` (line 482) using `_TOOL_PATTERN` and could cause the agent to call an arbitrary registered tool with attacker-controlled arguments. Since `execute_step` passes `tool_call.arguments` directly to `tool.execute(**tool_call.arguments)` (line 394) with no parameter allow-listing, a crafted query can invoke any registered tool with any kwargs it accepts.
- **Suggestion:** Strip or escape TOOL/Action/ARGS directive patterns from user-supplied input before building prompts. At minimum, validate that the resolved `tool_name` and all argument keys/values are within expected bounds for the specific tool before execution.

### 2. Prompt injection via workflow context interpolation
- **File:** `financial-report-insights/agents/workflows.py:94–132` (`_resolve_template`)
- **Severity:** Critical
- **Detail:** `_resolve_template` substitutes `{key}` placeholders from a shared `context` dict that accumulates the string outputs of every prior agent step. A compromised or misbehaving agent that returns a string containing `{ratio_analysis}` or a TOOL directive in its output can inject content into subsequent agents' prompts when the downstream step template is resolved. For example, if the `ratio_analyst` step output contains `TOOL: assess_distress\nARGS: total_assets=1`, the `trend_forecaster` step that receives `{ratio_analysis}` in its template will propagate that directive into the LLM context.
- **Suggestion:** Sanitize or HTML-encode context values before substitution. Consider wrapping substituted values in sentinel blocks (e.g., `--- BEGIN PRIOR OUTPUT ---\n{value}\n--- END PRIOR OUTPUT ---`) to prevent the LLM from treating embedded content as instructions.

---

## P1 — High (Fix This Sprint)

### 3. Unbounded parallel workflow with no per-step timeout
- **File:** `financial-report-insights/agents/workflows.py:359–396` (`ParallelWorkflow.run`)
- **Severity:** High
- **Detail:** The `ThreadPoolExecutor` submits agent steps via `executor.submit(_run_step, ...)` and collects results with `as_completed(future_to_step)`. `as_completed` has no timeout argument here, meaning a hung agent (e.g., waiting on a stalled LLM HTTP connection) will block the entire workflow indefinitely. The `BaseAgent.run` loop itself also has no wall-clock time limit — only a `max_steps` counter (line 441) which guards against step count, not elapsed time.
- **Suggestion:** Pass a `timeout` argument to `as_completed` and handle `concurrent.futures.TimeoutError`. Add an optional `step_timeout_seconds` parameter to `ParallelWorkflow` and `_run_step`. Additionally consider adding a wall-clock deadline inside `BaseAgent.run`.

### 4. Thread-safety violation: shared mutable `AgentMemory` in parallel workflows
- **File:** `financial-report-insights/agents/workflows.py:382–383` and `financial-report-insights/agents/base.py:90–164`
- **Severity:** High
- **Detail:** When the same agent instance (with its shared `AgentMemory`) is submitted to multiple concurrent futures in `ParallelWorkflow`, `execute_step` (base.py:413) calls `self.memory.add_message(...)` from multiple threads simultaneously. `AgentMemory._history` is a `collections.deque` which is not thread-safe for concurrent appends. In practice the factory functions in `specialized.py` create fresh agent instances per workflow, but nothing in the API prevents a caller from passing the same agent to multiple parallel steps.
- **Suggestion:** Add a threading lock inside `AgentMemory.add_message`/`store_fact`, or document clearly that agent instances must not be shared across parallel steps and add an assertion in `ParallelWorkflow.__init__` that agent names map 1:1 to steps.

### 5. Bare `except Exception` in `tool_forecast` silently swallows ML import errors
- **File:** `financial-report-insights/agents/tools.py:646` (`tool_forecast`)
- **Severity:** High
- **Detail:** The block `except Exception:` (line 646) catches every exception from the `ml.forecasting` import and model fitting, including `ImportError`, `AttributeError`, and logic errors in `SimpleARModel`, then silently falls back to linear extrapolation. A caller has no visibility into whether the AR(2) model was used or why it failed. A broken `ml.forecasting` module (e.g., after a refactor) will silently degrade forecast quality in production with no log output.
- **Suggestion:** Change to `except Exception as exc:` and log a warning with the exception string, as is done in `Tool.execute` at base.py:213. Separate `ImportError` (optional dependency missing) from genuine computation failures.

### 6. Tool argument parsing cannot handle values containing spaces or commas
- **File:** `financial-report-insights/agents/base.py:515` (`_parse_tool_arguments`)
- **Severity:** High
- **Detail:** Arguments are parsed by splitting on `r"[,\s]+"` (line 515) and then partitioning on `=`. Any value that contains a space or comma (e.g., a company name `label_a=Acme Corp, 2025`) will be incorrectly split into multiple fragments. This is exploitable: an attacker-controlled query can craft an ARGS line that injects unexpected key=value pairs by embedding commas in values.
- **Suggestion:** Use a proper key=value parser (e.g., `csv.reader` with `shlex.split` or a JSON-based argument format). Consider switching to structured JSON for tool calls: `ARGS: {"ratio_name": "roe", "label_a": "Acme Corp"}`.

### 7. `evaluate_retrieval` calls `doc.get("source", "")` with no type guard
- **File:** `financial-report-insights/evaluation/eval_harness.py:92–93`
- **Severity:** High
- **Detail:** `retrieved_docs` comes from `self.rag.retrieve(...)`, whose return type is not validated. If a document object is not a dict (e.g., a plain string, a dataclass, or a custom object), calling `.get()` will raise `AttributeError`, crashing the entire evaluation loop with no error recorded per-query. The same unguarded `.get("content", "")` pattern appears in `evaluate_answers` at line 164.
- **Suggestion:** Add `isinstance(doc, dict)` guards before calling `.get()`. Log and skip malformed documents rather than crashing. The `tool_search_documents` function in tools.py already demonstrates the correct pattern with `if isinstance(doc, dict)` at line 897.

---

## P2 — Medium (Fix Soon)

### 8. SYSTEM_PROMPT class constants are defined but never injected into LLM calls
- **File:** `financial-report-insights/agents/specialized.py:301–307, 358–365, 412–418, 469–475`
- **Severity:** Medium
- **Detail:** All four specialized agents define a `SYSTEM_PROMPT` class attribute but neither `BaseAgent` nor the specialized `run()` overrides prepend this prompt to any LLM call. The LLM therefore has no domain persona when generating responses — the entire persona definition is dead code. This reduces output quality and is likely a bug.
- **Suggestion:** Prepend `self.SYSTEM_PROMPT` (or the equivalent for `BaseAgent`) to the system context in `_format_step_prompt` or `_synthesise_answer`. A clean approach: pass it as the first message added to `AgentMemory` during `__init__`.

### 9. `_format_prompt` is defined but never called
- **File:** `financial-report-insights/agents/base.py:535–569` (`_format_prompt`)
- **Severity:** Medium
- **Detail:** `BaseAgent._format_prompt` is defined as a multi-turn ReAct prompt builder but is never invoked anywhere in the codebase. The agent loop in `run()` calls `execute_step()` (which calls `_format_step_prompt`) and `_synthesise_answer`. The dead `_format_prompt` method contains the more canonical ReAct loop logic and may represent an incomplete or abandoned implementation path.
- **Suggestion:** Either wire `_format_prompt` into the `run()` loop as intended, or delete it to reduce confusion and maintenance overhead.

### 10. Private `_tools` dict accessed directly across module boundary
- **File:** `financial-report-insights/agents/base.py:392` (`execute_step`)
- **Severity:** Medium
- **Detail:** `execute_step` accesses `self._tools._tools.get(tool_call.tool_name)` — bypassing the public `ToolRegistry.get()` interface and directly accessing the internal `_tools` dict attribute (line 392). `ToolRegistry.get()` raises `KeyError` on unknown names; the direct dict access returns `None`, which is then guarded at line 393. However, this pattern breaks the encapsulation contract, will fail silently if `ToolRegistry` is refactored, and is inconsistent with the validation already done by `_select_tool` at line 489.
- **Suggestion:** Replace `self._tools._tools.get(tool_call.tool_name)` with a try/except around `self._tools.get(tool_call.tool_name)`.

### 11. `_analyze_data` closure in `_build_report_writer_registry` ignores unknown `analysis_type` silently
- **File:** `financial-report-insights/agents/specialized.py:216–235` (`_analyze_data`)
- **Severity:** Medium
- **Detail:** When an LLM calls `analyze_data` with an unrecognized `analysis_type`, the function returns a helpful error string. However, this string is then fed back to the agent as an observation, and the agent will silently continue without re-trying. There is no indication in the `WorkflowResult` or logs that an unknown analysis type was requested. Combined with the prompt injection risk in finding #1, an attacker can call `analyze_data` with `analysis_type=<arbitrary>` and observe how the system responds via the returned message.
- **Suggestion:** Log a warning when an unknown `analysis_type` is received. Consider returning a structured error that callers can distinguish from genuine analysis output.

### 12. `GoldenQA` uses a mutable default factory for `expected_sources` on a frozen dataclass
- **File:** `financial-report-insights/evaluation/golden_qa.py:27`
- **Severity:** Medium
- **Detail:** `expected_sources: List[str] = field(default_factory=list)` on a `@dataclass(frozen=True)` creates a new list per instance, which is correct. However, `frozen=True` only prevents attribute reassignment — it does not prevent mutation of the list itself (`qa.expected_sources.append(...)` would succeed). Downstream code in `eval_harness.py:97` constructs `set(qa.expected_sources)` which is safe, but any code that mutates the list would corrupt the golden dataset without error.
- **Suggestion:** Use `tuple` instead of `list` for `expected_sources` in the `GoldenQA` dataclass (and update the type annotation to `Tuple[str, ...]`) to make true immutability possible. All call sites in the golden dataset use literal lists that can trivially be changed to tuples.

### 13. `run_full_eval` makes two full passes over the RAG system (doubled LLM load)
- **File:** `financial-report-insights/evaluation/eval_harness.py:229–230`
- **Severity:** Medium
- **Detail:** `run_full_eval` calls both `evaluate_retrieval` and `evaluate_answers` sequentially; each independently calls `self.rag.retrieve(qa.question, top_k=k)` for every QA pair. This means every query hits the embedding model and vector store twice per full evaluation run. For a 20-QA dataset this doubles embedding and retrieval costs with no benefit — the `evaluate_answers` path already has the retrieved docs it needs.
- **Suggestion:** Refactor `run_full_eval` to do a single retrieval pass per query, then compute both retrieval metrics and answer metrics from the same results.

### 14. `faithfulness_score` returns 0.0 when context chunks are empty, masking the difference from a genuinely unfaithful answer
- **File:** `financial-report-insights/evaluation/answer_metrics.py:63–66`
- **Severity:** Medium
- **Detail:** When `context_chunks` is an empty list, `faithfulness_score` returns `0.0` — the same score as a maximally unfaithful answer. A caller cannot distinguish "no context was retrieved" from "the answer contains no words from the context". This masks retrieval failures during evaluation.
- **Suggestion:** Return `None` (or a sentinel like `-1.0`) when context is empty, and document that the metric is undefined in that case. Update `RAGEvalHarness.evaluate_answers` to exclude such entries from the aggregate faithfulness average.

### 15. `AnswerGrounder.verify_citations` checks overlap on the full clean answer, not on per-citation surrounding context
- **File:** `financial-report-insights/evaluation/response_quality.py:393`
- **Severity:** Medium
- **Detail:** `verify_citations` strips all citation markers from the entire answer to get `clean_answer`, then checks whether the full `clean_answer` overlaps with the referenced document (`_overlap(clean_answer, content) > 0.0`). Because this uses the entire answer text, any document that shares even one content token with any part of the answer will be marked valid — even if the citation appears in a completely different sentence. A paper with any financial keyword will always pass this check.
- **Suggestion:** Use `_split_sentences` to compare each citation's surrounding sentence(s) against the referenced document, not the full answer.

---

## P3 — Low / Housekeeping

### 16. Duplicated stop-word list and `_tokenize`/`_content_tokens` helpers
- **File:** `financial-report-insights/evaluation/answer_metrics.py:18–39` and `financial-report-insights/evaluation/response_quality.py:18–44`
- **Severity:** Low
- **Detail:** The `_STOP_WORDS` frozenset, `_tokenize`, and `_content_tokens` functions are copy-pasted verbatim across two files. The comment in `response_quality.py:17` even notes "mirrors answer_metrics.py conventions", confirming this is intentional duplication. Any future change to the tokenization logic (e.g., adding new stop words, adjusting the regex) must be applied in both places.
- **Suggestion:** Extract shared helpers into a `evaluation/_text_utils.py` module and import from both files.

### 17. `ConditionalWorkflow` runs selected agents sequentially, not concurrently
- **File:** `financial-report-insights/agents/workflows.py:476–505` (`ConditionalWorkflow.run`)
- **Severity:** Low
- **Detail:** The docstring (line 407) states "Selected agents run concurrently" but the implementation iterates `for name in selected_names:` and calls `_run_step` synchronously. Multiple-agent routing (if a `router_func` returns more than one name) will run them one at a time, contradicting the documented behavior.
- **Suggestion:** Either update the docstring to reflect sequential execution, or wrap the loop body in a `ThreadPoolExecutor` as `ParallelWorkflow` does.

### 18. `_build_dependency_levels` marks all unresolvable steps as a "final wave" rather than failing fast
- **File:** `financial-report-insights/agents/workflows.py:329–337`
- **Severity:** Low
- **Detail:** When a circular dependency or unsatisfiable dependency is detected, the remaining steps are lumped into a final wave and executed anyway, with only a logger.warning. This means a misconfigured workflow (e.g., step A depends on step B which depends on step A) will execute all steps in undefined order, likely producing wrong results, rather than failing with a clear error.
- **Suggestion:** Raise a `ValueError` when a deadlock in the dependency graph is detected, rather than proceeding silently.

### 19. `ConfidenceScorer` weight constants not validated at construction time
- **File:** `financial-report-insights/evaluation/response_quality.py:204–207`
- **Severity:** Low
- **Detail:** `_W_RETRIEVAL + _W_COVERAGE + _W_DIVERSITY` equals `1.0` only by manual accounting. If a future developer changes one weight (e.g., `_W_RETRIEVAL = 0.6`) without adjusting the others, the `overall_confidence` computation will silently produce values outside `[0.0, 1.0]` (before clamping). There is no assertion or test that the weights sum to 1.0.
- **Suggestion:** Add a class-level or `__init_subclass__` assertion: `assert abs(cls._W_RETRIEVAL + cls._W_COVERAGE + cls._W_DIVERSITY - 1.0) < 1e-9`.

### 20. `tool_search_documents` leaks exception detail from the RAG instance to the agent
- **File:** `financial-report-insights/agents/tools.py:889–890`
- **Severity:** Low
- **Detail:** `return f"Document search failed: {exc}"` propagates the full exception message string (which may contain internal paths, connection strings, or stack trace fragments) back to the LLM as an observation. This is information disclosure into the agent's reasoning context.
- **Suggestion:** Log `exc` at WARNING level and return a generic message: `"Document search failed. See server logs for details."` — consistent with the API error sanitization pattern established elsewhere in this codebase.

### 21. `mrr` function name is misleading — it computes RR, not MRR (for a single query)
- **File:** `financial-report-insights/evaluation/retrieval_metrics.py:59–78`
- **Severity:** Low
- **Detail:** Mean Reciprocal Rank (MRR) is defined as the mean of reciprocal ranks across multiple queries. This function computes the reciprocal rank for a single query. The averaging over queries happens externally in `eval_harness.py:103` (`total_mrr += m`). The name is not wrong in the context of the harness, but calling it `mrr` at the single-query level can mislead readers into thinking it already aggregates.
- **Suggestion:** Rename to `reciprocal_rank` and update the import alias in `eval_harness.py` and `evaluation/__init__.py` to clarify intent.

### 22. `steps_int` cap of 1 in `tool_forecast` allows nonsensical 1-step forecasts
- **File:** `financial-report-insights/agents/tools.py:632`
- **Severity:** Low
- **Detail:** `steps_int = max(1, int(steps))` allows a caller to request a single-period forecast. A 1-step AR(2) forecast is technically valid but practically useless for trend analysis. No upper bound is enforced — an LLM could pass `steps=10000`, causing the function to return a very long string that may exceed context or memory limits.
- **Suggestion:** Clamp `steps_int = max(1, min(int(steps), 52))` (or another reasonable domain cap) and document the maximum.

---

## Files With No Issues Found

- `financial-report-insights/agents/__init__.py` — Clean re-export module; no logic.
- `financial-report-insights/evaluation/__init__.py` — Clean re-export module; no logic.
- `financial-report-insights/evaluation/retrieval_metrics.py` — Implementations of `precision_at_k`, `recall_at_k`, `mrr`, and `ndcg_at_k` are mathematically correct and handle all edge cases (empty lists, `k <= 0`, `idcg == 0`) cleanly. No issues beyond the naming note in finding #21.

---

## Summary

| Priority | Count | Key Themes |
|----------|-------|------------|
| P0 — Critical | 2 | Prompt injection via raw user input; prompt injection via context propagation |
| P1 — High | 5 | Unbounded parallel execution; thread-safety in shared memory; silent fallback masking ML errors; argument parsing fragility; unguarded RAG doc type |
| P2 — Medium | 8 | Dead SYSTEM_PROMPT code; dead `_format_prompt`; encapsulation violation; duplicate eval load; metric correctness edge cases |
| P3 — Low | 7 | Code duplication; docstring inaccuracy; silent workflow misconfiguration; weight validation; info disclosure; naming; unbounded input |
| **Total** | **22** | |

**Technical Debt Estimate:** 18–24 hours

The most urgent issues are the two P0 prompt injection vectors. The `_parse_tool_arguments` regex splitter (P1-#6) amplifies the P0 risk by making it easy to inject multiple key=value pairs via a crafted query. These three issues should be addressed together as a single security sprint item. The dead `SYSTEM_PROMPT` constants (P2-#8) and dead `_format_prompt` method (P2-#9) suggest the agent framework is partially implemented — the persona and multi-turn ReAct loop were designed but not wired up, which materially degrades output quality in production use.
