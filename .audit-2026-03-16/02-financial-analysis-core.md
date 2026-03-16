# Audit Report: Financial Analysis Core
**Date:** 2026-03-16
**Scope:** Financial analyzer, compliance, portfolio, underwriting, startup model, ratio framework, types
**Files Reviewed:**
- `financial-report-insights/financial_analyzer.py` (13,894 lines) — read in full via chunked reads and targeted greps
- `financial-report-insights/compliance_scorer.py` (728 lines)
- `financial-report-insights/portfolio_analyzer.py` (527 lines)
- `financial-report-insights/underwriting.py` (526 lines)
- `financial-report-insights/startup_model.py` (398 lines)
- `financial-report-insights/ratio_framework.py` (631 lines)
- `financial-report-insights/structured_types.py` (345 lines)

---

## P0 — Critical (Fix Immediately)

### P0-1: Partial Altman Z-Score produces misleading zone classifications
**File:** `financial-report-insights/financial_analyzer.py`, lines 2823–2859

When fewer than all 5 Altman components are available, the code sums only the available weighted components and still compares the partial sum to the full-formula thresholds (1.81, 2.99). A partial sum using only 3 components (e.g., x1=0.2, x2=0.3, x5=0.5 → partial z ≈ 0.87) will always fall below 1.81 and be flagged as `partial_distress`, even though a company with strong x3 (EBIT/TA) and x4 (equity/liabilities) values would be in the safe zone. This produces false distress signals that cascade into `composite_health_score` (line 3026 routes `partial_distress` → 0 leverage points), `audit_risk_assessment` in compliance_scorer (line 621), and `portfolio_risk_summary`.

The `zone` values `'partial'` and `'partial_distress'` are correct in intent, but `composite_health_score` at line 3026 gives `partial_distress` zero Z-Score points (same as actual distress), which is not justified for an incomplete dataset.

**Risk:** A company with missing retained_earnings and equity data gets penalised 25 composite health points it may not deserve.

**Suggestion:** When `len(available) < 5`, assign Z-Score health points based on the `'partial'` zone only (e.g., 10 pts), not `'partial_distress'` → 0. Alternatively, require all 5 components before scoring; return `None` health contribution otherwise.

---

### P0-2: `_safe_eval_formula` has no division-by-zero protection
**File:** `financial-report-insights/financial_analyzer.py`, lines 4800–4829

The AST-walking evaluator uses `operator.truediv` directly (line 4821). When a user's custom KPI formula divides by a field that is zero (e.g., `"net_income / revenue"` with `revenue=0`), this raises a native `ZeroDivisionError` that is caught at line 4895 and reported as an error string. However, the user sees only `"Division by zero"` rather than the field that caused it, and more importantly, the formula token validator at line 4877 uses `re.sub(r'[\d+\-*/().\s]', '', temp)` — the character class includes `+` literally (it should be `\\+` if intended as a literal plus, or it functions as a quantifier modifier which is not valid in a character class context and is implementation-defined). In CPython, `+` inside `[]` is treated as a literal, so the validator works, but this is an accident of implementation.

**Suggestion:** Check denominator before dividing in `_walk` for `ast.BinOp` / `ast.Div`, raising a `ValueError("Division by zero")` with the operand name if identifiable.

---

## P1 — High (Fix This Sprint)

### P1-1: `quick_ratio` in `ratio_framework.py` uses incorrect numerator
**File:** `financial-report-insights/ratio_framework.py`, lines 390–403

The `quick_ratio` entry in `RATIO_CATALOG` sets `numerator_field="current_assets"`. The acid-test ratio formula is (Current Assets − Inventory) / Current Liabilities. Using raw current assets is the same formula as the current ratio. The comment `# Simplified; should subtract inventory` acknowledges the error but does not fix it.

The `compute_ratio` engine only supports simple `numerator / denominator` via `_get_field_value` and `safe_divide` — it cannot express subtraction. Therefore `ratio_framework.quick_ratio` always equals `ratio_framework.current_ratio` for any data set.

**Risk:** Any UI or report that calls `compute_ratio(data, RATIO_CATALOG["quick_ratio"])` returns an inflated quick ratio, producing false liquidity signals. The correct `calculate_liquidity_ratios` in `CharlieAnalyzer` (line 2324) does compute it properly, so there are two inconsistent implementations of the same metric in the same codebase.

**Suggestion:** Either (a) add a `subtrahend_field` parameter to `RatioDefinition` so the framework can express `(A − B) / C`, or (b) remove `quick_ratio` from `RATIO_CATALOG` and route callers to `CharlieAnalyzer.calculate_liquidity_ratios`.

---

### P1-2: `roic` in `ratio_framework.py` uses total assets as denominator instead of invested capital
**File:** `financial-report-insights/ratio_framework.py`, lines 311–324

The `roic` entry uses `denominator_field="total_assets"`, which is the formula for ROA, not ROIC. ROIC = NOPAT / (Total Equity + Interest-Bearing Debt), and the comment acknowledges the error: `# Simplified; should be invested capital`. The result is that `RATIO_CATALOG["roic"]` produces a value algebraically equivalent to `RATIO_CATALOG["roa"]` (using EBIT as numerator instead of net income, but same denominator), making it misleading as a distinct metric.

**Suggestion:** Document clearly that this is an ROA proxy, rename it `ebit_to_assets`, or implement proper invested-capital computation outside the single-field framework.

---

### P1-3: Raw division for DIO and DPO without guarding against near-zero COGS
**File:** `financial-report-insights/financial_analyzer.py`, lines 4336 and 4347

In `working_capital_analysis`, DIO and DPO are computed as:
```python
dio = round(inv / cogs * 365, 1)   # line 4336
dpo = round(ap / cogs * 365, 1)    # line 4347
```
The guard `if cogs > 0` at the enclosing `if` statement prevents a ZeroDivisionError when `cogs` is exactly zero. However, a very small positive `cogs` (e.g., 1.0) with a large `inventory` would produce a wildly large DIO without any warning. More importantly, the guard uses `> 0` rather than `> 1e-12`, inconsistent with `safe_divide`'s own near-zero threshold. If `cogs` is a negative number (valid for some cost reversals), the condition `cogs > 0` will be False, silently returning `None` instead of a negative-DIO flag.

**Suggestion:** Replace both raw divisions with `safe_divide(inv, cogs, default=None)` and multiply by 365, matching the pattern used elsewhere. Also consider checking `cogs > 1e-12` for consistency with `safe_divide`.

---

### P1-4: `roa` adjustment in ratio_framework compares raw `total_debt` dollar value against 0.5
**File:** `financial-report-insights/ratio_framework.py`, line 288

The ROA definition includes:
```python
Adjustment("total_debt", Operator.LT, 0.5, 0.5, "Low leverage"),
```
`_apply_adjustments` fetches the value of `total_debt` directly from `FinancialData` (a dollar amount like 500,000,000). The condition `total_debt < 0.5` is almost never true for any real company, meaning this adjustment is dead code for all practical purposes. The intent was clearly to check a ratio (debt/assets < 0.5), but the framework resolves the field name directly to the raw field.

**Risk:** The ROA score is systematically 0.5 points lower than intended whenever a company carries any debt, because the intended adjustment never fires.

**Suggestion:** Either (a) reference a ratio key from `computed_ratios` (e.g., `"debt_to_assets"`) and run `run_all_ratios` before adjustments, or (b) replace this adjustment with a named computed ratio entry.

---

### P1-5: `AnalysisResults.__getitem__` rebuilds the full dict on every access
**File:** `financial-report-insights/structured_types.py`, lines 196–198

```python
def __getitem__(self, key: str) -> Any:
    return self.to_dict()[key]
```

Every dict-style key access on an `AnalysisResults` instance calls `to_dict()`, which in turn calls `.to_dict()` on all four ratio dataclasses. In `generate_insights` (financial_analyzer.py line 3321+), this is called in a tight loop through `analysis_results.get(...)` multiple times. The same pattern repeats in `generate_report` at line 3227+. Each such call reconstructs four inner dicts.

**Risk:** Not a correctness issue, but produces O(k) overhead per attribute access where k is the number of dict keys. For large reports with many insight generations this compounds. Also `__iter__` returns keys from a freshly built dict on every call.

**Suggestion:** Cache the `to_dict()` result, or use `dataclasses.asdict` once and store.

---

### P1-6: `dupont_analysis` interest burden uses `data.ebit or data.operating_income`, masking zero EBIT
**File:** `financial-report-insights/financial_analyzer.py`, line 2753

```python
interest_burden = safe_divide(ebt, data.ebit or data.operating_income)
```

When `data.ebit` is `0.0` (company breaking even), `0 or data.operating_income` evaluates to `data.operating_income` due to Python's falsy evaluation of `0`. This silently substitutes operating_income for a legitimately-zero EBIT, producing a wrong interest burden. The same pattern occurs at lines 2812, 5719, 6093, 6519, 6666, 6809.

For EBIT = 0, the interest burden (EBT/EBIT) is genuinely undefined (infinite), not a substitute for operating income. The fallback hides this edge case.

**Suggestion:** Use an explicit `None` check:
```python
ebit_val = data.ebit if data.ebit is not None else data.operating_income
interest_burden = safe_divide(ebt, ebit_val)
```

---

### P1-7: Monte Carlo simulation floors sampled multipliers at 0.01, preventing revenue from going negative
**File:** `financial-report-insights/financial_analyzer.py`, line 3864

```python
sample = max(sample, 0.01)  # Floor at 1% to avoid negatives
```

This prevents the simulation from modelling scenarios where revenue falls by more than 99% or where cost variables go negative. While it prevents a ZeroDivisionError downstream, it produces a truncated left-tail in the distribution, systematically underestimating downside risk. The 10th-percentile health score will always be higher than the true distribution would produce under severe stress.

For a financial risk tool, this is a material modelling bias rather than merely a performance guard.

**Suggestion:** Remove the hard floor or lower it significantly (e.g., 0.001). Handle potential near-zero denominators via `safe_divide` in downstream ratio calculations rather than by distorting the simulation inputs.

---

## P2 — Medium (Fix Soon)

### P2-1: `calculate_variance` returns `None` variance_percent when budget is zero with non-zero variance
**File:** `financial-report-insights/financial_analyzer.py`, line 2530

```python
variance_percent = safe_divide(variance, abs(budget), default=0.0 if variance == 0 else None)
```

When `variance != 0` and `budget == 0`, `variance_percent` is `None`. The `BudgetAnalysis` dataclass stores `total_variance_percent` as `float`, so downstream formatting code that calls `f"{vp:.1%}"` will raise a `TypeError` if `total_variance_percent` resolves to `None` at the aggregate level. The aggregate at line 2647 uses `safe_divide(total_variance, total_budget, default=0.0)`, so the aggregate total is guarded — but individual `VarianceResult.variance_percent` items can still be `None`.

**Suggestion:** Use `default=0.0` unconditionally, or document that callers must handle `None` and add a `None` guard in any UI formatting path.

---

### P2-2: `probability_weighted_scenarios` treats `None` z_score as zero in expected value
**File:** `financial-report-insights/financial_analyzer.py`, lines 3729–3731

```python
expected_z = sum(
    p * (r.scenario_z_score or 0)
    for p, r in zip(probs, results)
)
```

When a scenario produces insufficient data for a Z-Score (returns `None`), the `or 0` silently treats it as z = 0. Zero is in the distress zone (< 1.81), so this can push the `distress_probability` and `expected_z_score` downward, falsely suggesting distress when the issue is just missing data.

**Suggestion:** Skip None-z scenarios from the expected-z computation rather than coercing to 0, and add a warning when any scenario produces a None z_score.

---

### P2-3: `forecast_cashflow` raw division by `(1 + discount_rate) ** i` not protected against negative discount_rate
**File:** `financial-report-insights/financial_analyzer.py`, line 3969

```python
pv_fcf = fcf / ((1 + discount_rate) ** i)
```

If a caller passes `discount_rate = -1.0` or `discount_rate = -1.5`, the denominator `(1 + discount_rate)^i` becomes zero or negative, producing division by zero or nonsensical negative present values. There is no validation on the `discount_rate` parameter.

**Suggestion:** Add `if discount_rate <= -1.0: raise ValueError("discount_rate must be > -1.0")` at the start of `forecast_cashflow`.

---

### P2-4: `_compute_base_score` (ratio_framework) documentation says thresholds must be pre-sorted descending, but lower-is-better ratios need ascending order
**File:** `financial-report-insights/ratio_framework.py`, lines 113–129

The docstring says "Thresholds must be pre-sorted descending by threshold value" but for `higher_is_better=False`, the function iterates and returns on `ratio_value <= threshold`. With the `debt_to_equity` thresholds `[(0.3, 10.0), (0.5, 8.0), (1.0, 6.0), (2.0, 4.0)]`, a D/E of 0.4 correctly hits the first `<= 0.3` check as False, then `<= 0.5` as True and returns 8.0. This works correctly with ascending thresholds for `lower_is_better`, but the docstring's claim of "descending" ordering is technically wrong for that branch — it should say "ascending for lower-is-better, descending for higher-is-better". This creates a maintenance hazard: a future developer may sort thresholds descending for a lower-is-better ratio and break the scoring silently.

**Suggestion:** Clarify the docstring, or enforce the sort in `_compute_base_score` itself rather than requiring callers to pre-sort correctly.

---

### P2-5: `cash_conversion_cycle` in ratio_framework is a misleading misnomer
**File:** `financial-report-insights/ratio_framework.py`, lines 546–559

The `cash_conversion_cycle` entry computes `inventory / revenue` (a ratio, not days) and labels the unit as `"days"`. The true CCC formula is DSO + DIO − DPO and requires three separate components. This entry computes something closer to an inventory-to-revenue ratio.

**Risk:** Any report that labels this output as "Cash Conversion Cycle (days)" will present factually incorrect data.

**Suggestion:** Rename to `"inventory_intensity"` (unit `"x"`) or remove it from the catalog and route CCC computation to `CharlieAnalyzer.working_capital_analysis`.

---

### P2-6: `_score_profitability` and `_score_liquidity` in underwriting coerce `None` to `0` using `or 0`
**File:** `financial-report-insights/underwriting.py`, lines 186–187, 221–222, 237–238, 253–254

```python
nm = net_margin or 0
r = roa or 0
cr = current_ratio or 0
cashr = cash_ratio or 0
od = ocf_debt or 0
fm = fcf_margin or 0
ic = interest_coverage or 0
em = ebitda_margin or 0
```

When the ratio is `None` (insufficient data), this coerces to 0, which then falls through to the lowest scoring bracket (0 points). This is correct for conservative underwriting. However, when the ratio is legitimately `0.0` (e.g., exactly zero net income), it is also treated as missing data, and the condition `nm > 0` is False anyway so the score is 0 points in both cases. There is one subtle edge case: if `interest_coverage = 0.0` (EBIT exactly zero), the company scores 0 on stability despite having EBIT exactly at break-even — this is arguably correct but conflates "no data" with "zero coverage" without any signal to the caller.

**Note:** This is low severity because the scoring result (0 pts) is consistent, but the `or 0` pattern is semantically ambiguous.

---

### P2-7: `debt_capacity` in underwriting uses `or 0` for `ebitda`, silently treating None EBITDA as zero capacity
**File:** `financial-report-insights/underwriting.py`, lines 278–279

```python
current_debt = data.total_debt or 0
ebitda = data.ebitda or 0
```

`max_capacity = target * ebitda` (line 282) equals `0` when `ebitda` is `None`. Then `max_additional = max(0.0, 0 - current_debt) = 0` regardless of debt level. The `assessment` text says "Insufficient data to calculate current leverage" (because `current_leverage = safe_divide(data.total_debt, data.ebitda)` returns `None`), but `max_additional_debt` is set to `0.0` as if the company is fully leveraged, which is misleading.

**Suggestion:** Return early or set `max_additional_debt=None` when `data.ebitda is None`.

---

### P2-8: `NRR` calculation in `startup_model.py` is a misnomer
**File:** `financial-report-insights/startup_model.py`, lines 122–124

```python
nrr: Optional[float] = None
if gross_churn is not None:
    nrr = 1.0 - gross_churn
```

Net Revenue Retention (NRR) measures the change in revenue from an existing cohort of customers, accounting for upgrades, downgrades, and churn. `1 - gross_churn_rate` is simply customer retention rate (CRR), not NRR. A company with 5% customer churn but 50% expansion revenue from surviving customers would have NRR > 1.0 but would be reported here as NRR = 0.95. The field is exposed in `SaaSMetrics.net_revenue_retention`.

**Suggestion:** Rename the field to `gross_customer_retention_rate` or add a docstring clarifying that this is a simplified proxy, not true NRR.

---

### P2-9: `compare_periods` `lower_is_better` set is incomplete
**File:** `financial-report-insights/financial_analyzer.py`, lines 3172–3190

```python
lower_is_better = {'leverage_debt_to_equity', 'leverage_debt_to_assets'}
```

Only two metrics are in this set. `leverage_interest_coverage` is higher-is-better (correct). However, no efficiency metrics are in `lower_is_better` even though, for example, `DPO` increasing is neither purely good nor bad, and metrics like `cash_conversion_cycle` where lower is better are not considered. This is low-risk since the delta direction classification (`improvements`/`deteriorations`) is purely informational, but the set is not complete as a general reference.

**Suggestion:** Expand the set to include at minimum common lower-is-better metrics such as `leverage_debt_to_liabilities` if it ever appears in results.

---

## P3 — Low / Housekeeping

### P3-1: `AnalysisResults` `__iter__` builds a fresh dict on every iteration
**File:** `financial-report-insights/structured_types.py`, line 204–206

Minor extension of P1-5. The `__iter__` method returns `iter(self.to_dict())` which creates a full dict solely to iterate its keys. Used in `for k in results: ...` patterns in the codebase, this builds a dict per loop call.

---

### P3-2: `_detect_seasonality` method is not defined in the reviewed chunk — if stubbed, it returns `False` unconditionally
**File:** `financial-report-insights/financial_analyzer.py`, line 2486 call, line 2510 definition

The method is called at line 2486 but its body begins at line 2510. It is defined and present, no stub issue. Confirmed benign.

---

### P3-3: `compliance_pct` in `ComplianceReport` summary denominator includes excluded (None) checks
**File:** `financial-report-insights/compliance_scorer.py`, lines 550–551

```python
total = pass_count + fail_count
compliance_pct = (pass_count / total * 100) if total > 0 else 0.0
```

When all 6 regulatory thresholds return `None` (insufficient data), `total = 0` and `compliance_pct` defaults to `0.0`. The summary string at line 711 then reports `"0% (0/0 passed)"` which looks like total non-compliance rather than insufficient data.

**Suggestion:** Return a distinct sentinel like `None` for `compliance_pct` when `total == 0`, and update the summary string accordingly.

---

### P3-4: `evaluate_custom_kpis` regex character class for token sanitization contains ambiguous `+`
**File:** `financial-report-insights/financial_analyzer.py`, line 4877

```python
temp = re.sub(r'[\d+\-*/().\s]', '', temp)
```

Inside a character class `[...]`, `+` is a literal `+` in Python's `re` module, not a quantifier. This is technically correct behavior — `+` is stripped along with other operator characters. However, it looks like a possible typo for `\+` (escaped plus). The intent is to remove all valid formula characters, so `+` should indeed be included literally.

This is not a bug (it works correctly), but a code-clarity issue: the `+` at the start of the character class after `\d` is visually confusing because `\d+` reads as a regex quantified digit class in the docstring context.

**Suggestion:** Rewrite as `r'[\d\+\-\*/\(\)\.\s]'` with explicit escapes for clarity.

---

### P3-5: `roa` in `ratio_framework.py` has adjustment on raw `operating_income` dollar amount vs. 0
**File:** `financial-report-insights/ratio_framework.py`, line 287

```python
Adjustment("operating_income", Operator.GT, 0, 0.5, "Positive operating income"),
```

This adjustment fires when `operating_income > 0`, which is a raw dollar check. Unlike the `total_debt < 0.5` bug (P1-4), this one works correctly since it tests sign direction (positive/negative), not a magnitude threshold. However, the adjustment field `operating_income` falls through to `_get_field_value(data, adj.field)` which returns the raw dollar value. The `_apply_operator(value, Operator.GT, 0)` check on a dollar amount is directionally meaningful (positive vs. negative income), so this is acceptable.

---

### P3-6: `_scored_analysis` grade boundary uses `>= 8` for "Excellent" but the 0–10 scale can produce exactly `8.0` via threshold matching
**File:** `financial-report-insights/financial_analyzer.py`, lines 5282–5290

```python
if score >= 8:
    grade = "Excellent"
elif score >= 6:
    grade = "Good"
elif score >= 4:
    grade = "Adequate"
else:
    grade = "Weak"
```

This is internally consistent. The `ratio_framework.py` `get_grade` uses `low <= score < high` (exclusive upper), so a score of exactly `8.0` gets "Excellent" from `_scored_analysis` but "Good" from `ratio_framework.get_grade` (since `(6.0, 8.0): "Good"` is matched first with `6.0 <= 8.0 < 8.0` = False, then `(8.0, 10.0): "Excellent"` = True). Both grade at "Excellent" for 8.0, but the boundary semantics differ across the two implementations, which is a maintenance hazard for future contributors.

---

### P3-7: `forecast_cashflow` uses compound growth but `growth_rates` list contains a flat rate repeated
**File:** `financial-report-insights/financial_analyzer.py`, lines 3960–3978

The loop computes `current_rev = current_rev * (1 + revenue_growth)` correctly with compounding. However, `growth_rates.append(round(revenue_growth * 100, 2))` stores the same constant rate for every period. This is not wrong, but the `CashFlowForecast.growth_rates` field implies period-specific rates. For the current single-growth-rate implementation this is fine, but misleads any consumer expecting period-specific values.

---

## Files With No Issues Found

- **`structured_types.py`**: Type definitions are correct. `from_dict` and `to_dict` methods are internally consistent. `AnalysisResults` backward-compat methods work as documented (with the O(n) access cost noted in P1-5). No formula errors; no division operations.

- **`compliance_scorer.py`**: All ratio computations use `safe_divide`. The OCF/NI sign check at lines 237–256 correctly handles the negative-NI case (fixed in prior sprint). Balance sheet consistency tolerance (1%) at lines 426–432 is appropriate. The `accrual_gap` calculation at line 332–344 uses `safe_divide` with `default=0.0` and `abs()` correctly.

- **`portfolio_analyzer.py`**: `_hhi` handles the single-company case (returns 1.0) and empty/near-zero totals correctly. NaN correlation is excluded before averaging (lines 238–239). The `zero_companies` warning path at lines 244–246 is informational only. All ratio computations use `safe_divide`.

- **`startup_model.py`**: All divisions use `safe_divide`. The `burn_runway` method correctly avoids division when `net_burn <= 0` (cash-flow positive path). `funding_scenarios` checks `post_money > 0` before computing dilution. The `LTV = arpu / gross_churn` derivation guards `gross_churn > 0`. The `mrr_from_revenue` flag at line 166 is tracked but never used — harmless dead code.

---

## Summary

| Severity | Count | Highlights |
|----------|-------|------------|
| P0 Critical | 2 | Partial Z-Score produces false distress scores; AST evaluator unprotected division + ambiguous regex |
| P1 High | 7 | Quick ratio = current ratio bug; ROIC = ROA bug; raw DIO/DPO divisions; dead roa adjustment; O(n) access; EBIT `or 0` masking; Monte Carlo floor biases downside |
| P2 Medium | 9 | Various edge cases: None coercion, NRR misnomer, DSCR/DCF unvalidated inputs, CCC mislabeled |
| P3 Low | 7 | Clarity and housekeeping issues |

**Overall Quality Score: 6.5/10**

The codebase shows strong discipline in using `safe_divide` throughout and has successfully eliminated `eval()` calls. The main risks are two formula-level bugs in `ratio_framework.py` (quick ratio, ROIC) that cause silent incorrect values rather than crashes, one scoring model bias (partial Z-Score → 0 health points), and the Monte Carlo simulation floor that systematically truncates downside risk modelling. The `or 0` / `or data.operating_income` patterns for zero-valued fields are the most widespread low-level issue.

**Technical Debt Estimate: 12–16 hours** to address P0+P1 items; P2+P3 approximately 4–6 additional hours.
