# Continuous Learning Infrastructure

**Created:** 2025-02-13
**Purpose:** Enable iterative learning across sessions, preventing unbounded repetitive code generation and anti-patterns

## Overview

The RAG-LLM project now has a structured continuous learning system that tracks development patterns, prevents duplication, and maintains efficiency across extended development cycles. This prevents the anti-pattern of unbounded phase generation (Phases 100-300) that occurred previously.

## Four Core Learning Files

### 1. `.loki/memory/semantic/anti_patterns.json`

**Purpose:** Registry of detectable anti-patterns to prevent runaway development

**Contents:**
- **5 Critical Anti-Patterns:**
  - `unbounded_phase_generation` - Continuous new phases without consolidation
  - `context_explosion` - Single files exceeding 15K LOC
  - `orchestrator_drift` - CONTINUITY.md diverging from actual code
  - `ratio_duplication` - Algebraically equivalent ratios implemented multiple times
  - `missing_checkpoints` - No commits for 5+ phases

**For Each Anti-Pattern:**
- Description and severity (critical/high)
- Detection signals (observable conditions)
- Prevention rules (proactive measures)
- Example incidents (what went wrong historically)

**How to Use:**
1. Before starting new work, review detection signals in "pre_phase_launch" checklist
2. After completing phases, verify "post_phase_completion" checklist
3. If anti-pattern detected, stop and review prevention rules before continuing

### 2. `.loki/memory/semantic/efficiency_baselines.json`

**Purpose:** Quantitative thresholds for code health and development efficiency

**Key Baselines:**
- `tokens_per_phase`: 17,000 (baseline estimate from Phase 1-6 analysis)
- `target_tokens_per_phase`: 5,000 (after consolidation and deduplication)
- `max_phases_before_review`: 5 (trigger dedup audit every 5 phases)
- `max_file_loc`: 15,000 (split modules before exceeding)
- `test_coverage_target`: 80% (minimum on critical paths)
- `dedup_threshold`: 0.85 (cosine similarity threshold for ratio duplication)
- `commit_frequency`: Every 5 phases or sooner

**Efficiency Gates:**
- **Pre-Phase Gate:** Code health, git sync, ratio uniqueness, token budget checks
- **Post-Phase Gate:** Test passing, CONTINUITY sync, learning log entry
- **Review Gate:** Every 5 phases - consolidation and deduplication pass

**How to Use:**
1. Before launching new phase, verify pre-phase gate criteria
2. After completing phase, run post-phase gate checks
3. After completing every 5th phase, trigger review gate

### 3. `.loki/memory/semantic/learning_log.json`

**Purpose:** Audit trail of all development sessions and outcomes

**Session Entry Schema:**
```json
{
  "date": "ISO 8601 date",
  "session_id": "unique identifier",
  "task": "what was attempted",
  "outcome": "success | partial | blocked",
  "tokens_used": "estimated total",
  "ratios_implemented": "count",
  "tests_added": "count",
  "pattern_learned": "key insight",
  "anti_pattern_detected": "null or anti-pattern name",
  "files_modified": ["list"],
  "commits": ["commit hashes"],
  "dedup_actions": "any consolidations"
}
```

**Insights by Phase:**
- Cumulative metrics: ratios, tests, tokens
- Key patterns discovered
- Deduplication actions taken

**How to Use:**
1. At session start, search for similar tasks in learning_log to leverage past solutions
2. After session completes, append entry to sessions array
3. Track deduplication passes in dedup_history
4. Review insights_by_phase to spot patterns and inefficiencies

### 4. `.loki/memory/semantic/ratio_registry.json`

**Purpose:** Canonical registry of all financial ratios to prevent duplication

**For Each Ratio:**
- Canonical name and aliases
- Formula and variables
- Category (liquidity, leverage, profitability, etc.)
- Phase first introduced
- Interpretation and health thresholds
- Notes and variants

**Pre-Registered Ratios (Phase 1-6):**
1. `current_ratio` - Short-term liquidity
2. `quick_ratio` - Conservative liquidity
3. `debt_to_equity` - Leverage ratio
4. `return_on_assets` - Asset efficiency
5. `return_on_equity` - Shareholder return
6. `profit_margin` - Profitability per revenue
7. `asset_turnover` - Asset utilization
8. `altman_z_score` - Bankruptcy prediction
9. `piotroski_f_score` - Financial quality
10. `dupont_roe` - ROE decomposition

**Deduplication Rules:**
- Algebraic equivalences (e.g., debt_to_assets = 1 - equity_to_assets)
- Naming conventions (snake_case, prefixed categories)
- Implementation guidelines for new ratios

**How to Use:**
1. Before implementing new ratio, check registry for similar entries
2. If similarity > 0.85, use derived_from instead of creating new implementation
3. When adding new ratio:
   - Verify uniqueness against registry
   - Add entry with formula and first_introduced_phase
   - Create test case
   - Update registry in same commit
4. Run dedup checks every 5 phases to consolidate algebraically equivalent implementations

## Workflow Integration

### Pre-Phase Checklist
```
1. Run: pytest (verify all tests pass)
2. Run: git status (check for untracked files)
3. Check: CONTINUITY.md matches latest commit
4. Check: anti_patterns.json detection signals (no red flags?)
5. Check: ratio_registry.json for new ratios (not duplicating?)
6. Estimate: Token cost (targeting < 5000 consolidated, < 17000 new)
```

### Post-Phase Checklist
```
1. Run: pytest (verify all tests pass)
2. Update: CONTINUITY.md with phase summary and commit hash
3. Commit: git commit -m "Phase X: [description]"
4. Log: Add entry to learning_log.json
5. If phase 5, 10, 15, 20: Trigger review_gate (dedup pass)
```

### Every 5 Phases (Review Gate)
```
1. Compare all ratio formulas for algebraic equivalence
2. Scan test files for duplicate test logic
3. Identify and consolidate equivalent implementations
4. Update ratio_registry.json with dedup results
5. Commit: git commit -m "Consolidation: Phase X dedup pass"
```

## Alert Triggers

**STOP and INVESTIGATE if:**
- Tokens used exceed 25,000 for a single phase
- Any file grows beyond 8,000 LOC (split module needed)
- Test coverage drops below 80%
- More than 10 untracked files in git status
- CONTINUITY.md is not updated within 5 phases
- Any anti-pattern is detected in pre-phase checklist

**YELLOW FLAG if:**
- Tokens used exceed 17,000 (investigate consolidation)
- New phases implement > 10 ratios (likely duplication)
- More than 3 untracked files without commits

## Historical Context

**The Problem (Phases 100-300):**
- Generated 200+ test files without commits
- Implemented 200+ ratios, many algebraically identical
- Context explosion: no module splitting, single monolithic files
- No learning infrastructure: each phase re-implemented similar logic
- Result: Unbounded growth, impossible to maintain or debug

**The Solution:**
- Anti-patterns.json: Catch signals early (detection)
- Efficiency_baselines.json: Quantitative guardrails (prevention)
- Learning_log.json: Track outcomes across sessions (learning)
- Ratio_registry.json: One canonical implementation per unique ratio (consolidation)

## File Locations

```
C:\Users\dtmcg\RAG-LLM-project\
└── .loki\
    ├── memory\
    │   └── semantic\
    │       ├── anti_patterns.json           (5 anti-patterns + detection checklist)
    │       ├── efficiency_baselines.json    (baselines + gates + token tracking)
    │       ├── learning_log.json            (session audit trail)
    │       └── ratio_registry.json          (10 canonical ratios + dedup rules)
    ├── CONTINUITY.md                        (high-level orchestrator state)
    └── LEARNING_INFRASTRUCTURE.md           (this file)
```

## Next Steps

1. **Session Start:** Review learning_log.json for similar past tasks
2. **Phase Planning:** Run pre-phase checklist from anti_patterns.json
3. **Phase Execution:** Track tokens, ratios, tests in spreadsheet
4. **Phase Completion:** Run post-phase checklist, commit, log in learning_log.json
5. **Every 5 Phases:** Trigger review_gate (dedup pass), consolidate, commit

## Maintenance

- Keep learning_log.json updated (append session entries)
- Keep ratio_registry.json updated (add new ratios with dedup check)
- Review anti_patterns.json detection signals before each phase
- Update efficiency_baselines.json if baselines shift significantly
- Use these files to make decisions, not as bureaucratic overhead

---

**Key Principle:** These files enable continuous improvement across sessions. They are tools for preventing anti-patterns and learning from past work, not rules to be blindly followed. Use judgment, adapt baselines as needed, and focus on sustainable development practices.
