# Quick Reference Checklists

Copy and paste these into your session notes to ensure consistent application.

## PRE-PHASE CHECKLIST

Run before starting new phase development:

```
Phase: __________  Date: __________

[ ] 1. Tests passing
      Command: cd financial-report-insights && python -m pytest tests/ -v
      Result: ✓ All passing

[ ] 2. Git synchronized
      Command: git status
      Result: No untracked files (except tests/?__pycache__)

[ ] 3. CONTINUITY.md updated
      Check: Last entry matches latest commit hash
      Last commit: ____________

[ ] 4. Anti-patterns check (from anti_patterns.json detection_checklist)
      [ ] Phase number not exceeding 50 without dedup
      [ ] New functionality not in existing phases
      [ ] Tokens estimated < 5000 (consolidated) or < 17000 (exploration)
      [ ] Module files < 8000 LOC

[ ] 5. Ratio uniqueness check (from ratio_registry.json)
      New ratios planned: ________________
      [ ] Checked ratio_registry.json for similar entries
      [ ] If similarity > 0.85, consolidating instead of creating new
      [ ] Registry will be updated in commit

[ ] 6. Module health check
      [ ] financial_analyzer.py < 8000 LOC (current: _____)
      [ ] insights_page.py < 8000 LOC (current: _____)
      [ ] test_coverage >= 80% (current: ____%)

Notes: ________________________________________________________________

APPROVED TO PROCEED: ☐ YES  ☐ NO
```

## POST-PHASE CHECKLIST

Run after completing phase development:

```
Phase: __________  Date: __________

[ ] 1. All tests passing
      Command: cd financial-report-insights && python -m pytest tests/ -v
      Result: ✓ [X]/[X] tests passing

[ ] 2. CONTINUITY.md updated
      Summary: ________________________________________________________________
      Commit hash: ____________________________

[ ] 3. Commit created
      Message: ________________________________________________________________
      Command: git commit -m "[message]"

[ ] 4. Learning log entry added
      Entry appended to: C:\Users\dtmcg\RAG-LLM-project\.loki\memory\semantic\learning_log.json
      [ ] outcome: ________________
      [ ] tokens_used: ________________
      [ ] ratios_implemented: ________________
      [ ] tests_added: ________________
      [ ] pattern_learned: ________________________________________________________________
      [ ] anti_pattern_detected: ________________________

[ ] 5. Ratio registry updated (if new ratios added)
      New ratios added to: C:\Users\dtmcg\RAG-LLM-project\.loki\memory\semantic\ratio_registry.json
      Ratios: ________________________________________________________________

[ ] 6. Review gate trigger check
      [ ] NOT a review phase (1-4, 6-9, 11-14, etc.)
      [ ] Review phase (5, 10, 15, 20, etc.) -> See REVIEW GATE CHECKLIST below

Phase completed: ☐ YES  ☐ NO (notes: _____________________________________________________)
```

## REVIEW GATE CHECKLIST

Run after every 5th phase (5, 10, 15, 20, etc.):

```
Review Phase: __________  Date: __________
Covering phases: _____ through _____

[ ] 1. Ratio formula audit
      Command: Review ratio_registry.json for algebraic equivalence
      Check for: debt_to_assets ≈ 1 - equity_to_assets, etc.
      Consolidations needed: ________________________________________________________________

[ ] 2. Test duplication scan
      Command: grep -r "def test_" tests/ | grep -i [keyword] | sort
      Example: grep -i "revenue" tests/ to find similar revenue tests
      Duplicates found: ________________________________________________________________

[ ] 3. Implementation consolidation
      Files with duplicate logic: ________________________________________________________________
      Consolidation plan: ________________________________________________________________

[ ] 4. Ratio registry dedup
      Update ratio_registry.json with:
      [ ] Algebraic equivalences documented
      [ ] Alias relationships added
      [ ] derived_from links updated

[ ] 5. Token analysis
      Tokens used in phases [X-Y]: ________________
      Target: < 5000 per phase after consolidation
      Status: ☐ On target  ☐ Needs investigation

[ ] 6. Consolidation commit
      Command: git commit -m "Consolidation: Phase [X-Y] dedup pass"
      Message details: ________________________________________________________________

Review completed: ☐ YES  ☐ NO
Next review scheduled: Phase _________
```

## ANTI-PATTERN DETECTION QUICK SCAN

Run if you suspect an anti-pattern:

```
Date: __________

Suspected anti-pattern: ________________________________________________________________

UNBOUNDED PHASE GENERATION:
  [ ] Phases exceeding 50 without dedup
  [ ] Multiple similar test files (test_revenue_quality, test_revenue_resilience, etc.)
  [ ] Tokens > 17000 per phase
  [ ] No consolidation commits
  → Action: Stop, review efficiency_baselines.json, plan dedup

CONTEXT EXPLOSION:
  [ ] Any file > 8000 LOC
  [ ] Any file > 15000 LOC (critical!)
  [ ] > 50 functions in single file
  [ ] > 20 imports in single test file
  → Action: Plan module split, see anti_patterns.json prevention rules

ORCHESTRATOR DRIFT:
  [ ] CONTINUITY.md missing phases from code
  [ ] Code has features not in CONTINUITY.md
  [ ] CONTINUITY.md commit hash doesn't match git
  [ ] > 3 untracked files
  → Action: Sync CONTINUITY.md immediately, commit

RATIO DUPLICATION:
  [ ] Similar ratio names (margin, net_margin, net_profit_margin)
  [ ] Algebraically equivalent formulas
  [ ] Multiple implementations of same concept
  → Action: Check ratio_registry.json, consolidate with derived_from

MISSING CHECKPOINTS:
  [ ] 5+ phases without commits
  [ ] 10+ untracked test files
  [ ] Tests passing but commits stale
  → Action: Commit immediately, update CONTINUITY.md

Confirmed anti-pattern: ☐ YES  ☐ NO
Remediation plan: ________________________________________________________________
```

## FILE LOCATIONS FOR QUICK ACCESS

```
.loki/memory/semantic/anti_patterns.json          → Detection signals + prevention rules
.loki/memory/semantic/efficiency_baselines.json   → Baselines + gates + alert thresholds
.loki/memory/semantic/learning_log.json           → Session audit trail
.loki/memory/semantic/ratio_registry.json         → Canonical ratios + dedup rules

.loki/CONTINUITY.md                               → Orchestrator state (high-level)
.loki/LEARNING_INFRASTRUCTURE.md                  → Full documentation
```

## TOKEN TRACKING TEMPLATE

Track tokens to catch runaway generation:

```
Phase: __________  Date: __________

Estimated tokens: __________
Actual tokens used (from LLM logs): __________
Delta: __________

Ratios implemented: __________ (new) / __________ (total)
Tests added: __________ (new) / __________ (total)
Duplications detected: __________

Alert thresholds:
[ ] Tokens <= 5000 (target) ✓
[ ] Tokens <= 17000 (baseline) ✓
[ ] Ratios <= 10 new ✓
[ ] Duplications = 0 ✓

Status: ☐ Green  ☐ Yellow (investigate)  ☐ Red (stop and review)

If Yellow/Red, action: ________________________________________________________________
```

---

**How to Use These Checklists:**

1. **Copy to your session notes** at the start of each phase
2. **Check items off** as you complete them
3. **Fill in actual values** (commit hashes, token counts, etc.)
4. **Review anomalies** - if any checkbox fails, stop and investigate
5. **Store completed checklists** in session logs for audit trail

**Remember:** These are tools to help you think clearly, not bureaucracy. Use them to catch issues early and prevent the anti-pattern cycle.
