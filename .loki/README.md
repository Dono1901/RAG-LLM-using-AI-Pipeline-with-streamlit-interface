# Continuous Learning Infrastructure

**Status:** Active
**Created:** 2025-02-13
**Purpose:** Enable sustainable, anti-pattern-free development across extended project lifecycles

## Quick Start

If you're starting a new session:

1. **Read first:** [CHECKLIST_QUICK_REFERENCE.md](CHECKLIST_QUICK_REFERENCE.md) (5 min)
2. **Run pre-phase checklist** from that file
3. **Do your work**
4. **Run post-phase checklist**
5. **Update [memory/semantic/learning_log.json](memory/semantic/learning_log.json)**

Done! Your work is now tracked and next session can learn from it.

## Architecture

This learning infrastructure consists of:

### Core Files (in `.loki/memory/semantic/`)

| File | Purpose | Size |
|------|---------|------|
| **anti_patterns.json** | Detectable anti-patterns + prevention rules | 5 patterns, 2 checklists |
| **efficiency_baselines.json** | Quantitative guardrails (tokens, LOC, coverage, etc.) | 6 baselines, 3 gates |
| **learning_log.json** | Session audit trail + insights by phase | Append-only |
| **ratio_registry.json** | Canonical financial ratios + dedup rules | 10 ratios + rules |

### Documentation Files (in `.loki/`)

| File | Purpose |
|------|---------|
| **LEARNING_INFRASTRUCTURE.md** | Full documentation of the system |
| **CHECKLIST_QUICK_REFERENCE.md** | Copyable pre/post/review checklists |
| **README.md** | This file |
| **CONTINUITY.md** | High-level orchestrator state (version control) |

## The Five Anti-Patterns

This system guards against five patterns that caused past problems:

1. **Unbounded Phase Generation** - Continuous new phases without consolidation
   - Detection: > 50 phases, no dedup pass, tokens > 17K per phase
   - Prevention: Dedup every 5 phases, token budget discipline

2. **Context Explosion** - Single files exceeding 15K LOC
   - Detection: Any file > 8000 LOC
   - Prevention: Split modules before reaching 8000 LOC

3. **Orchestrator Drift** - CONTINUITY.md diverging from code
   - Detection: Mismatched commit hashes, untracked files
   - Prevention: Sync CONTINUITY.md with every commit

4. **Ratio Duplication** - Algebraically equivalent ratios implemented multiple times
   - Detection: Similar names or formulas
   - Prevention: Check ratio_registry.json before implementing

5. **Missing Checkpoints** - No commits for 5+ phases
   - Detection: Untracked test files, stale commits
   - Prevention: Commit every 5 phases minimum

## Efficiency Baselines

Quantitative targets to maintain sustainable development:

- **Tokens per phase:** 17,000 baseline → 5,000 target (after consolidation)
- **Max file size:** 15,000 LOC (split at 8,000 LOC)
- **Test coverage:** >= 80% on critical paths
- **Dedup threshold:** 0.85 cosine similarity = duplicate
- **Commit frequency:** Every 5 phases minimum
- **Review cycle:** Every 5 phases (dedup audit)

## Three Checkpoints Per Phase

### Pre-Phase (Before you start)
- ✓ Tests passing
- ✓ Git synchronized
- ✓ CONTINUITY.md updated
- ✓ Anti-patterns check
- ✓ Ratio uniqueness verified
- ✓ Module health OK

### Post-Phase (When you finish)
- ✓ Tests still passing
- ✓ CONTINUITY.md updated with phase summary
- ✓ Code committed
- ✓ Session logged in learning_log.json
- ✓ Ratio registry updated (if new ratios)

### Review Gate (Every 5 phases)
- ✓ Audit ratio formulas for equivalence
- ✓ Scan tests for duplicates
- ✓ Consolidate equivalent implementations
- ✓ Commit consolidation changes

## How Each File Works

### anti_patterns.json
Structure:
```json
{
  "anti_patterns": {
    "pattern_name": {
      "description": "...",
      "severity": "critical|high",
      "detection_signals": ["..."],
      "prevention_rules": ["..."],
      "example_incident": "..."
    }
  },
  "detection_checklist": {
    "pre_phase_launch": ["..."],
    "post_phase_completion": ["..."]
  }
}
```

**Use:** Before phase, scan detection signals. If any triggered, check prevention rules.

### efficiency_baselines.json
Structure:
```json
{
  "baselines": {
    "metric_name": {
      "value": 17000,
      "unit": "tokens|lines|percent",
      "description": "..."
    }
  },
  "efficiency_gates": {
    "pre_phase_gate": {"checks": [...]},
    "post_phase_gate": {"checks": [...]},
    "review_gate": {"trigger": "...", "steps": [...]}
  }
}
```

**Use:** Before phase, verify pre_phase_gate. After phase, verify post_phase_gate. Every 5 phases, run review_gate.

### learning_log.json
Structure:
```json
{
  "sessions": [
    {
      "date": "2025-02-13",
      "session_id": "...",
      "task": "...",
      "outcome": "success|partial|blocked",
      "tokens_used": 12000,
      "ratios_implemented": 3,
      "tests_added": 15,
      "pattern_learned": "...",
      "anti_pattern_detected": null,
      "files_modified": ["..."],
      "commits": ["..."],
      "dedup_actions": "..."
    }
  ]
}
```

**Use:** At session start, search for similar past tasks. At session end, append entry.

### ratio_registry.json
Structure:
```json
{
  "ratios": {
    "canonical_name": {
      "formula": "...",
      "variables": ["..."],
      "category": "...",
      "first_introduced_phase": "Phase_1",
      "description": "...",
      "interpretation": "..."
    }
  },
  "dedup_rules": [...]
}
```

**Use:** Before implementing new ratio, check registry. If similarity > 0.85 with existing, consolidate instead.

## Workflow Example

### Session: Implement Phase 7

**1. Pre-Phase (5 minutes)**
```bash
# Copy pre-phase checklist from CHECKLIST_QUICK_REFERENCE.md
# Run checks:
cd financial-report-insights && python -m pytest tests/ -v  # ✓ Pass
git status  # ✓ Clean
# Check CONTINUITY.md - matches latest commit ✓
# Check anti_patterns.json - no red flags ✓
# Check ratio_registry.json - new ratios are unique ✓
```

**2. Development (X hours)**
```python
# Implement Phase 7 features
# Add tests
# Update financial_analyzer.py or insights_page.py
```

**3. Post-Phase (5 minutes)**
```bash
# Copy post-phase checklist
cd financial-report-insights && python -m pytest tests/ -v  # ✓ Pass

# Update CONTINUITY.md
# vim .loki/CONTINUITY.md
# Add: "Phase 7 (2025-02-13): Implemented feature X, feature Y. Commit: abc1234"

# Commit code
git add -A
git commit -m "Phase 7: Implement feature X and Y

Features:
- Feature X description
- Feature Y description

Tests: 15 new tests, all passing"

# Log in learning_log.json
# Edit .loki/memory/semantic/learning_log.json
# Append session entry with outcome, tokens, patterns, etc.

# If new ratios: update ratio_registry.json
# vim .loki/memory/semantic/ratio_registry.json
# Add entries with formulas, first_introduced_phase: "Phase_7", etc.
```

**4. Every 5 Phases (Review Gate)**
```bash
# Copy review gate checklist
# Audit ratio_registry.json for equivalences
# Scan tests for duplicates
# Consolidate and commit

git commit -m "Consolidation: Phase X-Y dedup pass"
```

## File Locations

```
C:\Users\dtmcg\RAG-LLM-project\
├── .loki\
│   ├── README.md (you are here)
│   ├── LEARNING_INFRASTRUCTURE.md (full docs)
│   ├── CHECKLIST_QUICK_REFERENCE.md (copy checklists from here)
│   ├── CONTINUITY.md (orchestrator state - version controlled)
│   └── memory\
│       └── semantic\
│           ├── anti_patterns.json (5 patterns + detection)
│           ├── efficiency_baselines.json (baselines + gates)
│           ├── learning_log.json (session audit trail)
│           └── ratio_registry.json (10 canonical ratios)
│
├── financial-report-insights\ (your project code)
├── .git\ (version control)
└── ... (other project files)
```

## Integration with Development

These files are **supporting tools**, not obstacles:

- They help you catch issues early (detection)
- They guide decision-making (baselines, prevention)
- They enable learning across sessions (logging, registry)
- They prevent repeating mistakes (anti-patterns)

Use them to think clearly and work sustainably. Adapt baselines as needed. Focus on quality over quantity.

## Key Principles

1. **Anti-patterns are detectable** - If you know the signals, you can catch them before they're problems
2. **Efficiency has baselines** - Token budgets, file sizes, coverage targets are measurable
3. **Learning requires recording** - Session logs enable future sessions to build on past work
4. **Consolidation prevents duplication** - Ratio registry ensures one canonical implementation
5. **Checkpoints enable rollback** - Commits + CONTINUITY.md sync = recovery from bad directions

## Next Steps

1. Read [LEARNING_INFRASTRUCTURE.md](LEARNING_INFRASTRUCTURE.md) for deep context (15 min)
2. Copy [CHECKLIST_QUICK_REFERENCE.md](CHECKLIST_QUICK_REFERENCE.md) checklists to your notes
3. Run pre-phase checklist before starting Phase 7 or next work
4. After each phase, run post-phase checklist and log results
5. Every 5 phases, trigger review gate

---

**Questions?** See [LEARNING_INFRASTRUCTURE.md](LEARNING_INFRASTRUCTURE.md) for comprehensive documentation.

**Emergency?** Check [CHECKLIST_QUICK_REFERENCE.md](CHECKLIST_QUICK_REFERENCE.md) for anti-pattern detection quick scan.

**Ready?** Copy the pre-phase checklist and start your next phase!
