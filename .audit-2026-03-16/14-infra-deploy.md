# Audit Report: Infrastructure & Deployment
**Date:** 2026-03-16
**Scope:** Dockerfile, Docker Compose, GitHub Actions workflows, git hooks, deployment scripts
**Files Reviewed:**
- `financial-report-insights/Dockerfile`
- `financial-report-insights/docker-compose.yml`
- `financial-report-insights/.dockerignore`
- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`
- `.github/workflows/pr-checks.yml`
- `.github/workflows/pr-review.yml`
- `.github/workflows/release.yml`
- `.github/workflows/repo-health.yml`
- `.github/workflows/self-heal.yml`
- `.github/workflows/doc-review.yml`
- `.github/workflows/metrics-collector.yml`
- `.github/workflows/project-sync.yml`
- `.github/scripts/health_check.py`
- `.github/scripts/metrics_collector.py`
- `.github/scripts/doc_review.py`
- `.github/scripts/bootstrap_labels.sh`
- `.githooks/pre-commit`
- `.githooks/post-commit`

---

## P0 — Critical (Fix Immediately)

### P0-01: Security scanning is advisory-only — vulnerabilities never block merges or deploys
**Files:** `.github/workflows/ci.yml` lines 80–83, 200–204; `.github/workflows/deploy.yml` (no security job at all)

Both `bandit` and `pip-audit` are run with `|| true`, meaning a non-zero exit code is silently swallowed. The quality gate at `ci.yml:201–204` explicitly marks security failures as `::warning::` rather than blocking. The deploy workflow (`deploy.yml`) runs no security scan whatsoever before pushing an image to GHCR. This means a known CVE in a dependency, or a high-severity Bandit finding, will never prevent a release from shipping.

**Fix:** Remove `|| true` from both scan commands. Change the quality gate check from `::warning::` to `exit 1` for security failures. Add a security scan job to `deploy.yml` that must pass before `build-and-push`.

---

### P0-02: No container image vulnerability scan (Trivy / Grype) at any stage
**Files:** `.github/workflows/ci.yml` (build job, lines 147–171); `.github/workflows/deploy.yml` (lines 33–71); `.github/workflows/release.yml` (publish job, lines 56–95)

The built Docker image is never scanned for OS-level CVEs (base image packages, installed binaries). `python:3.12-slim` ships with a glibc and openssl version that can contain unpatched CVEs. There is no `aquasecurity/trivy-action` or `anchore/scan-action` step in any pipeline. A vulnerability in `tini`, `gcc`/`g++` build artifacts left in layers, or the Python runtime itself would go undetected.

**Fix:** Add a Trivy scan step after `docker/build-push-action` in `ci.yml` (build job) and in `release.yml` (publish job):
```yaml
- name: Scan image for CVEs
  uses: aquasecurity/trivy-action@0.19.0
  with:
    image-ref: rag-financial-insights:${{ github.sha }}
    format: sarif
    output: trivy-results.sarif
    exit-code: '1'
    severity: 'CRITICAL,HIGH'
```

---

### P0-03: `repo-health.yml` and `metrics-collector.yml` push directly to `main` without a PR
**Files:** `.github/workflows/repo-health.yml` lines 74–80; `.github/workflows/metrics-collector.yml` lines 39–45

Both workflows run `git push || true` to commit JSON files directly to `main` (or the default branch). The `|| true` means a push failure is silently ignored. This pattern bypasses any branch protection rules (required reviews, status checks) if the protection does not explicitly block the `github-actions[bot]` actor. It also creates a race condition when a developer has an in-flight push; the bot push can force a non-fast-forward situation, and the `|| true` hides the failure completely.

**Fix:** Use a `git push` without `|| true` so failures surface. Better: write metrics to a dedicated `metrics/` branch and open a PR to `main`, or use GitHub Pages / a separate artifact store so branch protection is not circumvented.

---

### P0-04: `self-heal.yml` has `contents: write` + `pull-requests: write` + `actions: read` — overly broad for a read/notify workflow
**Files:** `.github/workflows/self-heal.yml` lines 12–16

The workflow only reads CI run data, creates issues, and retries jobs. It never needs `contents: write`. Granting `contents: write` to a workflow that fires on `workflow_run` events from any workflow (including from forked PRs, depending on repository settings) is a supply-chain risk: a compromised action in the dependency chain could use this token to push code.

**Fix:** Remove `contents: write`. Scope to exactly what is needed:
```yaml
permissions:
  issues: write
  actions: read
  pull-requests: write
```

---

## P1 — High (Fix This Sprint)

### P1-01: `deploy.yml` has no explicit `permissions` block — defaults to broad read/write
**File:** `.github/workflows/deploy.yml` (entire file — no `permissions:` key)

Unlike `pr-checks.yml` and `pr-review.yml`, the deploy workflow declares no top-level `permissions`. GitHub's default when no permissions block is set is to inherit from the repository's default token permissions, which is often `contents: read/write` across all scopes. The `build-and-push` job does correctly narrow its own scope (`contents: read, packages: write`), but the `test` job has no job-level permissions and inherits the default, potentially allowing unintended writes during the test run.

**Fix:** Add a top-level restrictive default and only expand at the job level:
```yaml
permissions: {}  # deny all by default

jobs:
  test:
    permissions:
      contents: read
  build-and-push:
    permissions:
      contents: read
      packages: write
```

---

### P1-02: `deploy.yml` duplicate of `release.yml` — two workflows both push images to GHCR on version tags
**Files:** `.github/workflows/deploy.yml` lines 3–6 (`on: push: tags: "v*"`); `.github/workflows/release.yml` lines 5–7 (`on: push: tags: "v*.*.*"`)

Both workflows trigger on `v*` tags (deploy on any `v*`, release on `v*.*.*`). For any semver tag (e.g., `v1.2.3`) both fire simultaneously. They both push to `ghcr.io/${{ github.repository }}/rag-financial-insights`. This creates a race condition where two concurrent image pushes with the same tag may produce non-deterministic results, and the weaker `deploy.yml` (no LOC check, no Buildx multi-arch) can overwrite a correctly built release image.

**Fix:** Remove `deploy.yml` entirely — `release.yml` is the superset. The deploy workflow adds nothing that release does not already do better (it runs fewer checks, has no LOC guardrail, and lacks multi-arch). If a staging deploy workflow is needed, give it a distinct trigger (e.g., `push: branches: [staging]`).

---

### P1-03: `deploy.yml` test job does not run `pip install pytest-cov` — coverage is not measured before pushing
**File:** `.github/workflows/deploy.yml` lines 25–31

The test job installs only `pytest` (not `pytest-cov`, `pytest-xdist`, or security tools). It runs with `-v --tb=short` but has no coverage gate. This means a release can ship from a green test run that has dropped below the 60% coverage floor enforced in CI. The CI pipeline (`ci.yml` line 142) only warns on sub-60% coverage anyway (does not block), so this is a compounding gap.

**Fix:** Either remove `deploy.yml` (see P1-02), or reuse the same install and test invocation from `ci.yml`, including the `--fail-under=60` coverage gate.

---

### P1-04: `repo-health.yml` permissions `contents: write` is unnecessary for a health check/report workflow
**File:** `.github/workflows/repo-health.yml` lines 8–10

The workflow only needs to push one JSON file to the repo (`health-report.json`, line 78). `contents: write` grants far broader permission than that. Combined with `issues: write`, a compromised dependency in `pip install pip-audit` (line 24) could read repository secrets or exfiltrate code under the same runner token.

**Fix:** Narrow to `contents: write` only if you keep the direct push pattern (see P0-03 for why you should not). If you switch to the PR pattern, the health job itself needs only `contents: read`.

---

### P1-05: `healthcheck` in Dockerfile calls both services sequentially — a partial startup can pass
**File:** `financial-report-insights/Dockerfile` lines 53–54

```dockerfile
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health'); urllib.request.urlopen('http://localhost:8504/health')" || exit 1
```

If Streamlit is up but the FastAPI process has not bound yet, `urlopen` on port 8501 succeeds and the second call fails, which raises an exception that propagates to `|| exit 1`. However, if Streamlit is up and FastAPI has crashed permanently (e.g., port conflict), each individual health check invocation will timeout for 10 seconds before the overall `HEALTHCHECK --timeout=10s` fires, meaning the container can sit in an unhealthy-but-not-yet-failed state for up to `30 * 3 = 90` seconds before Docker marks it unhealthy. The two processes share the same health-check command; a failure in either one cannot be distinguished in logs.

**Fix:** Use separate timeouts and fail immediately on connection refused:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)
    urllib.request.urlopen('http://localhost:8504/health', timeout=4)
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
"
```

---

### P1-06: `docker-compose.yml` Neo4j healthcheck embeds `NEO4J_PASSWORD` in the command line
**File:** `financial-report-insights/docker-compose.yml` line 66

```yaml
test: ["CMD-SHELL", "cypher-shell -u neo4j -p ${NEO4J_PASSWORD:?...} 'RETURN 1' || exit 1"]
```

Passwords passed as command-line arguments are visible in `docker inspect`, `ps aux`, and container event logs. Any process with access to the Docker socket or the host process list can read the Neo4j password in plaintext.

**Fix:** Use a credentials file or environment variable inside the healthcheck:
```yaml
test: ["CMD-SHELL", "NEO4J_PASSWORD_FILE=/run/secrets/neo4j_password cypher-shell -u neo4j --password-file /run/secrets/neo4j_password 'RETURN 1' 2>/dev/null || cypher-shell -u neo4j -p \"$${NEO4J_PASSWORD}\" 'RETURN 1' || exit 1"]
```
Or, simpler, use the HTTP API endpoint which requires no password on the healthcheck shell:
```yaml
test: ["CMD", "wget", "-q", "--spider", "http://localhost:7474"]
```

---

### P1-07: `pre-commit` hook Python path is hardcoded to a Windows-specific location
**File:** `.githooks/pre-commit` line 52

```bash
PYTHON_EXE="/c/Users/dtmcg/AppData/Local/Microsoft/WindowsApps/python3.13.exe"
```

This path is absolute and user-specific. Any other developer on the team, any CI runner (Linux `ubuntu-latest`), and any macOS contributor will hit the `SKIP` branch (lines 69–71) because `$PYTHON_EXE` does not exist. This silently bypasses the test suite check for all non-Windows-dtmcg environments, defeating the guard.

**Fix:**
```bash
PYTHON_EXE=$(command -v python3 || command -v python || echo "")
if [ -z "$PYTHON_EXE" ]; then
    echo -e "${YELLOW}[SKIP] Python not found - skipping test check${NC}"
else
    ...
fi
```

---

### P1-08: `ci.yml` coverage gate is advisory (`|| echo "::warning::"`) — never blocks a merge
**File:** `.github/workflows/ci.yml` lines 138–142

```yaml
- name: Coverage summary
  if: matrix.python-version == '3.12'
  run: |
    pip install coverage
    python -m coverage report --fail-under=60 || echo "::warning::Coverage below 60%"
```

`--fail-under=60` exits with code 2 if coverage is below 60%, but the `|| echo` swallows that exit code. The step always exits 0. The quality gate at lines 176–206 checks `needs.test.result`, which is `success` regardless of coverage, so the gate passes. A PR that drops coverage from 95% to 1% will merge cleanly.

**Fix:** Remove the `|| echo` clause. If 60% is truly the floor, let the step fail:
```yaml
run: python -m coverage report --fail-under=60
```

---

## P2 — Medium (Fix Soon)

### P2-01: No `SBOM` (Software Bill of Materials) generation at release time
**File:** `.github/workflows/release.yml` (publish job, lines 56–95)

`docker/build-push-action@v6` supports `--sbom=true` and `--provenance=true` out of the box. Neither is enabled. For a financial application handling sensitive data, having a verifiable SBOM attached to each release image is best practice and increasingly a compliance requirement.

**Fix:** Add to the `build-and-push` step in `release.yml`:
```yaml
sbom: true
provenance: true
```

---

### P2-02: `docker-compose.yml` rag-app container has no CPU limit
**File:** `financial-report-insights/docker-compose.yml` lines 44–49

The `deploy.resources` block sets memory limits (`4G limit / 3G reservation`) but no `cpus` limit. A runaway PDF ingestion or Monte Carlo simulation can starve the Neo4j container and the host OS of CPU time. The Neo4j container also has no CPU limit.

**Fix:**
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: "2.0"
    reservations:
      memory: 3G
      cpus: "0.5"
```

---

### P2-03: `docker-compose.yml` Neo4j ports 7474 (HTTP) and 7687 (Bolt) are bound to `0.0.0.0` (all interfaces)
**File:** `financial-report-insights/docker-compose.yml` lines 54–56

```yaml
ports:
  - "7474:7474"
  - "7687:7687"
```

The Neo4j service is in the `backend` internal network, which is correct. However, the published ports override that network isolation by binding to the host's all-interfaces address. On a developer laptop connected to any network, the Neo4j instance is reachable from that network. The Neo4j password is the only barrier.

**Fix:** Bind to loopback only for local dev, or remove the port mapping entirely if the app container reaches Neo4j via the `backend` Docker network (which it already does via the `bolt://neo4j:7687` URI):
```yaml
ports:
  - "127.0.0.1:7474:7474"
  - "127.0.0.1:7687:7687"
```

---

### P2-04: `deploy.yml` and `release.yml` both build for `linux/amd64` only — no multi-arch support
**Files:** `.github/workflows/release.yml` line 95 (`platforms: linux/amd64`); `deploy.yml` has no `platforms` key (Buildx defaults to host arch)

For a containerized Python application, `linux/arm64` support is inexpensive to add and enables deployment on AWS Graviton, Apple Silicon Docker Desktop native mode, and cost-effective ARM cloud instances.

**Fix:** Add `linux/arm64` to the platforms list and confirm the `python:3.12-slim` base image (which is multi-arch) and all pip wheels support it.

---

### P2-05: `pre-commit` hook regex for detecting new ratio methods is greedy and fragile
**File:** `.githooks/pre-commit` lines 75–83

```bash
NEW_METHODS=$(git diff --cached "$ANALYZER" | grep "^+" | grep -oP 'def (calculate_\w+|analyze_\w+)' | sed 's/def //' || true)
```

This pattern matches lines added in the diff (starting with `+`), but also matches `+++ b/filename` header lines and any `+` inside docstrings that happen to contain `def calculate_`. More importantly, renamed or moved methods (a deletion `^-` paired with an addition `^+`) will be flagged as new unregistered methods even when the method already exists in the registry under the old name. The `|| true` at the end also suppresses grep errors silently.

**Fix:** Tighten the pattern to exclude diff headers (`^+++ ` prefix) and use `grep -oP 'def (calculate_\w+|analyze_\w+)\b'` with a word-boundary anchor. Remove `|| true` and let errors surface. Consider using `git diff --cached -U0` to reduce false positives from context lines.

---

### P2-06: `self-heal.yml` auto-retries CI without human review — can mask persistent failures
**File:** `.github/workflows/self-heal.yml` lines 107–145

The flaky-detection logic (lines 113–128) retries the entire CI run if 1–2 of the last 5 runs failed. There is no deduplication: if the same intermittent test keeps failing, the workflow will keep retrying it every time CI runs, consuming Actions minutes without resolution and suppressing the issue. There is no notification to the team that a retry was triggered, and no cap on total retries per branch.

**Fix:** At minimum, add a comment to the PR or commit when a retry is triggered, so the team knows. Add a guard that stops retrying if the same run-pattern has been retried more than once in 24 hours. Consider logging retries to a tracking issue.

---

### P2-07: `metrics-collector.yml` runs the full test suite (`--cov`) as a side-effect of metric collection — costs are hidden
**File:** `.github/workflows/metrics-collector.yml` lines 30–33

The workflow installs the full requirements stack and runs `python -m pytest tests/ -q --tb=no --cov=. --cov-report=xml || true` every Sunday. The `|| true` means test failures are silently ignored (the metrics are collected against a broken state). This also means Sunday's scheduled run consumes the same runner minutes as a full CI run, but with no blocking behaviour and no visibility into whether the tests actually passed.

**Fix:** Remove `|| true` — if tests fail during metric collection, the coverage value will be 0% or absent, which is a signal worth capturing (and surfacing as a regression). Alternatively, consume the `coverage.xml` artifact produced by the previous CI run rather than re-running tests.

---

### P2-08: `doc-review.yml` posts a PR comment on all `.py` file changes, including test files — noise
**File:** `.github/workflows/doc-review.yml` lines 8–10

```yaml
pull_request:
  paths:
    - "**.md"
    - "financial-report-insights/**/*.py"
```

The wildcard `**/*.py` matches test files (`tests/test_*.py`). Docstring coverage for test functions is not meaningful (and `doc_review.py:93` already skips `test_` functions), but the workflow still fires and posts/updates a comment on every PR that touches a test file. On a project with 4,900+ tests, this is constant noise.

**Fix:**
```yaml
paths:
  - "**.md"
  - "financial-report-insights/*.py"   # top-level modules only, excludes tests/
```

---

### P2-09: `health_check.py` calls `sys.exit(1)` when any critical issue is found — but `repo-health.yml` continues to create the issue regardless
**Files:** `.github/scripts/health_check.py` lines 239–241; `.github/workflows/repo-health.yml` lines 51–73

If `health_check.py` exits 1, the step `run: python .github/scripts/health_check.py` at `repo-health.yml:28` fails. GitHub Actions will skip subsequent steps by default. The issue-creation step at line 51 uses no `if: always()` condition, so it will be skipped when the health check finds critical issues — meaning the most important failures (critical LOC violations, CVEs) never create issues.

**Fix:** Add `if: always()` to both the issue-creation steps and the `git push` step in `repo-health.yml`, or capture the exit code with `continue-on-error: true` on the health check step and pass the result forward.

---

## P3 — Low / Housekeeping

### P3-01: `Dockerfile` runs two processes under a single `CMD bash -c` — not a true process supervisor
**File:** `financial-report-insights/Dockerfile` lines 60–74

The shell-level supervisor loop (`while kill -0 ...`) works but is fragile: it polls with 5-second intervals, meaning a crashed process can go undetected for up to 5 seconds. It also relies on bash signal handling (`trap`) which can behave differently across platforms. A dedicated supervisor like `supervisord`, `s6-overlay`, or `foreman` would be more robust and would log each process independently.

This is a housekeeping note rather than a blocking issue because `restart: unless-stopped` in compose provides the outer recovery loop.

---

### P3-02: `ci.yml` MyPy check is scoped to `structured_types.py` only — most of the codebase is not type-checked
**File:** `.github/workflows/ci.yml` lines 55–56

```yaml
- name: MyPy type check (structured_types)
  run: mypy structured_types.py --ignore-missing-imports --no-error-summary || true
```

Additionally, `|| true` means even this narrow check never blocks anything. The financial calculation core (`financial_analyzer.py`, `ratio_framework.py`, etc.) has no CI type-checking at all.

**Fix:** Remove `|| true`. Expand the checked surface incrementally: add one module per sprint. Start with `protocols.py` and `config.py` since they define the core interfaces.

---

### P3-03: `release.yml` changelog generation uses a bare `git log` without sanitising output for injection into `$GITHUB_OUTPUT`
**File:** `.github/workflows/release.yml` lines 112–122

```bash
echo "changes<<EOF" >> $GITHUB_OUTPUT
echo "$CHANGES" >> $GITHUB_OUTPUT
echo "EOF" >> $GITHUB_OUTPUT
```

If a commit message contains the literal string `EOF` on its own line, it terminates the heredoc early and the remaining commit messages are interpreted as shell commands in subsequent steps. This is a documented GitHub Actions output injection vector.

**Fix:** Use a unique, unguessable delimiter:
```bash
DELIMITER="CHANGELOG_$(uuidgen | tr -d '-')"
echo "changes<<${DELIMITER}" >> $GITHUB_OUTPUT
echo "$CHANGES" >> $GITHUB_OUTPUT
echo "${DELIMITER}" >> $GITHUB_OUTPUT
```

---

### P3-04: `post-commit` hook has no `set -e` — errors are silently swallowed
**File:** `.githooks/post-commit` line 1 (absent)

Unlike `pre-commit` which has `set -e` at line 9, `post-commit` has no error handling. Commands like `git rev-list` or `git log --grep` that fail (e.g., on a shallow clone) will produce empty output that is then used as-is in arithmetic (`$((COMMITS_SINCE % 5))`), causing a bash arithmetic error that is silently ignored.

**Fix:** Add `set -e` (or at minimum `set -o errexit`) after the shebang, and guard arithmetic operations against empty variables:
```bash
COMMITS_SINCE="${COMMITS_SINCE:-0}"
```

---

### P3-05: `.dockerignore` does not exclude `*.log` files or OS-specific artifacts (`.DS_Store`, `Thumbs.db`)
**File:** `financial-report-insights/.dockerignore`

Log files from local pytest runs (`*.log`, `test-results.xml`, `coverage.xml`) and OS artifacts are not excluded. If a developer runs tests locally before building the image, these files end up in the build context and potentially in the image layer that executes `COPY *.py ./` (though `COPY *.py` would not copy them — the risk is mainly build context bloat).

**Fix:** Add to `.dockerignore`:
```
*.log
*.xml
.DS_Store
Thumbs.db
desktop.ini
```

---

### P3-06: `bootstrap_labels.sh` does not verify `gh` CLI is authenticated before running
**File:** `.github/scripts/bootstrap_labels.sh` lines 7–22

The script uses `gh api user` to detect the user, but if the `gh` CLI is not authenticated the auto-detect block silently falls back to `REPO_FLAG=""`, which uses the default gh repo context. A developer running this script unauthenticated will silently apply labels to whatever repo `gh` considers default, potentially a different repo.

**Fix:** Add an explicit auth check at the top:
```bash
gh auth status >/dev/null 2>&1 || { echo "Error: gh CLI not authenticated. Run 'gh auth login'."; exit 1; }
```

---

### P3-07: `pr-checks.yml` secret detection pattern uses `grep -iP` which is not POSIX-portable
**File:** `.github/workflows/pr-checks.yml` lines 100–104

```bash
git diff origin/main...HEAD -- . | grep -iP '(password|secret|api_key|token)\s*=\s*["\x27][^"\x27]{8,}' | grep -v '\.example\|test\|mock\|fake'
```

`grep -P` (Perl-compatible regex) is available on `ubuntu-latest` but is not guaranteed on all runner images or future OS upgrades. Additionally, the pattern will generate false positives on any legitimate assignment of 8+ character strings to variables named `token`, `password`, etc. (e.g., Pydantic model field definitions). A dedicated tool like `truffleHog`, `detect-secrets`, or `gitleaks` is more robust.

**Fix:** Replace the hand-rolled grep with `gitleaks` or `truffleHog`:
```yaml
- uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

### P3-08: `ci.yml` and `pr-checks.yml` both run `ruff` — duplicated effort with no caching between workflows
**Files:** `.github/workflows/ci.yml` lines 49–53; `.github/workflows/pr-checks.yml` lines 34–36

Both workflows run `ruff check` and `ruff format --check` independently without sharing the pip cache (different workflow IDs mean different cache keys). On every PR push, ruff is installed and run twice. While fast, this is wasteful runner-minute usage.

**Fix:** Since `pr-checks.yml` already provides fast PR feedback, remove the lint job from `ci.yml` (which runs on `push` to `main/develop`) or gate it only on `push` to `main`, not on PR events (PRs are already covered by `pr-checks.yml`).

---

## Files With No Issues Found

- `.github/scripts/doc_review.py` — Logic is clean, uses `ast.parse` safely, avoids `eval`, handles all IO exceptions, outputs correctly to `GITHUB_OUTPUT`. The 30-item cap on TODO output is a reasonable safeguard.
- `.github/scripts/metrics_collector.py` — Rolling-window pruning, regression detection thresholds, and XML parsing are all well-implemented. The coverage XML parser correctly handles missing files.
- `.github/workflows/pr-review.yml` — Permissions are correctly scoped (`contents: read`, `pull-requests: write`, `issues: write`). Auto-labelling logic is safe; `github-script` executes under least-privilege token.
- `.github/workflows/project-sync.yml` — Permissions correctly scoped (`contents: read`, `issues: write`, `pull-requests: read`). Stale logic correctly skips PRs and automated health-report issues.
- `.githooks/post-commit` — No security concerns. Informational only; does not gate commits. (Minor robustness note in P3-04 above.)

---

## Summary

| Priority | Count | Key Theme |
|----------|-------|-----------|
| P0 — Critical | 4 | Security scans never block; bot commits bypass branch protection; overly broad token permissions |
| P1 — High | 8 | Duplicate deploy workflow; test/coverage gates advisory only; hardcoded Windows path breaks cross-platform; Neo4j password in process args |
| P2 — Medium | 9 | Missing SBOM; no CPU limits; Neo4j exposed on all interfaces; flaky-retry masking; heredoc injection vector |
| P3 — Low | 8 | Single-file MyPy; supervisord noise; portable grep; minor hook robustness |

**Most urgent actions (do these before the next release tag):**

1. Make Bandit and pip-audit block the pipeline (P0-01). A financial application that ships with known CVEs is a liability, not a risk.
2. Add Trivy image scanning to `ci.yml` and `release.yml` (P0-02). Container-level CVEs are invisible today.
3. Stop direct bot pushes to `main` (P0-03). They bypass all branch protection. Switch metrics to artifacts or a separate branch + PR flow.
4. Narrow `self-heal.yml` token permissions (P0-04). `contents: write` on a `workflow_run`-triggered workflow is a supply-chain attack surface.
5. Delete or merge `deploy.yml` into `release.yml` (P1-02) to eliminate the race condition on version tags.
6. Fix the hardcoded Python path in `pre-commit` (P1-07) so the test guard actually runs for all contributors.
