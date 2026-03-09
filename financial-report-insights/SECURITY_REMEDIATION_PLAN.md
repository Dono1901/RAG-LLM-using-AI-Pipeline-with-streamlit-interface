# Security Remediation Plan
## RAG-LLM Financial Report Insights

**Prepared:** 2026-02-22
**Status:** READY FOR IMPLEMENTATION
**Estimated Timeline:** 60-90 hours

---

## CRITICAL - IMMEDIATE ACTION (72 hours)

### Phase 1A: Credential Compromise Response (2 hours)

**Priority: URGENT - Prevents data breach**

- [ ] **Step 1: Rotate Neo4j Credentials**
  ```bash
  # 1. Log in to Neo4j Aura console
  # 2. Navigate to database settings
  # 3. Reset Neo4j_PASSWORD to a new strong random password
  # 4. Update securely in secrets manager (see Phase 3)
  ```
  **Responsible:** DevOps/Database Admin
  **Timeframe:** 15 minutes
  **Verification:** Test connection with new password

---

### Phase 1B: Remove Secrets from Git History (4 hours)

**Priority: URGENT - Prevents unauthorized access**

Choose ONE approach (BFG is safer for large repos):

**Option A: Using BFG Repo-Cleaner (RECOMMENDED)**
```bash
# 1. Install bfg
brew install bfg  # macOS
# OR: sudo apt-get install bfg  # Linux
# OR: Download from https://rtyley.github.io/bfg-repo-cleaner/

# 2. Clone a fresh copy (backup current repo first)
cd /tmp
git clone --mirror /c/Users/dtmcg/RAG-LLM-project/financial-report-insights.git

# 3. Remove .env from all history
bfg --delete-files .env financial-report-insights.git

# 4. Finish the rewrite
cd financial-report-insights.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Update your working repo
cd /c/Users/dtmcg/RAG-LLM-project/financial-report-insights
git push --mirror /tmp/financial-report-insights.git
```

**Option B: Using git-filter-branch (if BFG not available)**
```bash
cd /c/Users/dtmcg/RAG-LLM-project/financial-report-insights

# WARNING: This rewrites entire history and affects all collaborators
git filter-branch \
  --tree-filter 'rm -f .env' \
  --prune-empty \
  -- --all

git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push to remote
git push --force-with-lease --all
git push --force-with-lease --tags
```

**Responsible:** DevOps/Security Engineer
**Timeframe:** 2-4 hours (depending on repo size)
**Verification:**
```bash
# Verify .env is removed from all commits
git log --all --full-history -p -- .env
# Should return: "fatal: bad revision"

# Or if using git-filter-branch, nothing should print
```

**Post-Action:** Notify team of force push, request fresh clones

---

### Phase 1C: Add .gitignore (30 minutes)

**Status:** ✅ COMPLETED
**File:** `/c/Users/dtmcg/RAG-LLM-project/financial-report-insights/.gitignore`

- [x] Comprehensive .gitignore file created
- [x] Prevents future commits of secrets
- [ ] Commit to repository

```bash
cd /c/Users/dtmcg/RAG-LLM-project/financial-report-insights
git add .gitignore
git commit -m "chore: add .gitignore to prevent accidental secret commits"
git push
```

---

### Phase 1D: Update .env.example (30 minutes)

**Status:** ✅ COMPLETED
**File:** `/c/Users/dtmcg/RAG-LLM-project/financial-report-insights/.env.example`

- [x] Removed all actual credentials
- [x] Added explanatory comments
- [x] Added security guidance
- [ ] Commit to repository

```bash
cd /c/Users/dtmcg/RAG-LLM-project/financial-report-insights
git add .env.example
git commit -m "security: sanitize .env.example, remove exposed credentials"
git push
```

---

## HIGH PRIORITY - WEEK 1

### Phase 2A: Implement Per-Endpoint Rate Limiting (2 hours)

**File:** `api.py`
**Reference:** `SECURITY_FIXES_REQUIRED.py` (FIX #1)

- [ ] Replace lines 118-119 with per-endpoint rate limits
- [ ] Add `/health` endpoint bypass
- [ ] Add Retry-After headers (HTTP 429)
- [ ] Add X-RateLimit-* response headers
- [ ] Add audit logging for rate limit violations
- [ ] Test each endpoint with ab/wrk load testing

**Testing:**
```bash
# Install Apache Bench (ab) or wrk
# Test rate limiting on /query endpoint
ab -n 100 -c 10 http://localhost:8504/query

# Should see 429 responses after threshold
```

**Responsible:** Backend Developer
**Timeframe:** 2-3 hours including testing

---

### Phase 2B: Add File Upload Validation (3 hours)

**File:** `api.py`
**Reference:** `SECURITY_FIXES_REQUIRED.py` (FIX #2)

- [ ] Add `validate_uploaded_file()` function
- [ ] Whitelist allowed file extensions
- [ ] Validate file size (reduce from 200MB to 50MB)
- [ ] Add MIME type validation (optional: python-magic)
- [ ] Prevent path traversal in filenames
- [ ] Add `/documents/upload` endpoint with validation
- [ ] Add security logging
- [ ] Write unit tests for validation function

**Testing:**
```python
# Test valid files
test_pdf = Path("test.pdf")
validate_uploaded_file(test_pdf)  # Should pass

# Test invalid files
test_zip = Path("malware.zip")
validate_uploaded_file(test_zip)  # Should raise ValueError

test_exe = Path("virus.exe")
validate_uploaded_file(test_exe)  # Should raise ValueError
```

**Responsible:** Backend Developer + QA
**Timeframe:** 3-4 hours including testing

---

### Phase 2C: Harden Input Validation (2 hours)

**File:** `api.py` (AnalyzeRequest class)
**Reference:** `SECURITY_FIXES_REQUIRED.py` (FIX #3)

- [ ] Validate field names (alphanumeric + underscore only)
- [ ] Validate field name length (max 100 chars)
- [ ] Validate numeric value ranges
- [ ] Prevent NaN and Infinity values
- [ ] Add clear error messages
- [ ] Write unit tests for validator

**Testing:**
```python
# Valid request
req = AnalyzeRequest(financial_data={"revenue": 1000000, "expenses": 500000})
# Should pass

# Invalid request with NaN
req = AnalyzeRequest(financial_data={"revenue": float('nan')})
# Should raise ValueError

# Invalid field name
req = AnalyzeRequest(financial_data={"invalid-field": 100})
# Should raise ValueError
```

**Responsible:** Backend Developer
**Timeframe:** 2-3 hours

---

### Phase 2D: Pin All Dependency Versions (1 hour)

**Status:** ✅ COMPLETED
**Files:**
- `requirements.lock` - Exact versions
- `.env.example` - Updated settings

- [x] Created `requirements.lock` with pinned versions
- [ ] Update `requirements.txt` to match
- [ ] Test installation: `pip install -r requirements.lock`
- [ ] Update CI/CD to use `requirements.lock`
- [ ] Commit changes

```bash
cd /c/Users/dtmcg/RAG-LLM-project/financial-report-insights
cp requirements.txt requirements.txt.backup
cp requirements.lock requirements.txt
pip install -r requirements.lock --dry-run  # Verify before committing

git add requirements.txt requirements.lock
git commit -m "security: pin all dependency versions for reproducible builds"
git push
```

**Responsible:** DevOps/Backend
**Timeframe:** 1 hour

---

### Phase 2E: Update docker-compose.yml Security (2 hours)

**Reference:** `SECURITY_FIXES_REQUIRED.py` (FIX #6)

- [ ] Add `security_opt: [no-new-privileges:true]` to both services
- [ ] Add `cap_drop: [ALL]` to both services
- [ ] Add `cap_add: [NET_BIND_SERVICE]` to rag-app only
- [ ] Add `read_only: true` to both services
- [ ] Add `tmpfs` volumes for temporary data
- [ ] Remove rag-app from `backend` network (keep only `frontend`)
- [ ] Add CPU limits for both services
- [ ] Test: `docker compose build && docker compose up`

**Testing:**
```bash
# Verify security options are applied
docker inspect rag-financial-insights | grep -A5 "CapAdd\|CapDrop\|ReadOnly\|SecurityOpt"

# Should see:
# "CapDrop": ["ALL"],
# "CapAdd": ["NET_BIND_SERVICE"],
# "ReadonlyRootfs": true,
# "SecurityOpt": ["no-new-privileges:true"]
```

**Responsible:** DevOps/SRE
**Timeframe:** 1.5-2 hours

---

### Phase 2F: Implement Audit Logging (2 hours)

**File:** `api.py`
**Reference:** `SECURITY_FIXES_REQUIRED.py` (FIX #7)

- [ ] Add `audit_logging_middleware` for sensitive endpoints
- [ ] Log: method, path, client_ip, timestamp, query_string
- [ ] Log rate limit violations
- [ ] Log malformed requests (400, 413)
- [ ] Use structured logging (JSON format)
- [ ] Wire logs to centralized logging (future)
- [ ] Write tests for logging

**Responsible:** Backend Developer
**Timeframe:** 1.5-2 hours

---

## HIGH PRIORITY - WEEK 2

### Phase 3A: Implement Secrets Management (4 hours)

**Choose ONE approach:**

**Option 1: AWS Secrets Manager (if using AWS)**
```python
# config.py
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str) -> dict:
    """Fetch secret from AWS Secrets Manager."""
    client = boto3.client("secretsmanager")
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except ClientError as e:
        raise ValueError(f"Failed to fetch secret: {e}") from e

# Usage
neo4j_creds = get_secret("prod/neo4j/aura")
NEO4J_PASSWORD = neo4j_creds["password"]
```

**Option 2: HashiCorp Vault**
```python
# config.py
import hvac

def get_vault_secret(path: str, key: str) -> str:
    """Fetch secret from Vault."""
    client = hvac.Client(
        url=os.getenv("VAULT_ADDR"),
        token=os.getenv("VAULT_TOKEN")
    )
    secrets = client.secrets.kv.read_secret_version(path)
    return secrets["data"]["data"][key]

# Usage
NEO4J_PASSWORD = get_vault_secret("secret/data/neo4j", "password")
```

**Option 3: Docker Secrets (if using Docker Swarm)**
```bash
# Create secret
echo "oWPgE_o5Dq2QHQmWHM8tpv61RFEMBJ7DlUt3-nZn6Io" | \
  docker secret create neo4j_password -

# Use in docker-compose.yml
services:
  neo4j:
    secrets:
      - neo4j_password
    environment:
      NEO4J_PASSWORD_FILE: /run/secrets/neo4j_password
```

**Option 4: Kubernetes Secrets (if using K8s)**
```bash
kubectl create secret generic neo4j-password \
  --from-literal=password=YOUR_PASSWORD

# Then reference in deployment manifests
```

**Responsible:** DevOps/Platform Engineer
**Timeframe:** 3-4 hours

---

### Phase 3B: Update CORS Configuration (1.5 hours)

**File:** `config.py`, `api.py`
**Reference:** `SECURITY_FIXES_REQUIRED.py` (FIX #7)

- [ ] Add `cors_allow_credentials` setting (default: False)
- [ ] Add explicit origin validation
- [ ] Add CSP headers
- [ ] Add Permissions-Policy headers
- [ ] Add `max_age` for CORS preflight (3600 seconds)
- [ ] Document CORS configuration for teams
- [ ] Test CORS with actual frontend

**Testing:**
```bash
# Test CORS headers
curl -H "Origin: http://localhost:8501" \
     -H "Access-Control-Request-Method: POST" \
     -v http://localhost:8504/query

# Should see:
# Access-Control-Allow-Origin: http://localhost:8501
# Access-Control-Allow-Methods: GET, POST
```

**Responsible:** Backend Developer
**Timeframe:** 1-1.5 hours

---

### Phase 3C: Database Connection Security (2 hours)

- [ ] Enable Neo4j SSL/TLS for Aura connections
- [ ] Implement connection pooling with size limits
- [ ] Add connection timeout settings
- [ ] Implement automatic credential rotation (30-90 days)
- [ ] Add monitoring for failed connection attempts
- [ ] Document secure connection procedures

**Config Example:**
```python
# config.py
neo4j_connection_timeout: int = 30  # seconds
neo4j_max_pool_size: int = 50
neo4j_use_encryption: bool = True
neo4j_trust_certs: str = "TRUST_ALL_CERTIFICATES"  # or path to CA cert
```

**Responsible:** Database Admin/Backend
**Timeframe:** 1.5-2 hours

---

## MEDIUM PRIORITY - WEEK 2-3

### Phase 4A: Add Security Testing to CI/CD (3 hours)

**Tools to integrate:**
- [ ] **Bandit** - Python security linting
  ```bash
  pip install bandit
  bandit -r /c/Users/dtmcg/RAG-LLM-project/financial-report-insights/ -ll
  ```

- [ ] **Safety** - Dependency vulnerability checking
  ```bash
  pip install safety
  safety check --json
  ```

- [ ] **Semgrep** - SAST static analysis
  ```bash
  brew install semgrep
  semgrep --config=p/security-audit /c/Users/dtmcg/RAG-LLM-project/financial-report-insights/
  ```

- [ ] **Container scanning** - Trivy for Docker images
  ```bash
  brew install trivy
  trivy image rag-financial-insights:latest
  ```

**GitHub Actions Example:**
```yaml
# .github/workflows/security.yml
name: Security Scanning
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r . -ll -f json -o bandit-report.json

      - name: Run Safety
        run: |
          pip install safety
          safety check --json

      - name: Trivy Container Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'rag-financial-insights:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: trivy-results.sarif
```

**Responsible:** DevOps/CI-CD Engineer
**Timeframe:** 2.5-3 hours

---

### Phase 4B: Add Request Signing/Verification (3 hours)

For sensitive endpoints (/analyze, /export), implement request verification:

```python
# api.py
import hmac
import hashlib
from datetime import datetime, timedelta

class SignedRequest(BaseModel):
    payload: Dict[str, Any]
    signature: str
    timestamp: str

def verify_signature(
    payload: str,
    signature: str,
    shared_secret: str,
    max_age_seconds: int = 300
) -> bool:
    """Verify HMAC-SHA256 signature and prevent replay attacks."""
    try:
        # Check timestamp
        request_time = datetime.fromisoformat(timestamp)
        if datetime.utcnow() - request_time > timedelta(seconds=max_age_seconds):
            raise ValueError("Request timestamp too old (replay attack?)")

        # Verify signature
        expected_sig = hmac.new(
            shared_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_sig)
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return False

@app.post("/analyze")
async def analyze_signed(req: SignedRequest):
    """Analyze financial data with request signature verification."""
    if not verify_signature(
        str(req.payload),
        req.signature,
        os.getenv("SHARED_SECRET")
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # ... rest of endpoint
```

**Responsible:** Backend Developer + Security
**Timeframe:** 2.5-3 hours

---

### Phase 4C: Implement Rate Limiting Monitoring (2 hours)

- [ ] Add Prometheus metrics for rate limits
- [ ] Create Grafana dashboard for rate limit violations
- [ ] Set up alerting for suspicious patterns
- [ ] Log rate limit violations to centralized logging
- [ ] Create on-call runbook for DDoS response

**Prometheus Metrics Example:**
```python
from prometheus_client import Counter, Histogram

rate_limit_violations = Counter(
    'rate_limit_violations_total',
    'Total rate limit violations',
    ['endpoint', 'client_ip']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # ... rate limiting logic ...
    if rate_limited:
        rate_limit_violations.labels(
            endpoint=request.url.path,
            client_ip=client_ip
        ).inc()
```

**Responsible:** Observability/DevOps
**Timeframe:** 1.5-2 hours

---

## ONGOING - CONTINUOUS IMPROVEMENT

### Phase 5: Compliance & Audit (Monthly)

- [ ] **SOC2 Compliance** (if required)
  - Implement access logging
  - Document change management process
  - Perform annual penetration testing
  - Maintain audit trail for 90 days

- [ ] **ISO27001 Compliance** (if required)
  - Create information security policy
  - Implement asset inventory
  - Risk assessment and register
  - Implement security awareness training

- [ ] **GDPR/Privacy** (if applicable)
  - Document data processing activities
  - Implement data retention policies
  - Add data export/deletion endpoints
  - Implement privacy by design

**Responsibility:** Security, Compliance, Legal
**Frequency:** Quarterly review, Annual audit

---

### Phase 6: Vulnerability Management (Continuous)

**Automated:**
- [ ] Dependabot for weekly dependency updates
- [ ] GitGuardian for secrets scanning
- [ ] Snyk for vulnerability monitoring
- [ ] OWASP Dependency-Check integration

**Manual:**
- [ ] Monthly security review
- [ ] Quarterly penetration testing
- [ ] Semi-annual risk assessment
- [ ] Annual security architecture review

**Tools & Setup:**

```bash
# GitHub Dependabot (auto-create PRs for updates)
# In .github/dependabot.yml:

version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    allow:
      - dependency-type: "direct"
    reviewers:
      - "security-team"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

## TESTING & VERIFICATION CHECKLIST

After each phase, verify with:

### Security Testing
- [ ] Bandit scan completes with 0 HIGH/MEDIUM findings
- [ ] Safety check shows no known vulnerabilities
- [ ] Trivy container scan shows no CRITICAL findings
- [ ] OWASP ZAP scan on exposed endpoints
- [ ] Rate limiting tested with ab/wrk
- [ ] File upload validation tested with malicious files

### Functional Testing
- [ ] All endpoints respond with 200/appropriate status codes
- [ ] Rate limits correctly reject over-quota requests
- [ ] File upload rejects invalid types
- [ ] Input validation rejects bad data
- [ ] Logging captures all security events
- [ ] CORS headers correctly restrict origins
- [ ] Health checks pass

### Deployment Testing
- [ ] Docker image builds successfully
- [ ] docker-compose stack starts cleanly
- [ ] Security options applied (verify with docker inspect)
- [ ] Read-only filesystem works (temp files in tmpfs)
- [ ] Resource limits enforced
- [ ] Logging to stdout (JSON format)

```bash
# Comprehensive test script
#!/bin/bash

echo "=== Running Security Tests ==="
bandit -r . -ll -f json
safety check --json
semgrep --config=p/security-audit .

echo "=== Building Docker Image ==="
docker build -t rag-financial-insights:test .

echo "=== Starting Stack ==="
docker compose up -d

sleep 30

echo "=== Verifying Health Checks ==="
curl http://localhost:8501/_stcore/health
curl http://localhost:8504/health

echo "=== Testing Rate Limiting ==="
for i in {1..70}; do
  curl -s http://localhost:8504/query -X POST -H "Content-Type: application/json" \
    -d '{"text":"test","top_k":3}' -w "%{http_code}\n" | tail -1
done | sort | uniq -c

echo "=== All Tests Complete ==="
docker compose down
```

**Responsibility:** QA/DevOps
**Frequency:** Before each release + continuous

---

## TIMELINE SUMMARY

| Phase | Tasks | Owner | Duration | Status |
|-------|-------|-------|----------|--------|
| 1A | Rotate Neo4j | DevOps | 15 min | URGENT |
| 1B | Git history cleanup | Security | 2-4 hrs | URGENT |
| 1C | Add .gitignore | DevOps | 30 min | ✅ DONE |
| 1D | Update .env.example | Backend | 30 min | ✅ DONE |
| 2A | Rate limiting | Backend | 2-3 hrs | THIS WEEK |
| 2B | File upload validation | Backend | 3-4 hrs | THIS WEEK |
| 2C | Input validation | Backend | 2-3 hrs | THIS WEEK |
| 2D | Pin dependencies | DevOps | 1 hr | THIS WEEK |
| 2E | Docker hardening | DevOps | 2 hrs | THIS WEEK |
| 2F | Audit logging | Backend | 2 hrs | THIS WEEK |
| 3A | Secrets management | DevOps | 3-4 hrs | WEEK 2 |
| 3B | CORS hardening | Backend | 1-1.5 hrs | WEEK 2 |
| 3C | DB connection security | DBA | 1.5-2 hrs | WEEK 2 |
| 4A | CI/CD security | DevOps | 2.5-3 hrs | WEEK 2-3 |
| 4B | Request signing | Backend | 2.5-3 hrs | WEEK 2-3 |
| 4C | Monitoring | Observability | 1.5-2 hrs | WEEK 2-3 |

**Total Estimated Effort:** 60-90 hours
**Recommended Sprint Duration:** 2 sprints (2-4 weeks)

---

## SIGN-OFF & NEXT STEPS

- [ ] Review this plan with security team
- [ ] Assign owners to each phase
- [ ] Schedule sprint planning for Week 1 critical items
- [ ] Set up tracking (GitHub Issues, Jira, etc.)
- [ ] Create blockers for critical items
- [ ] Schedule 1x/week security review meeting
- [ ] Update incident response procedures
- [ ] Brief team on security changes

**Plan Prepared By:** Security Engineer
**Plan Review Date:** 2026-02-22
**Recommended Start Date:** 2026-02-24 (ASAP)
**Target Completion:** 2026-03-22 (4 weeks)
