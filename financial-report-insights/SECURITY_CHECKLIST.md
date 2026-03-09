# Security Hardening Checklist
## RAG-LLM Financial Report Insights

Quick reference for security remediation priority and status.

---

## CRITICAL - DO FIRST (Today, 2-4 hours)

### Credential Compromise Response
- [ ] **Rotate Neo4j Aura password** (15 min)
  - Go to https://neo4j.com/aura/
  - Database settings → Reset password
  - Generate 32-character random password
  - Store in secrets manager immediately
  - Verify connection with new password
  - Old password in .env is now invalid

- [ ] **Rewrite Git history to remove .env** (2-4 hours)
  - Save current work locally
  - Use BFG Repo-Cleaner or git-filter-branch
  - Force push to all remotes
  - Ask team to fresh clone
  - Update .env.example with no secrets

- [ ] **Add .gitignore** (30 min)
  - File created: `/c/Users/dtmcg/RAG-LLM-project/financial-report-insights/.gitignore`
  - `git add .gitignore && git commit -m "chore: add .gitignore"`

---

## HIGH PRIORITY - WEEK 1 (24-40 hours)

### Rate Limiting
- [ ] Replace `_RATE_LIMIT = 60` with per-endpoint limits
  - /health: 1000 req/min (monitoring)
  - /query: 10 req/min (LLM calls)
  - /analyze: 5 req/min (expensive)
  - /export: 3 req/min (resource intensive)
  - File: `api.py` lines 118-119
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #1

- [ ] Add Retry-After headers
- [ ] Add X-RateLimit-* response headers
- [ ] Add audit logging for violations

### File Upload Security
- [ ] Add `validate_uploaded_file()` function
  - Whitelist: .pdf, .docx, .xlsx, .xls, .txt only
  - Max size: 50MB (from 200MB)
  - MIME type validation (use python-magic)
  - Path traversal prevention
  - File: `api.py`
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #2

- [ ] Implement `/documents/upload` endpoint
- [ ] Test with malicious files (zip bombs, polyglots)

### Input Validation
- [ ] Harden `AnalyzeRequest` validator
  - Validate field names (alphanumeric + _ only)
  - Validate field name length (max 100 chars)
  - Validate numeric value ranges
  - Prevent NaN and Infinity
  - File: `api.py` lines 45-55
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #3

- [ ] Add unit tests for validators

### Dependency Pinning
- [ ] Update `requirements.txt` with pinned versions
  - File created: `requirements.lock` (ready to use)
  - Includes all 30+ dependencies with exact versions
  - Test: `pip install -r requirements.lock --dry-run`
  - Then: `cp requirements.lock requirements.txt`
  - Commit both files

### Docker Security
- [ ] Update `docker-compose.yml` for rag-app
  - Add: `security_opt: [no-new-privileges:true]`
  - Add: `cap_drop: [ALL]`
  - Add: `cap_add: [NET_BIND_SERVICE]`
  - Add: `read_only: true`
  - Add: `tmpfs: [/tmp, /app/.cache]`
  - Remove from `backend` network (keep `frontend` only)
  - Add CPU limits: 2-4 cores
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #6

- [ ] Update `docker-compose.yml` for neo4j
  - Same security options above
  - Add CPU limits: 1-2 cores

- [ ] Test: `docker compose build && docker compose up -d`
- [ ] Verify: `docker inspect rag-financial-insights | grep Security`

### Audit Logging
- [ ] Add `audit_logging_middleware()` to api.py
  - Log sensitive endpoints: /analyze, /export, /documents, /upload
  - Log rate limit violations
  - Log malformed requests (400, 413)
  - Log all 4xx/5xx errors
  - Use structured logging (JSON format)
  - File: `api.py`
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #7

### Security Headers
- [ ] Enhance `security_headers_middleware()`
  - Add CSP (Content-Security-Policy)
  - Add Permissions-Policy
  - Verify X-Content-Type-Options, X-Frame-Options
  - Add Pragma and Expires for caching
  - File: `api.py`
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #7

---

## HIGH PRIORITY - WEEK 2 (24-40 hours)

### Secrets Management
- [ ] Choose one approach:
  - [ ] AWS Secrets Manager (AWS accounts)
  - [ ] HashiCorp Vault (self-managed)
  - [ ] Docker Secrets (Docker Swarm)
  - [ ] Kubernetes Secrets (K8s deployments)

- [ ] Implement secret fetching in `config.py`
- [ ] Remove all secrets from .env files
- [ ] Test secret rotation procedures
- [ ] Document secret management process

### CORS Hardening
- [ ] Add explicit origin validation
  - Don't use wildcard (*)
  - Specify exact domains for production
  - Different config for dev/staging/prod

- [ ] Add `cors_allow_credentials` setting (default: False)
- [ ] Add max_age for CORS preflight
- [ ] Test with curl: `curl -H "Origin: ..." -v http://localhost:8504/query`

### Database Connection Security
- [ ] Enable TLS for Neo4j Aura connections
- [ ] Configure connection pooling
- [ ] Add connection timeouts
- [ ] Implement credential rotation (30-90 days)
- [ ] Monitor failed connection attempts

### CORS Testing
- [ ] Test allowed origins work
- [ ] Test blocked origins fail (403)
- [ ] Test CORS preflight requests
- [ ] Test credentialed requests when disabled

---

## MEDIUM PRIORITY - WEEK 2-3 (16-24 hours)

### CI/CD Security Scanning
- [ ] Install Bandit (Python security linter)
  ```bash
  pip install bandit
  bandit -r . -ll
  ```

- [ ] Install Safety (dependency vulnerability check)
  ```bash
  pip install safety
  safety check
  ```

- [ ] Install Semgrep (SAST static analysis)
  ```bash
  brew install semgrep
  semgrep --config=p/security-audit .
  ```

- [ ] Install Trivy (container scanning)
  ```bash
  brew install trivy
  trivy image rag-financial-insights:latest
  ```

- [ ] Integrate into GitHub Actions/CI pipeline
  - Example in `SECURITY_REMEDIATION_PLAN.md`

### Request Signing/Verification
- [ ] Implement HMAC-SHA256 signing for sensitive endpoints
  - /analyze, /export/xlsx, /export/pdf
  - Prevent replay attacks with timestamp validation
  - Verify signature on server side
  - Reference: `SECURITY_FIXES_REQUIRED.py` FIX #4

### Monitoring & Alerting
- [ ] Add Prometheus metrics for rate limits
- [ ] Create Grafana dashboard
- [ ] Set up alerts for suspicious patterns
- [ ] Create on-call runbook for DDoS response

---

## ONGOING - CONTINUOUS (Monthly)

### Vulnerability Management
- [ ] Enable Dependabot for weekly updates
- [ ] Review/merge security PRs weekly
- [ ] Use Safety to check for CVEs monthly
- [ ] Update container base images quarterly

### Compliance
- [ ] SOC2 Compliance (if required)
  - [ ] Implement access logging
  - [ ] Document change management
  - [ ] Perform annual penetration test
  - [ ] Maintain 90-day audit trail

- [ ] ISO27001 Compliance (if required)
  - [ ] Create security policies
  - [ ] Maintain asset inventory
  - [ ] Perform risk assessments
  - [ ] Implement security training

- [ ] GDPR/Privacy (if required)
  - [ ] Document data processing
  - [ ] Implement data retention
  - [ ] Add export/delete endpoints
  - [ ] Privacy by design

### Security Reviews
- [ ] Monthly: Dependency updates and vulnerabilities
- [ ] Quarterly: Security architecture review
- [ ] Semi-annual: Risk assessment
- [ ] Annual: Penetration testing

---

## TESTING VERIFICATION

### Before Each Release

#### Security Testing
- [ ] Bandit scan: 0 HIGH/MEDIUM findings
- [ ] Safety check: No known vulnerabilities
- [ ] Trivy scan: No CRITICAL findings
- [ ] OWASP ZAP: Automated scan on endpoints

#### Functional Testing
- [ ] All endpoints respond correctly
- [ ] Rate limits work (429 after threshold)
- [ ] File upload rejects invalid types
- [ ] Input validation rejects bad data
- [ ] Logging captures security events
- [ ] CORS headers correct

#### Deployment Testing
- [ ] Docker builds successfully
- [ ] docker-compose stack starts
- [ ] Health checks pass
- [ ] Security options applied
- [ ] Read-only filesystem works
- [ ] Resource limits enforced

### Load Testing Rate Limits
```bash
# Install Apache Bench
brew install httpd  # includes ab

# Test rate limiting
ab -n 100 -c 10 http://localhost:8504/query

# Should see 429 errors after ~10 requests
```

---

## CONFIGURATION CHECKLIST

### config.py Changes
- [ ] `max_file_size_mb = 50` (from 200)
- [ ] `rate_limit_requests_per_minute = 30` (default)
- [ ] `cors_allow_credentials = False`
- [ ] `cors_allow_methods = ["GET", "POST"]`

### .env.example
- [x] All credentials removed
- [x] Security guidance added
- [ ] Deploy to production with secure values

### docker-compose.yml
- [ ] rag-app: security options
- [ ] rag-app: resource limits
- [ ] rag-app: tmpfs volumes
- [ ] rag-app: removed from backend network
- [ ] neo4j: security options
- [ ] neo4j: resource limits

### Dockerfile
- [x] Multi-stage build (confirmed)
- [x] Non-root user (confirmed)
- [x] Health check (confirmed)
- [x] Process supervisor (confirmed)

---

## STATUS TRACKING

### Phase 1 - Credential Response (CRITICAL)
- [ ] Rotate Neo4j password - **ASSIGN TO:** ___________
- [ ] Rewrite Git history - **ASSIGN TO:** ___________
- [ ] Commit .gitignore - **ASSIGN TO:** ___________
- [ ] Commit .env.example - **ASSIGN TO:** ___________
**Target Date:** This week
**Status:** ⏳ PENDING

### Phase 2 - API Security (HIGH)
- [ ] Rate limiting - **ASSIGN TO:** ___________
- [ ] File upload validation - **ASSIGN TO:** ___________
- [ ] Input validation - **ASSIGN TO:** ___________
- [ ] Dependency pinning - **ASSIGN TO:** ___________
- [ ] Docker hardening - **ASSIGN TO:** ___________
- [ ] Audit logging - **ASSIGN TO:** ___________
**Target Date:** Week 1-2
**Status:** ⏳ PENDING

### Phase 3 - Infrastructure (HIGH)
- [ ] Secrets management - **ASSIGN TO:** ___________
- [ ] CORS hardening - **ASSIGN TO:** ___________
- [ ] DB connection security - **ASSIGN TO:** ___________
**Target Date:** Week 2
**Status:** ⏳ PENDING

### Phase 4 - Continuous (MEDIUM)
- [ ] CI/CD security scanning - **ASSIGN TO:** ___________
- [ ] Request signing - **ASSIGN TO:** ___________
- [ ] Monitoring & alerting - **ASSIGN TO:** ___________
**Target Date:** Week 2-3
**Status:** ⏳ PENDING

---

## QUICK REFERENCE

**Files to Review/Update:**
- `SECURITY_AUDIT.md` - Complete findings analysis
- `SECURITY_REMEDIATION_PLAN.md` - Implementation roadmap
- `SECURITY_FIXES_REQUIRED.py` - Code templates
- `.gitignore` - Protect repository
- `.env.example` - Safe configuration template
- `requirements.lock` - Pinned dependencies

**Key Metrics:**
- Critical findings: 3 (must fix immediately)
- High findings: 7 (must fix this month)
- Medium findings: 2 (fix this quarter)
- Total effort: 60-90 hours

**Contacts:**
- Security questions: See SECURITY_AUDIT.md
- Implementation help: See SECURITY_FIXES_REQUIRED.py
- Timeline/planning: See SECURITY_REMEDIATION_PLAN.md

---

## COMPLETION CHECKLIST

Week 1 Complete ✅:
- [ ] Critical credentials rotated
- [ ] .gitignore committed
- [ ] Rate limiting per endpoint
- [ ] File upload validation
- [ ] Dependency versions pinned
- [ ] Docker security hardened

Week 2 Complete ✅:
- [ ] Secrets management in place
- [ ] CORS hardened
- [ ] Audit logging enabled
- [ ] All tests passing

Month 1 Complete ✅:
- [ ] Security scanning in CI/CD
- [ ] Monitoring dashboard live
- [ ] Team trained on changes
- [ ] Documentation updated

---

**Last Updated:** 2026-02-22
**Review Date:** Every 90 days
**Escalation:** security-team@company.com
