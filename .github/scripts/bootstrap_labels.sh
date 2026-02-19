#!/bin/bash
# Bootstrap GitHub labels from labels.yml
# Usage: bash .github/scripts/bootstrap_labels.sh [owner/repo]
# Requires: gh CLI authenticated with repo scope
# If no repo argument given, auto-detects from gh auth + git remote

set -e

if [ -n "$1" ]; then
  REPO_FLAG="--repo $1"
else
  # Auto-detect: use the fork matching the authenticated gh user
  GH_USER=$(gh api user --jq '.login' 2>/dev/null || echo "")
  REPO_NAME=$(basename "$(git remote get-url origin)" .git 2>/dev/null || echo "")
  if [ -n "$GH_USER" ] && [ -n "$REPO_NAME" ]; then
    REPO_FLAG="--repo ${GH_USER}/${REPO_NAME}"
    echo "Auto-detected repo: ${GH_USER}/${REPO_NAME}"
  else
    REPO_FLAG=""
    echo "Using default gh repo detection"
  fi
fi

echo "Creating/updating labels..."

gh label create $REPO_FLAG "bug" --color "d73a4a" --description "Something isn't working" --force
gh label create $REPO_FLAG "enhancement" --color "a2eeef" --description "New feature or request" --force
gh label create $REPO_FLAG "documentation" --color "0075ca" --description "Improvements or additions to documentation" --force
gh label create $REPO_FLAG "tests" --color "bfd4f2" --description "Test additions or improvements" --force

gh label create $REPO_FLAG "component: analyzer" --color "5319e7" --description "Financial analyzer module" --force
gh label create $REPO_FLAG "component: ui" --color "e99695" --description "Streamlit UI / insights page" --force
gh label create $REPO_FLAG "component: api" --color "f9d0c4" --description "FastAPI REST/SSE layer" --force
gh label create $REPO_FLAG "component: docker" --color "0e8a16" --description "Docker / Docker Compose / DMR" --force
gh label create $REPO_FLAG "component: ci-cd" --color "fbca04" --description "GitHub Actions / CI/CD workflows" --force

gh label create $REPO_FLAG "triage" --color "ededed" --description "Needs initial assessment" --force
gh label create $REPO_FLAG "stale" --color "c2e0c6" --description "No activity for 90+ days" --force
gh label create $REPO_FLAG "needs-attention" --color "e11d48" --description "Requires immediate human review" --force

gh label create $REPO_FLAG "repo-health" --color "1d76db" --description "Auto-generated repository health report" --force
gh label create $REPO_FLAG "regression" --color "b60205" --description "Auto-detected metric regression" --force
gh label create $REPO_FLAG "security" --color "ee0701" --description "Security vulnerability or concern" --force
gh label create $REPO_FLAG "coverage-drop" --color "ff7619" --description "Auto-detected test coverage decrease" --force
gh label create $REPO_FLAG "tech-debt" --color "d4c5f9" --description "Technical debt identified by automation" --force
gh label create $REPO_FLAG "doc-health" --color "0e8a16" --description "Auto-generated documentation health report" --force
gh label create $REPO_FLAG "self-healed" --color "28a745" --description "Issue was auto-fixed by self-healing workflow" --force

echo "Done! 20 labels created/updated."
