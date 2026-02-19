#!/usr/bin/env python3
"""Repository health check script.

Called by repo-health.yml workflow. Produces a JSON report and markdown summary
of LOC per module, file threshold checks, dependency audit, and test counting.
Reads thresholds from .loki/memory/semantic/efficiency_baselines.json.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Find the repository root directory."""
    # In GitHub Actions, GITHUB_WORKSPACE is set
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace)
    # Fallback: walk up from this script
    return Path(__file__).resolve().parent.parent.parent


def load_baselines(repo_root: Path) -> dict:
    """Load efficiency baselines from .loki config."""
    baselines_path = repo_root / ".loki" / "memory" / "semantic" / "efficiency_baselines.json"
    if baselines_path.exists():
        with open(baselines_path) as f:
            data = json.load(f)
        return data.get("baselines", {})
    return {}


def count_loc(file_path: Path) -> int:
    """Count lines of code in a file."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def get_module_loc(src_dir: Path) -> dict[str, int]:
    """Get LOC for each Python module in the source directory."""
    loc_map = {}
    for py_file in sorted(src_dir.glob("*.py")):
        loc_map[py_file.name] = count_loc(py_file)
    return loc_map


def count_tests(test_dir: Path) -> tuple[int, int]:
    """Count test files and individual test functions."""
    test_files = 0
    test_functions = 0
    for py_file in test_dir.glob("test_*.py"):
        test_files += 1
        with open(py_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("def test_") or stripped.startswith("async def test_"):
                    test_functions += 1
    return test_files, test_functions


def audit_dependencies(req_file: Path) -> dict:
    """Audit dependencies from requirements.txt."""
    deps = []
    if req_file.exists():
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name
                    name = line.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
                    deps.append(name)
    return {"count": len(deps), "packages": deps}


def check_vulnerabilities(src_dir: Path) -> list[str]:
    """Run pip-audit if available, return findings."""
    findings = []
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "-r", str(src_dir / "requirements.txt"), "--format", "json"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for vuln in data:
                findings.append(f"{vuln.get('name', '?')} {vuln.get('version', '?')}: {vuln.get('id', '?')}")
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass  # pip-audit not installed or timed out
    return findings


def generate_report(repo_root: Path) -> dict:
    """Generate the full health check report."""
    src_dir = repo_root / "financial-report-insights"
    test_dir = src_dir / "tests"
    baselines = load_baselines(repo_root)

    # Thresholds from baselines
    loc_warn = baselines.get("max_file_loc", {}).get("value", 15000)
    loc_block = 20000
    coverage_target = baselines.get("test_coverage_target", {}).get("value", 80)

    # LOC per module
    module_loc = get_module_loc(src_dir)
    total_loc = sum(module_loc.values())

    # Test counts
    test_files, test_functions = count_tests(test_dir)

    # Dependency audit
    dep_info = audit_dependencies(src_dir / "requirements.txt")

    # Vulnerability check
    vulnerabilities = check_vulnerabilities(src_dir)

    # Threshold violations
    warnings = []
    critical = []
    for filename, loc in module_loc.items():
        if loc > loc_block:
            critical.append(f"{filename}: {loc} LOC exceeds {loc_block} BLOCK threshold")
        elif loc > loc_warn:
            warnings.append(f"{filename}: {loc} LOC exceeds {loc_warn} WARNING threshold")

    if vulnerabilities:
        critical.append(f"{len(vulnerabilities)} dependency vulnerability(ies) found")

    report = {
        "timestamp": "",  # Set by caller
        "module_loc": module_loc,
        "total_loc": total_loc,
        "test_files": test_files,
        "test_functions": test_functions,
        "dependency_count": dep_info["count"],
        "dependencies": dep_info["packages"],
        "vulnerabilities": vulnerabilities,
        "vulnerability_count": len(vulnerabilities),
        "thresholds": {
            "loc_warn": loc_warn,
            "loc_block": loc_block,
            "coverage_target": coverage_target,
        },
        "warnings": warnings,
        "critical": critical,
    }
    return report


def format_markdown(report: dict) -> str:
    """Format report as GitHub-flavored markdown."""
    lines = ["# Repository Health Report\n"]

    # Summary
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total LOC | {report['total_loc']:,} |")
    lines.append(f"| Modules | {len(report['module_loc'])} |")
    lines.append(f"| Test files | {report['test_files']} |")
    lines.append(f"| Test functions | {report['test_functions']} |")
    lines.append(f"| Dependencies | {report['dependency_count']} |")
    lines.append(f"| Vulnerabilities | {report['vulnerability_count']} |")
    lines.append("")

    # Module LOC breakdown
    lines.append("## Module LOC\n")
    lines.append("| Module | LOC | Status |")
    lines.append("|--------|-----|--------|")
    for name, loc in sorted(report["module_loc"].items(), key=lambda x: -x[1]):
        if loc > report["thresholds"]["loc_block"]:
            status = "CRITICAL"
        elif loc > report["thresholds"]["loc_warn"]:
            status = "WARNING"
        else:
            status = "OK"
        lines.append(f"| `{name}` | {loc:,} | {status} |")
    lines.append("")

    # Issues
    if report["critical"]:
        lines.append("## Critical Issues\n")
        for item in report["critical"]:
            lines.append(f"- {item}")
        lines.append("")

    if report["warnings"]:
        lines.append("## Warnings\n")
        for item in report["warnings"]:
            lines.append(f"- {item}")
        lines.append("")

    if report["vulnerabilities"]:
        lines.append("## Dependency Vulnerabilities\n")
        for v in report["vulnerabilities"]:
            lines.append(f"- {v}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from datetime import datetime, timezone

    repo_root = get_repo_root()
    report = generate_report(repo_root)
    report["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Write JSON report
    json_path = repo_root / ".github" / "metrics" / "health-report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write markdown summary
    md = format_markdown(report)
    md_path = repo_root / ".github" / "metrics" / "health-report.md"
    with open(md_path, "w") as f:
        f.write(md)

    # Print summary for workflow logs
    print(md)

    # Output for GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_critical={'true' if report['critical'] else 'false'}\n")
            f.write(f"has_warnings={'true' if report['warnings'] else 'false'}\n")
            f.write(f"total_loc={report['total_loc']}\n")
            f.write(f"test_count={report['test_functions']}\n")
            f.write(f"vulnerability_count={report['vulnerability_count']}\n")

    # Exit with error if critical issues
    if report["critical"]:
        sys.exit(1)
