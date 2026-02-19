#!/usr/bin/env python3
"""Metrics collector for CI/CD continuous learning system.

Parses coverage.xml, collects LOC/test counts/dependency info, appends to
.github/metrics/history.json (90-day rolling window), and detects regressions
using 7-day rolling averages.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean


def get_repo_root() -> Path:
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace)
    return Path(__file__).resolve().parent.parent.parent


def parse_coverage_xml(coverage_path: Path) -> float | None:
    """Parse coverage.xml and return line coverage percentage."""
    if not coverage_path.exists():
        return None
    try:
        tree = ET.parse(coverage_path)
        root = tree.getroot()
        line_rate = root.get("line-rate")
        if line_rate is not None:
            return round(float(line_rate) * 100, 2)
    except (ET.ParseError, ValueError):
        pass
    return None


def count_loc(src_dir: Path) -> tuple[dict[str, int], int]:
    """Count LOC per module and total."""
    module_loc = {}
    for py_file in sorted(src_dir.glob("*.py")):
        try:
            with open(py_file, encoding="utf-8", errors="replace") as f:
                module_loc[py_file.name] = sum(1 for _ in f)
        except OSError:
            pass
    return module_loc, sum(module_loc.values())


def count_tests(test_dir: Path) -> tuple[int, int]:
    """Count test files and test functions."""
    test_files = 0
    test_functions = 0
    for py_file in test_dir.glob("test_*.py"):
        test_files += 1
        try:
            with open(py_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("def test_") or stripped.startswith("async def test_"):
                        test_functions += 1
        except OSError:
            pass
    return test_files, test_functions


def count_dependencies(req_file: Path) -> int:
    """Count non-comment, non-empty lines in requirements.txt."""
    if not req_file.exists():
        return 0
    count = 0
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def collect_entry(repo_root: Path) -> dict:
    """Collect all metrics for a single entry."""
    src_dir = repo_root / "financial-report-insights"
    test_dir = src_dir / "tests"

    coverage_pct = parse_coverage_xml(src_dir / "coverage.xml")
    module_loc, total_loc = count_loc(src_dir)
    test_files, test_count = count_tests(test_dir)
    dep_count = count_dependencies(src_dir / "requirements.txt")

    # Get build duration from env (set by workflow)
    build_duration = os.environ.get("BUILD_DURATION_SECONDS")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "branch": os.environ.get("GITHUB_REF_NAME", "unknown"),
        "coverage_pct": coverage_pct,
        "test_count": test_count,
        "test_file_count": test_files,
        "module_loc": module_loc,
        "total_loc": total_loc,
        "dependency_count": dep_count,
        "build_duration_seconds": float(build_duration) if build_duration else None,
    }
    return entry


def load_history(history_path: Path) -> dict:
    """Load existing history or create empty structure."""
    if history_path.exists():
        try:
            with open(history_path) as f:
                data = json.load(f)
            if "entries" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"schema_version": 1, "description": "Rolling 90-day metrics history", "entries": []}


def prune_old_entries(history: dict, days: int = 90) -> dict:
    """Remove entries older than N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    history["entries"] = [e for e in history["entries"] if e.get("timestamp", "") >= cutoff]
    return history


def get_rolling_avg(entries: list[dict], key: str, days: int = 7) -> float | None:
    """Compute rolling average of a metric over last N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    values = [e[key] for e in entries if e.get("timestamp", "") >= cutoff and e.get(key) is not None]
    if not values:
        return None
    return mean(values)


def detect_regressions(entries: list[dict], new_entry: dict) -> list[str]:
    """Detect regressions comparing new entry against 7-day rolling averages."""
    regressions = []
    if len(entries) < 2:
        return regressions

    # Coverage drop > 2%
    avg_coverage = get_rolling_avg(entries, "coverage_pct")
    if avg_coverage is not None and new_entry.get("coverage_pct") is not None:
        drop = avg_coverage - new_entry["coverage_pct"]
        if drop > 2.0:
            regressions.append(
                f"Coverage dropped {drop:.1f}% (from {avg_coverage:.1f}% avg to {new_entry['coverage_pct']:.1f}%)"
            )

    # Test count decrease
    avg_tests = get_rolling_avg(entries, "test_count")
    if avg_tests is not None and new_entry.get("test_count") is not None:
        if new_entry["test_count"] < avg_tests - 5:
            regressions.append(
                f"Test count decreased (from {avg_tests:.0f} avg to {new_entry['test_count']})"
            )

    # Build time increase > 50%
    avg_duration = get_rolling_avg(entries, "build_duration_seconds")
    if avg_duration is not None and new_entry.get("build_duration_seconds") is not None:
        if new_entry["build_duration_seconds"] > avg_duration * 1.5:
            regressions.append(
                f"Build time increased >50% (from {avg_duration:.0f}s avg to {new_entry['build_duration_seconds']:.0f}s)"
            )

    # LOC spike > 1000 in any module
    if entries:
        last = entries[-1]
        last_loc = last.get("module_loc", {})
        new_loc = new_entry.get("module_loc", {})
        for module, loc in new_loc.items():
            prev = last_loc.get(module, loc)
            if loc - prev > 1000:
                regressions.append(
                    f"`{module}` LOC spiked by {loc - prev} lines ({prev} -> {loc})"
                )

    # Dependency bloat > 3 new deps
    avg_deps = get_rolling_avg(entries, "dependency_count")
    if avg_deps is not None and new_entry.get("dependency_count") is not None:
        if new_entry["dependency_count"] - avg_deps > 3:
            regressions.append(
                f"Dependency count spiked (from {avg_deps:.0f} avg to {new_entry['dependency_count']})"
            )

    return regressions


if __name__ == "__main__":
    repo_root = get_repo_root()
    history_path = repo_root / ".github" / "metrics" / "history.json"

    # Collect new entry
    new_entry = collect_entry(repo_root)

    # Load and update history
    history = load_history(history_path)
    regressions = detect_regressions(history["entries"], new_entry)
    history["entries"].append(new_entry)
    history = prune_old_entries(history)

    # Save updated history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Print summary
    print(f"Metrics collected at {new_entry['timestamp']}")
    print(f"  Coverage: {new_entry['coverage_pct']}%")
    print(f"  Tests: {new_entry['test_count']} ({new_entry['test_file_count']} files)")
    print(f"  Total LOC: {new_entry['total_loc']:,}")
    print(f"  Dependencies: {new_entry['dependency_count']}")

    # Output for GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_regressions={'true' if regressions else 'false'}\n")
            f.write(f"regression_count={len(regressions)}\n")

    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r}")

        # Write regression details for workflow
        regression_path = repo_root / ".github" / "metrics" / "regressions.json"
        with open(regression_path, "w") as f:
            json.dump({"regressions": regressions, "entry": new_entry}, f, indent=2)

        sys.exit(1)
