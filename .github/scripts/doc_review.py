#!/usr/bin/env python3
"""Documentation health review script.

Called by doc-review.yml workflow. Checks README freshness, scans for
TODO/FIXME comments, measures docstring coverage via ast.parse.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace)
    return Path(__file__).resolve().parent.parent.parent


def check_readme_freshness(repo_root: Path, max_days: int = 90) -> dict:
    """Check when README.md was last modified."""
    readme = repo_root / "README.md"
    if not readme.exists():
        return {"exists": False, "days_since_update": None, "stale": True}
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", "README.md"],
            capture_output=True, text=True, cwd=repo_root, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            import time
            last_modified = int(result.stdout.strip())
            days_ago = (time.time() - last_modified) / 86400
            return {
                "exists": True,
                "days_since_update": round(days_ago),
                "stale": days_ago > max_days,
            }
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return {"exists": True, "days_since_update": None, "stale": False}


def scan_todos(src_dir: Path) -> list[dict]:
    """Scan Python files for TODO/FIXME/HACK/XXX comments."""
    markers = ("TODO", "FIXME", "HACK", "XXX")
    findings = []
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    for marker in markers:
                        if marker in line and "#" in line:
                            findings.append({
                                "file": str(py_file.relative_to(src_dir.parent)),
                                "line": i,
                                "marker": marker,
                                "text": line.strip()[:120],
                            })
        except OSError:
            pass
    return findings


def measure_docstring_coverage(src_dir: Path) -> dict:
    """Measure docstring coverage for modules, classes, and functions."""
    total = 0
    documented = 0
    undocumented = []

    for py_file in sorted(src_dir.glob("*.py")):
        try:
            with open(py_file, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError):
            continue

        # Check module docstring
        total += 1
        if ast.get_docstring(tree):
            documented += 1
        else:
            undocumented.append(f"{py_file.name}: module")

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private/dunder methods and test functions
                if node.name.startswith("_") or node.name.startswith("test_"):
                    continue
                total += 1
                if ast.get_docstring(node):
                    documented += 1
                else:
                    undocumented.append(f"{py_file.name}:{node.lineno} {node.name}()")
            elif isinstance(node, ast.ClassDef):
                total += 1
                if ast.get_docstring(node):
                    documented += 1
                else:
                    undocumented.append(f"{py_file.name}:{node.lineno} class {node.name}")

    pct = round(documented / total * 100, 1) if total > 0 else 0.0
    return {
        "total_items": total,
        "documented": documented,
        "coverage_pct": pct,
        "undocumented_sample": undocumented[:20],  # Limit to avoid huge output
    }


def generate_report(repo_root: Path) -> dict:
    """Generate full documentation health report."""
    src_dir = repo_root / "financial-report-insights"

    readme_info = check_readme_freshness(repo_root)
    todos = scan_todos(src_dir)
    docstrings = measure_docstring_coverage(src_dir)

    # Summarize TODOs by marker
    todo_summary = {}
    for item in todos:
        marker = item["marker"]
        todo_summary[marker] = todo_summary.get(marker, 0) + 1

    return {
        "readme": readme_info,
        "todos": {
            "total": len(todos),
            "by_marker": todo_summary,
            "items": todos[:30],  # Cap output
        },
        "docstrings": docstrings,
    }


def format_markdown(report: dict) -> str:
    """Format report as GitHub-flavored markdown."""
    lines = ["# Documentation Health Report\n"]

    # README
    lines.append("## README Status\n")
    readme = report["readme"]
    if not readme["exists"]:
        lines.append("**README.md not found!**\n")
    elif readme["stale"]:
        lines.append(f"**README.md is stale** - last updated {readme['days_since_update']} days ago\n")
    else:
        lines.append(f"README.md last updated {readme['days_since_update']} days ago\n")

    # TODOs
    lines.append("## TODO/FIXME Scan\n")
    todos = report["todos"]
    lines.append(f"**Total markers found: {todos['total']}**\n")
    if todos["by_marker"]:
        lines.append("| Marker | Count |")
        lines.append("|--------|-------|")
        for marker, count in sorted(todos["by_marker"].items()):
            lines.append(f"| {marker} | {count} |")
        lines.append("")

    if todos["items"]:
        lines.append("<details><summary>Details (first 30)</summary>\n")
        for item in todos["items"]:
            lines.append(f"- `{item['file']}:{item['line']}` [{item['marker']}] {item['text'][:80]}")
        lines.append("\n</details>\n")

    # Docstrings
    lines.append("## Docstring Coverage\n")
    ds = report["docstrings"]
    lines.append(f"**{ds['coverage_pct']}%** ({ds['documented']}/{ds['total_items']} items documented)\n")
    if ds["undocumented_sample"]:
        lines.append("<details><summary>Undocumented items (sample)</summary>\n")
        for item in ds["undocumented_sample"]:
            lines.append(f"- `{item}`")
        lines.append("\n</details>\n")

    return "\n".join(lines)


if __name__ == "__main__":
    repo_root = get_repo_root()
    report = generate_report(repo_root)

    # Write JSON
    json_path = repo_root / ".github" / "metrics" / "doc-report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write markdown
    md = format_markdown(report)
    md_path = repo_root / ".github" / "metrics" / "doc-report.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(md)

    # GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"todo_count={report['todos']['total']}\n")
            f.write(f"docstring_coverage={report['docstrings']['coverage_pct']}\n")
            f.write(f"readme_stale={'true' if report['readme']['stale'] else 'false'}\n")
