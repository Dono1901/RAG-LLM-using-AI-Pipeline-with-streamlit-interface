"""
Startup validation and health check utilities.
Validates Ollama connectivity, model availability, and filesystem readiness.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def check_ollama_connection(host: str | None = None) -> Dict[str, str]:
    """Check if Ollama is reachable and return status."""
    try:
        import ollama
        if host:
            client = ollama.Client(host=host)
            client.list()
        else:
            ollama.list()
        return {"status": "ok", "detail": "Ollama is reachable"}
    except Exception as e:
        return {"status": "error", "detail": f"Cannot connect to Ollama: {e}"}


def check_model_available(model_name: str, host: str | None = None) -> Dict[str, str]:
    """Check if a specific Ollama model is pulled."""
    try:
        import ollama
        if host:
            client = ollama.Client(host=host)
            models = client.list()
        else:
            models = ollama.list()

        model_names = [m.get("name", "") for m in models.get("models", [])]
        # Match with or without tag
        if any(model_name in name for name in model_names):
            return {"status": "ok", "detail": f"Model '{model_name}' is available"}
        return {
            "status": "warning",
            "detail": f"Model '{model_name}' not found. Available: {model_names}. "
                       f"Run: ollama pull {model_name}"
        }
    except Exception as e:
        return {"status": "error", "detail": f"Cannot check models: {e}"}


def check_documents_folder(docs_path: str = "./documents") -> Dict[str, str]:
    """Check if documents folder exists and is writable."""
    path = Path(docs_path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            return {"status": "ok", "detail": f"Created documents folder: {path}"}
        except OSError as e:
            return {"status": "error", "detail": f"Cannot create documents folder: {e}"}

    if not os.access(path, os.W_OK):
        return {"status": "error", "detail": f"Documents folder not writable: {path}"}

    file_count = sum(1 for _ in path.glob("*") if _.is_file())
    return {"status": "ok", "detail": f"Documents folder ready ({file_count} files)"}


def check_neo4j_connection() -> Dict[str, str]:
    """Check Neo4j connectivity when configured."""
    uri = os.environ.get("NEO4J_URI", "").strip()
    if not uri:
        return {"status": "ok", "detail": "Neo4j not configured (optional)"}
    try:
        from graph_store import Neo4jStore
        store = Neo4jStore.connect()
        if store:
            store.close()
            return {"status": "ok", "detail": "Neo4j reachable"}
        return {"status": "warning", "detail": "Neo4j configured but connection failed"}
    except Exception:
        return {"status": "warning", "detail": "Neo4j check error"}


def check_cache_folders() -> Dict[str, str]:
    """Check if cache directories exist and are writable."""
    from config import settings
    cache_dirs = [settings.embedding_cache_dir, settings.llm_cache_dir]
    issues = []
    for d in cache_dirs:
        path = Path(d)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            issues.append(f"{d}: {e}")

    if issues:
        return {"status": "error", "detail": f"Cache folder issues: {'; '.join(issues)}"}
    return {"status": "ok", "detail": "Cache folders ready"}


def run_preflight_checks() -> List[Dict[str, str]]:
    """Run all startup validation checks. Returns list of check results."""
    from config import settings

    ollama_host = os.environ.get("OLLAMA_HOST")

    checks = [
        ("ollama_connection", check_ollama_connection(ollama_host)),
        ("model_available", check_model_available(settings.llm_model, ollama_host)),
        ("documents_folder", check_documents_folder()),
        ("cache_folders", check_cache_folders()),
        ("neo4j_connection", check_neo4j_connection()),
    ]

    results = []
    for name, result in checks:
        result["check"] = name
        level = {"ok": "INFO", "warning": "WARNING", "error": "ERROR"}.get(result["status"], "INFO")
        getattr(logger, level.lower())(f"[{name}] {result['detail']}")
        results.append(result)

    return results


def get_health_status() -> Dict:
    """Return aggregated health status for health check endpoints."""
    results = run_preflight_checks()
    errors = [r for r in results if r["status"] == "error"]
    warnings = [r for r in results if r["status"] == "warning"]

    if errors:
        return {"healthy": False, "status": "unhealthy", "checks": results}
    if warnings:
        return {"healthy": True, "status": "degraded", "checks": results}
    return {"healthy": True, "status": "healthy", "checks": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_preflight_checks()
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n{len(errors)} preflight check(s) failed. App may not work correctly.")
        raise SystemExit(1)
    print("\nAll preflight checks passed.")
