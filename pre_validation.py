#!/usr/bin/env python3
"""Pre-submission validation script for the CloudFinOps OpenEnv Hackathon.

Runs all disqualifying checks before submission. Exit code 0 = pass, 1 = fail.

Usage:
    python pre_validation.py
    python pre_validation.py --verbose
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: list[tuple[str, str, str]] = []  # (check, status, detail)


def record(check: str, status: str, detail: str = "") -> bool:
    results.append((check, status, detail))
    ok = status != FAIL
    return ok


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except ImportError:
        # Minimal YAML parser fallback for simple key-value files
        import re
        data: dict = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\w[\w_]*):\s*(.*)", line)
            if m:
                data[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        return data


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_openenv_yaml() -> bool:
    path = ROOT / "openenv.yaml"
    if not path.exists():
        return record("openenv.yaml exists", FAIL, "File not found")

    data = _load_yaml(path)

    ok = True
    ok &= record("openenv.yaml has spec_version",
                 PASS if "spec_version" in data else FAIL,
                 str(data.get("spec_version", "missing")))

    tasks = data.get("tasks", [])
    ok &= record("openenv.yaml has ≥3 tasks",
                 PASS if len(tasks) >= 3 else FAIL,
                 f"{len(tasks)} tasks: {[t.get('id', t) for t in tasks]}")

    env_vars = data.get("env", [])
    required_vars = {"API_BASE_URL", "MODEL_NAME", "HF_TOKEN"}
    found = {v.get("name", "") for v in env_vars}
    missing = required_vars - found
    ok &= record("Required env vars defined",
                 PASS if not missing else FAIL,
                 f"Missing: {missing}" if missing else f"All present: {required_vars}")

    return ok


def check_inference_py() -> bool:
    path = ROOT / "inference.py"
    if not path.exists():
        return record("inference.py at root", FAIL, "File not found")

    content = path.read_text()

    ok = True
    ok &= record("inference.py uses OpenAI client",
                 PASS if "from openai import OpenAI" in content else FAIL)

    ok &= record("inference.py uses API_BASE_URL",
                 PASS if "API_BASE_URL" in content else FAIL)

    ok &= record("inference.py uses MODEL_NAME",
                 PASS if "MODEL_NAME" in content else FAIL)

    ok &= record("inference.py uses HF_TOKEN",
                 PASS if "HF_TOKEN" in content else FAIL)

    ok &= record("inference.py emits [START] logs",
                 PASS if "[START]" in content else FAIL)

    ok &= record("inference.py emits [STEP] logs",
                 PASS if "[STEP]" in content else FAIL)

    ok &= record("inference.py emits [END] logs",
                 PASS if "[END]" in content else FAIL)

    return ok


def check_dockerfile() -> bool:
    paths = [ROOT / "server" / "Dockerfile", ROOT / "server" / "Dockerfile.standalone"]
    existing = [p for p in paths if p.exists()]

    if not existing:
        return record("Dockerfile exists", FAIL, "No Dockerfile found in server/")

    ok = True
    ok &= record("Dockerfile exists",
                 PASS,
                 f"Found: {[p.name for p in existing]}")

    # Check for standalone variant
    has_standalone = any(p.name == "Dockerfile.standalone" for p in existing)
    if not has_standalone:
        record("Standalone Dockerfile available",
               WARN,
               "Consider adding Dockerfile.standalone with python:3.11-slim base")

    return ok


def check_requirements() -> bool:
    path = ROOT / "server" / "requirements.txt"
    if not path.exists():
        return record("server/requirements.txt exists", FAIL)

    content = path.read_text().lower()
    ok = True

    for dep in ["openai", "httpx", "tenacity", "python-dotenv", "fastapi", "uvicorn"]:
        ok &= record(f"requirements.txt includes {dep}",
                     PASS if dep in content else FAIL)

    return ok


def check_env_example() -> bool:
    path = ROOT / ".env.example"
    if not path.exists():
        return record(".env.example exists", FAIL)

    content = path.read_text()
    ok = True
    for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        ok &= record(f".env.example defines {var}",
                     PASS if var in content else FAIL)

    return ok


def check_readme() -> bool:
    path = ROOT / "README.md"
    if not path.exists():
        return record("README.md exists", FAIL)

    content = path.read_text()
    ok = True
    ok &= record("README describes action space",
                 PASS if "Action Space" in content or "action space" in content.lower() else FAIL)

    ok &= record("README describes observation space",
                 PASS if "Observation Space" in content or "observation space" in content.lower() else FAIL)

    ok &= record("README has setup instructions",
                 PASS if "Quick Start" in content or "setup" in content.lower() else FAIL)

    return ok


def check_tasks_and_graders() -> bool:
    """Import engine and verify all tasks produce scores in [0.0, 1.0]."""
    try:
        from server.cloudfinops_env_environment import CloudFinOpsEngine, MAX_STEPS
        from models import CloudFinOpsAction
    except ImportError:
        return record("Engine imports", FAIL, "Cannot import engine for grading check")

    engine = CloudFinOpsEngine()
    ok = True

    for task_id in ["easy", "medium", "hard", "green"]:
        engine.reset(task_id)
        for _ in range(MAX_STEPS):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        score = engine.grade()
        in_range = 0.0 <= score <= 1.0
        ok &= record(f"Task '{task_id}' grader score in [0.0, 1.0]",
                     PASS if in_range else FAIL,
                     f"Score: {score}")

    return ok


def check_tests_exist() -> bool:
    tests_dir = ROOT / "tests"
    if not tests_dir.exists():
        return record("tests/ directory exists", FAIL)

    test_files = list(tests_dir.glob("test_*.py"))
    ok = record("Test files exist",
                PASS if test_files else FAIL,
                f"Found: {[f.name for f in test_files]}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    verbose = "--verbose" in sys.argv

    print("=" * 60)
    print("  CloudFinOps Pre-Submission Validator")
    print("=" * 60)
    print()

    checks = [
        ("OpenEnv Spec", check_openenv_yaml),
        ("Inference Script", check_inference_py),
        ("Dockerfile", check_dockerfile),
        ("Requirements", check_requirements),
        ("Env Example", check_env_example),
        ("README", check_readme),
        ("Tasks & Graders", check_tasks_and_graders),
        ("Tests Exist", check_tests_exist),
    ]

    all_ok = True
    for name, fn in checks:
        print(f"── {name} ──")
        ok = fn()
        all_ok &= ok
        print()

    # Summary
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    warned = sum(1 for _, s, _ in results if s == WARN)

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Warnings: {warned}")
    print()

    if failed > 0:
        print("  ❌ FAILED — Fix the issues above before submitting.")
        print()
        print("  Failed checks:")
        for check, status, detail in results:
            if status == FAIL:
                print(f"    • {check}: {detail}")
        print()
        sys.exit(1)
    else:
        print("  ✅ ALL CHECKS PASSED — Ready for submission!")
        print()
        if warned > 0:
            print("  Warnings (non-blocking):")
            for check, status, detail in results:
                if status == WARN:
                    print(f"    • {check}: {detail}")
            print()
        sys.exit(0)


if __name__ == "__main__":
    main()
