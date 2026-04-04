"""Pre-submission validation script for the OpenEnv Hackathon.

Runs all disqualifying checks locally before submitting to the platform.
Exit code 0 = all checks passed. Exit code 1 = one or more failures.

Usage:
    python pre_validation.py
    python pre_validation.py --docker   # include Docker build test
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results: List[Tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    return condition


def warn(name: str, detail: str) -> None:
    results.append((name, WARN, detail))


# ── 1. openenv.yaml ──────────────────────────────────────────────────────────

def validate_openenv_yaml() -> None:
    yaml_path = ROOT / "openenv.yaml"
    if not yaml_path.exists():
        check("openenv.yaml exists", False, "File not found")
        return
    check("openenv.yaml exists", True)

    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as e:
        check("openenv.yaml parses", False, str(e))
        return
    check("openenv.yaml parses", True)

    check("spec_version present", "spec_version" in data, f"Got keys: {list(data.keys())}")
    check("name present", bool(data.get("name")))
    check("description present", bool(data.get("description")), "Add a description field")

    tasks = data.get("tasks", [])
    check("≥3 tasks defined", len(tasks) >= 3, f"Found {len(tasks)} tasks")

    required_env_vars = {"API_BASE_URL", "MODEL_NAME", "HF_TOKEN"}
    defined_vars = {e["name"] for e in data.get("env", [])}
    missing = required_env_vars - defined_vars
    check("required env vars defined", not missing, f"Missing: {missing}")


# ── 2. inference.py at root ──────────────────────────────────────────────────

def validate_inference_py() -> None:
    inf = ROOT / "inference.py"
    check("inference.py at root", inf.exists())
    if not inf.exists():
        return

    source = inf.read_text()
    check("inference.py uses OpenAI client", "from openai import OpenAI" in source)
    check("inference.py references API_BASE_URL", "API_BASE_URL" in source)
    check("inference.py references MODEL_NAME", "MODEL_NAME" in source)
    check("inference.py references HF_TOKEN", "HF_TOKEN" in source)

    for tag in ["[START]", "[STEP]", "[END]"]:
        check(f"inference.py emits {tag}", tag in source, f"Missing {tag} log format")

    try:
        ast.parse(source)
        check("inference.py syntax", True)
    except SyntaxError as e:
        check("inference.py syntax", False, str(e))


# ── 3. Dockerfile exists ─────────────────────────────────────────────────────

def validate_dockerfile() -> None:
    df = ROOT / "server" / "Dockerfile"
    dfs = ROOT / "server" / "Dockerfile.standalone"
    check("Dockerfile exists", df.exists() or dfs.exists(),
          "Neither server/Dockerfile nor server/Dockerfile.standalone found")


# ── 4. Python syntax ─────────────────────────────────────────────────────────

def validate_syntax() -> None:
    python_files = [
        "server/cloudfinops_env_environment.py",
        "server/app.py",
        "models.py",
        "inference.py",
        "client.py",
        "__init__.py",
    ]
    for f in python_files:
        path = ROOT / f
        if not path.exists():
            check(f"Syntax: {f}", False, "File not found")
            continue
        try:
            ast.parse(path.read_text())
            check(f"Syntax: {f}", True)
        except SyntaxError as e:
            check(f"Syntax: {f}", False, str(e))


# ── 5. Engine smoke test ────────────────────────────────────────────────────

def validate_engine_smoke() -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from server.cloudfinops_env_environment import CloudFinOpsEngine
        from models import CloudFinOpsAction

        engine = CloudFinOpsEngine()

        for task in ["easy", "medium", "hard", "green"]:
            obs = engine.reset(task)
            check(f"Engine reset: {task}", True, f"{len(obs.servers)} servers")

            for _ in range(3):
                obs, reward, done, info = engine.step(CloudFinOpsAction(command="IGNORE"))

            score = engine.grade()
            check(f"Grader: {task} score in [0,1]", 0.0 <= score <= 1.0, f"score={score:.4f}")

    except Exception as e:
        check("Engine smoke test", False, str(e))


# ── 6. Docker build (optional, --docker flag) ────────────────────────────────

def validate_docker_build() -> None:
    dockerfile = ROOT / "server" / "Dockerfile.standalone"
    if not dockerfile.exists():
        dockerfile = ROOT / "server" / "Dockerfile"
    if not dockerfile.exists():
        warn("Docker build skipped", "No Dockerfile found")
        return

    try:
        result = subprocess.run(
            ["docker", "build", "-f", str(dockerfile), "-t", "cloudfinops-env-validate", str(ROOT)],
            capture_output=True, text=True, timeout=300,
        )
        check("Docker build succeeds", result.returncode == 0,
              result.stderr[-500:] if result.returncode != 0 else "")
    except FileNotFoundError:
        warn("Docker build skipped", "Docker not installed")
    except subprocess.TimeoutExpired:
        check("Docker build", False, "Timed out after 300s")


# ── 7. Required files ────────────────────────────────────────────────────────

def validate_required_files() -> None:
    for f in ["README.md", "openenv.yaml", "inference.py", "models.py", "pyproject.toml", ".env.example"]:
        check(f"Required file: {f}", (ROOT / f).exists())


# ── 8. README completeness ───────────────────────────────────────────────────

def validate_readme() -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        check("README.md exists", False)
        return
    check("README.md exists", True)

    content = readme.read_text().lower()
    check("README: environment description", "environment description" in content or "description" in content)
    check("README: action space", "action space" in content or "action" in content)
    check("README: observation space", "observation space" in content or "observation" in content)
    check("README: setup instructions", "setup" in content or "install" in content)
    check("README: baseline scores", "baseline" in content or "score" in content)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  CloudFinOps-Env Pre-Submission Validator")
    print("=" * 60)
    print()

    validate_openenv_yaml()
    validate_inference_py()
    validate_dockerfile()
    validate_syntax()
    validate_engine_smoke()
    validate_required_files()
    validate_readme()

    if "--docker" in sys.argv:
        validate_docker_build()

    print(f"{'Check':<45s} {'Status':<8s} Detail")
    print("-" * 80)
    for name, status, detail in results:
        symbol = "✅" if status == PASS else ("⚠️" if status == WARN else "❌")
        print(f"{symbol} {name:<42s} {status:<8s} {detail}")

    print()
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    warned = sum(1 for _, s, _ in results if s == WARN)
    total = len(results)

    print(f"  Results: {passed}/{total} passed, {failed} failed, {warned} warnings")
    print()

    if failed > 0:
        print("  ❌ FIX FAILURES BEFORE SUBMITTING — disqualification risk!")
        sys.exit(1)
    else:
        print("  ✅ All checks passed. Ready to submit!")
        sys.exit(0)


if __name__ == "__main__":
    main()
