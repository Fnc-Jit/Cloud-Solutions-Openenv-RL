"""Mandatory baseline evaluation script for the OpenEnv Hackathon.

Runs an LLM agent against the CloudFinOps environment through /reset and /step.
Uses the `openai` SDK and the following MANDATORY environment variables:

  API_BASE_URL  — The API endpoint for the LLM (e.g. https://api.groq.com/openai/v1)
  MODEL_NAME    — The model identifier to use for inference (e.g. llama-3.3-70b-versatile)
  HF_TOKEN      — Your Hugging Face / API key

Participants must use OpenAI Client for all LLM calls using above variables.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Auto-load .env file if present (judges can use .env.example as a template)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be exported in shell instead

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging as _logging

# Configure retry logger so before_sleep_log actually outputs
_retry_logger = _logging.getLogger("cloudfinops.retry")
_retry_logger.setLevel(_logging.WARNING)
if not _retry_logger.handlers:
    _handler = _logging.StreamHandler(sys.stderr)
    _handler.setFormatter(_logging.Formatter("  [RETRY] %(message)s"))
    _retry_logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Required environment variables — per hackathon rules
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "openai/gpt-4o")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
API_KEY: str = HF_TOKEN

# CloudFinOps environment URL (local Docker or HF Space)
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# Evaluation parameters — synced with engine.py MAX_STEPS
MAX_STEPS: int = 10
TASKS: List[str] = ["easy", "medium", "hard", "green", "expert"]
LLM_MAX_RETRIES: int = 3

def _validate_env() -> None:
    """Ensure the three mandatory credentials are set before proceeding."""
    if not API_KEY:
        print("\n  ❌ ERROR: HF_TOKEN is not set.")
        print("     Please set HF_TOKEN in your .env file or environment.")
        sys.exit(1)
    if not MODEL_NAME:
        print("\n  ❌ ERROR: MODEL_NAME is not set.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# OpenAI Client — mandatory per hackathon rules
# All LLM calls go through this client using the resolved provider vars.
# ---------------------------------------------------------------------------
_validate_env()

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# HTTP client for environment REST calls
http = httpx.Client(timeout=60.0)

# ---------------------------------------------------------------------------
# System Prompt — task-aware, structured for JSON output
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert cloud infrastructure engineer managing a fleet of AWS servers for cost optimisation.

You receive a JSON observation with:
- servers: list of objects — id, type, cpu_util, memory_util, cost_per_hour, status (running|terminated), cpu_history, memory_history
- traffic_load: global traffic % (0-100)
- spike_detected: boolean
- incidents: past SLA breaches
- budget_remaining: remaining USD
- time_step: current step (0-indexed)
- inbox: stakeholder messages
- carbon_kwh: cumulative carbon emissions

You MUST respond with ONLY a valid JSON object:
{"command": "TERMINATE"|"DOWNSCALE"|"UPSCALE"|"REDISTRIBUTE_LOAD"|"IGNORE", "target_id": "<server_id or null>", "reply": "<short reply to inbox or empty>"}

=== DECISION TREE — follow in order ===
1. If any server with id matching "idle-*" has status=running AND cpu_util=0 → TERMINATE it immediately.
2. If any server has status=running AND cpu_util >= 80 AND spike_detected=true → UPSCALE it.
3. If any server has status=running AND cpu_util <= 5 AND NOT an idle server → DOWNSCALE it.
4. If traffic load is uneven and no urgent action → REDISTRIBUTE_LOAD (target_id=null).
5. Only use IGNORE when ALL running servers are healthy and no idle servers remain.

=== RULES ===
- NEVER try to act on a server with status=terminated. Skip it.
- NEVER use IGNORE if there are servers with cpu_util=0 still running — terminate them.
- Only ONE action per step. Pick the most impactful.
- If inbox is not empty, always put a short reply in the reply field.
- For GreenOps tasks: prefer terminating c5.* and m5.* instances (dirty x86) over r6g.* (clean ARM).

Respond with ONLY the JSON. No explanation, no markdown, no extra text.
"""


# ---------------------------------------------------------------------------
# Spinner — shows a live animation while waiting for the LLM
# ---------------------------------------------------------------------------
@contextmanager
def _spinner(msg: str = "🤖 Asking LLM"):
    """Display an animated spinner in the terminal while the LLM is thinking."""
    stop_event = threading.Event()
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _spin():
        for frame in itertools.cycle(frames):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r  {msg} {frame} ")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(msg) + 10) + "\r")
        sys.stdout.flush()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_action(raw: str) -> Dict[str, Any]:
    """Extract a JSON action from LLM output, handling markdown fences."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"command": "IGNORE", "target_id": None, "reply": ""}


# How long to pause between steps — keeps us within Groq/HF rate limits
STEP_DELAY_S: float = float(os.environ.get("STEP_DELAY_S", "2.0"))


@retry(
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(_retry_logger, _logging.WARNING),
    reraise=True,
)
def _call_llm(obs_json: str, error_context: str = "") -> Dict[str, Any]:
    user_msg = f"Current observation:\n{obs_json}\n\nChoose your next action (respond with JSON only):"
    if error_context:
        user_msg += f"\n\nPREVIOUS ATTEMPT FAILED: {error_context}\nPlease fix and respond with valid JSON only."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as exc:
        # Detect 429 in the exception message and raise as a clean error
        # so that the caller can catch it cleanly without hanging in retry sleep
        msg = str(exc)
        if "429" in msg or "Too Many Requests" in msg or "rate_limit" in msg.lower():
            raise RuntimeError(f"Rate-limited (429): {msg.splitlines()[0]}") from exc
        raise

    raw_reply = completion.choices[0].message.content or ""
    action = parse_action(raw_reply)

    valid_commands = {"UPSCALE", "DOWNSCALE", "TERMINATE", "REDISTRIBUTE_LOAD", "IGNORE"}
    if action.get("command") not in valid_commands:
        raise ValueError(f"Invalid command '{action.get('command')}'")

    return action


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run_task(task_id: str) -> float:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Task: {task_id.upper()}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    log_start(task=task_id, env="cloudfinops-env", model=MODEL_NAME)

    resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()["observation"]

    rewards_list: List[float] = []
    steps_taken = 0
    score = 0.0
    done = False

    try:
        for step_num in range(1, MAX_STEPS + 1):
            obs_json = json.dumps(obs, indent=2)
            budget = obs.get("budget_remaining", 0)
            traffic = obs.get("traffic_load", 0)
            n_running = sum(1 for s in obs.get("servers", []) if s.get("status") == "running")
            print(f"\n--- Step {step_num}/{MAX_STEPS} ---", file=sys.stderr)
            print(f"  Budget: ${budget:.4f}  |  Traffic: {traffic}%  |  Running: {n_running} servers", file=sys.stderr)

            try:
                with _spinner("🤖 Asking LLM"):
                    action = _call_llm(obs_json)
                # Rate-limit guard: pause between steps to stay within quota
                if STEP_DELAY_S > 0:
                    time.sleep(STEP_DELAY_S)
            except Exception as exc:
                short = str(exc).splitlines()[0][:120]
                print(f"  [LLM Error after {LLM_MAX_RETRIES} retries] {short} - sending IGNORE", file=sys.stderr)
                action = {"command": "IGNORE", "target_id": None, "reply": ""}

            cmd = action.get("command", "IGNORE")
            target = action.get("target_id", "N/A")
            action_str = f"{cmd}({target})"
            reply_preview = (action.get("reply", "") or "")[:50]
            print(f"  Action: {cmd} -> {target}", file=sys.stderr)
            if reply_preview:
                print(f"  Reply:  \"{reply_preview}...\"", file=sys.stderr)

            error_msg = None
            try:
                resp = http.post(f"{ENV_BASE_URL}/step", json={"action": action})
                resp.raise_for_status()
                result = resp.json()
            except Exception as step_exc:
                print(f"  [Step Error] {step_exc} - sending IGNORE", file=sys.stderr)
                error_msg = str(step_exc)
                safe_action = {"command": "IGNORE", "target_id": None, "reply": ""}
                resp = http.post(f"{ENV_BASE_URL}/step", json={"action": safe_action})
                resp.raise_for_status()
                result = resp.json()

            obs = result["observation"]
            done = result["done"]
            reward = result["reward"]

            rewards_list.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)
            print(f"  Reward: {reward:+.1f}  |  Done: {done}", file=sys.stderr)

            if done:
                score = result.get("info", {}).get("grader_score", 0.0)
                print(f"\n  FINAL SCORE: {score:.4f}", file=sys.stderr)
                break

        if not done:
            print("\n  Max steps reached.", file=sys.stderr)

    finally:
        # ALWAYS emit [END] — even on crash (hackathon requirement)
        success = score > 0.0
        log_end(success=success, steps=steps_taken, rewards=rewards_list)

    return score


def main() -> None:
    start_time = time.time()

    masked_key = ('*' * 4 + API_KEY[-4:]) if len(API_KEY) > 4 else '****'

    print("=" * 60, file=sys.stderr)
    print("  CloudFinOps-Env Baseline Evaluator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Model:       {MODEL_NAME}", file=sys.stderr)
    print(f"  API:         {API_BASE_URL}", file=sys.stderr)
    print(f"  API Key:     {masked_key}", file=sys.stderr)
    print(f"  Env:         {ENV_BASE_URL}", file=sys.stderr)
    print(f"  Max Steps:   {MAX_STEPS}", file=sys.stderr)
    print(f"  LLM Retries: {LLM_MAX_RETRIES}", file=sys.stderr)
    print(file=sys.stderr)
    print("  Live Dashboard:", file=sys.stderr)
    print(f"  {ENV_BASE_URL}/dashboard", file=sys.stderr)

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as exc:
            print(f"  Task '{task_id}' failed: {exc}", file=sys.stderr)
            scores[task_id] = 0.0

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("  FINAL RESULTS", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for tid, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "PASS" if score > 0.0 else "FAIL"
        print(f"  {status:>4s} {tid:>8s}: {score:.4f}  {bar}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':>10s}: {avg:.4f}", file=sys.stderr)
    print(f"  {'TIME':>10s}: {elapsed:.1f}s", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    for tid, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Score for {tid} out of range: {score}"

    print("\n  All scores within valid 0.0-1.0 range.", file=sys.stderr)


if __name__ == "__main__":
    main()