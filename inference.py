"""Mandatory baseline evaluation script for the OpenEnv Hackathon.

Runs an LLM agent against the CloudFinOps environment through /reset and /step.
Uses the `openai` SDK and the following MANDATORY environment variables:

  API_BASE_URL  — The API endpoint for the LLM (OpenAI-compatible).
  MODEL_NAME    — The model identifier to use for inference.
  HF_TOKEN      — Your Hugging Face / API key.

Participants must use OpenAI Client for all LLM calls using above variables.

Quick-start (Hugging Face):

  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=openai/gpt-oss-120b
  export HF_TOKEN=hf_xxxxx
  python inference.py

To test with Groq (same 3 vars, just point at Groq):

  export API_BASE_URL=https://api.groq.com/openai/v1
  export MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
  export HF_TOKEN=gsk_xxxxx
  python inference.py
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

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging as _logging

# ---------------------------------------------------------------------------
# Mandatory environment variables (checked by automated LLM evaluator)
#
#   API_BASE_URL  — The API endpoint for the LLM (OpenAI-compatible).
#   MODEL_NAME    — The model identifier to use for inference.
#   HF_TOKEN      — Your Hugging Face / API key.
#
# These MUST be set before running. No hardcoded defaults.
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN: str     = os.environ.get("HF_TOKEN", "")

# CloudFinOps environment URL (local Docker or HF Space)
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# Evaluation parameters — synced with engine.py MAX_STEPS
MAX_STEPS: int = 10
TASKS: List[str] = ["easy", "medium", "hard", "green"]
LLM_MAX_RETRIES: int = 3

# Hackathon validator requires scores strictly in (0, 1) — not 0.0 or 1.0
_SCORE_EPS = 0.01
def _clamp_score(val: float) -> float:
    """Clamp to (0, 1) open interval. Use 0.01 margin so :.2f never rounds to 0.00 or 1.00."""
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, val))


def _validate_env() -> None:
    """Ensure the three mandatory environment variables are set."""
    if not API_BASE_URL:
        print("\n  ❌ ERROR: API_BASE_URL is not set.")
        print("     The API endpoint for the LLM (OpenAI-compatible).")
        print("     Example:  export API_BASE_URL=https://router.huggingface.co/v1")
        sys.exit(1)
    if not MODEL_NAME:
        print("\n  ❌ ERROR: MODEL_NAME is not set.")
        print("     The model identifier to use for inference.")
        print("     Example:  export MODEL_NAME=openai/gpt-4o")
        sys.exit(1)
    if not HF_TOKEN:
        print("\n  ❌ ERROR: HF_TOKEN is not set.")
        print("     Your Hugging Face / API key.")
        print("     Example:  export HF_TOKEN=hf_xxxxx")
        sys.exit(1)


# ---------------------------------------------------------------------------
# OpenAI Client — mandatory per hackathon rules
# All LLM calls go through this client using the three env vars above.
# ---------------------------------------------------------------------------
_validate_env()

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# HTTP client for environment REST calls
http = httpx.Client(timeout=60.0)

# ---------------------------------------------------------------------------
# System Prompt — task-aware, structured for JSON output
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior AWS Solutions Architect and FinOps specialist with deep expertise in cloud cost optimization, capacity planning, and sustainability (GreenOps). You manage a fleet of AWS EC2 instances and must make ONE optimal infrastructure decision per time-step.

═══════════════════════════════════════════════
  OBSERVATION SCHEMA (fields you will receive)
═══════════════════════════════════════════════

servers: list of objects, each with:
  - id           : unique identifier (e.g. "web-0", "idle-2", "compute-1")
  - type         : AWS instance type (t3.micro/medium/large, c5.large/xlarge, r6g.medium/large/xlarge, m5.large/xlarge)
  - cpu_util     : current CPU utilization % (0-100)
  - memory_util  : current memory utilization % (0-100)
  - cost_per_hour: hourly cost in USD
  - status       : "running" or "terminated"  ← CRITICAL: never act on terminated servers
  - cpu_history  : list of last 3 CPU readings (trend detection)
  - memory_history: list of last 3 memory readings

traffic_load   : global traffic percentage (0-100)
spike_detected : boolean — true = traffic surge in progress
incidents      : list of past SLA breaches (cpu >= 100 = breach)
budget_remaining: remaining USD for the episode
time_step      : current step (0-indexed, max 10 steps)
inbox          : stakeholder messages requiring acknowledgement
carbon_kwh     : cumulative carbon emissions in kWh

═══════════════════════════════════════════════
  RESPONSE FORMAT (strict JSON, nothing else)
═══════════════════════════════════════════════

{"command": "<COMMAND>", "target_id": "<server_id_or_null>", "reply": "<string>"}

Commands: TERMINATE | DOWNSCALE | UPSCALE | REDISTRIBUTE_LOAD | IGNORE

═══════════════════════════════════════════════
  ⚠️ ABSOLUTE RULES — VIOLATIONS = INSTANT PENALTY
═══════════════════════════════════════════════

1. **CHECK STATUS FIRST**: Before EVERY action, scan the servers list. If a server's status is "terminated", it is DEAD. Do NOT target it. Targeting a terminated server = -2.0 penalty.
2. **NEVER REPEAT A FAILED ACTION**: If you targeted a server in a previous step and got -2.0 reward, that server is terminated or invalid. Move on to the NEXT target.
3. **ONE action per step**. Choose the highest-priority action from the decision framework below.
4. **Only target servers with status="running"**. Filter out ALL terminated servers before choosing.

═══════════════════════════════════════════════
  PRIORITY DECISION FRAMEWORK (follow top→down)
═══════════════════════════════════════════════

P1 — ZOMBIE CLEANUP (highest priority)
  1. Look at ALL servers in the observation.
  2. Filter to ONLY servers where status="running" AND cpu_util == 0 AND id starts with "idle-".
  3. Pick the FIRST one from that filtered list.
  4. TERMINATE it.
  If no idle running servers remain, skip to P2.

P2 — SLA BREACH PREVENTION
  IF any server has status="running" AND cpu_util >= 80 AND spike_detected == true
  → UPSCALE the server with the HIGHEST cpu_util first.
  Rationale: CPU >= 100 causes SLA breach → episode terminates with massive penalty.
  IMPORTANT: Check cpu_history — if trend is rising (each value > previous), upscale even at cpu_util >= 70.

P3 — COST OPTIMIZATION (downscale underutilized)
  IF any server has status="running" AND cpu_util <= 5 AND id does NOT start with "idle-"
  → DOWNSCALE the server with the LOWEST cpu_util (halves cost, increases CPU by 1.8×).
  WARNING: After downscale, CPU jumps to ~1.8× original. Only downscale if cpu_util * 1.8 < 80.

P4 — CARBON REDUCTION (GreenOps scenarios)
  IF inbox mentions carbon/emissions/sustainability AND c5.* or m5.* servers are running
  → TERMINATE the highest-cost c5.* or m5.* instance (dirty x86).
  PRESERVE r6g.* instances (clean ARM Graviton — 3× lower emissions).
  Priority order for termination: m5.xlarge > c5.xlarge > m5.large > c5.large.

P5 — LOAD BALANCING
  IF traffic_load > 50 AND there is high variance in cpu_util across running servers
  → REDISTRIBUTE_LOAD (target_id=null). Equalizes CPU/memory across all running servers.

P6 — STRATEGIC IGNORE
  ONLY if ALL of these are true:
    • No running servers with cpu_util == 0
    • No servers approaching SLA breach (cpu_util < 75 or spike_detected == false)
    • No obviously over-provisioned servers (cpu_util > 5 for all)
    • Budget is healthy (budget_remaining > 1.0)
  → IGNORE

═══════════════════════════════════════════════
  TASK-SPECIFIC STRATEGIES
═══════════════════════════════════════════════

EASY ("Zombie Cleanup"):
  Goal: Terminate ALL 3 idle-* servers. That's it.
  Strategy: Step 1 → TERMINATE idle-0. Step 2 → TERMINATE idle-1. Step 3 → TERMINATE idle-2.
  NEVER terminate the same server twice. After terminating idle-0, its status becomes "terminated".
  Move to idle-1 next, then idle-2. Don't touch active web-* or compute-* servers.
  Perfect score requires: All 3 idle terminated + zero active servers terminated.

MEDIUM ("CTO Budget Squeeze"):
  Goal: Cut costs by 50%+. You have 12 over-provisioned servers.
  Strategy: DOWNSCALE the lowest-CPU servers first (they're all at 3-9% CPU).
  Then TERMINATE any that are still barely used. Watch SLA — downscale pushes CPU to 1.8×.
  Target DIFFERENT servers each step. Don't downscale the same server repeatedly.

HARD ("Black Friday Chaos"):
  Goal: Keep uptime (avoid SLA breaches) while managing costs under a spike.
  Strategy: UPSCALE the DB servers (r6g.*) immediately — they're under spike pressure.
  Then manage batch servers (low priority, safe to downscale/terminate).
  Traffic grows logarithmically each step — act fast on databases.

GREEN ("Green Initiative"):
  Goal: Reduce carbon emissions by 40%+. TERMINATE dirty x86 (c5/m5), keep ARM (r6g).
  Strategy: TERMINATE compute-* and batch-* instances (c5/m5 types) in order of highest cost.
  Order: batch-2 (m5.xlarge) → compute-2 (c5.xlarge) → batch-0 → batch-1 → compute-0 → compute-1.
  Keep idle-0 for last (low carbon). NEVER terminate arm-* (r6g) instances.
  NEVER downscale — downscaling doesn't reduce carbon, only TERMINATE removes emissions.

═══════════════════════════════════════════════
  CRITICAL RULES (violations = score penalties)
═══════════════════════════════════════════════

1. NEVER act on a server with status="terminated" → automatic -2.0 penalty.
2. NEVER use IGNORE if idle-* servers with cpu_util=0 are still running.
3. ONE action per step. Choose the highest-priority action from the framework.
4. If inbox is non-empty, ALWAYS include a short professional reply (earns +2.0 bonus).
5. Upscaling costs -5.0 reward but prevents SLA breach (-100.0). Always upscale if breach is imminent.
6. After terminating a server, its CPU load redistributes to remaining servers — plan for this.
7. Budget < 0 ends the episode with penalty. Track cost_per_hour × running_servers vs budget_remaining.
8. cpu_util >= 100 = SLA BREACH = episode over with -100 penalty. Prevent at all costs.
9. Downscale only safe targets: cpu_util * 1.8 must stay below 80 to avoid triggering breach cascade.
10. Each server can only be upscaled twice max. Don't waste upscales.
11. Upscale takes effect NEXT step (queued). Plan one step ahead.
12. REDISTRIBUTE_LOAD averages all CPU/memory — useful when some servers are hot and others cold.

Respond with ONLY valid JSON. No markdown fences, no explanation, no commentary.
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


# ---------------------------------------------------------------------------
# LLM call with 429-aware retry logic
# ---------------------------------------------------------------------------
import httpx as _httpx

# How long to pause between steps — keeps us within Groq/HF rate limits
STEP_DELAY_S: float = float(os.environ.get("STEP_DELAY_S", "2.0"))

_retry_logger = _logging.getLogger("cloudfinops.retry")


@retry(
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(_retry_logger, _logging.WARNING),
    reraise=True,
)
def _call_llm(messages: List[Dict[str, str]], error_context: str = "") -> Dict[str, Any]:
    """Call the LLM with full conversation history for better decision-making."""
    call_messages = list(messages)
    if error_context:
        call_messages.append({
            "role": "user",
            "content": f"PREVIOUS ATTEMPT FAILED: {error_context}\nPlease fix and respond with valid JSON only."
        })

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=call_messages,
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as exc:
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


def _build_obs_message(obs: Dict[str, Any], step_num: int, task_id: str, action_history: List[str]) -> str:
    """Build an observation message with context about server status and action history."""
    # Build a quick status summary to help the LLM
    running = [s for s in obs.get("servers", []) if s.get("status") == "running"]
    terminated = [s for s in obs.get("servers", []) if s.get("status") == "terminated"]

    parts = [f"TASK: {task_id.upper()} | Step {step_num}/{MAX_STEPS}"]

    if action_history:
        parts.append("\nYOUR PREVIOUS ACTIONS THIS EPISODE:")
        for ah in action_history:
            parts.append(f"  {ah}")

    if terminated:
        term_ids = [s.get("id") for s in terminated]
        parts.append(f"\n⚠️ ALREADY TERMINATED (do NOT target these): {term_ids}")

    parts.append(f"\nRUNNING SERVERS ({len(running)}):")
    for s in running:
        parts.append(f"  {s['id']}: type={s['type']} cpu={s.get('cpu_util', 0)}% mem={s.get('memory_util', 0)}% cost=${s.get('cost_per_hour', 0)}")

    parts.append(f"\nBudget: ${obs.get('budget_remaining', 0):.4f} | Traffic: {obs.get('traffic_load', 0)}% | Spike: {obs.get('spike_detected', False)}")
    parts.append(f"Carbon: {obs.get('carbon_kwh', 0):.4f} kWh")

    inbox = obs.get("inbox", [])
    if inbox:
        parts.append(f"\nINBOX ({len(inbox)} messages — reply to earn +2.0 bonus):")
        for msg in inbox:
            parts.append(f"  • {msg}")

    parts.append("\nChoose your next action (respond with JSON only):")
    return "\n".join(parts)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


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
    score = _SCORE_EPS
    done = False

    try:
        # Conversation history for multi-turn LLM calls
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        action_history: List[str] = []

        for step_num in range(1, MAX_STEPS + 1):
            budget = obs.get("budget_remaining", 0)
            traffic = obs.get("traffic_load", 0)
            n_running = sum(1 for s in obs.get("servers", []) if s.get("status") == "running")
            print(f"\n--- Step {step_num}/{MAX_STEPS} ---", file=sys.stderr)
            print(f"  Budget: ${budget:.4f}  |  Traffic: {traffic}%  |  Running: {n_running} servers", file=sys.stderr)

            # Build context-rich observation message
            user_msg = _build_obs_message(obs, step_num, task_id, action_history)
            messages.append({"role": "user", "content": user_msg})

            try:
                with _spinner("🤖 Asking LLM"):
                    action = _call_llm(messages)
                # Rate-limit guard: pause between steps to stay within quota
                if STEP_DELAY_S > 0:
                    time.sleep(STEP_DELAY_S)
            except Exception as exc:
                short = str(exc).splitlines()[0][:120]
                print(f"  [LLM Error after {LLM_MAX_RETRIES} retries] {short} - sending IGNORE", file=sys.stderr)
                action = {"command": "IGNORE", "target_id": None, "reply": ""}

            # Record the LLM's response in conversation history
            messages.append({"role": "assistant", "content": json.dumps(action)})

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

            # Track action history for context
            action_history.append(f"Step {step_num}: {action_str} → reward={reward:+.1f}")

            rewards_list.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)
            print(f"  Reward: {reward:+.1f}  |  Done: {done}", file=sys.stderr)

            if done:
                score = _clamp_score(result.get("info", {}).get("grader_score", _SCORE_EPS))
                print(f"\n  FINAL SCORE: {score:.4f}", file=sys.stderr)
                break

        if not done:
            print("\n  Max steps reached.", file=sys.stderr)

    finally:
        # ALWAYS emit [END] — even on crash (hackathon requirement)
        success = score > _SCORE_EPS
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)

    return score


def main() -> None:
    start_time = time.time()

    masked_key = ('*' * 4 + HF_TOKEN[-4:]) if len(HF_TOKEN) > 4 else '****'

    print("=" * 60, file=sys.stderr)
    print("  CloudFinOps-Env Baseline Evaluator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Model:       {MODEL_NAME}", file=sys.stderr)
    print(f"  API:         {API_BASE_URL}", file=sys.stderr)
    print(f"  HF_TOKEN:    {masked_key}", file=sys.stderr)
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
            scores[task_id] = _SCORE_EPS

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
        assert 0.0 < score < 1.0, f"Score for {tid} out of range: {score}"

    print("\n  All scores within valid (0, 1) range.", file=sys.stderr)


if __name__ == "__main__":
    main()