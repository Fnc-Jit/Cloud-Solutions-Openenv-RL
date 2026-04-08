---
title: CloudFinOps Env
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ☁️ CloudFinOps-Env

> **An RL environment combining cloud cost-optimization, SLA incident management, and carbon emissions tracking (GreenOps).**

[![Validate](https://github.com/Fnc-Jit/Cloud-Solutions_Re/actions/workflows/validate.yml/badge.svg)](https://github.com/Fnc-Jit/Cloud-Solutions_Re/actions/workflows/validate.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green.svg)](https://huggingface.co/openenv)
[![Tests](https://img.shields.io/badge/tests-84%20passed-brightgreen.svg)]()

![CloudFinOps Dashboard](assets/dashboard.png)
![CloudFinOps Dashboard Details](assets/dashboard_details.png)

Built for the **Meta AI × Hugging Face OpenEnv Hackathon**. Agents manage a fleet of AWS-style servers, balancing cost, performance, carbon emissions, and stakeholder communication through a REST API.

## Environment Description

CloudFinOps-Env simulates day-to-day cloud operations where an agent must optimize spend, preserve reliability, and reduce emissions under realistic constraints.

## Motivation

Real cloud teams constantly trade off cost, SLA risk, and sustainability. This environment is designed to benchmark whether an agent can make practical infrastructure decisions that humans actually make in operations, SRE, and FinOps workflows.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CloudFinOps-Env                           │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ models.py│──▶│  engine +    │◀──│    server/app.py   │  │
│  │ Pydantic │   │ environment  │   │ FastAPI + OpenEnv   │  │
│  │ Schemas  │   │  .py         │   │ + Dashboard        │  │
│  └──────────┘   └──────────────┘   └────────────────────┘  │
│                        ▲                    ▲               │
│                        │                    │               │
│                   ┌────┴────┐         ┌─────┴─────┐        │
│                   │  Tests  │         │inference.py│        │
│                   │ 84 unit │         │ LLM Agent  │        │
│                   │  tests  │         │ Evaluator  │        │
│                   └─────────┘         └───────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏢 **AWS Instance Catalog** | 10 realistic instance types (`t3.micro` → `m5.xlarge`) with real-world pricing |
| 📊 **Trailing Metrics History** | `cpu_history` / `memory_history` — last 3 steps per server for LLM trend detection |
| 🌍 **GreenOps Carbon Tracking** | Per-instance `carbon_kwh` emissions, ARM (r6g) vs x86 (c5/m5) efficiency modeling |
| 🎯 **4 Difficulty Tiers** | Easy → Medium → Hard → Green, each with unique objectives and grading |
| 📬 **Human-in-the-Loop** | Inbox messages from stakeholders; replying earns bonus points |
| ⏱️ **Delayed Scaling** | UPSCALE queues for next step — agents must plan ahead |
| 🔒 **Deterministic Noise** | Hash-seeded metric jitter — fully reproducible episodes |
| 📈 **Live Dashboard** | Real-time glassmorphism web UI at `/dashboard` with sparklines |
| 🧠 **LLM Conversational Memory** | Multi-turn conversation history so the agent remembers previous actions and avoids repeating mistakes |
| 🧪 **84 Unit Tests** | Comprehensive pytest suite with 20 test classes + GitHub Actions CI |

---

## Task Descriptions with Expected Difficulty

### 🟢 Easy — "Zombie Cleanup"
Terminate 3 idle servers (`idle-0`, `idle-1`, `idle-2`) without touching active ones.  
**Budget:** $5.00 | **Servers:** 10 | **Grading:** +1/3 per zombie killed, -0.25 per wrongful termination.

### 🟡 Medium — "CTO Budget Squeeze"
Cut cloud costs by ≥50% across 12 over-provisioned servers.  
**Budget:** $10.00 | **Servers:** 12 | **Grading:** Proportional to `cost_saved_pct / 50%`.

### 🔴 Hard — "Black Friday Chaos"
Handle a traffic spike with exponential ramp. Keep DB servers alive while managing budget.  
**Budget:** $4.00 | **Servers:** 8 | **Grading:** Uptime (60%) + Cost Efficiency (40%) + Inbox Bonus.

### 🌍 Green — "The Green Initiative"
Reduce carbon emissions by 40% by migrating workloads from dirty x86 instances (c5, m5) to efficient ARM Graviton (r6g).  
**Budget:** $8.00 | **Servers:** 10 | **Grading:** Carbon Reduction (50%) + Uptime (30%) + Cost (10%) + Inbox (10%).

---

## 📊 Carbon Intensity per Instance Type

| Instance | Architecture | Carbon (kWh/step) | Category |
|----------|-------------|-------------------|----------|
| `t3.micro` | x86 | 0.005 | Web |
| `t3.medium` | x86 | 0.012 | Web |
| `t3.large` | x86 | 0.022 | Web |
| `c5.large` | x86 | **0.035** | Compute |
| `c5.xlarge` | x86 | **0.065** | Compute |
| `r6g.medium` | ARM Graviton | 0.008 | DB |
| `r6g.large` | ARM Graviton | 0.015 | DB |
| `r6g.xlarge` | ARM Graviton | 0.028 | DB |
| `m5.large` | x86 | **0.040** | Batch |
| `m5.xlarge` | x86 | **0.075** | Batch |

> ARM Graviton instances produce **2–3× less carbon** than equivalent x86 instances.

---

## 📈 Trailing Metrics History

Each server's observation includes the **last 3 steps** of CPU and memory utilisation:

```json
{
  "id": "web-0",
  "type": "t3.medium",
  "cpu_util": 85.0,
  "cpu_history": [60.2, 72.5, 85.0],
  "memory_util": 45.0,
  "memory_history": [38.1, 41.7, 45.0]
}
```

This lets LLM agents **detect trends** (e.g., "CPU rising 3 steps in a row → act preemptively") without needing explicit memory systems.

---

## Setup and Usage Instructions

Before submitting, ensure the following **3 mandatory environment variables** are defined:

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | **Yes** | The API endpoint for the LLM (OpenAI-compatible) |
| `MODEL_NAME` | **Yes** | The model identifier to use for inference |
| `HF_TOKEN` | **Yes** | Your Hugging Face / API key |

These are the **only** variables the automated evaluator checks. To test with different providers, simply point these at the desired endpoint:

### 1. Clone
```bash
git clone https://github.com/Fnc-Jit/Cloud-Solutions_Re.git
cd cloudfinops_env
```

### 2. Build Docker Image
```bash
docker build -t cloudfinops-env:latest -f server/Dockerfile .
```

### 3. Start the Environment Server
```bash
docker run -p 8000:8000 cloudfinops-env:latest
```

### 4. Open the Live Dashboard
Open `http://localhost:8000/dashboard` in your browser.

### 5. Run the Agent Evaluator
```bash
docker run \
  -e ENV_BASE_URL=http://host.docker.internal:8000 \
  cloudfinops-env:latest python3 /app/env/inference.py
```

> Defaults for `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are already set in `inference.py`. Override via `-e` flags if needed.

### 6. Run Tests
```bash
docker run --rm cloudfinops-env:latest python3 -m pytest tests/ -v
```

### Alternative: Running Locally with uv

```bash
# Install dependencies
uv sync

# Start the server
uv run server

# Or with uvicorn directly
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Alternative: Running Locally (Direct)

```bash
# Defaults are set in inference.py — just run:
python inference.py

# To override any variable:
# export API_BASE_URL=https://router.huggingface.co/v1
# export MODEL_NAME=openai/gpt-4o
# export HF_TOKEN=hf_your_token_here
# python inference.py
```

### Alternative: pip + OpenEnv Core Scaffold

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "openenv-core[core]>=0.2.2" pytest
pip install -e .
```

### Alternative: Using the Python SDK

```python
from cloudfinops_env import CloudFinOpsAction, CloudFinOpsEnv

# Connect to a running server
with CloudFinOpsEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Budget: {result.observation.budget_remaining}")

    result = env.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
    print(f"Reward: {result.reward}")
```

---

## Deploying to Hugging Face Spaces

```bash
# From the environment directory
openenv push

# Or specify options
openenv push --repo-id your-username/cloudfinops-env --private
```

---

## 🖥️ API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (OpenEnv standard) |
| `POST` | `/reset` | Reset environment for a task (`{"task_id": "easy"}`) |
| `POST` | `/step` | Submit an action and advance the engine |
| `GET` | `/state` | Current observation (read-only, no side effects) |
| `GET` | `/schema` | Action/Observation JSON schemas (OpenEnv standard) |
| `WS` | `/ws` | WebSocket persistent session (OpenEnv standard) |
| `GET` | `/dashboard` | Real-time glassmorphism web dashboard |
| `GET` | `/history` | Agent action history for current episode |

---

## 🎮 Action Space

| Command | Effect | Reward |
|---------|--------|--------|
| `TERMINATE` | Kill a server immediately | +10 |
| `UPSCALE` | Queue upgrade (applies next step) | -5 |
| `DOWNSCALE` | Halve cost, but CPU load × 1.8 | +5 |
| `REDISTRIBUTE_LOAD` | Spread CPU evenly across fleet | +3 |
| `IGNORE` | Do nothing this step | 0 |

**Penalties:**
- Invalid target: **-2**
- SLA breach (CPU ≥ 100%): **-100** + episode ends
- Budget overrun: **-20** + episode ends
- High ongoing cost (>$0.50/step): **-1** per step

---

## 📊 Observation Space

```json
{
  "servers": [...],
  "traffic_load": 30.0,
  "spike_detected": false,
  "incidents": [],
  "budget_remaining": 5.0,
  "time_step": 0,
  "inbox": ["Ops Team: ..."],
  "carbon_kwh": 0.0
}
```

Each server includes:
- `id`, `type`, `cpu_util`, `memory_util`, `cost_per_hour`, `status`
- `cpu_history`: last 3 CPU values
- `memory_history`: last 3 memory values

---

## 🏆 Baseline Scores

The enclosed baseline evaluator (`inference.py`) establishes the reference performance for agents.

> **Score Range:** All task scores are strictly within the open interval **(0, 1)** — exclusive of both 0.0 and 1.0. The grading engine enforces this constraint via epsilon clamping (`0.001 ≤ score ≤ 0.999`).

![Baseline Evaluation Results](assets/baseline_scores.png)

| Task | Difficulty | Baseline Score (Llama 4 Scout 17B) | Success Status |
|------|------------|--------------------------------|----------------|
| `easy` | Easy | 0.9990 | ✅ |
| `medium` | Medium | 0.4954 | ✅ |
| `hard` | Hard | 0.7000 | ✅ |
| `green` | Green | 0.9499 | ✅ |
| **Average** | — | **0.7861** | ✅ |

> **Note:** Run the evaluator yourself using the setup and usage instructions above to see the exact real-time scores for your chosen LLM.

## Documentation Coverage

This README explicitly includes all required sections:

1. Environment description and motivation
2. Action and observation space definitions
3. Task descriptions with expected difficulty
4. Setup and usage instructions
5. Baseline scores

---

## 🧪 Testing

The project includes **84 unit tests** across 20 test classes:

| Test Class | Tests | What it covers |
|-----------|-------|----------------|
| `TestReset` | 6 | Clean state, all 4 tasks, invalid task handling |
| `TestDeterministicNoise` | 3 | Reproducibility, seed isolation, amplitude bounds |
| `TestActions` | 9 | TERMINATE, UPSCALE, DOWNSCALE, REDISTRIBUTE, IGNORE, inbox |
| `TestSLABreach` | 1 | Breach detection, episode termination |
| `TestGrading` | 4 | All 4 graders, score ranges, carbon reduction scoring |
| `TestCarbonTracking` | 4 | Accumulation, reduction after terminate, catalog coverage |
| `TestTrailingHistory` | 3 | Initial values, growth, max depth enforcement |
| `TestEpisodeBoundaries` | 3 | Max steps, budget overrun, post-done behavior |
| `TestClamp` | 4 | Utility function edge cases |
| `TestCascadingFailures` | 3 | Load redistribution after terminates → SLA breach risk |
| `TestBudgetEdgeCases` | 4 | Zero budget, negative budget, burn rate reduction |
| `TestInvalidActions` | 6 | Double-terminate, ghost servers, max tier, null target |
| `TestDownscaleSafety` | 3 | 1.8× CPU multiplier, cost halving, breach risk |
| `TestGreenOpsEdgeCases` | 4 | Carbon rate tracking, ARM vs x86, strategy comparison |
| `TestInboxMechanics` | 6 | Reply bonus, inbox clearing, empty reply, message content |
| `TestOptimalPlaySequences` | 4 | Known-optimal action sequences for each task type |
| `TestDeterministicReplay` | 3 | Same actions → same scores, observation reproducibility |
| `TestGraderBoundaries` | 5 | Score clamping, zero-score conditions, baseline scores |
| `TestUpscaleQueueTiming` | 4 | Queued upscale, next-step application, terminated server |
| `TestTrafficSimulation` | 4 | Logarithmic growth (hard), stable traffic, spike flags |

Run via Docker:
```bash
docker run --rm cloudfinops-env:latest python3 -m pytest tests/ -v --tb=short
```

Or locally:
```bash
python3 -m pytest tests/ -v --tb=short
```

---

## 🌐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | **Yes** | The API endpoint for the LLM (OpenAI-compatible) |
| `MODEL_NAME` | **Yes** | The model identifier to use for inference |
| `HF_TOKEN` | **Yes** | Your Hugging Face / API key |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:8000`) |

---

## 🔄 CI/CD

GitHub Actions runs automatically on every push/PR:
1. **Unit Tests** — `pytest tests/ -v`
2. **Syntax Check** — AST parse all Python files
3. **OpenEnv Spec** — Verify `openenv.yaml` has ≥3 tasks
4. **Docker Build** — Full image build + container smoke test

## ✅ Hackathon Submission Checklist

Before final submission, verify all of the following:

1. `openenv.yaml` defines spec metadata, env vars, and 3+ tasks.
2. `inference.py` is at repository root and uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
3. Inference stdout emits only required protocol lines:
  - `[START] task=<task_name> env=<benchmark> model=<model_name>`
  - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
  - `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`
4. `docker build` and `docker run` start successfully and endpoints respond.
5. Space root URL returns HTTP 200 and `/reset` responds successfully.

---

## 🏅 Hackathon Evaluation Phases

### Phase 1: Automated Validation *(Pass/Fail Gate)*
HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

### Phase 2: Agentic Evaluation *(Scored)*
Baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

### Phase 3: Human Review
Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

---

## 🏆 Key Design Decisions

1. **Deterministic Noise** — Hash-seeded jitter ensures reproducible episodes while maintaining realistic metric variation.
2. **Delayed Scaling** — UPSCALE takes effect next step, forcing agents to plan ahead (not just react).
3. **Carbon Emissions** — Models real-world ARM vs x86 efficiency gap, rewarding sustainable infrastructure.
4. **Trailing Metrics** — Designed for LLM agents with limited context memory — trend detection without explicit memory.
5. **Human-in-the-Loop** — Inbox/reply system tests whether agents can communicate with humans while managing infra.
6. **Upscale Tier Path** — Enforces realistic upgrade constraints (`t3.micro` → `t3.medium` → `t3.large`, max 2 upgrades).
7. **LLM Conversational Memory** — The inference agent maintains multi-turn conversation history across all steps within an episode. Each observation sent to the LLM includes: (a) a structured summary of running vs terminated servers, (b) a log of all previous actions and their rewards, and (c) explicit warnings about already-terminated servers. This prevents the most common LLM failure mode — repeatedly targeting the same server — and enables the agent to reason about its own action history for better sequential decision-making.
8. **Strict Score Range** — All grader scores are clamped to the open interval (0, 1) via epsilon bounds (`0.001 ≤ score ≤ 0.999`), ensuring compliance with hackathon validation requirements.

---

## 📁 Project Structure

```
cloudfinops_env/
├── __init__.py              # Module exports (CloudFinOpsAction, CloudFinOpsObservation, CloudFinOpsEnv)
├── models.py                # Pydantic schemas inheriting from openenv base types
├── client.py                # CloudFinOpsEnv(EnvClient) — Python SDK client
├── openenv.yaml             # OpenEnv manifest (spec_version: 1)
├── pyproject.toml           # Project metadata, deps, entry point
├── README.md                # This file
├── inference.py             # LLM baseline evaluator
├── .env.example             # Reference template for environment variables
├── .gitignore
├── server/
│   ├── __init__.py          # Server module exports
│   ├── cloudfinops_env_environment.py  # Physics engine + OpenEnv Environment wrapper
│   ├── app.py               # FastAPI app via openenv create_app()
│   ├── dashboard.html       # Real-time glassmorphism web dashboard
│   ├── Dockerfile           # Multi-stage build using openenv-base
│   └── requirements.txt     # Server-specific deps
├── tests/
│   ├── __init__.py
│   └── test_engine.py       # 84 pytest unit tests
└── assets/
    ├── dashboard.png
    └── dashboard_details.png
```

---

## 📜 License

MIT License — Built with ❤️ By Jitraj for the Meta AI × Hugging Face OpenEnv Hackathon 2025.
