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

[![Validate](https://github.com/Fnc-Jit/Cloud-Solutions_Openenv-RL/actions/workflows/validate.yml/badge.svg)](https://github.com/Fnc-Jit/Cloud-Solutions_Openenv-RL/actions/workflows/validate.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green.svg)](https://huggingface.co/openenv)
[![Tests](https://img.shields.io/badge/tests-40%2B%20passed-brightgreen.svg)]()

![CloudFinOps Dashboard](assets/dashboard.png)
![CloudFinOps Dashboard Details](assets/dashboard_details.png)

Built for the **Meta AI × Hugging Face OpenEnv Hackathon**. Agents manage a fleet of AWS-style servers, balancing cost, performance, carbon emissions, and stakeholder communication through a REST API and WebSocket interface.

## Environment Description

CloudFinOps-Env simulates day-to-day cloud operations where an agent must optimize spend, preserve reliability, and reduce emissions under realistic constraints. The environment exposes the standard OpenEnv `step()` / `reset()` / `state()` API over HTTP and WebSocket, making it consumable by any LLM agent or RL algorithm.

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
│  │ Schemas  │   │  .py         │   │ + Dashboard + WS    │  │
│  └──────────┘   └──────────────┘   └────────────────────┘  │
│                        ▲                    ▲               │
│                        │                    │               │
│                   ┌────┴────┐         ┌─────┴─────┐        │
│                   │  Tests  │         │inference.py│        │
│                   │ 40 unit │         │ LLM Agent  │        │
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
| 🎯 **5 Tasks with Graders** | Easy → Medium → Hard → Green → Expert, each with unique objectives and 0.0–1.0 scoring |
| 📬 **Human-in-the-Loop** | Inbox messages from stakeholders; early replies earn bonus points |
| ⏱️ **Delayed Scaling** | UPSCALE queues for next step — agents must plan ahead |
| 🔒 **Deterministic Noise** | Hash-seeded metric jitter — fully reproducible episodes |
| 📈 **Live Dashboard** | Real-time glassmorphism web UI at `/dashboard` with sparklines |
| 🔌 **WebSocket Support** | Persistent `/ws` session for low-latency reset/step/state without HTTP overhead |
| 🧪 **40+ Unit Tests** | Comprehensive pytest suite + GitHub Actions CI + pre-submission validator |

---

## Task Descriptions

### 🟢 Easy — "Zombie Cleanup"
Terminate 3 idle servers (`idle-0`, `idle-1`, `idle-2`) without touching active ones.  
**Budget:** $5.00 | **Servers:** 10 | **Grading:** +1/3 per zombie killed, -0.25 per wrongful termination.

### 🟡 Medium — "CTO Budget Squeeze"
Cut cloud costs by ≥50% across 12 over-provisioned servers.  
**Budget:** $10.00 | **Servers:** 12 | **Grading:** Proportional to `cost_saved_pct / 50%`.

### 🔴 Hard — "Black Friday Chaos"
Handle a traffic spike with exponential ramp. Keep DB servers alive while managing budget.  
**Budget:** $4.00 | **Servers:** 8 | **Grading:** Uptime (60%) + Cost Efficiency (40%) + Inbox Bonus (reply within first 3 steps).

### 🌍 Green — "The Green Initiative"
Reduce carbon emissions by 40% by migrating workloads from dirty x86 instances (c5, m5) to efficient ARM Graviton (r6g).  
**Budget:** $8.00 | **Servers:** 10 | **Grading:** Carbon Reduction (50%) + Uptime (30%) + Cost (10%) + Inbox (10%).

### ⚡ Expert — "Multi-Region Disaster Recovery"
eu-west-1 region is cascading into failure. Terminate zombies, shut down failing EU servers, and preserve healthy regions (us-east-1, ap-south-1) under a tight budget.  
**Budget:** $6.00 | **Servers:** 14 across 3 regions | **Grading:** Zombies (20%) + EU handled (30%) + Healthy regions uptime (30%) + Inbox (20%).

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

### 1. Clone & Configure
```bash
git clone https://github.com/Fnc-Jit/Cloud-Solutions_Openenv-RL.git
cd Cloud-Solutions_Openenv-RL
cp .env.example .env
# Edit .env with your API keys
```

### 2. Build Docker Image

**Option A: Standalone (recommended — no external base image dependency)**
```bash
docker build -f server/Dockerfile.standalone -t cloudfinops-env:latest .
```

**Option B: OpenEnv base image**
```bash
docker build -f server/Dockerfile -t cloudfinops-env:latest .
```

### 3. Start the Environment Server
```bash
docker run --env-file .env -p 8000:8000 cloudfinops-env:latest
```

### 4. Open the Live Dashboard
Open `http://localhost:8000/dashboard` in your browser.

### 5. Run the Agent Evaluator
```bash
docker run --env-file .env -e ENV_BASE_URL=http://host.docker.internal:8000 cloudfinops-env:latest python3 inference.py
```

### 6. Run Tests
```bash
docker run --rm cloudfinops-env:latest python3 -m pytest tests/ -v
```

### 7. Pre-Submission Validation
```bash
python3 pre_validation.py           # All checks
python3 pre_validation.py --docker  # Includes Docker build test
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

### Alternative: Using Make

```bash
make test              # Run pytest suite
make validate          # Pre-submission checks
make docker            # Build Docker image
make run               # Start server locally
make push              # Deploy to HF Spaces
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
| `GET` | `/` | Root health probe |
| `GET` | `/health` | Health check (OpenEnv standard) |
| `POST` | `/reset` | Reset environment for a task (`{"task_id": "easy"}`) |
| `POST` | `/step` | Submit an action and advance the engine |
| `GET` | `/state` | Current observation (read-only, no side effects) |
| `GET` | `/schema` | Action/Observation JSON schemas (OpenEnv standard) |
| `WS` | `/ws` | WebSocket persistent session (reset/step/state) |
| `GET` | `/dashboard` | Real-time glassmorphism web dashboard |
| `GET` | `/history` | Agent action history for current episode |

---

## 🎮 Action Space

| Command | Effect | Reward |
|---------|--------|--------|
| `TERMINATE` | Kill a server immediately | +10 (or -2 if invalid/already terminated) |
| `UPSCALE` | Queue upgrade (applies next step, max 2 per server) | -5 (or -1 if maxed/no path) |
| `DOWNSCALE` | Halve cost, but CPU load ×1.8, memory ×1.3 | +5 |
| `REDISTRIBUTE_LOAD` | Spread CPU evenly across fleet | +3 |
| `IGNORE` | Do nothing this step | 0 |

**Reply Bonus:** Providing a non-empty `reply` when inbox has messages → **+2** and clears inbox. Early reply (within first 3 steps) also counts toward Hard/Expert grading.

**Penalties:**
- Invalid target or terminated server: **-2**
- SLA breach (CPU ≥ 100%): **-100** + episode ends immediately
- Budget overrun (≤ $0): **-20** + episode ends
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

## 🔌 WebSocket Protocol

The `/ws` endpoint accepts JSON messages for persistent session interaction:

**Client → Server:**
```json
{"type": "reset", "task_id": "easy"}
{"type": "step", "action": {"command": "TERMINATE", "target_id": "idle-0", "reply": ""}}
{"type": "state"}
```

**Server → Client:**
```json
{"type": "reset", "observation": {...}, "reward": null, "done": false}
{"type": "step", "observation": {...}, "reward": 10.0, "done": false, "info": {...}}
{"type": "state", "observation": {...}}
{"type": "error", "detail": "..."}
```

---

## 🏆 Baseline Scores

The enclosed baseline evaluator (`inference.py`) establishes the reference performance for agents.

| Task | Difficulty | Baseline Score (OpenAI GPT-4o) | Success Status |
|------|------------|--------------------------------|----------------|
| `easy` | Easy | 0.9500 | ✅ |
| `medium` | Medium | 0.8200 | ✅ |
| `hard` | Hard | 0.7600 | ✅ |
| `green` | Green | 0.8800 | ✅ |
| `expert` | Expert | 0.6500 | ✅ |

> **Note:** Run the evaluator yourself using the setup and usage instructions above to see the exact real-time scores for your chosen LLM.

## Documentation Coverage

This README includes all required sections per the OpenEnv Hackathon specification:

1. Environment description and motivation
2. Action and observation space definitions
3. Task descriptions with expected difficulty
4. Setup and usage instructions
5. Baseline scores

---

## 🧪 Testing

The project includes **40+ unit tests** across 10 test classes:

| Test Class | Tests | What it covers |
|-----------|-------|----------------|
| `TestReset` | 6 | Clean state, all tasks, invalid task handling |
| `TestDeterministicNoise` | 3 | Reproducibility, seed isolation, amplitude bounds |
| `TestActions` | 10 | TERMINATE, UPSCALE, DOWNSCALE, REDISTRIBUTE, IGNORE, inbox |
| `TestSLABreach` | 1 | Breach detection, episode termination |
| `TestGrading` | 4+ | All graders, score ranges, carbon reduction scoring |
| `TestCarbonTracking` | 4 | Accumulation, reduction after terminate, catalog coverage |
| `TestTrailingHistory` | 3 | Initial values, growth, max depth enforcement |
| `TestEpisodeBoundaries` | 3 | Max steps, budget overrun, post-done behavior |
| `TestClamp` | 4 | Utility function edge cases |
| `TestAPI` | 2 | Space ping, reset response validation |

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

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | The API endpoint for the LLM (OpenAI-compatible) |
| `MODEL_NAME` | Yes | `openai/gpt-4o` | The model identifier to use for inference |
| `HF_TOKEN` | Yes | — | Your Hugging Face / API key |
| `ENV_BASE_URL` | No | `http://localhost:8000` | Environment server URL |
| `STEP_DELAY_S` | No | `2.0` | Seconds to pause between LLM calls (rate limiting) |

> The inference script uses **only** the three mandatory variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) as required by the hackathon rules. Any OpenAI-compatible endpoint works — set `API_BASE_URL` to your provider's URL.

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
   - `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`
4. `docker build` and `docker run` start successfully and endpoints respond.
5. Space root URL returns HTTP 200 and `/reset` responds successfully.

Run `python3 pre_validation.py` to check all of these automatically.

---

## 🏆 Key Design Decisions

1. **Deterministic Noise** — Hash-seeded jitter ensures reproducible episodes while maintaining realistic metric variation.
2. **Delayed Scaling** — UPSCALE takes effect next step, forcing agents to plan ahead (not just react).
3. **Carbon Emissions** — Models real-world ARM vs x86 efficiency gap, rewarding sustainable infrastructure.
4. **Trailing Metrics** — Designed for LLM agents with limited context memory — trend detection without explicit memory.
5. **Human-in-the-Loop** — Inbox/reply system tests whether agents can communicate with humans while managing infra. Early replies are rewarded more heavily.
6. **Upscale Tier Path** — Enforces realistic upgrade constraints (`t3.micro` → `t3.medium` → `t3.large`, max 2 upgrades).
7. **Persistent Singleton** — Custom FastAPI app with lifespan-managed singleton environment (OpenEnv's `create_app()` factory is stateless, which breaks multi-step episodes).
8. **Standalone Dockerfile** — `Dockerfile.standalone` uses `python:3.11-slim` with no external base image dependency, guaranteeing the Docker build check passes in any CI environment.

---

## 📁 Project Structure

```
Cloud-Solutions_Openenv-RL/
├── __init__.py              # Module exports (CloudFinOpsAction, CloudFinOpsObservation, CloudFinOpsEnv)
├── models.py                # Pydantic schemas inheriting from openenv base types
├── client.py                # CloudFinOpsEnv(EnvClient) — Python SDK client
├── openenv.yaml             # OpenEnv manifest (spec_version: 1)
├── pyproject.toml           # Project metadata, deps, entry point
├── README.md                # This file
├── inference.py             # LLM baseline evaluator
├── pre_validation.py        # Pre-submission validator (run before submitting)
├── Makefile                 # Common operations: make test, make validate, make docker
├── .env.example             # Template environment variables
├── .gitignore
├── server/
│   ├── __init__.py          # Server module exports
│   ├── cloudfinops_env_environment.py  # Physics engine + OpenEnv Environment wrapper
│   ├── app.py               # FastAPI app with REST + WebSocket endpoints
│   ├── dashboard.html       # Real-time glassmorphism web dashboard
│   ├── Dockerfile           # Multi-stage build using openenv-base
│   ├── Dockerfile.standalone# Standalone build using python:3.11-slim
│   └── requirements.txt     # Server-specific deps
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Test path configuration
│   ├── test_engine.py       # 38 pytest unit tests for engine mechanics
│   └── test_api.py          # 2 pytest tests for API endpoints
└── assets/
    ├── dashboard.png
    └── dashboard_details.png
```

---

## 📜 License

MIT License — Built with ❤️ By Jitraj for the Meta AI × Hugging Face OpenEnv Hackathon 2026.
