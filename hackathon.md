# OpenEnv Hackathon — Complete Rules & Reference Guide

> **Target audience:** AI agents and participants. This document consolidates every rule, requirement, checklist, and best-practice from the official hackathon portal.

---

## Timeline

| Phase | Dates |
|---|---|
| Registration | 14 March – 3 April |
| Declaration (Solo/Team) | Before Round 1 |
| Preparation | Now – 25 March |
| Round 1 | 25 March – 8 April |
| Results | 10 April |
| Finale | 25–26 April |

**Round 1 submission deadline: 8 April 2026, 11:59 PM IST**

---

## Team & Solo Rules

### How to Declare

- Before starting the assessment, each participant must declare whether they are competing **solo** or as part of a **team**.
- This choice is made in Step 1 of the platform flow.

### Solo

- Compete individually — work and submit entirely on your own.
- Solo status is locked for Round 1 only.

### Team

- Teams consist of **2–3 members**.
- **Only one person (the team lead)** fills out the team form.
- Teammates are added by their email address.
- Each teammate must have their own registered account (registered with their email first).
- If a teammate has already added you to their team, your screen updates automatically — you do not need to do anything.

> ⚠️ **Once confirmed, teams cannot be changed.**

### Submission

- **Only the team leader can make the final submission.**

---

## Round 1 — Problem Statement

### The Task

Build a complete, real-world **OpenEnv environment** that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

### Key Requirements at a Glance

- Must simulate a **real-world task** (not games or toys)
- Implement the **full OpenEnv spec**: typed models, `step()` / `reset()` / `state()`, `openenv.yaml`
- **Minimum 3 tasks** with agent graders (easy → medium → hard), with scores/rewards in the `0.0–1.0` range
- **Meaningful reward function** with partial progress signals
- **Baseline inference script** with reproducible scores
- **Deploy to Hugging Face Spaces** with a working `Dockerfile`
- **README** with environment description, action/observation spaces, and setup instructions

---

## Pre-Submission Checklist

> All items must pass or the submission is **disqualified**.

| Check | Requirement |
|---|---|
| HF Space deploys | Automated ping to the Space URL — must return `200` and respond to `reset()` |
| OpenEnv spec compliance | Validate `openenv.yaml`, typed models, `step()` / `reset()` / `state()` endpoints |
| Dockerfile builds | Automated `docker build` on the submitted repo must succeed |
| Baseline reproduces | Run the submitted inference script — must complete without error and produce scores |
| 3+ tasks with graders | Enumerate tasks, run each grader, verify scores/reward in `0.0–1.0` range |

---

## Mandatory Additional Instructions

### Required Environment Variables

Before submitting, ensure the following variables are defined in your environment configuration:

| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

### Inference Script Rules

- The inference script **must be named `inference.py`** and placed in the **root directory** of the project.
- Participants **must use the OpenAI Client** for all LLM calls, using the variables above.
- Participants **must emit structured stdout logs** strictly following the `[START]`, `[STEP]`, and `[END]` format defined in the sample `inference.py`.
  - Any deviation in field names, ordering, or formatting will result in **incorrect evaluation scoring**.
  - Refer to the Sample Inference Script for the complete format specification and examples.

### Infrastructure Restrictions

- Runtime of the inference script must be **less than 20 minutes**.
- The environment and inference must run on a machine with **vCPU = 2, memory = 8 GB**.

---

## Evaluation Criteria

| Criterion | Description |
|---|---|
| Runtime correctness | Runs without errors |
| Interface compliance | Follows OpenEnv standard |
| Task design | Clear, realistic, testable |
| Grading logic | Reward system makes sense |

**Advancement:** 20,000 → 3,000 teams advance to the next round.

---

## How to Submit (Step-by-Step)

1. **Application Form** — Choose 1 of the 4–5 problem statements revealed on the platform.
2. **Scaffold** — Run the scaffold command to generate the project structure.
3. **Build** — Define your environment in the generated files.
4. **Test locally** — Run local tests to verify correctness.
5. **Deploy** — Deploy to Hugging Face Spaces.
6. **Submit** — Paste your HF Spaces URL on the platform before the deadline.

> **Deadline: 8 April 2026, 11:59 PM IST**

---

## Prerequisites

Install everything before **1 April**.

### Required

| Tool | Notes |
|---|---|
| Python 3.10+ | Install 3.10, 3.11, or 3.12 |
| Git + GitHub account | For pushing submissions to GitHub or HF |
| Hugging Face CLI | For deploying to HF Spaces |
| OpenEnv | The core framework |
| Google Colab | Prep course runs in Colab; free tier works |
| Docker | For isolated container testing |

### Recommended

| Tool | Notes |
|---|---|
| VS Code | Best Python + Docker support |

---

## Preparatory Course

4 modules · ~3.5 hours total. Read each module's README first, then open the notebook in Colab. No local setup needed.

| Module | Priority | Duration |
|---|---|---|
| Module 1: Why OpenEnv? | Essential for Round 1 | 45 min |
| Module 2: Using Existing Environments | Essential for Round 1 | 50 min |
| Module 3: Deploying Environments | Essential for Round 1 | 45 min |
| Module 4: Building Your Own Environment | **Most Important for Round 1** | 60 min |

---

## Validator

Run the **pre-submission validation script** before submitting to catch disqualifying issues early.

---

## Community & Support

- **Discord** — All announcements, mentor access, and team matching happen here. Join via the platform.
- **Email** — [email protected]

---

## FAQs Summary

**How does the team/solo declaration work?**
Declare before starting the assessment. Solo locks you in for Round 1; teams are 2–3 members with one team lead.

**Who fills the team form?**
Only the team lead.

**What if someone already added me to their team?**
Your screen updates automatically — no action needed from you.

**Can I change my team or switch to solo after confirming?**
No. Teams cannot be changed once confirmed.

**Do I need to complete the prep course?**
It is strongly recommended — all four modules are marked Essential or Most Important for Round 1.

**Can I update my submission?**
The submission window opens on 28 March. You can update until the deadline of 8 April 11:59 PM.

**What framework must be used?**
OpenEnv. All environments must comply with its full specification.

**What happens after Round 1?**
Results are announced on 10 April. Top 3,000 teams advance to the Finale on 25–26 April.

**Where can I get help?**
Join the Discord community or email [email protected].
