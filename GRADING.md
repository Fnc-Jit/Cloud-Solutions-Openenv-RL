# Grading System

This document explains the multi-objective grading logic for each CloudFinOps task. All final scores are clamped to the `0.0–1.0` range.

---

## Easy — "Zombie Cleanup"

**Objective:** Terminate 3 idle servers (`idle-0`, `idle-1`, `idle-2`) without touching active ones.

| Component | Weight | Formula |
|-----------|--------|---------|
| Idle terminations | Primary | `terminated_idle / 3.0` |
| Wrongful terminations | Penalty | `active_terminated × 0.25` |
| SLA breach | Penalty | `-0.5` if any server hits 100% CPU |

```
score = (terminated_idle / 3.0) - (active_terminated × 0.25)
if sla_breached: score -= 0.5
score = clamp(score, 0.0, 1.0)
```

**Perfect strategy:** Terminate all 3 idle servers, leave everything else running.

---

## Medium — "CTO Budget Squeeze"

**Objective:** Cut cloud costs by ≥50% across 12 over-provisioned servers.

| Component | Weight | Formula |
|-----------|--------|---------|
| Cost savings efficiency | Primary | `min(cost_saved_pct / 0.50, 1.0)` |
| SLA breach | Penalty | `-0.5` if any server hits 100% CPU |

```
cost_saved_pct = 1.0 - (total_cost_spent / initial_budget)
efficiency = min(cost_saved_pct / 0.50, 1.0)
score = clamp(efficiency - crash_penalty, 0.0, 1.0)
```

**Perfect strategy:** Downscale or terminate enough servers to spend less than 50% of the $10 budget without causing SLA breaches.

---

## Hard — "Black Friday Chaos"

**Objective:** Handle a traffic spike with exponential CPU ramp. Keep DB servers alive while managing a tight $4 budget.

| Component | Weight | Formula |
|-----------|--------|---------|
| Uptime (no SLA breach) | 60% | `1.0` if no breach, `0.0` otherwise |
| Cost efficiency | 30% | `clamp(cost_saved_pct, 0.0, 1.0)` |
| Inbox cleared | 10% | `0.1` if all stakeholder messages replied to |
| Fleet survival | 10% | `0.1` if ≥6 servers still running at end |

```
score = (uptime × 0.6) + (cost_efficiency × 0.3) + inbox_bonus + survival_bonus
score = clamp(score, 0.0, 1.0)
```

**Perfect strategy:** Upscale DB servers preemptively, reply to inbox messages, keep at least 6 servers running, and stay within budget.

---

## Green — "The Green Initiative"

**Objective:** Reduce carbon emissions by 40% by migrating workloads from dirty x86 (c5, m5) to efficient ARM Graviton (r6g) instances.

| Component | Weight | Formula |
|-----------|--------|---------|
| Carbon reduction | 50% | `min(reduction_pct / 0.40, 1.0)` |
| Uptime (no SLA breach) | 30% | `1.0` if no breach, `0.0` otherwise |
| Cost efficiency | 10% | `clamp(cost_saved_pct, 0.0, 1.0)` |
| Inbox cleared | 10% | `0.1` if all stakeholder messages replied to |

```
reduction_pct = 1.0 - (current_carbon_rate / initial_carbon_rate)
carbon_score = min(reduction_pct / 0.40, 1.0)
score = (carbon_score × 0.5) + (uptime × 0.3) + (cost_efficiency × 0.1) + inbox_bonus
score = clamp(score, 0.0, 1.0)
```

**Perfect strategy:** Terminate or downscale x86 instances (c5, m5) while keeping ARM instances (r6g) running, reply to inbox messages, and avoid SLA breaches.

---

## Carbon Intensity Reference

| Instance | Architecture | Carbon (kWh/step) | Relative Impact |
|----------|-------------|-------------------|-----------------|
| `t3.micro` | x86 | 0.005 | 1× |
| `t3.medium` | x86 | 0.012 | 2.4× |
| `t3.large` | x86 | 0.022 | 4.4× |
| `c5.large` | x86 | 0.035 | 7× |
| `c5.xlarge` | x86 | 0.065 | 13× |
| `r6g.medium` | ARM Graviton | 0.008 | 1.6× |
| `r6g.large` | ARM Graviton | 0.015 | 3× |
| `r6g.xlarge` | ARM Graviton | 0.028 | 5.6× |
| `m5.large` | x86 | 0.040 | 8× |
| `m5.xlarge` | x86 | 0.075 | 15× |

ARM Graviton instances produce **2–3× less carbon** than equivalent x86 instances at the same tier.

---

## Reward Signals (Per-Step)

These are intermediate rewards returned by `step()` to guide RL training. They are **not** the final grade.

| Signal | Value | Trigger |
|--------|-------|---------|
| TERMINATE | +10 | Server terminated |
| DOWNSCALE | +5 | Server downscaled |
| REDISTRIBUTE_LOAD | +3 | Load redistributed evenly |
| Inbox reply | +2 | Reply sent when inbox not empty |
| UPSCALE | -5 | Server upgrade queued |
| Invalid target | -2 | Action on nonexistent/terminated server |
| High cost/step | -1 | Total running cost > $0.50/step |
| Budget overrun | -20 | Budget goes negative |
| SLA breach | -100 | Any server CPU ≥ 100% |

The final `grader_score` in the `info` dict is only set when `done=True`.
