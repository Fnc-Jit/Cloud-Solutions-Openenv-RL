"""CloudFinOps Physics Simulator & Grading Engine.

Simulates a realistic cloud cost-optimization scenario using AWS-style
instance types, real-world pricing, stochastic metric noise, carbon
emissions tracking (GreenOps), trailing metrics history, and
multi-objective grading.

Wrapped in the OpenEnv Environment interface for SDK compatibility.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import sys as _sys
import os as _os

_parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

try:
    from ..models import CloudFinOpsAction, CloudFinOpsObservation, ServerState, RewardInfo
except (ImportError, SystemError):
    from models import CloudFinOpsAction, CloudFinOpsObservation, ServerState, RewardInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Realistic AWS-style hourly pricing (on-demand, approximate)
INSTANCE_CATALOG: Dict[str, Dict[str, Any]] = {
    "t3.micro":    {"vcpu": 2,  "mem_gb": 1,   "cost": 0.0104, "category": "web"},
    "t3.medium":   {"vcpu": 2,  "mem_gb": 4,   "cost": 0.0416, "category": "web"},
    "t3.large":    {"vcpu": 2,  "mem_gb": 8,   "cost": 0.0832, "category": "web"},
    "c5.large":    {"vcpu": 2,  "mem_gb": 4,   "cost": 0.0850, "category": "compute"},
    "c5.xlarge":   {"vcpu": 4,  "mem_gb": 8,   "cost": 0.1700, "category": "compute"},
    "r6g.medium":  {"vcpu": 1,  "mem_gb": 8,   "cost": 0.0504, "category": "db"},
    "r6g.large":   {"vcpu": 2,  "mem_gb": 16,  "cost": 0.1008, "category": "db"},
    "r6g.xlarge":  {"vcpu": 4,  "mem_gb": 32,  "cost": 0.2016, "category": "db"},
    "m5.large":    {"vcpu": 2,  "mem_gb": 8,   "cost": 0.0960, "category": "batch"},
    "m5.xlarge":   {"vcpu": 4,  "mem_gb": 16,  "cost": 0.1920, "category": "batch"},
}

# Upscale path — each instance can only be upgraded through these tiers
UPSCALE_PATH: Dict[str, str] = {
    "t3.micro":   "t3.medium",
    "t3.medium":  "t3.large",
    "c5.large":   "c5.xlarge",
    "r6g.medium": "r6g.large",
    "r6g.large":  "r6g.xlarge",
    "m5.large":   "m5.xlarge",
}

# GreenOps: Carbon intensity per instance type (kWh per step)
CARBON_INTENSITY: Dict[str, float] = {
    "t3.micro":   0.005,
    "t3.medium":  0.012,
    "t3.large":   0.022,
    "c5.large":   0.035,
    "c5.xlarge":  0.065,
    "r6g.medium": 0.008,
    "r6g.large":  0.015,
    "r6g.xlarge": 0.028,
    "m5.large":   0.040,
    "m5.xlarge":  0.075,
}

# Trailing metrics history depth
HISTORY_DEPTH = 3

MAX_STEPS = 10
SLA_CPU_LIMIT = 100.0  # CPU >= this => SLA breach


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _deterministic_noise(seed_str: str, amplitude: float = 2.0) -> float:
    """Return a deterministic pseudo-random noise value in [-amplitude, +amplitude]."""
    h = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    normalised = (h / 0xFFFFFFFF) * 2.0 - 1.0
    return normalised * amplitude


# ---------------------------------------------------------------------------
# Task Blueprints
# ---------------------------------------------------------------------------

def _easy_servers() -> List[ServerState]:
    """10 servers: 7 active web instances + 3 completely idle instances."""
    servers: List[ServerState] = []
    for i in range(4):
        servers.append(ServerState(
            id=f"web-{i}",
            type="t3.medium",
            cpu_util=round(25.0 + i * 7.5, 1),
            memory_util=round(20.0 + i * 5.0, 1),
            cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],
            status="running",
        ))
    for i in range(3):
        servers.append(ServerState(
            id=f"compute-{i}",
            type="c5.large",
            cpu_util=round(30.0 + i * 10.0, 1),
            memory_util=round(25.0 + i * 6.0, 1),
            cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],
            status="running",
        ))
    for i in range(3):
        servers.append(ServerState(
            id=f"idle-{i}",
            type="t3.micro",
            cpu_util=0.0,
            memory_util=0.0,
            cost_per_hour=INSTANCE_CATALOG["t3.micro"]["cost"],
            status="running",
        ))
    return servers


def _medium_servers() -> List[ServerState]:
    """12 over-provisioned servers at very low CPU across web/db/batch types."""
    servers: List[ServerState] = []
    configs = [
        ("t3.medium", "web"),
        ("r6g.large", "db"),
        ("m5.large", "batch"),
    ]
    for i in range(12):
        inst_type, _ = configs[i % 3]
        cat = INSTANCE_CATALOG[inst_type]
        servers.append(ServerState(
            id=f"{cat['category']}-{i}",
            type=inst_type,
            cpu_util=round(3.0 + (i % 4) * 1.5, 1),
            memory_util=round(5.0 + i * 0.8, 1),
            cost_per_hour=cat["cost"],
            status="running",
        ))
    return servers


def _hard_servers() -> List[ServerState]:
    """8 servers: DB (high load), web (medium), batch (low)."""
    return [
        ServerState(id="db-0",    type="r6g.large",  cpu_util=70.0, memory_util=60.0, cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],  status="running"),
        ServerState(id="db-1",    type="r6g.large",  cpu_util=63.0, memory_util=55.0, cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],  status="running"),
        ServerState(id="web-0",   type="t3.medium",  cpu_util=55.0, memory_util=40.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],  status="running"),
        ServerState(id="web-1",   type="t3.medium",  cpu_util=50.0, memory_util=35.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],  status="running"),
        ServerState(id="web-2",   type="c5.large",   cpu_util=60.0, memory_util=45.0, cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],   status="running"),
        ServerState(id="batch-0", type="m5.large",   cpu_util=20.0, memory_util=15.0, cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
        ServerState(id="batch-1", type="m5.large",   cpu_util=15.0, memory_util=10.0, cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
        ServerState(id="batch-2", type="m5.large",   cpu_util=10.0, memory_util=8.0,  cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
    ]


def _green_servers() -> List[ServerState]:
    """10 servers: mix of dirty x86 (c5, m5) and clean ARM (r6g) instances."""
    return [
        ServerState(id="compute-0", type="c5.large",   cpu_util=45.0, memory_util=30.0, cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],   status="running"),
        ServerState(id="compute-1", type="c5.large",   cpu_util=50.0, memory_util=35.0, cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],   status="running"),
        ServerState(id="compute-2", type="c5.xlarge",  cpu_util=40.0, memory_util=25.0, cost_per_hour=INSTANCE_CATALOG["c5.xlarge"]["cost"],  status="running"),
        ServerState(id="batch-0",   type="m5.large",   cpu_util=35.0, memory_util=20.0, cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
        ServerState(id="batch-1",   type="m5.large",   cpu_util=25.0, memory_util=15.0, cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
        ServerState(id="batch-2",   type="m5.xlarge",  cpu_util=30.0, memory_util=18.0, cost_per_hour=INSTANCE_CATALOG["m5.xlarge"]["cost"],  status="running"),
        ServerState(id="arm-0",     type="r6g.large",  cpu_util=15.0, memory_util=10.0, cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],  status="running"),
        ServerState(id="arm-1",     type="r6g.large",  cpu_util=12.0, memory_util=8.0,  cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],  status="running"),
        ServerState(id="arm-2",     type="r6g.medium", cpu_util=10.0, memory_util=5.0,  cost_per_hour=INSTANCE_CATALOG["r6g.medium"]["cost"], status="running"),
        ServerState(id="idle-0",    type="t3.micro",   cpu_util=0.0,  memory_util=0.0,  cost_per_hour=INSTANCE_CATALOG["t3.micro"]["cost"],   status="running"),
    ]


def _expert_servers() -> List[ServerState]:
    """14 servers across 3 simulated regions. Region eu-west-1 is failing —
    agents must redistribute load to us-east-1 and ap-south-1 while managing
    a tight budget and escalating traffic."""
    return [
        # eu-west-1 — failing region (high CPU, degrading)
        ServerState(id="eu-web-0",    type="t3.large",    cpu_util=88.0, memory_util=75.0, cost_per_hour=INSTANCE_CATALOG["t3.large"]["cost"],    status="running"),
        ServerState(id="eu-web-1",    type="t3.large",    cpu_util=82.0, memory_util=70.0, cost_per_hour=INSTANCE_CATALOG["t3.large"]["cost"],    status="running"),
        ServerState(id="eu-db-0",     type="r6g.large",   cpu_util=90.0, memory_util=80.0, cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],   status="running"),
        ServerState(id="eu-compute-0",type="c5.xlarge",   cpu_util=78.0, memory_util=65.0, cost_per_hour=INSTANCE_CATALOG["c5.xlarge"]["cost"],   status="running"),
        # us-east-1 — healthy region with capacity
        ServerState(id="us-web-0",    type="t3.medium",   cpu_util=25.0, memory_util=20.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],   status="running"),
        ServerState(id="us-web-1",    type="t3.medium",   cpu_util=30.0, memory_util=22.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],   status="running"),
        ServerState(id="us-db-0",     type="r6g.xlarge",  cpu_util=35.0, memory_util=28.0, cost_per_hour=INSTANCE_CATALOG["r6g.xlarge"]["cost"],  status="running"),
        ServerState(id="us-compute-0",type="c5.large",    cpu_util=20.0, memory_util=15.0, cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],    status="running"),
        ServerState(id="us-batch-0",  type="m5.large",    cpu_util=10.0, memory_util=8.0,  cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],    status="running"),
        # ap-south-1 — warm standby
        ServerState(id="ap-web-0",    type="t3.medium",   cpu_util=15.0, memory_util=12.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],   status="running"),
        ServerState(id="ap-db-0",     type="r6g.medium",  cpu_util=10.0, memory_util=8.0,  cost_per_hour=INSTANCE_CATALOG["r6g.medium"]["cost"],  status="running"),
        ServerState(id="ap-batch-0",  type="m5.large",    cpu_util=5.0,  memory_util=4.0,  cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],    status="running"),
        # Zombies in failing region
        ServerState(id="eu-idle-0",   type="t3.micro",    cpu_util=0.0,  memory_util=0.0,  cost_per_hour=INSTANCE_CATALOG["t3.micro"]["cost"],    status="running"),
        ServerState(id="eu-idle-1",   type="t3.micro",    cpu_util=0.0,  memory_util=0.0,  cost_per_hour=INSTANCE_CATALOG["t3.micro"]["cost"],    status="running"),
    ]


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "servers_fn": _easy_servers,
        "budget": 5.0,
        "traffic_load": 30.0,
        "spike": False,
        "inbox": [
            "Ops Team: We're seeing 3 zombie instances racking up charges. Please terminate unused servers to save costs.",
        ],
    },
    "medium": {
        "servers_fn": _medium_servers,
        "budget": 10.0,
        "traffic_load": 20.0,
        "spike": False,
        "inbox": [
            "CTO: Our cloud bill is through the roof. Cut costs by at least 50% immediately — no excuses.",
            "Finance: Q3 budget review in 2 days. We need measurable savings by then.",
        ],
    },
    "hard": {
        "servers_fn": _hard_servers,
        "budget": 4.0,
        "traffic_load": 75.0,
        "spike": True,
        "inbox": [
            "Marketing: Massive ad campaign going live RIGHT NOW! Expect 10× normal traffic.",
            "SRE On-Call: DB-0 is approaching capacity. Consider upscaling before it breaches SLA.",
        ],
    },
    "green": {
        "servers_fn": _green_servers,
        "budget": 8.0,
        "traffic_load": 40.0,
        "spike": False,
        "inbox": [
            "CTO: We've committed to a 40% carbon reduction by EOQ. Migrate workloads from x86 to ARM Graviton where possible.",
            "Sustainability Lead: Our c5 and m5 instances produce 3× the emissions of r6g. Please prioritise migration.",
        ],
    },
    "expert": {
        "servers_fn": _expert_servers,
        "budget": 6.0,
        "traffic_load": 60.0,
        "spike": True,
        "inbox": [
            "SRE Alert: eu-west-1 region experiencing cascading failures. DB and compute nodes at critical capacity.",
            "VP Engineering: Activate disaster recovery plan. Shift load to us-east-1 and ap-south-1 immediately.",
            "Finance: Budget is constrained — terminate zombies first, then redistribute. No budget overruns.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Core Engine (internal — not exposed directly to OpenEnv)
# ---------------------------------------------------------------------------

class CloudFinOpsEngine:
    """Deterministic physics engine for the CloudFinOps RL environment."""

    def __init__(self) -> None:
        self.servers: List[ServerState] = []
        self.task_id: str = "easy"
        self.time_step: int = 0
        self.budget_remaining: float = 0.0
        self.initial_budget: float = 0.0
        self.traffic_load: float = 0.0
        self.spike_detected: bool = False
        self.incidents: List[Dict[str, Any]] = []
        self.inbox: List[str] = []
        self.done: bool = False
        self.sla_breached: bool = False
        self.total_cost_spent: float = 0.0
        self.terminated_ids: List[str] = []
        self.upscaled_ids: List[str] = []
        self.upscale_counts: Dict[str, int] = {}
        self.pending_scales: Dict[str, str] = {}
        self._reward_accum: float = 0.0
        self.carbon_kwh: float = 0.0
        self.initial_carbon_rate: float = 0.0
        self._cpu_history: Dict[str, List[float]] = {}
        self._mem_history: Dict[str, List[float]] = {}
        self._inbox_replied_early: bool = False

    def reset(self, task_id: str) -> CloudFinOpsObservation:
        cfg = TASK_CONFIGS.get(task_id)
        if cfg is None:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_CONFIGS)}")

        self.task_id = task_id
        self.servers = cfg["servers_fn"]()
        self.budget_remaining = cfg["budget"]
        self.initial_budget = cfg["budget"]
        self.traffic_load = cfg["traffic_load"]
        self.spike_detected = cfg["spike"]
        self.inbox = list(cfg["inbox"])
        self.incidents = []
        self.time_step = 0
        self.done = False
        self.sla_breached = False
        self.total_cost_spent = 0.0
        self.terminated_ids = []
        self.upscaled_ids = []
        self.upscale_counts = {}
        self.pending_scales = {}
        self._reward_accum = 0.0
        self.carbon_kwh = 0.0
        self.initial_carbon_rate = sum(
            CARBON_INTENSITY.get(s.type, 0.01)
            for s in self.servers if s.status == "running"
        )
        self._cpu_history = {s.id: [s.cpu_util] for s in self.servers}
        self._mem_history = {s.id: [s.memory_util] for s in self.servers}
        self._inbox_replied_early = False
        return self._obs()

    def step(self, action: CloudFinOpsAction) -> Tuple[CloudFinOpsObservation, float, bool, Dict[str, Any]]:
        if self.done:
            return self._obs(), 0.0, True, {"message": "Episode already done."}

        reward = 0.0
        self.time_step += 1

        self._apply_pending_scales()
        reward += self._process_action(action)
        self._simulate_traffic()
        self._apply_noise()
        self._redistribute_load()

        step_cost = sum(s.cost_per_hour for s in self.servers if s.status == "running")
        self.budget_remaining -= step_cost
        self.total_cost_spent += step_cost

        step_carbon = sum(
            CARBON_INTENSITY.get(s.type, 0.01)
            for s in self.servers if s.status == "running"
        )
        self.carbon_kwh += step_carbon

        self._update_history()

        for s in self.servers:
            if s.status == "running" and s.cpu_util >= SLA_CPU_LIMIT:
                self.sla_breached = True
                self.incidents.append({
                    "type": "SLA_BREACH",
                    "server": s.id,
                    "instance_type": s.type,
                    "cpu_at_breach": round(s.cpu_util, 1),
                    "step": self.time_step,
                })
                reward -= 100.0

        if step_cost > 0.50:
            reward -= 1.0

        if self.budget_remaining < 0:
            reward -= 20.0

        if self.time_step >= MAX_STEPS or self.sla_breached or self.budget_remaining <= 0:
            self.done = True

        self._reward_accum += reward

        info: Dict[str, Any] = {"step_reward": reward, "cumulative_reward": self._reward_accum}
        if self.done:
            final_score = self.grade()
            info["grader_score"] = final_score

        return self._obs(), reward, self.done, info

    def get_state(self) -> CloudFinOpsObservation:
        return self._obs()

    def grade(self) -> float:
        if self.task_id == "easy":
            return self._grade_easy()
        elif self.task_id == "medium":
            return self._grade_medium()
        elif self.task_id == "hard":
            return self._grade_hard()
        elif self.task_id == "green":
            return self._grade_green()
        elif self.task_id == "expert":
            return self._grade_expert()
        else:
            return 0.0

    def _grade_easy(self) -> float:
        idle_ids = {f"idle-{i}" for i in range(3)}
        terminated_idle = len(idle_ids & set(self.terminated_ids))
        active_terminated = len(set(self.terminated_ids) - idle_ids)
        score = (terminated_idle / 3.0) - (active_terminated * 0.25)
        if self.sla_breached:
            score -= 0.5
        return _clamp(score, 0.0, 1.0)

    def _grade_medium(self) -> float:
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        target = 0.50
        efficiency = min(cost_saved_pct / target, 1.0) if target > 0 else 0.0
        crash_penalty = 0.5 if self.sla_breached else 0.0
        return _clamp(efficiency - crash_penalty, 0.0, 1.0)

    def _grade_hard(self) -> float:
        uptime_score = 0.0 if self.sla_breached else 1.0
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        cost_efficiency = _clamp(cost_saved_pct, 0.0, 1.0)
        base = uptime_score * 0.6 + cost_efficiency * 0.4
        inbox_bonus = 0.1 if self._inbox_replied_early else 0.0
        return round(_clamp(base + inbox_bonus, 0.0, 1.0), 4)

    def _grade_green(self) -> float:
        if self.initial_carbon_rate > 0:
            current_rate = sum(
                CARBON_INTENSITY.get(s.type, 0.01)
                for s in self.servers if s.status == "running"
            )
            reduction_pct = 1.0 - (current_rate / self.initial_carbon_rate)
            target = 0.40
            carbon_score = min(reduction_pct / target, 1.0) if target > 0 else 0.0
        else:
            carbon_score = 0.0

        uptime_score = 0.0 if self.sla_breached else 1.0
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        cost_efficiency = _clamp(cost_saved_pct, 0.0, 1.0)
        inbox_bonus = 0.1 if not self.inbox else 0.0
        base = carbon_score * 0.5 + uptime_score * 0.3 + cost_efficiency * 0.1
        return round(_clamp(base + inbox_bonus, 0.0, 1.0), 4)

    def _grade_expert(self) -> float:
        """Multi-Region Disaster Recovery grading.
        - Zombies terminated (eu-idle-0, eu-idle-1): 20%
        - EU region servers terminated or downscaled (eu-web-*, eu-db-0, eu-compute-0): 30%
        - No SLA breach on healthy regions (us-*, ap-*): 30%
        - Inbox replies sent (clears all 3 messages): 20%
        """
        zombie_ids = {"eu-idle-0", "eu-idle-1"}
        terminated_zombies = len(zombie_ids & set(self.terminated_ids))
        zombie_score = terminated_zombies / 2.0

        eu_target_ids = {"eu-web-0", "eu-web-1", "eu-db-0", "eu-compute-0"}
        eu_handled = sum(1 for sid in eu_target_ids if sid in self.terminated_ids)
        eu_score = eu_handled / len(eu_target_ids)

        healthy_ids = {s.id for s in self.servers if s.id.startswith(("us-", "ap-"))}
        healthy_breached = any(
            inc["server"] in healthy_ids for inc in self.incidents
        )
        uptime_score = 0.0 if (self.sla_breached or healthy_breached) else 1.0

        inbox_bonus = 0.2 if not self.inbox else 0.0

        base = zombie_score * 0.2 + eu_score * 0.3 + uptime_score * 0.3
        return round(_clamp(base + inbox_bonus, 0.0, 1.0), 4)

    def _obs(self) -> CloudFinOpsObservation:
        server_copies = []
        for s in self.servers:
            sc = s.model_copy()
            sc.cpu_history = list(self._cpu_history.get(s.id, []))[-HISTORY_DEPTH:]
            sc.memory_history = list(self._mem_history.get(s.id, []))[-HISTORY_DEPTH:]
            server_copies.append(sc)

        return CloudFinOpsObservation(
            servers=server_copies,
            traffic_load=round(self.traffic_load, 2),
            spike_detected=self.spike_detected,
            incidents=list(self.incidents),
            budget_remaining=round(self.budget_remaining, 4),
            time_step=self.time_step,
            inbox=list(self.inbox),
            carbon_kwh=round(self.carbon_kwh, 4),
        )

    def _find_server(self, server_id: Optional[str]) -> Optional[ServerState]:
        if server_id is None:
            return None
        for s in self.servers:
            if s.id == server_id:
                return s
        return None

    def _process_action(self, action: CloudFinOpsAction) -> float:
        reward = 0.0
        server = self._find_server(action.target_id)

        if action.command == "IGNORE":
            return 0.0

        if server is None:
            return -2.0

        if server.status == "terminated":
            return -2.0

        if action.command == "TERMINATE":
            server.status = "terminated"
            server.cpu_util = 0.0
            server.memory_util = 0.0
            self.terminated_ids.append(server.id)
            reward += 10.0

        elif action.command == "UPSCALE":
            next_type = UPSCALE_PATH.get(server.type)
            if next_type is None:
                reward -= 1.0
            else:
                count = self.upscale_counts.get(server.id, 0)
                if count >= 2:
                    reward -= 1.0
                else:
                    self.pending_scales[server.id] = next_type
                    self.upscaled_ids.append(server.id)
                    self.upscale_counts[server.id] = count + 1
                    reward -= 5.0

        elif action.command == "DOWNSCALE":
            server.cost_per_hour = round(server.cost_per_hour * 0.5, 4)
            server.cpu_util = _clamp(server.cpu_util * 1.8)
            server.memory_util = _clamp(server.memory_util * 1.3)
            reward += 5.0

        elif action.command == "REDISTRIBUTE_LOAD":
            running = [s for s in self.servers if s.status == "running"]
            if len(running) > 1:
                avg_cpu = sum(s.cpu_util for s in running) / len(running)
                avg_mem = sum(s.memory_util for s in running) / len(running)
                for s in running:
                    s.cpu_util = round(_clamp(avg_cpu), 1)
                    s.memory_util = round(_clamp(avg_mem), 1)
                reward += 3.0

        if action.reply and self.inbox:
            reward += 2.0
            self.inbox = []
            if self.time_step <= 3:
                self._inbox_replied_early = True

        return reward

    def _apply_pending_scales(self) -> None:
        for sid, new_type in list(self.pending_scales.items()):
            server = self._find_server(sid)
            if server and server.status == "running":
                new_cost = INSTANCE_CATALOG[new_type]["cost"]
                server.type = new_type
                server.cost_per_hour = new_cost
                server.cpu_util = _clamp(server.cpu_util * 0.5)
                server.memory_util = _clamp(server.memory_util * 0.6)
        self.pending_scales.clear()

    def _simulate_traffic(self) -> None:
        if self.task_id == "hard":
            self.traffic_load = _clamp(self.traffic_load + 5.0 * math.log1p(self.time_step))
            self.spike_detected = True
            for s in self.servers:
                if s.status == "running" and s.type.startswith("r6g"):
                    s.cpu_util = _clamp(s.cpu_util + 4.0 * math.log1p(self.time_step))
        elif self.task_id == "expert":
            self.traffic_load = _clamp(self.traffic_load + 3.0 * math.log1p(self.time_step))
            self.spike_detected = True
            for s in self.servers:
                if s.status == "running" and s.id.startswith("eu-"):
                    s.cpu_util = _clamp(s.cpu_util + 6.0 * math.log1p(self.time_step))
        else:
            self.traffic_load = _clamp(self.traffic_load + 0.5)

    def _apply_noise(self) -> None:
        for s in self.servers:
            if s.status != "running":
                continue
            seed = f"{self.task_id}:{s.id}:{self.time_step}"
            cpu_noise = _deterministic_noise(seed + ":cpu", amplitude=2.5)
            mem_noise = _deterministic_noise(seed + ":mem", amplitude=1.5)
            s.cpu_util = round(_clamp(s.cpu_util + cpu_noise), 1)
            s.memory_util = round(_clamp(s.memory_util + mem_noise), 1)

    def _redistribute_load(self) -> None:
        running = [s for s in self.servers if s.status == "running"]
        terminated_this = [s for s in self.servers if s.status == "terminated"]
        if not running:
            return
        orphan_cpu = sum(s.cpu_util for s in terminated_this)
        if orphan_cpu > 0 and len(running) > 0:
            per_server = orphan_cpu / len(running)
            for s in running:
                s.cpu_util = round(_clamp(s.cpu_util + per_server), 1)

    def _update_history(self) -> None:
        for s in self.servers:
            if s.id not in self._cpu_history:
                self._cpu_history[s.id] = []
                self._mem_history[s.id] = []
            self._cpu_history[s.id].append(s.cpu_util)
            self._mem_history[s.id].append(s.memory_util)
            if len(self._cpu_history[s.id]) > HISTORY_DEPTH:
                self._cpu_history[s.id] = self._cpu_history[s.id][-HISTORY_DEPTH:]
            if len(self._mem_history[s.id]) > HISTORY_DEPTH:
                self._mem_history[s.id] = self._mem_history[s.id][-HISTORY_DEPTH:]


# ---------------------------------------------------------------------------
# OpenEnv Environment Wrapper
# ---------------------------------------------------------------------------

class CloudFinOpsEnvironment(Environment):
    """OpenEnv-compatible wrapper around the CloudFinOps physics engine.

    Maps the OpenEnv interface (reset/step/state) to the underlying engine.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._engine = CloudFinOpsEngine()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_reward: float = 0.0
        self._done: bool = False
        self._current_task: str = "easy"
        # Action history for dashboard
        self.action_history: List[Dict[str, Any]] = []

    def reset(self, task_id: str = "easy") -> CloudFinOpsObservation:
        """Reset the environment for the given task."""
        self._current_task = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_reward = 0.0
        self.action_history = []
        obs = self._engine.reset(task_id)
        obs.done = False
        obs.reward = 0.0
        return obs

    def step(self, action: CloudFinOpsAction) -> CloudFinOpsObservation:  # type: ignore[override]
        """Execute a step in the environment."""
        self._state.step_count += 1
        obs, reward, done, info = self._engine.step(action)
        self._last_reward = reward
        self._done = done

        # Record action in history
        self.action_history.append({
            "step": obs.time_step,
            "command": action.command,
            "target_id": action.target_id,
            "reply": action.reply or "",
            "reward": reward,
            "done": done,
            "budget": obs.budget_remaining,
            "score": info.get("grader_score"),
        })

        obs.done = done
        obs.reward = reward
        obs.metadata = info
        return obs

    @property
    def state(self) -> State:
        return self._state

    @property
    def engine(self) -> CloudFinOpsEngine:
        """Expose the underlying engine for direct access (e.g. grading)."""
        return self._engine
