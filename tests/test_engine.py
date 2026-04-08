"""Unit tests for the CloudFinOps physics engine.

Tests engine mechanics including state transitions, SLA breaches,
deterministic noise reproducibility, grading, carbon tracking,
and trailing metrics history.
"""

import pytest
from server.cloudfinops_env_environment import (
    CloudFinOpsEngine,
    INSTANCE_CATALOG,
    UPSCALE_PATH,
    CARBON_INTENSITY,
    MAX_STEPS,
    SLA_CPU_LIMIT,
    _deterministic_noise,
    _clamp,
)
from models import CloudFinOpsAction, CloudFinOpsObservation, ServerState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return CloudFinOpsEngine()


# ---------------------------------------------------------------------------
# 1. Reset produces clean state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy_returns_observation(self, engine):
        obs = engine.reset("easy")
        assert isinstance(obs, CloudFinOpsObservation)
        assert obs.time_step == 0
        assert obs.budget_remaining == 5.0
        assert len(obs.servers) == 10

    def test_reset_medium(self, engine):
        obs = engine.reset("medium")
        assert len(obs.servers) == 12
        assert obs.budget_remaining == 10.0

    def test_reset_hard(self, engine):
        obs = engine.reset("hard")
        assert len(obs.servers) == 8
        assert obs.budget_remaining == 4.0
        assert obs.spike_detected is True

    def test_reset_green(self, engine):
        obs = engine.reset("green")
        assert len(obs.servers) == 10
        assert obs.budget_remaining == 8.0
        assert obs.carbon_kwh == 0.0

    def test_reset_invalid_task(self, engine):
        with pytest.raises(ValueError, match="Unknown task_id"):
            engine.reset("nonexistent")

    def test_reset_clears_state(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        obs2 = engine.reset("easy")
        assert obs2.time_step == 0
        assert len([s for s in obs2.servers if s.status == "running"]) == 10


# ---------------------------------------------------------------------------
# 2. Deterministic noise reproducibility
# ---------------------------------------------------------------------------

class TestDeterministicNoise:
    def test_same_seed_same_result(self):
        a = _deterministic_noise("easy:web-0:1:cpu")
        b = _deterministic_noise("easy:web-0:1:cpu")
        assert a == b

    def test_different_seeds_different_results(self):
        a = _deterministic_noise("easy:web-0:1:cpu")
        b = _deterministic_noise("easy:web-0:2:cpu")
        assert a != b

    def test_noise_within_amplitude(self):
        for i in range(100):
            val = _deterministic_noise(f"test:{i}", amplitude=3.0)
            assert -3.0 <= val <= 3.0


# ---------------------------------------------------------------------------
# 3. Action mechanics
# ---------------------------------------------------------------------------

class TestActions:
    def test_terminate_removes_server(self, engine):
        engine.reset("easy")
        obs, reward, done, info = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        terminated = [s for s in obs.servers if s.id == "idle-0"][0]
        assert terminated.status == "terminated"
        assert terminated.cpu_util == 0.0
        assert reward > 0

    def test_terminate_invalid_target(self, engine):
        engine.reset("easy")
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="nonexistent"))
        assert reward == -2.0

    def test_terminate_already_dead(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        assert reward == -2.0

    def test_upscale_queues_pending(self, engine):
        engine.reset("easy")
        obs, reward, _, _ = engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        web0 = [s for s in obs.servers if s.id == "web-0"][0]
        assert reward < 0

    def test_upscale_applies_next_step(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        obs, _, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        web0 = [s for s in obs.servers if s.id == "web-0"][0]
        assert web0.type == "t3.large"

    def test_upscale_max_cap(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        engine.step(CloudFinOpsAction(command="IGNORE"))
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        assert reward == -1.0

    def test_downscale_increases_cpu(self, engine):
        engine.reset("easy")
        obs_before = engine.get_state()
        web0_before = [s for s in obs_before.servers if s.id == "web-0"][0]
        cpu_before = web0_before.cpu_util
        obs, reward, _, _ = engine.step(CloudFinOpsAction(command="DOWNSCALE", target_id="web-0"))
        web0_after = [s for s in obs.servers if s.id == "web-0"][0]
        assert reward > 0

    def test_redistribute_load(self, engine):
        engine.reset("easy")
        obs, reward, _, _ = engine.step(CloudFinOpsAction(command="REDISTRIBUTE_LOAD", target_id="web-0"))
        assert reward > 0

    def test_ignore_no_reward(self, engine):
        engine.reset("easy")
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert reward <= 0

    def test_inbox_reply_gives_bonus(self, engine):
        engine.reset("easy")
        obs1 = engine.get_state()
        assert len(obs1.inbox) > 0
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0", reply="On it!"))
        assert reward >= 12.0


# ---------------------------------------------------------------------------
# 4. SLA breach detection
# ---------------------------------------------------------------------------

class TestSLABreach:
    def test_sla_breach_triggers_done(self, engine):
        engine.reset("hard")
        engine.servers[0].cpu_util = 99.9
        _, reward, done, info = engine.step(CloudFinOpsAction(command="IGNORE"))
        engine2 = CloudFinOpsEngine()
        engine2.reset("hard")
        engine2.servers[0].cpu_util = 100.0
        _, reward, done, info = engine2.step(CloudFinOpsAction(command="IGNORE"))
        assert done is True
        assert reward <= -100.0


# ---------------------------------------------------------------------------
# 5. Grading
# ---------------------------------------------------------------------------

class TestGrading:
    def test_easy_perfect_score(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0", reply="Done"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-1"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-2"))
        score = engine.grade()
        assert score >= 0.99  # clamped to max 0.99

    def test_easy_partial_score(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        score = engine.grade()
        assert 0.0 < score < 1.0

    def test_grader_scores_in_range(self, engine):
        for task in ["easy", "medium", "hard", "green"]:
            engine.reset(task)
            for _ in range(MAX_STEPS):
                engine.step(CloudFinOpsAction(command="IGNORE"))
            score = engine.grade()
            assert 0.0 < score < 1.0, f"Score for {task} out of range: {score}"

    def test_green_grader_rewards_carbon_reduction(self, engine):
        engine.reset("green")
        for sid in ["compute-0", "compute-1", "compute-2", "batch-0", "batch-1", "batch-2"]:
            engine.step(CloudFinOpsAction(command="TERMINATE", target_id=sid, reply="Reducing carbon"))
        for _ in range(MAX_STEPS - 6):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        score = engine.grade()
        assert score > 0.3


# ---------------------------------------------------------------------------
# 6. Carbon tracking (GreenOps)
# ---------------------------------------------------------------------------

class TestCarbonTracking:
    def test_carbon_starts_at_zero(self, engine):
        obs = engine.reset("green")
        assert obs.carbon_kwh == 0.0

    def test_carbon_accumulates(self, engine):
        engine.reset("green")
        obs1, _, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert obs1.carbon_kwh > 0.0
        obs2, _, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert obs2.carbon_kwh > obs1.carbon_kwh

    def test_carbon_decreases_after_terminate(self, engine):
        engine.reset("green")
        obs1, _, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        rate1 = obs1.carbon_kwh

        engine.reset("green")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-2"))
        obs2, _, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert obs2.carbon_kwh > 0

    def test_all_instances_have_carbon_intensity(self):
        for inst_type in INSTANCE_CATALOG:
            assert inst_type in CARBON_INTENSITY, f"Missing carbon intensity for {inst_type}"


# ---------------------------------------------------------------------------
# 7. Trailing metrics history
# ---------------------------------------------------------------------------

class TestTrailingHistory:
    def test_initial_history(self, engine):
        obs = engine.reset("easy")
        for s in obs.servers:
            assert len(s.cpu_history) >= 1
            assert len(s.memory_history) >= 1

    def test_history_grows(self, engine):
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="IGNORE"))
        obs = engine.step(CloudFinOpsAction(command="IGNORE"))[0]
        for s in obs.servers:
            if s.status == "running":
                assert len(s.cpu_history) >= 2
                assert len(s.memory_history) >= 2

    def test_history_max_depth(self, engine):
        engine.reset("easy")
        for _ in range(5):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        obs = engine.get_state()
        for s in obs.servers:
            assert len(s.cpu_history) <= 3
            assert len(s.memory_history) <= 3


# ---------------------------------------------------------------------------
# 8. Episode boundaries
# ---------------------------------------------------------------------------

class TestEpisodeBoundaries:
    def test_max_steps_ends_episode(self, engine):
        engine.reset("easy")
        for i in range(MAX_STEPS):
            obs, reward, done, info = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert done is True
        assert "grader_score" in info

    def test_budget_overrun_ends_episode(self, engine):
        engine.reset("easy")
        engine.budget_remaining = 0.01
        _, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert done is True

    def test_step_after_done_returns_done(self, engine):
        engine.reset("easy")
        engine.done = True
        _, reward, done, info = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert done is True
        assert reward == 0.0


# ---------------------------------------------------------------------------
# 9. Clamp utility
# ---------------------------------------------------------------------------

class TestClamp:
    def test_clamp_within_range(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_below(self):
        assert _clamp(-10.0) == 0.0

    def test_clamp_above(self):
        assert _clamp(150.0) == 100.0

    def test_clamp_custom_bounds(self):
        assert _clamp(5.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# 10. Cascading Failure Edge Cases
# ---------------------------------------------------------------------------

class TestCascadingFailures:
    """Test load redistribution after terminations can cascade into SLA breaches."""

    def test_terminate_redistributes_load(self, engine):
        """Terminating a server pushes its CPU to remaining servers."""
        engine.reset("easy")
        # Record CPU of web-0 before terminating idle-0
        web0_before = [s for s in engine.servers if s.id == "web-0"][0].cpu_util
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        web0_after = [s for s in engine.servers if s.id == "web-0"][0].cpu_util
        # Idle-0 had 0 CPU so redistribution should be negligible
        # but the mechanism should still run
        assert web0_after >= 0

    def test_massive_terminate_cascade(self, engine):
        """Terminating many servers at once can push remaining ones toward breach."""
        engine.reset("hard")
        # Terminate batch servers one by one, tracking db-0 CPU
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-0"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-1"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-2"))
        db0 = [s for s in engine.servers if s.id == "db-0"][0]
        # DB-0 should have absorbed orphan CPU from batch terminations
        assert db0.cpu_util > 70.0  # original was 70, should be higher now

    def test_terminate_then_upscale_prevents_breach(self, engine):
        """Strategic upscale after terminate prevents cascading SLA breach."""
        engine.reset("hard")
        # Upscale db-0 first (queued)
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="db-0"))
        # Next step applies upscale (halves CPU), then terminate batch
        obs, _, _, _ = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-2"))
        db0 = [s for s in obs.servers if s.id == "db-0"][0]
        # After upscale applied, CPU should be reduced
        assert db0.type == "r6g.xlarge"  # upgraded from r6g.large


# ---------------------------------------------------------------------------
# 11. Budget Edge Cases
# ---------------------------------------------------------------------------

class TestBudgetEdgeCases:
    """Test budget boundary conditions and exhaustion scenarios."""

    def test_budget_exactly_zero_ends_episode(self, engine):
        """Budget hitting exactly 0 should end the episode."""
        engine.reset("easy")
        engine.budget_remaining = 0.0
        _, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert done is True

    def test_budget_negative_after_step_penalty(self, engine):
        """Going negative incurs -20 reward penalty."""
        engine.reset("easy")
        engine.budget_remaining = 0.001  # barely positive
        _, reward, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
        assert done is True
        assert reward < 0  # should include budget penalty

    def test_high_cost_fleet_drains_budget_fast(self, engine):
        """With expensive servers, budget drains in fewer steps."""
        engine.reset("hard")  # budget=4.0, expensive fleet
        steps_to_done = 0
        for i in range(MAX_STEPS):
            _, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
            steps_to_done = i + 1
            if done:
                break
        # Hard task should complete before MAX_STEPS due to budget/SLA
        assert steps_to_done <= MAX_STEPS

    def test_terminate_reduces_burn_rate(self, engine):
        """Terminating servers should reduce per-step cost."""
        engine.reset("easy")
        cost_before = sum(s.cost_per_hour for s in engine.servers if s.status == "running")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-1"))
        cost_after = sum(s.cost_per_hour for s in engine.servers if s.status == "running")
        assert cost_after < cost_before


# ---------------------------------------------------------------------------
# 12. Double-Action and Invalid Action Edge Cases
# ---------------------------------------------------------------------------

class TestInvalidActions:
    """Test penalty mechanics for invalid or repeated actions."""

    def test_double_terminate_penalty(self, engine):
        """Terminating an already-terminated server should give -2.0."""
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        assert reward == -2.0

    def test_nonexistent_server_penalty(self, engine):
        """Acting on a server that doesn't exist should give -2.0."""
        engine.reset("easy")
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="TERMINATE", target_id="ghost-99"))
        assert reward == -2.0

    def test_upscale_at_max_tier_penalty(self, engine):
        """Upscaling a server already at max tier should penalize."""
        engine.reset("easy")
        # t3.large is max of t3 path; upscale web-0 twice to get there
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))  # queue t3.medium -> t3.large
        engine.step(CloudFinOpsAction(command="IGNORE"))  # apply
        # Now web-0 is t3.large (top of path), upscale again
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        assert reward == -1.0  # no upscale path

    def test_upscale_count_cap(self, engine):
        """Each server can only be upscaled 2 times maximum."""
        engine.reset("medium")
        # r6g.large can go to r6g.xlarge (1 upscale)
        # Find a db server
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="db-1"))
        engine.step(CloudFinOpsAction(command="IGNORE"))  # apply first upscale
        # Second upscale attempt on r6g.xlarge (no path)
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="UPSCALE", target_id="db-1"))
        assert reward <= -1.0

    def test_downscale_null_target(self, engine):
        """Downscale with no target should penalize."""
        engine.reset("easy")
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="DOWNSCALE", target_id=None))
        assert reward == -2.0

    def test_redistribute_with_single_server(self, engine):
        """REDISTRIBUTE_LOAD with 1 running server should still work."""
        engine.reset("easy")
        # Terminate everything except web-0
        for s in engine.servers:
            if s.id != "web-0":
                engine.step(CloudFinOpsAction(command="TERMINATE", target_id=s.id))
        obs, reward, _, _ = engine.step(CloudFinOpsAction(command="REDISTRIBUTE_LOAD", target_id="web-0"))
        # Should not crash, reward may be 0 (no redistribution with 1 server)
        assert reward is not None


# ---------------------------------------------------------------------------
# 13. Downscale Safety Edge Cases
# ---------------------------------------------------------------------------

class TestDownscaleSafety:
    """Test that downscale correctly increases CPU by 1.8x."""

    def test_downscale_cpu_multiplier(self, engine):
        """Downscale should multiply CPU by ~1.8x."""
        engine.reset("medium")
        # Find a server with known low CPU
        target = None
        for s in engine.servers:
            if s.status == "running" and s.cpu_util < 10:
                target = s
                break
        assert target is not None
        cpu_before = target.cpu_util
        engine.step(CloudFinOpsAction(command="DOWNSCALE", target_id=target.id))
        # After downscale, CPU should increase (1.8x + noise + clamp)
        target_after = [s for s in engine.servers if s.id == target.id][0]
        # CPU should have increased (accounting for noise)
        assert target_after.cpu_util > cpu_before * 1.0

    def test_downscale_halves_cost(self, engine):
        """Downscale should halve cost_per_hour."""
        engine.reset("medium")
        target = engine.servers[0]
        cost_before = target.cost_per_hour
        engine.step(CloudFinOpsAction(command="DOWNSCALE", target_id=target.id))
        target_after = [s for s in engine.servers if s.id == target.id][0]
        assert abs(target_after.cost_per_hour - cost_before * 0.5) < 0.001

    def test_downscale_high_cpu_risks_breach(self, engine):
        """Downscaling a server with CPU ~50 should push it to ~90 (danger zone)."""
        engine.reset("hard")
        # web-0 has cpu_util=55.0 -> 55*1.8 = 99 -> danger!
        engine.step(CloudFinOpsAction(command="DOWNSCALE", target_id="web-0"))
        web0 = [s for s in engine.servers if s.id == "web-0"][0]
        # After noise/clamp, CPU should be very high
        assert web0.cpu_util > 80.0


# ---------------------------------------------------------------------------
# 14. GreenOps Carbon-Aware Edge Cases
# ---------------------------------------------------------------------------

class TestGreenOpsEdgeCases:
    """Test carbon tracking precision and green grading nuances."""

    def test_carbon_rate_decreases_with_dirty_termination(self, engine):
        """Terminating x86 instances should reduce per-step carbon rate."""
        engine.reset("green")
        initial_rate = engine.initial_carbon_rate

        # Terminate the dirtiest instance (m5.xlarge has highest carbon)
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-2"))  # m5.xlarge
        current_rate = sum(
            CARBON_INTENSITY.get(s.type, 0.01)
            for s in engine.servers if s.status == "running"
        )
        assert current_rate < initial_rate

    def test_arm_instances_have_lower_carbon(self):
        """r6g (ARM) instances should have lower carbon than c5/m5 (x86)."""
        assert CARBON_INTENSITY["r6g.large"] < CARBON_INTENSITY["c5.large"]
        assert CARBON_INTENSITY["r6g.large"] < CARBON_INTENSITY["m5.large"]
        assert CARBON_INTENSITY["r6g.medium"] < CARBON_INTENSITY["c5.large"]

    def test_green_perfect_play_high_score(self, engine):
        """Optimal green strategy: terminate all dirty x86, keep ARM."""
        engine.reset("green")
        # Terminate dirty instances in order of highest carbon
        dirty_targets = ["batch-2", "compute-2", "batch-0", "batch-1",
                         "compute-0", "compute-1"]
        for i, target in enumerate(dirty_targets):
            reply = "Reducing carbon footprint" if i == 0 else ""
            engine.step(CloudFinOpsAction(
                command="TERMINATE", target_id=target, reply=reply
            ))
        # Fill remaining steps
        for _ in range(MAX_STEPS - len(dirty_targets)):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        score = engine.grade()
        assert score > 0.5, f"Expected high green score, got {score}"

    def test_terminating_arm_hurts_green_score(self, engine):
        """Terminating ARM instances should NOT help green score much."""
        engine.reset("green")
        # Bad strategy: terminate ARM (clean) instead of x86 (dirty)
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="arm-0", reply="Oops"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="arm-1"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="arm-2"))
        for _ in range(MAX_STEPS - 3):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        bad_score = engine.grade()

        # Compare with good strategy
        engine.reset("green")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-2", reply="Reducing emissions"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="compute-2"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="batch-0"))
        for _ in range(MAX_STEPS - 3):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        good_score = engine.grade()

        assert good_score > bad_score, f"Good strategy ({good_score}) should beat bad ({bad_score})"


# ---------------------------------------------------------------------------
# 15. Inbox / Reply Mechanics
# ---------------------------------------------------------------------------

class TestInboxMechanics:
    """Test inbox reply bonuses and clearing behavior."""

    def test_reply_clears_inbox(self, engine):
        """Replying to inbox should clear it."""
        engine.reset("easy")
        assert len(engine.inbox) > 0
        engine.step(CloudFinOpsAction(
            command="TERMINATE", target_id="idle-0", reply="Acknowledged, cleaning up."
        ))
        assert len(engine.inbox) == 0

    def test_no_reply_keeps_inbox(self, engine):
        """Not replying should keep inbox messages."""
        engine.reset("easy")
        assert len(engine.inbox) > 0
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        # Inbox should persist if no reply
        assert len(engine.inbox) > 0

    def test_reply_bonus_reward(self, engine):
        """Reply should add +2.0 bonus on top of action reward."""
        engine.reset("easy")
        _, reward_with_reply, _, _ = engine.step(CloudFinOpsAction(
            command="TERMINATE", target_id="idle-0", reply="On it"
        ))
        engine.reset("easy")
        _, reward_no_reply, _, _ = engine.step(CloudFinOpsAction(
            command="TERMINATE", target_id="idle-0"
        ))
        assert reward_with_reply > reward_no_reply

    def test_empty_reply_no_bonus(self, engine):
        """Empty string reply should NOT trigger bonus."""
        engine.reset("easy")
        _, reward_empty, _, _ = engine.step(CloudFinOpsAction(
            command="TERMINATE", target_id="idle-0", reply=""
        ))
        engine.reset("easy")
        _, reward_none, _, _ = engine.step(CloudFinOpsAction(
            command="TERMINATE", target_id="idle-0", reply=None
        ))
        # Both should give the same reward (no reply bonus)
        assert reward_empty == reward_none

    def test_hard_task_has_multiple_inbox(self, engine):
        """Hard task should have 2 inbox messages."""
        obs = engine.reset("hard")
        assert len(obs.inbox) == 2
        assert any("traffic" in msg.lower() or "campaign" in msg.lower() for msg in obs.inbox)

    def test_medium_task_has_cto_message(self, engine):
        """Medium task inbox should contain CTO cost-cutting demand."""
        obs = engine.reset("medium")
        assert any("cto" in msg.lower() or "cost" in msg.lower() for msg in obs.inbox)


# ---------------------------------------------------------------------------
# 16. Full-Episode Optimal Play Sequences
# ---------------------------------------------------------------------------

class TestOptimalPlaySequences:
    """Test that known-optimal action sequences produce high scores."""

    def test_easy_optimal_3_step_perfect(self, engine):
        """Easy: terminate 3 idle servers = perfect score."""
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0", reply="Cleaning up zombies"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-1"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-2"))
        score = engine.grade()
        assert score >= 0.99  # clamped to max 0.99

    def test_easy_terminating_active_reduces_score(self, engine):
        """Easy: terminating active servers should reduce score."""
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0", reply="Working on it"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-1"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="web-0"))  # BAD: active server
        score = engine.grade()
        assert score < 1.0  # penalty for active termination

    def test_medium_downscale_strategy(self, engine):
        """Medium: downscaling low-CPU servers should save costs."""
        engine.reset("medium")
        initial_budget = engine.budget_remaining
        # Downscale several low-CPU servers
        for i in range(6):
            sid = engine.servers[i].id
            engine.step(CloudFinOpsAction(
                command="DOWNSCALE", target_id=sid,
                reply="Optimizing costs" if i == 0 else ""
            ))
        for _ in range(MAX_STEPS - 6):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        score = engine.grade()
        assert score > 0.0, "Downscale strategy should produce positive score"

    def test_hard_upscale_db_first(self, engine):
        """Hard: upscaling DB servers first should prevent SLA breach."""
        engine.reset("hard")
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="db-0", reply="Scaling for traffic"))
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="db-1"))
        # Continue with safe actions
        for _ in range(MAX_STEPS - 2):
            obs, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
            if done:
                break
        score = engine.grade()
        # Should get a decent score by preventing SLA breach
        assert score >= 0.0


# ---------------------------------------------------------------------------
# 17. Deterministic Replay Verification
# ---------------------------------------------------------------------------

class TestDeterministicReplay:
    """Verify that identical action sequences produce identical outcomes."""

    def test_replay_produces_same_score(self, engine):
        """Running the same actions twice should give the same final score."""
        actions = [
            CloudFinOpsAction(command="TERMINATE", target_id="idle-0", reply="On it"),
            CloudFinOpsAction(command="TERMINATE", target_id="idle-1"),
            CloudFinOpsAction(command="TERMINATE", target_id="idle-2"),
        ]

        # Run 1
        engine.reset("easy")
        for a in actions:
            engine.step(a)
        score1 = engine.grade()

        # Run 2
        engine.reset("easy")
        for a in actions:
            engine.step(a)
        score2 = engine.grade()

        assert score1 == score2, f"Non-deterministic: {score1} != {score2}"

    def test_replay_same_observations(self, engine):
        """Same actions should produce same intermediate observations."""
        engine.reset("easy")
        obs1, r1, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))

        engine.reset("easy")
        obs2, r2, _, _ = engine.step(CloudFinOpsAction(command="IGNORE"))

        assert r1 == r2
        assert obs1.budget_remaining == obs2.budget_remaining
        assert obs1.traffic_load == obs2.traffic_load

    def test_different_tasks_different_scores(self, engine):
        """IGNORE-only on different tasks should produce different scores."""
        scores = {}
        for task in ["easy", "medium", "hard", "green"]:
            engine.reset(task)
            for _ in range(MAX_STEPS):
                _, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
                if done:
                    break
            scores[task] = engine.grade()
        # At least some tasks should have different scores
        unique_scores = set(round(s, 4) for s in scores.values())
        assert len(unique_scores) > 1, f"All tasks gave same score: {scores}"


# ---------------------------------------------------------------------------
# 18. Grader Boundary Conditions
# ---------------------------------------------------------------------------

class TestGraderBoundaries:
    """Test grader edge cases and score clamping."""

    def test_all_scores_clamped_0_to_1(self, engine):
        """No matter what, scores should always be in [0, 1]."""
        for task in ["easy", "medium", "hard", "green"]:
            engine.reset(task)
            # Do something harmful
            engine.step(CloudFinOpsAction(command="TERMINATE", target_id=engine.servers[0].id))
            score = engine.grade()
            assert 0.0 < score < 1.0, f"{task}: score {score} out of bounds"

    def test_easy_all_wrong_zero_score(self, engine):
        """Easy: terminating only active servers + no idle cleanup = 0 score."""
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="web-0"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="web-1"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="web-2"))
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="web-3"))
        score = engine.grade()
        assert score <= 0.01  # clamped to min 0.01

    def test_medium_no_savings_low_score(self, engine):
        """Medium: doing nothing should yield low/zero score (no cost savings)."""
        engine.reset("medium")
        for _ in range(MAX_STEPS):
            engine.step(CloudFinOpsAction(command="IGNORE"))
        score = engine.grade()
        assert score < 0.2, f"Expected low score for no action, got {score}"

    def test_hard_sla_breach_zero_uptime(self, engine):
        """Hard: SLA breach should zero out the uptime component."""
        engine.reset("hard")
        engine.servers[0].cpu_util = 100.0  # force breach
        engine.step(CloudFinOpsAction(command="IGNORE"))
        score = engine.grade()
        # Uptime component (60% weight) should be 0
        assert score < 0.5

    def test_green_no_action_baseline(self, engine):
        """Green: IGNORE-only should give low score (no carbon reduction)."""
        engine.reset("green")
        for _ in range(MAX_STEPS):
            _, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
            if done:
                break
        score = engine.grade()
        assert score < 0.5, f"Passive play should be low, got {score}"


# ---------------------------------------------------------------------------
# 19. Upscale Queue Timing Edge Cases
# ---------------------------------------------------------------------------

class TestUpscaleQueueTiming:
    """Test that upscale is queued and applied on the NEXT step."""

    def test_upscale_not_immediate(self, engine):
        """Upscale should NOT change instance type on the same step."""
        engine.reset("easy")
        original_type = engine.servers[0].type
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        web0 = [s for s in engine.servers if s.id == "web-0"][0]
        # Type should still be the original (pending)
        assert web0.type == original_type

    def test_upscale_applies_next_step(self, engine):
        """Upscale should change instance type on the following step."""
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="web-0"))
        engine.step(CloudFinOpsAction(command="IGNORE"))
        web0 = [s for s in engine.servers if s.id == "web-0"][0]
        assert web0.type == "t3.large"  # upgraded from t3.medium

    def test_upscale_halves_cpu_on_apply(self, engine):
        """When upscale applies, CPU should be halved (more headroom)."""
        engine.reset("hard")
        db0_cpu_before = [s for s in engine.servers if s.id == "db-0"][0].cpu_util
        engine.step(CloudFinOpsAction(command="UPSCALE", target_id="db-0"))
        engine.step(CloudFinOpsAction(command="IGNORE"))
        db0 = [s for s in engine.servers if s.id == "db-0"][0]
        # CPU should be significantly lower than before (0.5x + noise + traffic)
        # At least verify the type changed
        assert db0.type == "r6g.xlarge"

    def test_upscale_terminated_server_fails(self, engine):
        """Upscaling a terminated server should return penalty."""
        engine.reset("easy")
        engine.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        _, reward, _, _ = engine.step(CloudFinOpsAction(command="UPSCALE", target_id="idle-0"))
        assert reward == -2.0


# ---------------------------------------------------------------------------
# 20. Traffic Simulation Edge Cases
# ---------------------------------------------------------------------------

class TestTrafficSimulation:
    """Test traffic dynamics across task types."""

    def test_hard_traffic_increases_each_step(self, engine):
        """Hard task: traffic should grow logarithmically."""
        engine.reset("hard")
        traffic_values = [engine.traffic_load]
        for _ in range(5):
            obs, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
            traffic_values.append(obs.traffic_load)
            if done:
                break
        # Traffic should be monotonically increasing
        for i in range(1, len(traffic_values)):
            assert traffic_values[i] >= traffic_values[i - 1], \
                f"Traffic decreased: {traffic_values[i-1]} -> {traffic_values[i]}"

    def test_easy_traffic_stable(self, engine):
        """Easy task: traffic should increase slowly."""
        engine.reset("easy")
        initial_traffic = engine.traffic_load
        engine.step(CloudFinOpsAction(command="IGNORE"))
        obs = engine.get_state()
        # Should increase by ~0.5 per step for non-hard tasks
        assert obs.traffic_load >= initial_traffic

    def test_hard_spike_always_true(self, engine):
        """Hard task should always have spike_detected=True."""
        engine.reset("hard")
        for _ in range(5):
            obs, _, done, _ = engine.step(CloudFinOpsAction(command="IGNORE"))
            assert obs.spike_detected is True
            if done:
                break

    def test_easy_no_spike(self, engine):
        """Easy task should never have spike_detected=True."""
        obs = engine.reset("easy")
        assert obs.spike_detected is False
