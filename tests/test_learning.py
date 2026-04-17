# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Learning smoke test — bit-exact regression guard.

One training run on 2x2 foraging with dqn_fc, asserted at three tiers:
    tier 1 — gradient flows   (mean reward, early window)
    tier 2 — policy improves  (mean reward, mid window)
    tier 3 — greedy converges (test-block efficiency_pct)

The run is deterministic by seed. Any numeric drift from a locked value is a
code change that affects learning. Relock deliberately by pasting the observed
value printed on failure.

Separate from tests/validation/test_validation.py on purpose: validation encodes
empirical performance claims (threshold-based, relaxed as the system evolves);
this test locks reproducibility (exact-match, moves only on intentional relock).

Fixture is module-scoped so the three tier tests share one training run; this
is why the config is inlined rather than reusing base_test_config (function-scoped).
"""

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run


# ── Locked empirical values ───────────────────────────────────────────────────
# Populate by running once with `make tests-local`; each tier prints its
# observed value on failure. Paste the number back here and re-run.
_TIER1_MEAN_REWARD = 6.68
_TIER2_MEAN_REWARD = 8.213333333333333
_TIER3_EFFICIENCY = 31.5

_TOL = 1e-1


# ── Training windows ──────────────────────────────────────────────────────────
# Windows are episode ranges [start, end) on the train block's per-episode
# reward series. Chosen to bracket learning regimes: tier 1 in the exploration
# phase (epsilon still high), tier 2 after the policy has clearly converged.
_TIER1_WINDOW = (200, 300)
_TIER2_WINDOW = (1200, 1500)


# ── Smoke config ──────────────────────────────────────────────────────────────
# greedy_until=0.5 pushes epsilon to zero at episode 750 (half of 1500 train),
# so tier 2's window (1200–1500) observes a fully-greedy policy.

_SMOKE_CFG = {
    "train": {
        "run": {
            "seed": 0,
            "trials": 1,
            "max_workers": 1,
            "write_outputs": False,
            "experiment": {"episodes": 1500, "steps": 20, "test_mode": False},
        },
        "agent_params": {
            "batch_size": 350,
            "replay_buffer_size": 30000,
            "gamma": 0.99,
            "learning_rate": 0.0005,
            "agents": {"agent_0": {"model": "dqn_fc"}},
        },
        "models": {"DQN": {"metadata": {"greedy_until": 0.5}}},
        "env_params": {"map_size": 2, "goal": "reach_food"},
    },
    "test": {
        "run": {
            "experiment": {"episodes": 100, "steps": 10, "test_mode": True},
            "analytics": {"optimal_efficiency": {"min_efficiency_pct": 0.0}},
        },
        "models": {"DQN": {"metadata": {"greedy_until": 0.0}}},
        "env_params": {"map_size": 2, "goal": "reach_food"},
    },
}


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def smoke_result():
    """One training run shared across all tier assertions."""
    return run(OmegaConf.create(_SMOKE_CFG))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _window_mean_reward(events, start, end):
    """Mean per-episode total reward over episode range [start, end)."""
    ep_rewards = events.groupby("Episode")["Reward"].sum().sort_index()
    return float(ep_rewards.iloc[start:end].mean())


def _assert_locked(observed, locked, tier, var_name):
    if locked is None:
        pytest.fail(
            f"{tier}: locked value not set.\n"
            f"  Observed: {observed!r}\n"
            f"  Lock it by editing this file:\n"
            f"    {var_name} = {observed!r}"
        )
    assert observed == pytest.approx(locked, abs=_TOL), (
        f"{tier} drifted: observed={observed!r}, locked={locked!r}, "
        f"delta={observed - locked:+.9f}. "
        f"If intentional (and upward), relock. If downward, investigate."
    )


# ── Tier tests ────────────────────────────────────────────────────────────────


def test_tier1_gradient_flows(smoke_result):
    events = smoke_result["results"]["train"]["events"]
    observed = _window_mean_reward(events, *_TIER1_WINDOW)
    _assert_locked(observed, _TIER1_MEAN_REWARD, "tier1", "_TIER1_MEAN_REWARD")


def test_tier2_policy_improves(smoke_result):
    events = smoke_result["results"]["train"]["events"]
    observed = _window_mean_reward(events, *_TIER2_WINDOW)
    _assert_locked(observed, _TIER2_MEAN_REWARD, "tier2", "_TIER2_MEAN_REWARD")


def test_tier3_greedy_converges(smoke_result):
    observed = float(
        smoke_result["analytics"]["optimal_efficiency"]["test"]["efficiency_pct"]
    )
    _assert_locked(observed, _TIER3_EFFICIENCY, "tier3", "_TIER3_EFFICIENCY")


if __name__ == "__main__":
    pytest.main([__file__])
