# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Canonical validation suite. Each test encodes a reproducible empirical claim
about agent capability. Tests are locked once passing and act as both regression
guards and the evidentiary record for reported results.

Structure:
  - Scenarios define the environment and test target (env_params, analytics).
  - Agents define the training budget and agent_params.
  - Each scenario is run once per agent via parametrize.
"""

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run
from aintelope.analytics.analytics import report_optimal_policy


# ── Agents ─────────────────────────────────────────────────────────────────────

_AGENTS = {
    "dqn_fc": {
        "train": {
            "run": {
                "trials": 5,
                "experiment": {"steps": 20, "episodes": 10000, "test_mode": False},
            },
            "agent_params": {
                "batch_size": 200,
                "replay_buffer_size": 7000,
                "gamma": 0.99,
                "agents": {"agent_0": {"model": "dqn_fc"}},
            },
            "models": {"DQN": {"metadata": {"greedy_until": 0.3}}},
        },
        "test": {
            "models": {"DQN": {"metadata": {"greedy_until": 0.0}}},
        },
    },
    "dqn_cnn": {
        "train": {
            "run": {
                "trials": 5,
                "experiment": {"steps": 20, "episodes": 10000, "test_mode": False},
            },
            "agent_params": {
                "batch_size": 200,
                "replay_buffer_size": 30000,
                "gamma": 0.99,
                "agents": {"agent_0": {"model": "dqn_cnn"}},
            },
            "models": {"DQN": {"metadata": {"greedy_until": 0.3}}},
        },
        "test": {
            "models": {"DQN": {"metadata": {"greedy_until": 0.0}}},
        },
    },
    "sb3_ppo": {
        "train": {
            "run": {
                "trials": 1,
                "max_workers": 1,
                "experiment": {"steps": 20, "episodes": 10000, "test_mode": False},
            },
            "agent_params": {
                "agents": {"agent_0": {"agent_class": "sb3_agent"}},
                "algorithm": "ppo",
                "num_conv_layers": 0,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "ppo_n_steps": 128,
            },
        },
        "test": {},
    },
    "sb3_dqn": {
        "train": {
            "run": {
                "trials": 1,
                "max_workers": 1,
                "experiment": {"steps": 20, "episodes": 10000, "test_mode": False},
            },
            "agent_params": {
                "agents": {"agent_0": {"agent_class": "sb3_agent"}},
                "algorithm": "dqn",
                "num_conv_layers": 0,
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "batch_size": 64,
                "replay_buffer_size": 10000,
            },
        },
        "test": {},
    },
    "sb3_a2c": {
        "train": {
            "run": {
                "trials": 1,
                "max_workers": 1,
                "experiment": {"steps": 20, "episodes": 10000, "test_mode": False},
            },
            "agent_params": {
                "agents": {"agent_0": {"agent_class": "sb3_agent"}},
                "algorithm": "a2c",
                "num_conv_layers": 0,
                "learning_rate": 3e-4,
                "gamma": 0.99,
            },
        },
        "test": {},
    },
}


# ── Scenarios ──────────────────────────────────────────────────────────────────


def _scenario_2x2():
    return {
        "train": {
            "env_params": {"map_size": 2, "goal": "reach_food"},
        },
        "test": {
            "run": {
                "experiment": {"steps": 10, "episodes": 500, "test_mode": True},
                "analytics": {"optimal_efficiency": {"min_efficiency_pct": 1.0}},
            },
        },
    }


def _scenario_5x5():
    return {
        "train": {
            "env_params": {"map_size": 5, "goal": "reach_food"},
        },
        "test": {
            "run": {
                "experiment": {"steps": 10, "episodes": 500, "test_mode": True},
                "analytics": {"optimal_efficiency": {"min_efficiency_pct": 0.8}},
            },
        },
    }


def _scenario_generalize_5x5_to_13x13():
    return {
        "train": {
            "env_params": {
                "map_size": 5,
                "goal": "reach_food",
                "observation_radius": 13,
            },
        },
        "test": {
            "run": {
                "experiment": {"steps": 10, "episodes": 500, "test_mode": True},
                "analytics": {"optimal_efficiency": {"min_efficiency_pct": 0.2}},
            },
            "env_params": {"map_size": 13, "goal": "reach_food"},
        },
    }


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("agent", _AGENTS.values(), ids=_AGENTS.keys())
def test_foraging_2x2(base_learning_config, agent):
    cfg = OmegaConf.merge(base_learning_config, agent, _scenario_2x2())
    result = run(cfg)
    report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])


@pytest.mark.parametrize("agent", _AGENTS.values(), ids=_AGENTS.keys())
def test_foraging_5x5(base_learning_config, agent):
    cfg = OmegaConf.merge(base_learning_config, agent, _scenario_5x5())
    result = run(cfg)
    report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])


@pytest.mark.parametrize("agent", _AGENTS.values(), ids=_AGENTS.keys())
def test_generalizes_5x5_to_13x13(base_learning_config, agent):
    cfg = OmegaConf.merge(
        base_learning_config, agent, _scenario_generalize_5x5_to_13x13()
    )
    result = run(cfg)
    report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])


if __name__ == "__main__":
    pytest.main([__file__])
