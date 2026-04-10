# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Canonical validation suite. Each test encodes a reproducible empirical claim 
about agent capability. Tests are locked once passing and act as both regression 
guards and the evidentiary record for reported results.
"""

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run
from aintelope.analytics.analytics import (
    assert_learning_improvement,
    report_optimal_policy,
)


# @pytest.mark.skip("99% with base DQN-FC 2x2")
def test_foraging_dqn_fc_2x2(base_learning_config):
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 5,
                    "experiment": {
                        "steps": 20,
                        "episodes": 15000,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "batch_size": 200,
                    "replay_buffer_size": 7000,
                    "gamma": 0.99,
                    "agents": {
                        "agent_0": {
                            "model": "dqn_fc",
                        },
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.3},
                    },
                },
                "env_params": {
                    "goal": "reach_food",
                    "map_max": 2,
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
                        "episodes": 500,
                        "test_mode": True,
                    },
                    "analytics": {
                        "optimal_efficiency": {"min_efficiency_pct": 1.0},
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.0},
                    },
                },
            },
        },
    )
    result = run(cfg)
    report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])


# @pytest.mark.skip("89% with base DQN-FC 5x5")
def test_foraging_dqn_fc_5x5(base_learning_config):
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 5,
                    "experiment": {
                        "steps": 20,
                        "episodes": 7500,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "batch_size": 350,
                    "replay_buffer_size": 30000,
                    "gamma": 0.99,
                    "agents": {
                        "agent_0": {
                            "model": "dqn_fc",
                        },
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.3},
                    },
                },
                "env_params": {
                    "map_max": 5,
                    "goal": "reach_food",
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
                        "episodes": 500,
                        "test_mode": True,
                    },
                    "analytics": {
                        "optimal_efficiency": {"min_efficiency_pct": 0.8},
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.0},
                    },
                },
            },
        },
    )
    result = run(cfg)
    report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])


# @pytest.mark.skip("Curriculum DQN-CNN 5x5 to 13x13")
def test_generalizes_dqn(base_learning_config):
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 5,
                    "experiment": {
                        "steps": 20,
                        "episodes": 6500,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "batch_size": 550,
                    "replay_buffer_size": 30000,
                    "gamma": 0.99,
                    "agents": {
                        "agent_0": {
                            "model": "dqn_cnn",
                        },
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.3},
                    },
                },
                "env_params": {
                    "map_max": 5,
                    "render_agent_radius": 15,
                    "goal": "reach_food",
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
                        "episodes": 100,
                        "test_mode": True,
                    },
                    "analytics": {
                        "optimal_efficiency": {"min_efficiency_pct": 0.2},
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.0},
                    },
                },
                "env_params": {
                    "map_max": 13,
                    "render_agent_radius": 15,
                    "goal": "reach_food",
                },
            },
        },
    )
    result = run(cfg)
    report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])


if __name__ == "__main__":
    pytest.main([__file__])
