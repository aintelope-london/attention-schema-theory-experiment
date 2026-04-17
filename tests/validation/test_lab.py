# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Development sandbox. Tests here are works in progress — not yet validated, not part of the canonical evidence record. Graduate to test_validation.py once a test is reproducibly passing and the claim is ready to be locked.
"""

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run
from aintelope.analytics.analytics import (
    assert_learning_improvement,
    report_optimal_policy,
)


#@pytest.mark.skip(reason="ModelBased debugging pending")
def test_model_based_learns(base_learning_config):
    """main_agent with ModelBased architecture shows learning improvement."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 500,
                    },
                },
                "agent_params": {
                    "roi_mode": None,
                    "agents": {
                        "agent_0": {
                            "architecture": {
                                "action": {
                                    "type": "ModelBased",
                                    "inputs": ["dynamic", "value"],
                                },
                                "reward": {
                                    "type": "RewardInference",
                                    "inputs": ["observation"],
                                },
                                "dynamic": {
                                    "type": "NextState-NN",
                                    "inputs": ["observation"],
                                },
                                "value": {
                                    "type": "StateValue-NN",
                                    "inputs": ["observation"],
                                },
                            },
                        },
                    },
                },
                "env_params": {},
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


@pytest.mark.skip("89% with base DQN-FC 5x5")
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
                    "map_size": 5,
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


if __name__ == "__main__":
    pytest.main([__file__])
