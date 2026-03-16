# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run
from aintelope.analytics.analytics import (
    assert_learning_improvement,
    report_optimal_policy,
)


@pytest.mark.skip
def test_sb3_ppo_learns(base_learning_config):
    """SB3 PPO agent shows learning improvement over training."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 300,
                    },
                },
                "agent_params": {
                    "agent_0": {"agent_class": "sb3_ppo_agent"},
                    "num_conv_layers": 0,
                    "learning_rate": 0.002,
                    "ppo_n_steps": 10,
                },
                "env_params": {
                    "map_max": 4,
                    "combine_interoception_and_vision": True,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


@pytest.mark.skip
def test_dqn_learns(base_learning_config):
    """main_agent with DQN, roi_mode=null (vestigial ROI channel, no cone)."""
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
                    "learning_rate": 0.002,
                },
                "env_params": {
                    "map_max": 4,
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


@pytest.mark.skip
def test_dqn_roi_learns(base_learning_config):
    """main_agent with DQN + active ROI cone."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 1000,
                    },
                },
                "agent_params": {
                    "roi_mode": "cone",
                    "learning_rate": 0.002,
                },
                "env_params": {
                    "map_max": 4,
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


@pytest.mark.skip(reason="ModelBased debugging pending")
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
                    "learning_rate": 0.002,
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
                "env_params": {
                    "map_max": 4,
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


def test_main_agent_dqn_optimal(base_learning_config):
    """DQN agent reaches near-optimal policy on simple scenario.

    Two-block run: train block builds the policy, test block measures it
    at zero epsilon (pure exploitation) so efficiency is not noise-floored.
    """
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 200,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "roi_mode": None,
                    "learning_rate": 0.002,
                },
                "env_params": {
                    "map_max": 4,
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 50,
                        "test_mode": True,
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
