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


@pytest.mark.skip(reason="SB3 PPO shows learning improvement on 4x4 map, 300 episodes")
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
                    "ppo_n_steps": 10,
                },
                "env_params": {
                    "combine_interoception_and_vision": True,
                    "env_layout_seed_repeat_sequence_length": 12,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


@pytest.mark.skip(
    reason="DQN (no ROI) shows learning improvement on 4x4 map, 500 episodes"
)
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
                },
                "env_params": {
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 12,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


@pytest.mark.skip(
    reason="DQN + ROI cone shows learning improvement on 4x4 map, 1000 episodes"
)
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
                },
                "env_params": {
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 12,
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
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 12,
                },
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


# @pytest.mark.skip()
def test_main_agent_dqn_optimal(base_learning_config):
    """DQN (no ROI) reaches near-optimal food-finding on 4x4 map.

    Two-block run: train block builds the policy, test block measures it
    at zero epsilon (pure exploitation) so efficiency is not noise-floored.
    """
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 20,
                        "episodes": 4000,
                        "test_mode": False,
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.3},
                    },
                },
                "agent_params": {
                    "roi_mode": None,
                },
                "env_params": {
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 24,
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
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


@pytest.mark.skip(reason="DQN + ROI cone reaches near-optimal food-finding on 5x5 map")
def test_roi_agent_dqn_optimal(base_learning_config):
    """DQN + ROI cone reaches near-optimal policy on simple scenario.

    Two-block run: train block builds the policy, test block measures it
    at zero epsilon (pure exploitation) so efficiency is not noise-floored.
    """
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 10,
                        "episodes": 7000,
                        "test_mode": False,
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.1},
                    },
                },
                "agent_params": {
                    "roi_mode": "cone",
                },
                "env_params": {
                    "map_max": 5,
                    "combine_interoception_and_vision": False,
                    "env_layout_seed_repeat_sequence_length": 12,
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
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
