# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Learning diagnostics tests.

Run via: make tests-learning
Not collected by the default pytest sweep or CI.
Writes outputs/ for post-run inspection of learning curves and reports.
"""

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run
from aintelope.analytics.analytics import assert_learning_improvement


def test_sb3_ppo_learns(base_learning_config):
    """SB3 PPO agent shows learning improvement over training."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 300,
                    },
                },
                "agent_params": {
                    "agent_0": {
                        "agent_class": "sb3_ppo_agent",
                    },
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
    assert_learning_improvement(result["analytics"])


def test_main_agent_dqn_learns(base_learning_config):
    """main_agent with DQN architecture shows learning improvement over training."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 500,
                    },
                },
                "agent_params": {
                    "agent_0": {
                        "agent_class": "main_agent",
                        "architecture": {
                            "action": {"type": "DQN", "inputs": ["q_net"]},
                            "reward": {
                                "type": "RewardInference",
                                "inputs": ["observation"],
                            },
                            "q_net": {"type": "DQN-NN", "inputs": ["observation"]},
                        },
                    },
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
    assert_learning_improvement(result["analytics"])


def test_main_agent_model_based_learns(base_learning_config):
    """main_agent with ModelBased architecture shows learning improvement over training."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 500,
                    },
                },
                "agent_params": {
                    "agent_0": {
                        "agent_class": "main_agent",
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
    assert_learning_improvement(result["analytics"])