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


@pytest.mark.skip(reason="SB3 PPO debug pending")
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
                    "agents": {
                        "agent_0": {"agent_class": "sb3_ppo_agent"},
                    },
                    "num_conv_layers": 0,
                    "ppo_n_steps": 10,
                },
                "env_params": {
                    "combine_interoception_and_vision": True,
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


# @pytest.mark.skip("100% with base DQN-FC 2x2")
def test_dqn_fc_2x2(base_learning_config):
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 5,
                    "experiment": {
                        "steps": 20,
                        "episodes": 5000,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "batch_size": 150,
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
                    "env": "savanna-safetygrid-v1",
                    "goal": "reach_food",
                    "map_max": 4,
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
                        "episodes": 500,
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


@pytest.mark.skip("89% with base DQN-FC 5x5")
def test_dqn_fc_5x5(base_learning_config):
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
                    "map_max": 7,
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


@pytest.mark.skip("86% with roi DQN-FC")
def test_dqn_fc_roi(base_learning_config):
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 5,
                    "experiment": {
                        "steps": 20,
                        "episodes": 6000,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "batch_size": 350,
                    "replay_buffer_size": 30000,
                    "gamma": 0.99,
                    "agents": {
                        "agent_0": {
                            "model": "dqn_fc_roi",
                        },
                    },
                },
                "models": {
                    "DQN": {
                        "metadata": {"greedy_until": 0.3},
                    },
                },
                "env_params": {
                    "map_max": 7,
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


@pytest.mark.skip("Curriculum DQN-CNN 5x5 to 13x13")
def test_dqn_curri(base_learning_config):
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 1,
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
                    "map_max": 7,
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


@pytest.mark.skip(reason="Test new env with 2x2")
def test_gridworld_dqn_fc_learns(base_learning_config):
    """main_agent with DQN-FC shows learning improvement on gridworld.

    Smoke test for the new gridworld environment: verifies that reward
    (food interoception via RewardInference) increases over training on
    a small fixed map. Episodes do not terminate on food contact, so the
    agent has the full step budget to improve its food-finding rate.
    """
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "trials": 3,
                    "experiment": {
                        "steps": 20,
                        "episodes": 5000,
                        "test_mode": False,
                    },
                },
                "agent_params": {
                    "batch_size": 150,
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
                    "env": "gridworld-v1",
                    "map_size": 2,
                    "observation_radius": 3,
                    "termination": "food",
                    "objects": {
                        "food": {"count": 1},
                        "predator": {"count": 0},
                    },
                },
            },
            "test": {
                "run": {
                    "experiment": {
                        "steps": 10,
                        "episodes": 500,
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
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


if __name__ == "__main__":
    pytest.main([__file__])
