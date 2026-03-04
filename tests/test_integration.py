"""Integration test: full orchestrator cycle."""

import pytest
import pandas as pd
from omegaconf import OmegaConf
from aintelope.analytics.analytics import assert_learning_improvement
from aintelope.__main__ import run


def test_integration(base_test_config):
    """Run single-block config through orchestrator without errors."""
    run(base_test_config)


def test_agent_learns(base_test_config):
    """SB3 agent shows learning improvement over training."""
    learning = OmegaConf.merge(
        base_test_config,
        {
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 300,
                        "write_outputs": False,
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
    result = run(learning)
    events = pd.concat(result["events"], ignore_index=True)
    assert_learning_improvement(events[~events["IsTest"]])


def test_main_agent_model_free_learns(base_test_config):
    """main_agent with DQN architecture shows learning improvement over training."""
    cfg = OmegaConf.merge(
        base_test_config,
        {
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 500,
                        "write_outputs": False,
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
    events = pd.concat(result["events"], ignore_index=True)
    assert_learning_improvement(events[~events["IsTest"]])

#test_ <- add back when ready to do this
def main_agent_model_based_learns(base_test_config):
    """main_agent with ModelBased architecture shows learning improvement over training."""
    cfg = OmegaConf.merge(
        base_test_config,
        {
            "test": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 300,
                        "write_outputs": False,
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
    events = pd.concat(result["events"], ignore_index=True)
    assert_learning_improvement(events[~events["IsTest"]])
