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
            "train": {
                "run": {"episodes": 300},
                "agent_params": {
                    "agent_0": {
                        "agent_class": "sb3_ppo_agent",
                    },
                    "num_conv_layers": 0,
                    "learning_rate": 0.002,
                    "ppo_n_steps": 32,
                },
                "env_params": {
                    "map_max": 4,
                    "num_iters": 10,
                    "combine_interoception_and_vision": True,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
            "test": {
                "run": {"episodes": 10, "test_mode": True},
                "agent_params": {
                    "agent_0": {
                        "agent_class": "sb3_ppo_agent",
                    },
                    "num_conv_layers": 0,
                    "learning_rate": 0.001,
                    "ppo_n_steps": 32,
                },
                "env_params": {
                    "map_max": 4,
                    "num_iters": 10,
                    "combine_interoception_and_vision": True,
                    "env_layout_seed_repeat_sequence_length": 5,
                },
            },
        },
    )
    result = run(learning)
    events = pd.concat(result["events"], ignore_index=True)
    assert_learning_improvement(events[~events["IsTest"]])
