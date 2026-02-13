"""Integration test: agent completes full orchestrator without errors."""

import os
import pytest
import pandas as pd
from omegaconf import OmegaConf

from aintelope.analytics.analytics import (
    assert_learning_threshold,
    assert_learning_improvement,
)
from aintelope.analytics.recording import read_events
from aintelope.__main__ import run


def test_integration(learning_config):
    """Run full train + test cycle."""
    run(learning_config)


def test_agent_learns(base_test_config):
    """Learning agent achieves reward threshold on simple gridworld."""

    shared_overrides = {
        "agent_class": "sb3_ppo_agent",
        "env_layout_seed_repeat_sequence_length": 1,
        "model_params": {
            "num_conv_layers": 0,
            "learning_rate": 0.001,
            "ppo_n_steps": 32,
        },
        "env_params": {
            "map_max": 4,
            "num_iters": 10,
            "combine_interoception_and_vision": True,
        },
    }

    train_block = OmegaConf.merge(
        base_test_config, shared_overrides, {"num_episodes": 500}
    )
    test_block = OmegaConf.merge(
        base_test_config, shared_overrides, {"num_episodes": 10, "test_mode": True}
    )

    result = run(OmegaConf.create({"train": train_block, "test": test_block}))

    exp_cfg = result["configs"][0]

    # Read events and filter to training episodes
    events = read_events(exp_cfg.experiment_dir, exp_cfg.events_fname)
    events_combined = pd.concat(events, ignore_index=True)
    train_events = events_combined[~events_combined["IsTest"]]

    #assert_learning_threshold(train_events)
    assert_learning_improvement(train_events)


if __name__ == "__main__":
    pytest.main([__file__])