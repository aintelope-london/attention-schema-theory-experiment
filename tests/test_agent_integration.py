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
from tests.conftest import as_orchestrator


def test_agent_completes_orchestrator(learning_config):
    """Agent runs full train + test cycle without errors."""
    run(as_orchestrator(learning_config))


def test_agent_learns(base_test_config):
    """Learning agent achieves reward threshold on simple gridworld."""

    # Minimal learning scenario overrides
    learning_overrides = {
        "agent_class": "sb3_ppo_agent",
        "num_episodes": 500,
        "test_episodes": 10,
        "show_plot": False,
        "do_not_enforce_checkpoint_file_existence_during_test": True,
        "env_layout_seed_repeat_sequence_length": 1,
        "model_params": {
            "num_conv_layers": 0,  # use MLP, not CNN
            "learning_rate": 0.001,
            "ppo_n_steps": 32,
        },
        "env_params": {
            "map_max": 4,
            "num_iters": 10,
            "combine_interoception_and_vision": True,
        },
    }
    cfg = OmegaConf.merge(base_test_config, OmegaConf.create(learning_overrides))

    result = run(as_orchestrator(cfg))

    # Get config and summary for first experiment
    exp_cfg = result["configs"][0]
    summary = result["summaries"][0]

    # Build paths to events file
    experiment_dir = exp_cfg.experiment_dir
    events_fname = exp_cfg.events_fname

    # Read events and filter to training episodes
    events = read_events(experiment_dir, events_fname)
    events_combined = pd.concat(events, ignore_index=True)
    train_events = events_combined[~events_combined["IsTest"]]

    # Assert learning improvement from early to late training
    assert_learning_improvement(train_events)


if __name__ == "__main__":
    pytest.main([__file__])
