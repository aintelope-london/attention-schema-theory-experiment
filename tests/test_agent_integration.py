"""Integration test: agent completes full pipeline without errors."""

import sys
import os
import pytest

from omegaconf import OmegaConf
from aintelope.analytics.analytics import assert_learning_threshold
from aintelope.__main__ import run


def test_agent_completes_pipeline():
    """Agent runs full train + test cycle without errors."""
    run("config_tests.yaml")

def test_agent_learns(base_test_config):
    """Learning agent achieves reward threshold on simple gridworld."""

    # Minimal learning scenario overrides
    learning_overrides = {
        "hparams": {
            "agent_class": "sb3_ppo_agent",
            "num_episodes": 5,
            "test_episodes": 5,
            "show_plot": False,
            "do_not_enforce_checkpoint_file_existence_during_test": True,
            "model_params": {
                "num_conv_layers": 0,  # use MLP, not CNN
            },
            "env_params": {
                "map_max": 5,
                "num_iters": 10,
                "combine_interoception_and_vision": True,
            },
        }
    }
    cfg = OmegaConf.merge(base_test_config, OmegaConf.create(learning_overrides))

    test_summaries = run(cfg)

    REWARD_THRESHOLD = 5.0  # TODO: derive analytically
    assert_learning_threshold(test_summaries[0], REWARD_THRESHOLD)

if __name__ == "__main__":  # and os.name == "nt":
    pytest.main([__file__])  # run tests only in this file
