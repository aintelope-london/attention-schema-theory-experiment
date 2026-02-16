# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os

import pytest
from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers

register_resolvers()


@pytest.fixture
def base_test_config():
    """Minimal params diff for fast test execution.
    Merged over default_config.yaml by run_experiments().
    """
    return OmegaConf.create(
        {
            "run": {
                "episodes": 1,
                "save_logs": False,
            },
            "env_params": {
                "num_iters": 10,
                "map_max": 5,
            },
        }
    )


@pytest.fixture
def base_env_cfg():
    """Full cfg for direct environment construction in tests.
    Loads default_config.yaml with minimal test overrides.
    """
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    return OmegaConf.merge(
        cfg,
        {
            "env_params": {
                "num_iters": 10,
                "map_max": 5,
            },
        },
    )


@pytest.fixture
def base_env_params(base_env_cfg):
    """Flat env_params dict for tests that need raw params."""
    return dict(base_env_cfg.env_params)


@pytest.fixture
def learning_config(base_test_config):
    """Two-block config: train then test."""
    train_block = OmegaConf.merge(
        base_test_config,
        {"run": {"episodes": 50}, "env_params": {"num_iters": 100}},
    )
    test_block = OmegaConf.merge(
        base_test_config,
        {"run": {"episodes": 10, "test_mode": True}, "env_params": {"num_iters": 100}},
    )
    return OmegaConf.create({"train": train_block, "test": test_block})
