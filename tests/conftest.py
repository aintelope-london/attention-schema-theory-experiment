# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import pathlib
from typing import Dict, Tuple, Union

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers

register_resolvers()


def constants() -> DictConfig:
    constants_dict = {
        "PROJECT": "aintelope",
        "BASELINE": "run-training-baseline",
    }
    return OmegaConf.create(constants_dict)


@pytest.fixture
def base_test_config() -> Union[DictConfig, ListConfig]:
    """Base test configuration built on default_config.yaml.

    Loads full config to satisfy external env dependencies,
    then overrides for fast test execution.
    """
    full_params = OmegaConf.load(
        os.path.join("aintelope", "config", "default_config.yaml")
    )

    full_params.hparams.unit_test_mode = True
    full_params.hparams.num_episodes = min(5, full_params.hparams.num_episodes)
    full_params.hparams.env_params.num_iters = min(
        50, full_params.hparams.env_params.num_iters
    )
    full_params.hparams.warm_start_steps = min(10, full_params.hparams.warm_start_steps)

    return full_params


@pytest.fixture
def dqn_learning_config(base_test_config) -> Union[DictConfig, ListConfig]:
    """Config for baseline ML learning tests (DQN).

    Extends base_test_config with settings suitable for
    verifying that learning actually occurs.
    """
    config = base_test_config.copy()

    config.hparams.num_episodes = 50
    config.hparams.env_params.num_iters = 100

    return config
