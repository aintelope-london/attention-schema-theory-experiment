# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import pytest
from omegaconf import OmegaConf

from aintelope.nonpipeline import aintelope_main


def test_training_pipeline_main():
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "config_experiment.yaml"))

    OmegaConf.update(cfg, "hparams.env", "savanna-safetygrid-sequential-v1")
    OmegaConf.update(
        cfg,
        "hparams.env_entry_point",
        "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv",
    )
    OmegaConf.update(cfg, "hparams.env_type", "zoo")
    OmegaConf.update(cfg, "hparams.unit_test_mode", True)
    OmegaConf.update(cfg, "hparams.num_episodes", 5)
    OmegaConf.update(cfg, "hparams.test_episodes", 1)
    OmegaConf.update(cfg, "hparams.env_params.num_iters", 50)
    OmegaConf.update(cfg, "hparams.warm_start_steps", 10)

    aintelope_main(cfg)


@pytest.mark.parametrize("execution_number", range(1))
def test_training_pipeline_main_with_dead_agents(execution_number):
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "config_experiment.yaml"))

    OmegaConf.update(cfg, "hparams.env", "savanna-safetygrid-sequential-v1")
    OmegaConf.update(
        cfg,
        "hparams.env_entry_point",
        "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv",
    )
    OmegaConf.update(cfg, "hparams.env_type", "zoo")
    OmegaConf.update(cfg, "hparams.env_params.seed", execution_number)
    OmegaConf.update(cfg, "hparams.env_params.test_death", True)
    OmegaConf.update(cfg, "hparams.unit_test_mode", True)
    OmegaConf.update(cfg, "hparams.num_episodes", 5)
    OmegaConf.update(cfg, "hparams.test_episodes", 1)
    OmegaConf.update(cfg, "hparams.env_params.num_iters", 50)
    OmegaConf.update(cfg, "hparams.warm_start_steps", 10)

    aintelope_main(cfg)


def test_training_pipeline_baseline():
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "config_experiment.yaml"))

    OmegaConf.update(cfg, "hparams.env", "savanna-safetygrid-sequential-v1")
    OmegaConf.update(
        cfg,
        "hparams.env_entry_point",
        "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv",
    )
    OmegaConf.update(cfg, "hparams.env_type", "zoo")
    OmegaConf.update(cfg, "hparams.agent_class", "q_agent")
    OmegaConf.update(cfg, "hparams.unit_test_mode", True)
    OmegaConf.update(cfg, "hparams.num_episodes", 5)
    OmegaConf.update(cfg, "hparams.test_episodes", 1)
    OmegaConf.update(cfg, "hparams.env_params.num_iters", 50)
    OmegaConf.update(cfg, "hparams.warm_start_steps", 10)

    aintelope_main(cfg)


@pytest.mark.parametrize("execution_number", range(1))
def test_training_pipeline_baseline_with_dead_agents(execution_number):
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "config_experiment.yaml"))

    OmegaConf.update(cfg, "hparams.env", "savanna-safetygrid-sequential-v1")
    OmegaConf.update(
        cfg,
        "hparams.env_entry_point",
        "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv",
    )
    OmegaConf.update(cfg, "hparams.env_type", "zoo")
    OmegaConf.update(cfg, "hparams.agent_class", "q_agent")
    OmegaConf.update(cfg, "hparams.env_params.seed", execution_number)
    OmegaConf.update(cfg, "hparams.env_params.test_death", True)
    OmegaConf.update(cfg, "hparams.unit_test_mode", True)
    OmegaConf.update(cfg, "hparams.num_episodes", 5)
    OmegaConf.update(cfg, "hparams.test_episodes", 1)
    OmegaConf.update(cfg, "hparams.env_params.num_iters", 50)
    OmegaConf.update(cfg, "hparams.warm_start_steps", 10)

    aintelope_main(cfg)


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])
