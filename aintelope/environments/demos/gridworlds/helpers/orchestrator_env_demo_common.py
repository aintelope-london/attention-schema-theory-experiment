# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

from __future__ import absolute_import, division, print_function

import os
import pathlib

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers

from aintelope.environments.ai_safety_gridworlds.aintelope_savanna import *
from ai_safety_gridworlds.environments.shared.safety_game_moma import override_flags

from aintelope.environments.savanna_safetygrid import GridworldZooBaseEnv
from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import (
    GridworldZooParallelEnv,
)


class SavannaGridworldDemoEnv(GridworldZooBaseEnv, GridworldZooParallelEnv):
    def __init__(self, cfg: Optional[Dict] = None):
        env_params = cfg.env_params
        if env_params is None:
            env_params = {}
        GridworldZooBaseEnv.__init__(self, cfg)
        GridworldZooParallelEnv.__init__(self, **self.super_initargs)


def main(orchestrator_env_conf_name):
    try:
        register_resolvers()

        experiment_cfg = OmegaConf.load(
            os.path.join("aintelope", "config", "default_config.yaml")
        )

        if orchestrator_env_conf_name is not None:
            orchestrator_cfg = OmegaConf.load(
                os.path.join("aintelope", "config", "example_config.yaml")
            )

            OmegaConf.update(
                experiment_cfg, "experiment_name", orchestrator_env_conf_name
            )
            OmegaConf.update(
                experiment_cfg,
                "hparams",
                orchestrator_cfg[orchestrator_env_conf_name],
                force_add=True,
            )

        FLAGS = define_flags()
        zoo_env = SavannaGridworldDemoEnv(experiment_cfg)
        gridworlds_env = zoo_env._env

        for episode_no in range(0, 1000):
            gridworlds_env.reset()
            ui = safety_ui_ex.make_human_curses_ui_with_noop_keys(
                GAME_BG_COLOURS, GAME_FG_COLOURS, noop_keys=FLAGS.noops
            )
            ui.play(gridworlds_env)

    except Exception as ex:
        print(ex)
        print(traceback.format_exc())
