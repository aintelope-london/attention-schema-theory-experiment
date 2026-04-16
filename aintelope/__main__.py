# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import os
import sys
import threading
from typing import Union

from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import (
    init_config,
    select_gpu,
    set_memory_limits,
    set_priorities,
)


def run(config: Union[str, DictConfig] = "default_config.yaml", gui: bool = False):
    if sys.gettrace() is None:
        set_priorities()

    set_memory_limits()

    if isinstance(config, str):
        config = OmegaConf.load(os.path.join("aintelope", "config", config))

    if gui:
        from aintelope.gui.view import run_gui

        config = run_gui(config)
        if config is None:
            print("GUI cancelled.")
            return

    first_cfg = init_config(config)

    gpu_thread = threading.Thread(target=select_gpu)
    gpu_thread.start()

    from aintelope.orchestrator import run_experiments

    gpu_thread.join()

    result = run_experiments(first_cfg, config)

    if gui:
        from aintelope.gui.view import run_gui

        run_gui(first_cfg, initial_tab="results")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="aintelope")
    parser.add_argument(
        "config",
        nargs="?",
        default="default_config.yaml",
        help="Config filename in aintelope/config/",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI for config editing, then results after run",
    )
    parser.add_argument(
        "--search",
        metavar="FILENAME",
        help="Run hyperparameter search with the given search config",
    )
    args = parser.parse_args()

    if args.search:
        from aintelope.utils.param_search import run_search

        run_search(args.search)
    else:
        run(args.config, gui=args.gui)