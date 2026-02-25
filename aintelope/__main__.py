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
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
)


def run(config: Union[str, DictConfig] = "default_config.yaml", gui: bool = False):
    """Single entrypoint for the whole project.

    Args:
        config: Either a filename (relative to aintelope/config/) or a DictConfig.
        gui: If True, launch config GUI before run, and results viewer after.

    Usage:
        python -m aintelope --gui
        python -m aintelope custom_config.yaml
        In tests: run(my_dictconfig)
    """
    register_resolvers()

    if sys.gettrace() is None:
        set_priorities()

    set_memory_limits()

    if isinstance(config, str):
        config = OmegaConf.load(os.path.join("aintelope", "config", config))

    if gui:
        from aintelope.gui.config_viewer import run_gui

        config = run_gui(config)
        if config is None:
            print("GUI cancelled.")
            return

    # Start GPU init after GUI — heavy imports happen below, not before
    gpu_thread = threading.Thread(target=select_gpu)
    gpu_thread.start()

    from aintelope.orchestrator import run_experiments

    gpu_thread.join()

    result = run_experiments(config)

    if gui:
        from aintelope.gui.results_viewer import run_results_viewer

        run_results_viewer()

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
        "--results",
        action="store_true",
        help="Launch results viewer standalone",
    )
    args = parser.parse_args()

    if args.results:
        from aintelope.gui.results_viewer import run_results_viewer

        run_results_viewer()
    else:
        run(args.config, gui=args.gui)
