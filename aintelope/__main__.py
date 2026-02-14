# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import sys
from typing import Union

from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import (
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
)
from aintelope.orchestrator import run_experiments
from aintelope.gui.results_window import run_results


def run(
    config: Union[str, DictConfig] = "default_config.yaml",
    gui: bool = False,
    results: bool = False,
):
    """Single entrypoint for the whole project.

    Args:
        config: Either a filename (relative to aintelope/config/) or a DictConfig.
        gui: If True, launch GUI for orchestrator configuration.

    Usage:
        python -m aintelope --gui
        python -m aintelope custom_config.yaml
        In tests: run(my_dictconfig)
    """
    register_resolvers()

    # Do not set low priority while debugging.
    # Unit tests also set sys.gettrace() to not-None.
    if sys.gettrace() is None:
        set_priorities()

    set_memory_limits()

    # Need to choose GPU early before torch fully starts up.
    select_gpu()

    if isinstance(config, str):
        config = OmegaConf.load(os.path.join("aintelope", "config", config))

    if gui:
        from aintelope.gui.main_window import run_gui

        config = run_gui(config)
        if config is None:
            print("GUI cancelled.")
            return

    result = run_experiments(config)

    if results:
        run_results()

    return result


def gui_main():
    """Entry point for aintelope-gui console script."""
    run(gui=True)


def results_main():
    """Entry point for aintelope-results console script."""
    run(results=True)


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
        help="Launch GUI for orchestrator configuration",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Launch results viewer",
    )
    args = parser.parse_args()
    run(args.config, gui=args.gui, results=args.results)
