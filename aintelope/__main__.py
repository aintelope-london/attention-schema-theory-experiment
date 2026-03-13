import os
import sys
import threading
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import (
    TeeStream,
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
        from aintelope.gui.config_viewer import run_gui

        config = run_gui(config)
        if config is None:
            print("GUI cancelled.")
            return

    first_cfg = init_config(config)
    if first_cfg.run.write_outputs:
        outputs_dir = Path(first_cfg.run.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(outputs_dir / "stdout.txt", "w")
        sys.stdout = TeeStream(sys.__stdout__, log_file)

    try:
        gpu_thread = threading.Thread(target=select_gpu)
        gpu_thread.start()

        from aintelope.orchestrator import run_experiments

        gpu_thread.join()

        result = run_experiments(first_cfg, config)
    finally:
        if first_cfg.run.write_outputs:
            sys.stdout = sys.__stdout__
            log_file.close()

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
