# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import io
import os
import sys
import threading
import time
import datetime
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
    t0 = time.monotonic()
    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    # Buffer stdout from the very start so init_config output is captured too.
    _orig_stdout = sys.stdout
    _early_buf = io.StringIO()
    sys.stdout = TeeStream(_orig_stdout, _early_buf)

    first_cfg = init_config(config)

    log_file = None
    if first_cfg.run.write_outputs:
        outputs_dir = Path(first_cfg.run.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(outputs_dir / "stdout.txt", "w")
        log_file.write(_early_buf.getvalue())
        sys.stdout = TeeStream(_orig_stdout, log_file)

    _early_buf.close()

    try:
        gpu_thread = threading.Thread(target=select_gpu)
        gpu_thread.start()

        from aintelope.orchestrator import run_experiments

        gpu_thread.join()

        result = run_experiments(first_cfg, config)
    finally:
        elapsed = time.monotonic() - t0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        runtime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        if first_cfg.run.write_outputs:
            sys.stdout = _orig_stdout
            log_file.close()

            stdout_path = outputs_dir / "stdout.txt"
            report_path = outputs_dir / "report.txt"

            stdout_content = stdout_path.read_text() if stdout_path.exists() else ""
            report_content = report_path.read_text() if report_path.exists() else ""

            combined = (
                f"Started:  {start_str}\n"
                f"Runtime:  {runtime_str}\n"
                f"\n"
                f"{stdout_content}"
                f"\n{report_content}"
            )
            report_path.write_text(combined)
            stdout_path.unlink(missing_ok=True)
        else:
            sys.stdout = _orig_stdout

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
