# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from aintelope.config.config_utils import (
    archive_code,
    set_console_title,
    prepare_experiment_cfg,
    to_picklable,
    from_picklable,
)
from aintelope.analytics.diagnostics import collector
from aintelope.experiment import run_experiment
from aintelope.utils.seeding import set_global_seeds
from aintelope.utils.progress import ProgressReporter
from aintelope.utils.concurrency import find_workers
from aintelope.analytics.analytics import analyze
from aintelope.analytics.recording import write_results, write_csv, write_cfg


def run_trial(cfg_dict, main_config_dict, i_trial):
    """Run all experiment blocks for a single trial.

    Returns {block_name: {events, states, learning_df, performance_df, manifesto, cfg_dict}}.
    Args must be dicts for multiprocessing pickling.
    """
    cfg = from_picklable(cfg_dict)
    main_config = from_picklable(main_config_dict)

    trial_seed = cfg.run.seed + i_trial * 10000
    set_global_seeds(trial_seed)

    trial_results = {}
    for experiment_name in main_config:
        experiment_cfg = prepare_experiment_cfg(
            cfg, main_config[experiment_name], experiment_name, trial_seed
        )
        reporter = ProgressReporter(["episode"], on_update=None)
        result = run_experiment(experiment_cfg, i_trial=i_trial, reporter=reporter)
        trial_results[experiment_name] = {
            "events": result["events"],
            "states": result["states"],
            "learning_df": result["learning_df"],
            "performance_df": result["performance_df"],
            "manifesto": result["manifesto"],
            "cfg_dict": to_picklable(experiment_cfg),
        }
    return trial_results


def run_experiments(cfg, main_config):
    """Main orchestrator entry point."""
    t0 = time.monotonic()

    if cfg.run.write_outputs:
        collector.init(cfg.run.outputs_dir)
        collector.collect({"_config": OmegaConf.to_yaml(main_config)})

    set_console_title(cfg.run.outputs_dir)

    block_data = {}

    workers = find_workers(cfg.run.max_workers, cfg.run.trials)
    cfg_dict = to_picklable(cfg)
    main_config_dict = to_picklable(main_config)

    ctx = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    try:
        futures = {
            executor.submit(run_trial, cfg_dict, main_config_dict, i_trial): i_trial
            for i_trial in range(cfg.run.trials)
        }
        for future in as_completed(futures):
            for block_name, data in future.result().items():
                if block_name not in block_data:
                    block_data[block_name] = {
                        "events": [],
                        "states": [],
                        "learning": [],
                        "performance": [],
                        "manifesto": None,
                        "cfg_dict": None,
                    }
                block_data[block_name]["events"].append(data["events"])
                block_data[block_name]["states"].append(data["states"])
                block_data[block_name]["learning"].append(data["learning_df"])
                block_data[block_name]["performance"].append(data["performance_df"])
                block_data[block_name]["manifesto"] = (
                    block_data[block_name]["manifesto"] or data["manifesto"]
                )
                block_data[block_name]["cfg_dict"] = (
                    block_data[block_name]["cfg_dict"] or data["cfg_dict"]
                )
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        for p in executor._processes.values():
            p.kill()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    results = {
        block: {
            "events": pd.concat(data["events"], ignore_index=True),
            "states": pd.concat(data["states"], ignore_index=True),
            "learning_df": pd.concat(data["learning"], ignore_index=True)
            if data["learning"]
            else pd.DataFrame(),
            "manifesto": data["manifesto"],
            "cfg": from_picklable(data["cfg_dict"]),
        }
        for block, data in block_data.items()
    }

    analytics = analyze(results)

    try:
        if cfg.run.write_outputs:
            all_events = [ev for data in block_data.values() for ev in data["events"]]
            all_states = [st for data in block_data.values() for st in data["states"]]
            write_results(cfg.run.outputs_dir, all_events, all_states)

            for block_name, data in block_data.items():
                if data["performance"]:
                    perf_df = pd.concat(data["performance"], ignore_index=True)
                    write_csv(
                        Path(cfg.run.outputs_dir)
                        / block_name
                        / "performance_report.csv",
                        perf_df,
                    )
                    write_cfg(
                        Path(cfg.run.outputs_dir) / block_name,
                        from_picklable(data["cfg_dict"]),
                    )

    finally:
        if cfg.run.write_outputs:
            elapsed = time.monotonic() - t0
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            runtime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
            collector.collect({"_elapsed": runtime_str})
            collector.finalize(cfg.run.outputs_dir)

    return {
        "outputs_dir": cfg.run.outputs_dir,
        "results": results,
        "analytics": analytics,
    }
