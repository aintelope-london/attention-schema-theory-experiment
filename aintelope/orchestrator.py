# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from aintelope.config.config_utils import (
    archive_code,
    set_console_title,
    prepare_experiment_cfg,
    init_config,
    to_picklable,
    from_picklable,
)
from aintelope.experiment import run_experiment
from aintelope.utils.seeding import set_global_seeds
from aintelope.utils.progress import ProgressReporter
from aintelope.utils.concurrency import find_workers
from aintelope.analytics.analytics import analyze, write_analytics
from aintelope.analytics.recording import write_results


def run_trial(cfg_dict, main_config_dict, i_trial):
    """Run all experiments for a single trial.

    Args must be dicts for multiprocessing pickling.
    Returns dict with configs, events, states, learning_dfs, and manifestos.
    """
    cfg = from_picklable(cfg_dict)
    main_config = from_picklable(main_config_dict)

    trial_seed = cfg.run.seed + i_trial
    set_global_seeds(trial_seed)

    configs = []
    all_events = []
    all_states = []
    all_learning = []
    all_manifestos = []

    for _, experiment_name in enumerate(main_config):
        experiment_cfg = prepare_experiment_cfg(
            cfg, main_config[experiment_name], experiment_name, trial_seed
        )
        reporter = ProgressReporter(["episode"], on_update=None)

        result = run_experiment(
            experiment_cfg,
            i_trial=i_trial,
            reporter=reporter,
        )

        all_events.append(result["events"])
        all_states.append(result["states"])
        all_learning.append(result["learning_df"])
        all_manifestos.append(result["manifesto"])
        configs.append(experiment_cfg)

    return {
        "configs": configs,
        "events": all_events,
        "states": all_states,
        "learning": all_learning,
        "manifestos": all_manifestos,
    }


def run_experiments(cfg, main_config):
    """Main orchestrator entry point."""

    set_console_title(cfg.run.outputs_dir)

    configs = []
    all_events = []
    all_states = []
    all_learning = []
    all_manifestos = []

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
            result = future.result()
            configs.extend(result["configs"])
            all_events.extend(result["events"])
            all_states.extend(result["states"])
            all_learning.extend(result["learning"])
            all_manifestos.extend(result["manifestos"])
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        for p in executor._processes.values():
            p.kill()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    combined_events = pd.concat(all_events, ignore_index=True)
    combined_learning = (
        pd.concat(all_learning, ignore_index=True) if all_learning else pd.DataFrame()
    )

    manifesto = all_manifestos[0] if all_manifestos else None
    analytics_result = analyze(cfg, combined_events, combined_learning, manifesto)

    if cfg.run.write_outputs:
        write_results(cfg.run.outputs_dir, all_events, all_states)
        archive_code(cfg)
        write_analytics(analytics_result, folder=cfg.run.outputs_dir)

    return {
        "outputs_dir": cfg.run.outputs_dir,
        "configs": configs,
        "events": all_events,
        "states": all_states,
        "analytics": analytics_result.metrics,
        "result": analytics_result,
    }
