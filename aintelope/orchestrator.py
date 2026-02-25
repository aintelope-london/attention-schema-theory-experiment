# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import os

from concurrent.futures import ProcessPoolExecutor, as_completed

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
from aintelope.analytics.recording import write_results


def run_trial(cfg_dict, main_config_dict, i_trial):
    """Run all experiments for a single trial.

    Args must be dicts for multiprocessing pickling.
    Returns dict with configs and events lists.
    """
    cfg = from_picklable(cfg_dict)
    main_config = from_picklable(main_config_dict)

    trial_seed = cfg.run.seed + i_trial
    set_global_seeds(trial_seed)

    configs = []
    all_events = []
    all_states = []

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
        configs.append(experiment_cfg)

    return {"configs": configs, "events": all_events, "states": all_states}


def run_experiments(main_config):
    """Main orchestrator entry point."""

    cfg = init_config(main_config)
    set_console_title(cfg.run.outputs_dir)

    configs = []
    all_events = []
    all_states = []

    workers = find_workers(cfg.run.max_workers, cfg.run.trials)

    cfg_dict = to_picklable(cfg)
    main_config_dict = to_picklable(main_config)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_trial, cfg_dict, main_config_dict, i_trial): i_trial
            for i_trial in range(cfg.run.trials)
        }
        for future in as_completed(futures):
            result = future.result()
            configs.extend(result["configs"])
            all_events.extend(result["events"])
            all_states.extend(result["states"])

    if cfg.run.write_outputs:
        write_results(cfg.run.outputs_dir, all_events, all_states)
        archive_code(cfg)

    return {
        "Outputs_dir": cfg.run.outputs_dir,
        "configs": configs,
        "events": all_events,
        "states": all_states,
    }
