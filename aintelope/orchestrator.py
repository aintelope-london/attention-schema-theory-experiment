# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import copy

import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from omegaconf import OmegaConf

from aintelope.config.config_utils import (
    archive_code,
    get_score_dimensions,
    set_console_title,
)
from aintelope.experiments import run_experiment
from aintelope.utils.seeding import set_global_seeds
from aintelope.utils.progress import ProgressReporter


def find_workers() -> int:
    """Return max available workers (GPUs if available, else CPUs)."""
    gpu_count = torch.cuda.device_count()
    return gpu_count if gpu_count > 0 else os.cpu_count()


def run_trial(cfg_dict, main_config_dict, i_trial):
    """Run all experiments for a single trial.

    Args must be dicts (not OmegaConf) for multiprocessing pickling.
    Returns dict with configs and events lists.
    """
    cfg = OmegaConf.create(cfg_dict)
    main_config = OmegaConf.create(main_config_dict)

    trial_seed = cfg.run.seed + i_trial
    set_global_seeds(trial_seed)

    configs = []
    all_events = []

    for _, experiment_name in enumerate(main_config):
        experiment_cfg = copy.deepcopy(cfg)
        experiment_cfg = OmegaConf.merge(cfg, main_config[experiment_name])
        OmegaConf.update(
            experiment_cfg, "experiment_name", experiment_name, force_add=True
        )
        OmegaConf.update(experiment_cfg.run, "seed", trial_seed, force_add=True)

        score_dimensions = get_score_dimensions(experiment_cfg)
        reporter = ProgressReporter(["episode"], on_update=None)

        events = run_experiment(
            experiment_cfg,
            score_dimensions=score_dimensions,
            i_trial=i_trial,
            reporter=reporter,
        )

        if cfg.run.save_logs:
            block_output_dir = os.path.join(cfg.run.outputs_dir, experiment_name)
            events.write(block_output_dir)

        all_events.append(events.to_dataframe())
        configs.append(experiment_cfg)

    return {"configs": configs, "events": all_events}


def run_experiments(main_config):
    """Main orchestrator entry point."""
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    # resolve timestamp, freeze
    outputs_dir = cfg.run.outputs_dir
    OmegaConf.update(cfg, "run.outputs_dir", outputs_dir)  

    set_console_title(cfg.run.outputs_dir)

    configs = []
    all_events = []

    workers = find_workers()

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    main_config_dict = OmegaConf.to_container(main_config, resolve=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_trial, cfg_dict, main_config_dict, i_trial): i_trial
            for i_trial in range(cfg.run.trials)
        }
        for future in as_completed(futures):
            result = future.result()
            configs.extend(result["configs"])
            all_events.extend(result["events"])

    archive_code(cfg)

    return {
        "Outputs_dir": cfg.run.outputs_dir,
        "configs": configs,
        "events": all_events,
    }
