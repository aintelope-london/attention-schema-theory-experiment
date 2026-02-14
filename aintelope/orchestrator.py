# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import copy
import logging
import json
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from omegaconf import OmegaConf

# this one is cross-platform
from filelock import FileLock

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    archive_code,
    get_score_dimensions,
    set_console_title,
)
from aintelope.experiments import run_experiment
from aintelope.utils.seeding import set_global_seeds
from aintelope.utils.progress import ProgressReporter

logger = logging.getLogger("aintelope.__main__")


def find_workers() -> int:
    """Return max available workers (GPUs if available, else CPUs)."""
    gpu_count = torch.cuda.device_count()
    return gpu_count if gpu_count > 0 else os.cpu_count()


def run_trial(cfg_dict, main_config_dict, i_trial):
    """Run all experiments for a single trial.

    Args must be dicts (not OmegaConf) for multiprocessing pickling.
    Returns dict with summaries, configs, and events lists.
    """
    # Reconstruct OmegaConf objects
    cfg = OmegaConf.create(cfg_dict)
    main_config = OmegaConf.create(main_config_dict)

    trial_seed = cfg.hparams.run_params.seed + i_trial
    set_global_seeds(trial_seed)

    summaries = []
    configs = []
    all_events = []

    timestamp = str(cfg.timestamp)

    for _, experiment_name in enumerate(main_config):
        experiment_cfg = copy.deepcopy(cfg)
        # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment

        experiment_cfg.hparams = OmegaConf.merge(
            experiment_cfg.hparams, main_config[experiment_name]
        )
        OmegaConf.update(experiment_cfg.hparams, "seed", trial_seed, force_add=True)

        logger.info("Running training with the following configuration")
        logger.info(os.linesep + str(OmegaConf.to_yaml(experiment_cfg, resolve=True)))

        params_set_title = experiment_cfg.hparams.params_set_title
        logger.info(f"params_set: {params_set_title}, experiment: {experiment_name}")

        score_dimensions = get_score_dimensions(experiment_cfg)
        reporter = ProgressReporter(["episode"], on_update=None)

        events = run_experiment(
            experiment_cfg,
            experiment_name=experiment_name,
            score_dimensions=score_dimensions,
            i_trial=i_trial,
            reporter=reporter,
        )

        if cfg.hparams.run_params.save_logs:
            events.write(os.path.normpath(experiment_cfg.experiment_dir))

        all_events.append(events.to_dataframe())

        # Not using timestamp_pid_uuid here since it would make the title too long. In case of manual execution with plots, the pid-uuid is probably not needed anyway.
        title = timestamp + " : " + params_set_title + " : " + experiment_name
        summary = analytics(
            experiment_cfg,
            score_dimensions,
            title=title,
            experiment_name=experiment_name,
            group_by_trial=cfg.hparams.trials >= 1,
            events=all_events,
        )
        summaries.append(summary)
        configs.append(experiment_cfg)

    return {"summaries": summaries, "configs": configs, "events": all_events}


def run_experiments(main_config):
    """Main orchestrator entry point."""
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    timestamp = str(cfg.timestamp)
    timestamp_pid_uuid = str(cfg.timestamp_pid_uuid)
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"timestamp_pid_uuid: {timestamp_pid_uuid}")

    set_console_title(cfg.hparams.params_set_title + " : " + timestamp_pid_uuid)

    summaries = []
    configs = []
    all_events = []

    workers = find_workers()

    # Convert to dicts for pickling
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    main_config_dict = OmegaConf.to_container(main_config, resolve=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_trial, cfg_dict, main_config_dict, i_trial): i_trial
            for i_trial in range(cfg.hparams.trials)
        }
        # Add gc here somewhere garbacecollector?
        for future in as_completed(futures):
            result = future.result()
            summaries.extend(result["summaries"])
            configs.extend(result["configs"])
            all_events.extend(result["events"])

    # Write the orchestrator results to file only when entire orchestrator has run. Else crashing the program during orchestrator run will cause the aggregated results file to contain partial data which will be later duplicated by re-run.
    # TODO: alternatively, cache the results of each experiment separately
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)
        aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
        with aggregated_results_file_lock:
            with open(aggregated_results_file, mode="a", encoding="utf-8") as fh:
                for summary in summaries:
                    # Do not write directly to file. If JSON serialization error occurs during json.dump() then a broken line would be written into the file (I have verified this). Therefore using json.dumps() is safer.
                    json_text = json.dumps(summary)
                    fh.write(
                        json_text + "\n"
                    )  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present.
                fh.flush()

    archive_code(cfg)

    return {"summaries": summaries, "configs": configs, "events": all_events}


def analytics(
    cfg,
    score_dimensions,
    title,
    experiment_name,
    group_by_trial,
    events,
):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    num_train_episodes = cfg.hparams.episodes
    num_train_trials = cfg.hparams.trials

    savepath = os.path.join(log_dir, "plot_" + experiment_name)

    (
        test_totals,
        test_averages,
        test_variances,
        test_sfella_totals,
        test_sfella_averages,
        test_sfella_variances,
        sfella_score_total,
        sfella_score_average,
        sfella_score_variance,
        score_dimensions_out,
    ) = plotting.aggregate_scores(
        events,
        num_train_trials,
        score_dimensions,
        group_by_trial=group_by_trial,
    )

    test_summary = {
        "timestamp": cfg.timestamp,
        "timestamp_pid_uuid": cfg.timestamp_pid_uuid,
        "experiment_name": experiment_name,
        "title": title,  # timestamp + " : " + params_set_title + " : " + experiment_name
        "params_set_title": cfg.hparams.params_set_title,
        "num_train_trials": num_train_trials,
        "score_dimensions": score_dimensions_out,
        "group_by_trial": group_by_trial,
        "test_totals": test_totals,
        "test_averages": test_averages,
        "test_variances": test_variances,
        # per score dimension results
        "test_sfella_totals": test_sfella_totals,
        "test_sfella_averages": test_sfella_averages,
        "test_sfella_variances": test_sfella_variances,
        # over score dimensions results
        # TODO: rename to test_*
        "sfella_score_total": sfella_score_total,
        "sfella_score_average": sfella_score_average,
        "sfella_score_variance": sfella_score_variance,
    }

    plotting.prettyprint(test_summary)

    plotting.plot_performance(
        events,
        num_train_episodes,
        num_train_trials,
        score_dimensions,
        save_path=savepath,
        title=title,
        group_by_trial=group_by_trial,
    )

    return test_summary


# if __name__ == "__main__":
#    run()
