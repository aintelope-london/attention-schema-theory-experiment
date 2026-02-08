# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import copy
import logging
import sys
import torch
import gc
import json

from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten
from flatten_dict.reducers import make_reducer

from diskcache import Cache

# this one is cross-platform
from filelock import FileLock

from aintelope.utils import RobustProgressBar, Semaphore, wait_for_enter

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    archive_code,
    get_score_dimensions,
    set_console_title,
)
from aintelope.experiments import run_experiment


logger = logging.getLogger("aintelope.__main__")

gpu_count = torch.cuda.device_count()
worker_count_multiplier = 1  # when running orchestrator search, then having more workers than GPU-s will cause all sorts of Python and CUDA errors under Windows for some reason, even though there is plenty of free RAM and GPU memory. Yet, when the orchestrator processes are run manually, there is no concurrency limit except the real hardware capacity limits. # TODO: why?
num_workers = max(1, gpu_count) * worker_count_multiplier


def run_experiments(orchestrator_config):
    """
    extra_cfg: filename, DictConfig or nothing
    """
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    timestamp = str(cfg.timestamp)
    timestamp_pid_uuid = str(cfg.timestamp_pid_uuid)
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"timestamp_pid_uuid: {timestamp_pid_uuid}")

    set_console_title(cfg.hparams.params_set_title + " : " + timestamp_pid_uuid)

    summaries = []
    configs = []

    # use additional semaphore here since the user may launch multiple processes manually
    semaphore_name = (
        "AIntelope_orchestrator_semaphore"
        + (
            "_" + cfg.hparams.params_set_title
            if cfg.hparams.params_set_title in ["handwritten_rules", "random"]
            else ""
        )
        + ("_debug" if sys.gettrace() is not None else "")
    )
    print("Waiting for semaphore...")
    with Semaphore(
        semaphore_name,
        max_count=num_workers,
        disable=(
            os.name != "nt"
            or gpu_count == 0
            or True  # TODO: config flag for disabling the semaphore
        ),  # Linux does not unlock semaphore after a process gets killed, therefore disabling Semaphore under Linux until this gets resolved.
    ) as semaphore:
        print("Semaphore acquired...")
        # In case of 0 orchestrator cycles (num_trials == 0), each environment has its own model. In this case run training and testing inside the same cycle immediately after each other.
        # In case of (num_trials > 0), train a SHARED model over all environments in the orchestrator steps for num_trials. Then test that shared model for one additional cycle.
        # Therefore, the + 1 cycle is for testing. In case of (num_trials == 0), run testing inside the same cycle immediately after each environment's training ends.
        max_trial = cfg.hparams.num_trials + 1
        with RobustProgressBar(
            max_value=max_trial
        ) as trial_bar:  # this is a slow task so lets use a progress bar
            for i_trial in range(0, max_trial):
                # In case of (num_trials == 0), each environment has its own model. In this case run training and testing inside the same cycle immediately after each other.
                # In case of (num_trials > 0), train a SHARED model over all environments in the orchestrator steps for num_trials. Then test that shared model for one additional cycle
                train_mode = (
                    i_trial < cfg.hparams.num_trials or cfg.hparams.num_trials == 0
                )
                test_mode = i_trial == cfg.hparams.num_trials

                with RobustProgressBar(
                    max_value=len(orchestrator_config)
                ) as orchestrator_bar:  # this is a slow task so lets use a progress bar
                    for env_conf_i, env_conf_name in enumerate(orchestrator_config):
                        experiment_cfg = copy.deepcopy(
                            cfg
                        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment

                        experiment_cfg.hparams = OmegaConf.merge(
                            experiment_cfg.hparams, orchestrator_config[env_conf_name]
                        )
                        """
                        OmegaConf.update(
                            experiment_cfg, "experiment_name", env_conf_name
                        )

                        OmegaConf.update(
                            experiment_cfg,
                            "hparams",
                            orchestrator_config[env_conf_name],
                            force_add=True,
                        )
                        """
                        logger.info("Running training with the following configuration")
                        logger.info(
                            os.linesep
                            + str(OmegaConf.to_yaml(experiment_cfg, resolve=True))
                        )

                        # Training
                        params_set_title = experiment_cfg.hparams.params_set_title
                        logger.info(
                            f"params_set: {params_set_title}, experiment: {env_conf_name}"
                        )

                        score_dimensions = get_score_dimensions(experiment_cfg)

                        num_actual_train_episodes = -1
                        if (
                            train_mode and test_mode
                        ):  # In case of (num_trials == 0), each environment has its own model. In this case run training and testing inside the same cycle immediately after each other.
                            num_actual_train_episodes = run_experiment(
                                experiment_cfg,
                                experiment_name=env_conf_name,
                                score_dimensions=score_dimensions,
                                test_mode=False,
                                i_trial=i_trial,
                            )
                        elif test_mode:
                            pass  # TODO: optional: obtain num_actual_train_episodes. But this is not too important: in case of training a model over one or more orchestrator cycles, the final test cycle gets its own i_trial index, therefore it is clearly distinguishable anyway

                        run_experiment(
                            experiment_cfg,
                            experiment_name=env_conf_name,
                            score_dimensions=score_dimensions,
                            test_mode=test_mode,
                            i_trial=i_trial,
                            num_actual_train_episodes=num_actual_train_episodes,
                        )

                        # torch.cuda.empty_cache()
                        # gc.collect()

                        if test_mode:
                            # Not using timestamp_pid_uuid here since it would make the title too long. In case of manual execution with plots, the pid-uuid is probably not needed anyway.
                            title = (
                                timestamp
                                + " : "
                                + params_set_title
                                + " : "
                                + env_conf_name
                            )
                            summary = analytics(
                                experiment_cfg,
                                score_dimensions,
                                title=title,
                                experiment_name=env_conf_name,
                                group_by_trial=cfg.hparams.num_trials >= 1,
                                gridsearch_params=None,
                                show_plot=experiment_cfg.hparams.show_plot,
                            )
                            summaries.append(summary)
                            configs.append(experiment_cfg)

                        orchestrator_bar.update(env_conf_i + 1)

                    # / for env_conf_name in orchestrator_config:
                # / with RobustProgressBar(max_value=len(orchestrator_config)) as orchestrator_bar:

                trial_bar.update(i_trial + 1)

            # / for i_trial in range(0, max_trial):
        # / with RobustProgressBar(max_value=max_trial) as trial_bar:
    # / with Semaphore('name', max_count=num_workers, disable=False) as semaphore:

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

    torch.cuda.empty_cache()
    gc.collect()

    # keep plots visible until the user decides to close the program
    if experiment_cfg.hparams.show_plot:
        # uses less CPU on Windows than input() function. Note that the graph window will be frozen, but will still show graphs
        wait_for_enter("\norchestrator done. Press [enter] to continue.")

    return {"summaries": summaries, "configs": configs}


def analytics(
    cfg,
    score_dimensions,
    title,
    experiment_name,
    group_by_trial,
    gridsearch_params=DictConfig,
    show_plot=False,
):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_trials = cfg.hparams.num_trials

    savepath = os.path.join(log_dir, "plot_" + experiment_name)
    events = recording.read_events(experiment_dir, events_fname)

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
        "title": title,  # timestamp + " : " + params_set_title + " : " + env_conf_name
        "params_set_title": cfg.hparams.params_set_title,
        "gridsearch_params": OmegaConf.to_container(gridsearch_params, resolve=True)
        if gridsearch_params is not None
        else None,  # Object of type DictConfig is not JSON serializable, neither can yaml.dump in plotting.prettyprint digest it, so need to convert it to ordinary dictionary
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
        show_plot=show_plot,
    )

    return test_summary


# if __name__ == "__main__":
#    run()
