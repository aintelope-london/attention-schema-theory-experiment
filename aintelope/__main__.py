import copy
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    archive_code,
    get_pipeline_score_dimensions,
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
)
from aintelope.experiments import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:
  
    archive_code(cfg)

    timestamp = str(cfg.timestamp)
    logger.info(f"timestamp: {timestamp}")

    pipeline_config = OmegaConf.load("aintelope/config/config_pipeline.yaml")
    # score_dimensions = get_pipeline_score_dimensions(cfg, pipeline_config)

    for i_pipeline_cycle in range(0, cfg.hparams.num_pipeline_cycles + 1):  # last +1 cycle is for testing

        is_last_pipeline_cycle = (i_pipeline_cycle == cfg.hparams.num_pipeline_cycles)

        for env_conf_name in pipeline_config:
            experiment_cfg = copy.deepcopy(
                cfg
            )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
            OmegaConf.update(experiment_cfg, "experiment_name", env_conf_name)
            OmegaConf.update(
                experiment_cfg, "hparams", pipeline_config[env_conf_name], force_add=True
            )
            logger.info("Running training with the following configuration")
            logger.info(OmegaConf.to_yaml(experiment_cfg))

            # Training
            params_set_title = experiment_cfg.hparams.params_set_title
            logger.info(f"params_set: {params_set_title}, experiment: {env_conf_name}")

            score_dimensions = get_score_dimensions(experiment_cfg)
            run_experiment(experiment_cfg, experiment_name=env_conf_name, score_dimensions=score_dimensions, is_last_pipeline_cycle=is_last_pipeline_cycle, i_pipeline_cycle=i_pipeline_cycle)

            # torch.cuda.empty_cache()
            # gc.collect()

            if is_last_pipeline_cycle:
                title = timestamp + " : " + params_set_title + " : " + env_conf_name
                analytics(experiment_cfg, score_dimensions, title=title, experiment_name=env_conf_name)

        #/ for env_conf_name in pipeline_config:
    #/ for pipeline_cycle_i in range(0, cfg.hparams.num_pipeline_cycles):


    for env_conf_name in pipeline_config:
        experiment_cfg = copy.deepcopy(
            cfg
        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
        OmegaConf.update(experiment_cfg, "experiment_name", env_conf_name)
        OmegaConf.update(
            experiment_cfg, "hparams", pipeline_config[env_conf_name], force_add=True
        )
			
        # Testing
        OmegaConf.update(experiment_cfg, "hparams.traintest_mode", "test")
        params_set_title = experiment_cfg.hparams.params_set_title
        logger.info(f"params_set: {params_set_title}, experiment: {env_conf_name}")

        score_dimensions = get_score_dimensions(experiment_cfg)
        run_experiment(experiment_cfg, score_dimensions)
        analytics(experiment_cfg, score_dimensions, title=title, experiment_name=env_conf_name)




def analytics(cfg, score_dimensions, title, experiment_name):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_pipeline_cycles = cfg.hparams.num_pipeline_cycles

    savepath = os.path.join(log_dir, "plot_" + experiment_name)
    events = recording.read_events(experiment_dir, events_fname)
    plotting.plot_performance(events, num_train_episodes, num_train_pipeline_cycles, score_dimensions, save_path=savepath, title=title)


if __name__ == "__main__":
    register_resolvers()

    set_priorities()
    set_memory_limits() 
    select_gpu()

    aintelope_main()
